import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
from einops import rearrange
from sklearn.model_selection import train_test_split

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

from torch.utils.tensorboard import SummaryWriter

from src import LocalTransformer

# load config

from config import *
from utils import save_config

writer = SummaryWriter(log_dir=LOG_DIR)
save_config(writer)

# create log dir

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GEN_DIR, exist_ok=True)

# set seed

random.seed(SEED)
torch.manual_seed(SEED)

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield (
                data['input_ids'].to(DEVICE),
                data['attention_mask'].bool().to(DEVICE)
            )

def decode_tokens(tokens, tokenizer):
    return tokenizer([tokens])

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def extract_pitches_from_tokens(tokens, tokenizer):
    pass

# load/train tokenizer and maestro data

if USE_PRETRAINED_TOKENIZER and TOKENIZER_LOAD_PATH.exists():
    print(f"\nloading pretrained tokenizer from {TOKENIZER_LOAD_PATH}\n")
    tokenizer = REMI(params=TOKENIZER_LOAD_PATH)
else:
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    print(f"\nusing base tokenizer with vocab size: {len(tokenizer)}\n")

if SPLIT_DATA:

    split_parent_path = MAESTRO_DATA_PATH / "splits"
    df = pd.read_csv(MAESTRO_CSV)

    if SORT_BY == 'compositions':
        # select top k composers by number of compositions
        composer_n_compositions = df.groupby('canonical_composer')['canonical_title'].nunique()
        top_k_composers = composer_n_compositions.sort_values(ascending=False).head(TOP_K_COMPOSERS).index

    elif SORT_BY == 'duration':
        # select top k composers by duration
        composer_duration = df.groupby('canonical_composer')['duration'].sum()
        top_k_composers = composer_duration.sort_values(ascending=False).head(TOP_K_COMPOSERS).index
    else:
        raise ValueError(f"Invalid sort_by: {SORT_BY}. Must be 'compositions' or 'duration'.")

    if TO_SKIP:
        print(f"\nskipping composers: {TO_SKIP}\n")
        top_k_composers = top_k_composers[~top_k_composers.isin(TO_SKIP)]

    df = df[df['canonical_composer'].isin(top_k_composers)]
    print(f"\nselected {len(top_k_composers)} composers: {top_k_composers.tolist()}\n")

    # shuffle data (ensure that no composition is shared b/w train and valid)

    if SHUFFLE:
        print("\nshuffling data...\n")

        for composer, df_composer in df.groupby('canonical_composer'):

            titles = df_composer['canonical_title'].unique()
            titles_train, titles_test = train_test_split(
                titles, test_size=TEST_SIZE, random_state=SEED, shuffle=True
            )

            df.loc[df['canonical_title'].isin(titles_train), 'split'] = 'train'
            df.loc[df['canonical_title'].isin(titles_test), 'split'] = 'validation'

            assert set(titles_train) & set(titles_test) == set(), 'overlapping titles b/w train and test'

        # # save new split summary

        # split_summary = plot_data_split(df, LOG_DIR)
        # print("\n# compositions per composer:\n")
        # print(split_summary)

    # train tokenizer

    if TRAIN_TOKENIZER:
        print(f"\ntraining tokenizer to target vocab size: {VOCAB_SIZE}\n")

        train_paths = df[df['split'] == 'train']['midi_filename'].apply(
            lambda x: str(MAESTRO_DATA_PATH / x)
        ).tolist()

        # train the tokenizer with Byte Pair Encoding to build the vocabulary

        tokenizer.train(
            vocab_size=VOCAB_SIZE,
            files_paths=train_paths,
        )
        tokenizer.save(TOKENIZER_SAVE_PATH)

    # remove existing split

    if split_parent_path.exists():
        shutil.rmtree(split_parent_path)
        print(f"\nremoved existing split: {split_parent_path}\n")

    # create new split

    print("\ncreating new split...\n")

    failures = []
    for split, df_split in df.groupby("split"):
        split_path = split_parent_path / split

        for composer, df_composer in df_split.groupby("canonical_composer"):
            # if any(skip_composer in str(composer) for skip_composer in TO_SKIP):
            #     continue

            composer_path = split_path / composer
            midi_file_paths = df_composer["midi_filename"].apply(
                lambda x: MAESTRO_DATA_PATH / x
            ).tolist()
            
            for midi_file_path in midi_file_paths:
                try:
                    split_files_for_training(
                        files_paths=[midi_file_path],
                        tokenizer=tokenizer,
                        save_dir=composer_path / midi_file_path.name,
                        max_seq_len=MAX_SEQ_LEN,
                        num_overlap_bars=2,
                    )
                except FileNotFoundError:
                    failures.append(midi_file_path)
                    continue

    if failures:
        print(f"\nfailed to split {len(failures)} files")
        print(failures)

leaf = lambda split : f"splits/{split}/*/*/*.mid?"
get_composer_label = lambda dummy1, dummy2, x: x.parent.parent.name # signature expected by DatasetMIDI

midi_paths_train = list(MAESTRO_DATA_PATH.glob(leaf("train")))
midi_paths_valid = list(MAESTRO_DATA_PATH.glob(leaf("validation")))
midi_paths_test = list(MAESTRO_DATA_PATH.glob(leaf("test")))

if AUGMENT_DATA:
    augment_dataset(
        MAESTRO_DATA_PATH / "splits" / "train",
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
    )

    # augments = list(set(MAESTRO_DATA_PATH.glob(leaf("train"))) - set(midi_paths_train))

    # df_augments = pd.DataFrame(augments, columns=["path"])
    # df_augments['composer'] = df_augments['path'].apply(lambda x: x.parent.parent.name)

    midi_paths_train = list(MAESTRO_DATA_PATH.glob(leaf("train")))
    print(f"\ntrain samples (augmentations added): {len(midi_paths_train)}")

# NOTE: for testing only
# split individual recordings instead of compositions 
# note that there are multiple recordings of the same composition in the MAESTRO dataset
_SHUFFLE_RECORDINGS = True
if _SHUFFLE_RECORDINGS:
    print("\nshuffling recordings...\n")
    all_paths = midi_paths_train + midi_paths_valid + midi_paths_test
    df_all = pd.DataFrame(all_paths, columns=["path"])
    df_all['composer'] = df_all['path'].apply(lambda x: x.parent.parent.name)
    X_train, X_test = train_test_split(
        df_all['path'], stratify=df_all['composer'], random_state=SEED, shuffle=True
    )

    midi_paths_train = X_train.tolist()
    midi_paths_valid = X_test.tolist()

# define datasets and dataloaders

# get_composer_label = (
#     lambda dummy1, dummy2, x: x.parent.name # signature expected by DatasetMIDI
#     if LABEL_COMPOSER else None
# )

kwargs_dataset = {
    "max_seq_len": MAX_SEQ_LEN, 
    "tokenizer": tokenizer,
    "bos_token_id": tokenizer["BOS_None"],
    "eos_token_id": tokenizer["EOS_None"],
    # "func_to_get_labels": get_composer_label
}

train_dataset = DatasetMIDI(midi_paths_train, **kwargs_dataset)
val_dataset = DatasetMIDI(midi_paths_valid, **kwargs_dataset)

collator_right_pad = DataCollator(pad_token_id=tokenizer["PAD_None"], pad_on_left=False)
collator_left_pad = DataCollator(pad_token_id=tokenizer["PAD_None"], pad_on_left=True)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator_right_pad))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_left_pad))

# instantiate model

model = LocalTransformer(
    num_tokens = len(tokenizer), # vocab size
    dim = DIM,
    depth = DEPTH,
    causal = CAUSAL,
    local_attn_window_size = ATTN_WINDOW_SIZE,
    # local_attn_window_sizes = ATTN_WINDOW_SIZES,
    max_seq_len = MAX_SEQ_LEN,
    use_dynamic_pos_bias = USE_DYNAMIC_POS_BIAS,
    ignore_index = tokenizer["PAD_None"]
).to(DEVICE)

print(f"\nmodel size: {sum(p.numel() for p in model.parameters()):,}")

# optimizer and loss

optim = Adam(model.parameters(), lr=LEARNING_RATE) # TODO: try AdamW
criterion = lambda logits, labels: F.cross_entropy(
    rearrange(logits, 'b n c -> b c n'),
    labels,
    ignore_index = tokenizer["PAD_None"]
)

# training

constant_seed = random.choice(val_dataset)['input_ids'].to(DEVICE)
constant_seed_midi = decode_tokens(constant_seed.tolist(), tokenizer)
constant_seed_midi.dump_midi(GEN_DIR / f"0_const_seed.mid")

best_val_loss = float('inf')
for i in tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    
    train_loss = 0.
    # train_melody_consistency_total = 0.

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        inp, mask = next(train_loader)

        inp, labels = inp[:, :-1], inp[:, 1:]
        mask = mask[:, :-1]

        logits = model(inp, mask=mask)

        loss = criterion(logits, labels)
        loss.backward()

        train_loss += loss.item()

        # filtered_logits = top_k(logits[:, -1], thres = FILTER_THRES)

        # if TEMPERATURE == 0.:
        #     sampled = filtered_logits.argmax(dim = -1, keepdim = True)
        # else:
        #     probs = F.softmax(filtered_logits / TEMPERATURE, dim = -1)
        #     sampled = torch.multinomial(probs, 1)

        # train_melody_consistency_total += calculate_melody_consistency_from_logits(logits, labels, tokenizer)
    
    train_loss /= GRADIENT_ACCUMULATE_EVERY
    # train_melody_consistency = train_melody_consistency_total / GRADIENT_ACCUMULATE_EVERY

    print(f'training loss: {train_loss:.4f}, perplexity: {torch.exp(torch.tensor(train_loss)).item():.4f}')
    
    # log to tensorboard
    writer.add_scalar('Loss/Train', train_loss, i)
    writer.add_scalar('Perplexity/Train', torch.exp(torch.tensor(train_loss)).item(), i)
    
    # writer.add_scalar('Melody_Consistency/Train', train_melody_consistency, i)
    # print(f'training melody consistency: {train_melody_consistency:.4f}')
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            inp, mask = next(val_loader)
            inp, labels = inp[:, :-1], inp[:, 1:]
            mask = mask[:, :-1]

            logits = model(inp, mask=mask)
            val_loss = criterion(logits, labels)
            
            # val_melody_consistency = calculate_batch_melody_consistency(inp, model, tokenizer)
            # print(f'validation melody consistency: {val_melody_consistency:.4f}')
            
            # Log validation metrics
            writer.add_scalar('Loss/Validation', val_loss.item(), i)
            writer.add_scalar('Perplexity/Validation', torch.exp(val_loss).item(), i)
            # writer.add_scalar('Melody_Consistency/Validation', val_melody_consistency, i)
            
            # Save model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'epoch': i,
                'train_loss': train_loss,  # Correct training loss
                'val_loss': val_loss.item(),  # Correct validation loss
                'perplexity': torch.exp(val_loss).item(),
                # 'val_melody_consistency': val_melody_consistency
            }

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(checkpoint, LOG_DIR / f'best_model.pt')


    if i % GENERATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            # generate with constant seed

            gen_const = model.generate(constant_seed[None, ...], GENERATE_LENGTH, temperature=TEMPERATURE)
            gen_const_midi = decode_tokens(gen_const[0].tolist(), tokenizer)
            gen_const_midi.dump_midi(GEN_DIR / f"{i}_const_generated.mid")

            combined_const = torch.cat((constant_seed, gen_const[0]))
            combined_const_midi = decode_tokens(combined_const.tolist(), tokenizer)
            combined_const_midi.dump_midi(GEN_DIR / f"{i}_const_combined.mid")

            # melody_consistency_const = calculate_melody_consistency_score(
            #     constant_seed.tolist(), gen_const[0].tolist(), tokenizer
            # )

            # generate with random seed

            seed = random.choice(val_dataset)['input_ids'].to(DEVICE)
            generated = model.generate(seed[None, ...], GENERATE_LENGTH, temperature=TEMPERATURE)
            combined = torch.cat((seed, generated[0]))

            seed_midi = decode_tokens(seed.tolist(), tokenizer)
            generated_midi = decode_tokens(generated[0].tolist(), tokenizer)
            combined_midi = decode_tokens(combined.tolist(), tokenizer)

            seed_midi.dump_midi(GEN_DIR / f"{i}_seed.mid")
            generated_midi.dump_midi(GEN_DIR / f"{i}_generated.mid")
            combined_midi.dump_midi(GEN_DIR / f"{i}_combined.mid")

            print("\ngeneration complete\n")

writer.close()
