import os
import json
import random
import shutil
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

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

_SHUFFLE_RECORDINGS = False
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

    print(f"\nselected {len(midi_paths_train)} train samples")
    print(f"\nselected {len(midi_paths_valid)} valid samples")

# define datasets and dataloaders

kwargs_dataset = {
    "max_seq_len": MAX_SEQ_LEN, 
    "tokenizer": tokenizer,
    "bos_token_id": tokenizer["BOS_None"],
    "eos_token_id": tokenizer["EOS_None"],
}

train_dataset = DatasetMIDI(midi_paths_train, **kwargs_dataset)
val_dataset = DatasetMIDI(midi_paths_valid, **kwargs_dataset)

collator_right_pad = DataCollator(pad_token_id=tokenizer["PAD_None"], pad_on_left=False)
collator_left_pad = DataCollator(pad_token_id=tokenizer["PAD_None"], pad_on_left=True)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator_right_pad))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_left_pad))

# instantiate model

global_attn = None
if USE_GLOBAL_ATTENTION:
    from src.transformer import LocalMHA
    global_attn = LocalMHA(
        dim=DIM,
        window_size=MAX_SEQ_LEN,  # global attention
        dim_head=DIM_HEAD,
        heads=HEADS,
        dropout=ATTN_DROPOUT,
        causal=CAUSAL,
        prenorm=True,
        qk_rmsnorm=True,
        qk_scale=8,
        use_xpos=USE_XPOS,
        exact_windowsize=False
    )

model = LocalTransformer(
    num_tokens = len(tokenizer), # vocab size
    dim = DIM,
    dim_head = DIM_HEAD,
    heads = HEADS,
    ff_mult = FF_MULT,
    depth = DEPTH,
    causal = CAUSAL,
    attn_window_sizes = ATTN_WINDOW_SIZES,
    max_seq_len = MAX_SEQ_LEN,
    use_xpos = USE_XPOS,
    use_dynamic_pos_bias = USE_DYNAMIC_POS_BIAS,
    ignore_index = tokenizer["PAD_None"],
    attn_dropout = ATTN_DROPOUT,
    ff_dropout = FF_DROPOUT,
    conv_dropout = CONV_DROPOUT,
    conv_expansion_factor = CONV_EXPANSION_FACTOR,
    conv_kernel_size = CONV_KERNEL_SIZE,
    global_attn_layer = global_attn,
    layers_insert_global_attn = GLOBAL_ATTN_LAYERS if USE_GLOBAL_ATTENTION else None,
).to(DEVICE)

print(f"\nmodel size: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# optimizer and loss

optim = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = None
if LR_SCHEDULER is not None:
    if LR_SCHEDULER == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optim, T_max=NUM_BATCHES, eta_min=1e-6)
    else:
        raise ValueError(f"Invalid LR_SCHEDULER: {LR_SCHEDULER}")

pad_id = tokenizer["PAD_None"]
nll_sum = lambda logits, labels: F.cross_entropy(
    rearrange(logits, 'b n c -> b c n'),
    labels,
    ignore_index=pad_id,
    reduction='sum'
)
nll_mean = lambda logits, labels: F.cross_entropy(
    rearrange(logits, 'b n c -> b c n'),
    labels,
    ignore_index=pad_id,
    reduction = 'mean'
)

# training

start_step = 0
if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
    print(f"\nresuming from checkpoint: {RESUME_CHECKPOINT}\n")

    ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    # load optimizer and lr scheduler state

    try:
        optim.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception:
        print("optimizer state from checkpoint could not be loaded; continuing with fresh optimizer")
    

    # if scheduler is not None and 'scheduler_state_dict' in ckpt:
    #     try:
    #         scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    #         print("scheduler state loaded from checkpoint")
    #     except Exception:
    #         print("scheduler state from checkpoint could not be loaded; continuing with fresh scheduler")

    start_step = int(ckpt.get('step', 0)) + 1

    # NOTE: for backward compatibility
    if start_step == 1:
        start_step = int(ckpt.get('epoch', 0)) + 1

# constant seed for generation

constant_seed = random.choice(val_dataset)
const_seed_inp = constant_seed['input_ids'][-128:].to(DEVICE)
constant_seed_midi = decode_tokens(const_seed_inp.tolist(), tokenizer)
constant_seed_midi.dump_midi(GEN_DIR / f"0_const_seed.mid")

# training loop

best_val_loss = float('inf')
for i in tqdm(range(start_step, NUM_BATCHES), desc='training'):
    model.train()
    
    total_loss, total_tokens = 0.0, 0
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        inp, mask = next(train_loader)

        inp, labels = inp[:, :-1], inp[:, 1:]
        mask = mask[:, :-1]

        logits = model(inp, mask=mask)

        # accumulate summed NLL and token count
        loss_sum = nll_sum(logits, labels)
        total_loss += loss_sum

        total_tokens += (labels != pad_id).sum()

        # use mean loss to keep gradients invariant to token count
        loss_mean = nll_mean(logits, labels)
        (loss_mean / GRADIENT_ACCUMULATE_EVERY).backward()

    train_avg_nll = total_loss / total_tokens
    train_ppl = torch.exp(train_avg_nll)

    print(f'training loss: {train_avg_nll:.4f}, perplexity: {train_ppl:.4f}')
    
    # log to tensorboard
    writer.add_scalar('Loss/Train', train_avg_nll, i)
    writer.add_scalar('Perplexity/Train', train_ppl, i)

    if MAX_GRAD_NORM is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    optim.step()
    optim.zero_grad()

    if scheduler is not None:
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning_Rate', current_lr, i)

        scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            inp, mask = next(val_loader)
            inp, labels = inp[:, :-1], inp[:, 1:]
            mask = mask[:, :-1]

            logits = model(inp, mask=mask)
            val_loss = nll_mean(logits, labels)
            val_ppl = torch.exp(val_loss)
            
            # log validation metrics

            writer.add_scalar('Loss/Validation', val_loss, i)
            writer.add_scalar('Perplexity/Validation', val_ppl, i)
            print(f"validation loss: {val_loss:.4f}, perplexity: {val_ppl:.4f}")
            
            # save model checkpoint
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'step': i,
                'train_loss': train_avg_nll,
                'val_loss': val_loss,
                'perplexity': val_ppl,
            }
            
            # add scheduler state if scheduler exists

            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            # save model checkpoint

            torch.save(checkpoint, LOG_DIR / f"latest_model.pt")
            if val_loss < best_val_loss:
                print(f"\nsaving best model checkpoint (step {i})...\n")
                best_val_loss = val_loss
                torch.save(checkpoint, LOG_DIR / f'best_model.pt')

    if (i % VALIDATE_ALL_EVERY == 0 and i!=0) or i == start_step:
        print(f"\nvalidating on entire validation set...\n")
        model.eval()
        with torch.no_grad():
            val_epoch_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collator_left_pad
            )

            val_total_loss, val_total_tokens = 0.0, 0
            for batch in tqdm(val_epoch_loader, desc=f"Validation Epoch {i}", leave=False):
                inp = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].bool().to(DEVICE)

                inp, labels = inp[:, :-1], inp[:, 1:]
                mask = mask[:, :-1]

                logits = model(inp, mask=mask)
                loss_sum = nll_sum(logits, labels)
                val_total_loss += loss_sum
                val_total_tokens += (labels != pad_id).sum()

            val_avg_nll = val_total_loss / val_total_tokens
            val_ppl = torch.exp(val_avg_nll)

            print(f"validation loss: {val_avg_nll:.4f}, perplexity: {val_ppl:.4f}")

            # log validation metrics

            writer.add_scalar('Loss/Validation', val_avg_nll, i)
            writer.add_scalar('Perplexity/Validation', val_ppl, i)
            
            # save model checkpointh

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'step': i,
                'train_loss': train_avg_nll,
                'val_loss': val_avg_nll,
                'perplexity': val_ppl,
            }
            
            # add scheduler state

            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            if val_avg_nll < best_val_loss:
                best_val_loss = val_avg_nll
                torch.save(checkpoint, LOG_DIR / f'best_model.pt')


    if i % GENERATE_EVERY == -1: # NOTE: disabled for now
        model.eval()
        with torch.no_grad():
            for temp in TEMPERATURES:

                # generate with constant seed

                gen_const = model.generate(
                    const_seed_inp[None, ...],
                    GENERATE_LENGTH,
                    temperature=temp,
                    filter_thres=FILTER_THRES,
                    use_kv_cache=False,
                )

                gen_const_midi = decode_tokens(gen_const[0].tolist(), tokenizer)
                gen_const_midi.dump_midi(GEN_DIR / f"{i}_{temp}_const_generated.mid")

                combined_const = torch.cat((const_seed_inp, gen_const[0]))
                combined_const_midi = decode_tokens(combined_const.tolist(), tokenizer)
                combined_const_midi.dump_midi(GEN_DIR / f"{i}_{temp}_const_combined.mid")

                # generate with random seed

                seed = random.choice(val_dataset)
                seed_inp = seed['input_ids'][-128:].to(DEVICE)
                
                generated = model.generate(
                    seed_inp[None, ...],
                    GENERATE_LENGTH,
                    temperature=temp,
                    filter_thres=FILTER_THRES,
                    use_kv_cache=False,
                )

                combined = torch.cat((seed_inp, generated[0]))

                seed_midi = decode_tokens(seed_inp.tolist(), tokenizer)
                generated_midi = decode_tokens(generated[0].tolist(), tokenizer)
                combined_midi = decode_tokens(combined.tolist(), tokenizer)

                seed_midi.dump_midi(GEN_DIR / f"{i}_{temp}_seed.mid")
                generated_midi.dump_midi(GEN_DIR / f"{i}_{temp}_generated.mid")
                combined_midi.dump_midi(GEN_DIR / f"{i}_{temp}_combined.mid")

                print(f"\ngeneration complete w/ temperature: {temp}\n")

writer.close()
