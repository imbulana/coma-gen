import os
import random
import tqdm
import pandas as pd
import shutil

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src import LocalTransformer

from pathlib import Path

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

# load config

from config import *

writer = SummaryWriter(log_dir=LOG_DIR)

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

# load tokenizer and maestro data

config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

if SPLIT_DATA:
    split_parent_path = MAESTRO_DATA_PATH / "splits"
    df = pd.read_csv(MAESTRO_CSV)
    for split, df_split in df.groupby("split"):
        split_path = split_parent_path / split

        if split_path.exists():
            shutil.rmtree(split_path)
            print(f"removed existing split: {split_path}")

        split_path.mkdir(parents=True, exist_ok=True)

        midi_file_paths = df_split["midi_filename"].apply(
            lambda x: MAESTRO_DATA_PATH / x
        ).tolist()

        split_files_for_training(
            files_paths=midi_file_paths,
            tokenizer=tokenizer,
            save_dir=split_path,
            max_seq_len=MAX_SEQ_LEN,
            num_overlap_bars=2,
        )


midi_paths_train = list(MAESTRO_DATA_PATH.glob("splits/train/*/*.mid?"))
midi_paths_valid = list(MAESTRO_DATA_PATH.glob("splits/validation/*/*.mid?"))
midi_paths_test = list(MAESTRO_DATA_PATH.glob("splits/test/*/*.mid?"))

print(f"\ntrain samples: {len(midi_paths_train)}")
print(f"valid samples: {len(midi_paths_valid)}")
print(f"test samples: {len(midi_paths_test)}")

if AUGMENT_DATA:
    augment_dataset(
        MAESTRO_DATA_PATH / "splits" / "train",
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
    )

    midi_paths_train = list(MAESTRO_DATA_PATH.glob("splits/train/*/*.mid?"))
    print(f"\ntrain samples (augmentations added): {len(midi_paths_train)}")

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
    dim = 144,
    depth = 4,
    causal = True,
    local_attn_window_size = 64,
    # local_attn_window_sizes = ATTN_WINDOW_SIZES,
    max_seq_len = MAX_SEQ_LEN,
    use_dynamic_pos_bias = True,
    ignore_index = tokenizer["PAD_None"]
).to(DEVICE)

print(f"\nmodel size: {sum(p.numel() for p in model.parameters()):,}")

# optimizer

optim = Adam(model.parameters(), lr=LEARNING_RATE) # TODO: try AdamW

# training

constant_seed = random.choice(val_dataset)['input_ids'].to(DEVICE)
constant_seed_midi = decode_tokens(constant_seed.tolist(), tokenizer)
constant_seed_midi.dump_midi(GEN_PATH / f"0_const_seed.mid")

best_val_loss = float('inf')
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    
    train_loss = 0.
    # train_melody_consistency_total = 0.

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        inp, mask = next(train_loader)

        inp, labels = inp[:, :-1], inp[:, 1:]
        mask = mask[:, :-1]

        logits = model(inp, mask=mask)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = tokenizer["PAD_None"]
        )

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
            val_loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                labels,
                ignore_index = tokenizer["PAD_None"]
            )
            
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
            gen_const_midi.dump_midi(GEN_PATH / f"{i}_const_generated.mid")

            combined_const = torch.cat((constant_seed, gen_const[0]))
            combined_const_midi = decode_tokens(combined_const.tolist(), tokenizer)
            combined_const_midi.dump_midi(GEN_PATH / f"{i}_const_combined.mid")

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

            seed_midi.dump_midi(GEN_PATH / f"{i}_seed.mid")
            generated_midi.dump_midi(GEN_PATH / f"{i}_generated.mid")
            combined_midi.dump_midi(GEN_PATH / f"{i}_combined.mid")

            print("\ngeneration complete\n")

writer.close()
