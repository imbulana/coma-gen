import os
import random
import tqdm
import pandas as pd
import shutil

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src import LocalTransformer

from pathlib import Path

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

# constants

MAESTRO_DATA_PATH = Path("data/maestro-v3.0.0").resolve()
MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0.csv"

# SPLIT_DATA = True if not os.path.exists(MAESTRO_DATA_PATH / "splits") else False
SPLIT_DATA = False
LABEL_COMPOSER = False # for classification task
AUGMENT_DATA = False

GEN_PATH = Path("generated").resolve()
GEN_PATH.mkdir(exist_ok=True)

LOG_DIR = Path("logs").resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR.mkdir(exist_ok=True)

NUM_BATCHES = int(1e4)
BATCH_SIZE = 8
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 150
GENERATE_EVERY  = 50
GENERATE_LENGTH = 512
MAX_SEQ_LEN = 1024
DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

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
    return tokenizer.decode([tokens])

# load tokenizer and maestro data

tokenizer = REMI(params=Path("tokenizer.json"))

if SPLIT_DATA:
    split_parent_path = MAESTRO_DATA_PATH / "splits"
    df = pd.read_csv(MAESTRO_CSV)
    for split, df_split in df.groupby("split"):
        split_path = split_parent_path / split

        if split_path.exists():
            shutil.rmtree(split_path)
            print(f"removed existing split: {split_path}")

        split_path.mkdir(parents=True, exist_ok=True)

        if LABEL_COMPOSER:
            for composer, df_composer in df_split.groupby("canonical_composer"):
                composer_path = split_path / composer
                os.makedirs(composer_path, exist_ok=True)

                midi_file_paths = df_composer["midi_filename"].apply(
                    lambda x: MAESTRO_DATA_PATH / x
                ).tolist()

                split_files_for_training(
                    files_paths=midi_file_paths,
                    tokenizer=tokenizer,
                    save_dir=composer_path,
                    max_seq_len=MAX_SEQ_LEN,
                    num_overlap_bars=2,
                )
        else:
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

# instantiate GPT-like decoder model

model = LocalTransformer(
    num_tokens = len(tokenizer), # vocab size
    dim = 512,
    depth = 6,
    causal = True,
    local_attn_window_size = 256,
    max_seq_len = MAX_SEQ_LEN,
    use_dynamic_pos_bias = True,
    ignore_index = tokenizer["PAD_None"]
).to(DEVICE)

print(f"\nmodel size: {sum(p.numel() for p in model.parameters()):,}")

# optimizer

optim = Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        inp, mask = next(train_loader)
        loss = model(inp, mask=mask, return_loss=True)
        loss.backward()

    print(f'training loss: {loss.item():.4f}, perplexity: {torch.exp(loss).item():.4f}')
    
    # log to tensorboard
    writer.add_scalar('Loss/Train', loss.item(), i)
    writer.add_scalar('Perplexity/Train', torch.exp(loss).item(), i)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    best_val_loss = float('inf')
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            inp, mask = next(val_loader)
            loss = model(inp, mask=mask, return_loss=True)
            print(f'validation loss: {loss.item():.4f}, perplexity: {torch.exp(loss).item():.4f}')
            
            # Log validation metrics
            writer.add_scalar('Loss/Validation', loss.item(), i)
            writer.add_scalar('Perplexity/Validation', torch.exp(loss).item(), i)
            
            # Save model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'epoch': i,
                'train_loss': loss.item(),
                'val_loss': loss.item(),
                'perplexity': torch.exp(loss).item()
            }

            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                torch.save(checkpoint, LOG_DIR / f'checkpoint_{i}.pt')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)['input_ids'].to(DEVICE)
        prime = decode_tokens(inp.tolist(), tokenizer)
        print('\nprime:', prime)

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        combined = torch.cat((inp, sample[0]))

        sample_decoded = decode_tokens(sample[0].tolist(), tokenizer)
        sample_decoded.dump_midi(GEN_PATH / f'{i}_sample.mid')

        combined_decoded = decode_tokens(combined.tolist(), tokenizer)
        combined_decoded.dump_midi(GEN_PATH / f'{i}_combined.mid')
        print('generated:', combined_decoded, '\n')

writer.close()
