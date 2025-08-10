import torch
from pathlib import Path
from datetime import datetime

SEED = 42
DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

# data paths

MAESTRO_DATA_PATH = Path("data/maestro-v3.0.0").resolve()
MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0.csv"
# MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0_standardized.csv"

# data splitting

SPLIT_DATA = False
MAX_SEQ_LEN = 1024
SHUFFLE = True
SORT_BY = 'compositions' # must be in ['compositions', 'duration']
TEST_SIZE = 0.1
TOP_K_COMPOSERS = 5 # select top K composers by SORT_BY type to train/test on
TO_SKIP = [] # composers to skip
AUGMENT_DATA = False

# tokenizer

USE_PRETRAINED_TOKENIZER = False
TOKENIZER_LOAD_PATH = Path("tokenizer.json").resolve() # pretrained tokenizer path

TRAIN_TOKENIZER = False # whether to train the tokenizer to a target vocab size with byte pair encoding
VOCAB_SIZE = 5000 # target vocab size for tokenizer training
TOKENIZER_SAVE_PATH = Path("tokenizer.json").resolve()

BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}

# model

DIM = 512
DIM_HEAD = 64
HEADS = 8
FF_MULT = 4
DEPTH = 4
CAUSAL = True
USE_XPOS = True
USE_DYNAMIC_POS_BIAS = False
ATTN_WINDOW_SIZES = [32, 64]

CONV_EXPANSION_FACTOR = 2
CONV_KERNEL_SIZE = 31

ATTN_DROPOUT = 0.1
FF_DROPOUT = 0.1
CONV_DROPOUT = 0.1

# training

NUM_BATCHES = int(1e5)
BATCH_SIZE = 8

GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY = 150 # 1 epoch = NUM_BATCHES / (BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY)
VALIDATE_ALL_EVERY = int(1e5)

GENERATE_EVERY  = 50
GENERATE_LENGTH = 512

LEARNING_RATE = 2e-4
LR_SCHEDULER = None # must be in ["CosineAnnealingLR", "MultiStepLR", None]
# MILESTONES = [7, 14] # for MultiStepLR
# WEIGHT_DECAY = 2e-4 # 4e-4

MAX_GRAD_NORM = None # for gradient clipping, set to None to disable

# tensorboard logs

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs") / f"k={TOP_K_COMPOSERS}_{datetime_str}"
GEN_DIR = LOG_DIR / "gen"

# generator params

TEMPERATURE = .9
FILTER_THRES = .9

# checkpointing / resume
# Set to a checkpoint path to resume training, e.g.
# RESUME_CHECKPOINT = Path("logs/k=5_20250101_120000/best_model.pt").resolve()
RESUME_CHECKPOINT = None