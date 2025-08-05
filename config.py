from pathlib import Path
from datetime import datetime
import torch

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

NUM_BATCHES = int(1e5)
BATCH_SIZE = 8
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 150 # ~1 epoch for maestro
GENERATE_EVERY  = 25
GENERATE_LENGTH = 1024
MAX_SEQ_LEN = 1024
DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}

# generate params

TEMPERATURE = 0.7
FILTER_THRES = 0.9