from miditok import REMI, TokenizerConfig
from pathlib import Path

# tokenizer's configuration

MIDI_PATHS = list(Path("data/maestro-v3.0.0").resolve().glob("*/*.mid?"))
TOKENIZER_PATH = Path("tokenizer.json").resolve()
VOCAB_SIZE = 30000

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
    "use_programs": False,  # no multitrack here
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# create tokenizer
tokenizer = REMI(config)

# train the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary, here 30k tokens
tokenizer.train(
    vocab_size=VOCAB_SIZE,
    files_paths=MIDI_PATHS,
)
tokenizer.save_params(TOKENIZER_PATH)