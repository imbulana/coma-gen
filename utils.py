import json

from config import *

# data utils

def build_config_dict():
    return {
        "SEED": SEED,
        "DEVICE": str(DEVICE),

        "SPLIT_DATA": SPLIT_DATA,
        "SHUFFLE": SHUFFLE,
        "TEST_SIZE": TEST_SIZE,
        "TOP_K_COMPOSERS": TOP_K_COMPOSERS,
        "TO_SKIP": TO_SKIP,
        "AUGMENT_DATA": AUGMENT_DATA,

        "TRAIN_TOKENIZER": TRAIN_TOKENIZER,
        "VOCAB_SIZE": VOCAB_SIZE,

        "BEAT_RES": str(BEAT_RES),
        "TOKENIZER_PARAMS": {k: str(v) for k, v in TOKENIZER_PARAMS.items()},

        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,

        "MODEL_PARAMS": {
            "DIM": DIM,
            "DIM_HEAD": DIM_HEAD,
            "HEADS": HEADS,
            "FF_MULT": FF_MULT,
            "DEPTH": DEPTH,
            "CAUSAL": CAUSAL,
            "USE_DYNAMIC_POS_BIAS": USE_DYNAMIC_POS_BIAS,
            "ATTN_WINDOW_SIZES": ATTN_WINDOW_SIZES,
            "CONV_EXPANSION_FACTOR": CONV_EXPANSION_FACTOR,
            "CONV_KERNEL_SIZE": CONV_KERNEL_SIZE,
            "CONV_DROPOUT": CONV_DROPOUT,
            "FF_DROPOUT": FF_DROPOUT,
            "ATTN_DROPOUT": ATTN_DROPOUT,
            "USE_XPOS": USE_XPOS,
            "USE_DYNAMIC_POS_BIAS": USE_DYNAMIC_POS_BIAS,
        },

        "GENERATE_PARAMS": {
            "TEMPERATURE": TEMPERATURE,
            "FILTER_THRES": FILTER_THRES,
        },
    }

def save_config(writer):
    writer.add_text("config", json.dumps(build_config_dict(), indent=2))
