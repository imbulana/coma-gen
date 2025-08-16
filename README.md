Conformer with multi-scale local attention for symbolic music generation.

Model Architecture (see [`src/transformer.py`](src/transformer.py)):

- **Embedding**: REMI token embedding + learned positional embedding

- **Decoder**: Stack of conformer-like blocks[^1] (1/2 * FeedForward → Multi-Scale Local Attention → Conformer Conv Module → 1/2 * FeedForward) blocks with hyper-connections and residual streams:
    - **Local Attention**: Multi-scale local self-attention with multiple window sizes (e.g., [32, 64]).
        - Each scale uses windowed attention (not full sequence) with optional rotary position embeddings (xpos) or dynamic position bias
        - Scales aggregated via learnable weighted sum
        - Query-Key RMSNorm with learnable scales for improved training stability
    - **Conformer Conv Module**: 
    - LayerNorm → Pointwise conv (1D, expansion factor 2) → GLU activation → Depthwise conv (causal) → Swish → Channel LayerNorm → Pointwise conv → Dropout

    - **Global Attention**: Optional global attention layers can be inserted at specified positions (disabled by default)
    - **Hyper-connections**: Each component wrapped with residual stream expansion/reduction functions

- **Output**: LayerNorm → Linear projection to vocabulary size

## Setup

Create a conda environment with python 3.11

```bash
conda create -n coma-gen python=3.11
conda activate coma-gen
```

Install requirements

```bash
pip install -r requirements.txt
```

## Dataset

Download the Maestro 3.0 dataset

```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip 'maestro-v3.0.0-midi.zip'
rm 'maestro-v3.0.0-midi.zip'
mv 'maestro-v3.0.0' 'data/maestro-v3.0.0'
```

## Usage

Train the tokenizer with

```bash
python3 train_tokenizer.py
```

Adjust training params in [`train.py`](/train.py) and begin training the transformer with

```bash
python3 train.py
```

Tensorboard logs will be saved in the `LOG_DIR` directory.

## References

local attention transformer: https://github.com/lucidrains/local-attention

conformer: https://github.com/jreremy/conformer, https://github.com/lucidrains/conformer