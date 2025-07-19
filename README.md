## Setup

Create a conda environment with python 3.11

```bash
conda create -n coma python=3.11
conda activate coma
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