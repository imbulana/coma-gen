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

```

## Usage

Adjust training params in [`train.py`](/train.py) and begin a training with

```bash
python3 train.py
```

Tensorboard logs will be saved in the `LOG_DIR` directory.

## References

local attention transformer: https://github.com/lucidrains/local-attention