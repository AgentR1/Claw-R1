# Installation

Claw-R1 relies on [verl](https://github.com/volcengine/verl) for the training backend. Follow the steps below to set up your environment.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment management)
- Python 3.10+
- CUDA (for GPU training)

## Setup

### 1. Create a Conda Environment

```bash
conda create -n clawr1 python=3.10 -y
conda activate clawr1
```

### 2. Clone and Install veRL

Install the nightly version of veRL from source (recommended):

```bash
git clone https://github.com/volcengine/verl && cd verl
pip install --no-deps -e .
cd ..
```

## Verify Installation

After installation, ensure your environment is ready:

```bash
python -c "import verl; print('veRL installed successfully')"
```
