# Installation

Claw-R1 uses [veRL](https://github.com/volcengine/verl) as its training backend and [Ray](https://docs.ray.io/) for distributed execution.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Conda recommended |
| CUDA | 11.8+ | Required for GPU training |
| Conda / Mamba | latest | For environment isolation |
| Git | any | For cloning repositories |

## Step 1 — Create the Environment

```bash
conda create -n clawr1 python=3.10 -y
conda activate clawr1
```

## Step 2 — Install veRL

Claw-R1 requires the nightly version of veRL installed from source:

```bash
git clone https://github.com/volcengine/verl && cd verl
pip install --no-deps -e .
cd ..
```

## Step 3 — Install Claw-R1

```bash
git clone https://github.com/AgentR1/Claw-R1 && cd Claw-R1
pip install -e .
```

## Step 4 — Install Ray and FastAPI

```bash
pip install "ray[default]" fastapi uvicorn
```

## Verify Installation

Run the following checks to ensure everything is installed correctly:

```bash
# Check veRL
python -c "import verl; print('veRL:', verl.__version__)"

# Check Ray
python -c "import ray; print('Ray:', ray.__version__)"

# Check Claw-R1 gateway
python -m claw_r1.gateway.gateway --help
```

!!! tip "GPU Memory"
    The Gateway Server is CPU-only and lightweight. Rollout workers (vLLM) and Training workers require separate GPU pools. See [Async Training](../components/async-training.md) for multi-GPU configuration.

## What's Next

- [Quick Start](quickstart.md) — run your first training loop
- [Configuration Reference](../configuration/index.md) — customize the setup for your hardware
