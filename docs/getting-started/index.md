# Getting Started

Welcome to the Claw-R1 documentation. This section guides you from installation to your first training run.

<div class="grid cards" markdown>

-   :material-package-down:{ .lg .middle } **Installation**

    ---

    Set up your conda environment, install veRL, and get Claw-R1 running in minutes.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run your first white-box or black-box agent training with a minimal working example.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

</div>

## Prerequisites

Before you begin, make sure you have:

- A machine with one or more NVIDIA GPUs (CUDA required for training)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/) for environment management
- Python 3.10 or higher
- Git

## Architecture at a Glance

Claw-R1 separates concerns into three independent processes that communicate over the network:

```
Agent (any HTTP client)
    │
    │  POST /v1/chat/completions  (OpenAI-compatible)
    ▼
Gateway Server  ──── Ray RPC ────►  DataPool (Ray Actor)
                                         │
                                    fetch_batch()
                                         │
                                         ▼
                                   Async Trainer  ──► vLLM (weight sync)
```

This design lets you run the agent, gateway, and trainer on completely separate machines, with no coupling between service latency and training throughput.
