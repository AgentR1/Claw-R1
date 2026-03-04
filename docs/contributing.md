# Contributing

Thank you for your interest in contributing to Claw-R1! This document explains how to get started, the project structure, and our development workflow.

!!! warning "Early Stage"
    Claw-R1 is under active development. APIs and interfaces may change significantly before the first stable release.

## Getting the Code

```bash
git clone https://github.com/AgentR1/Claw-R1
cd Claw-R1
conda create -n clawr1-dev python=3.10 -y
conda activate clawr1-dev
pip install -e ".[dev]"
```

## Project Structure

```
Claw-R1/
├── claw_r1/
│   ├── agent_flow/         # White-box agent base classes
│   ├── config/             # Hydra YAML configurations
│   ├── data_pool/          # DataPool Ray Actor + data model
│   ├── gateway/            # FastAPI Gateway server
│   ├── async_main.py       # Entry point for async training
│   ├── async_rollouter.py  # AsyncRollouter Ray Actor
│   ├── async_trainer.py    # AsyncTrainer Ray Actor
│   ├── core_algos.py       # PPO / GAE / GRPO algorithms
│   ├── param_sync.py       # Weight synchronization
│   └── reward_loop.py      # RewardLoopWorker Ray Actor
├── docs/                   # MkDocs documentation (this site)
├── mkdocs.yml              # Documentation configuration
└── pyproject.toml          # Project metadata and linting config
```

## Code Style

Claw-R1 uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check
ruff check claw_r1/

# Format
ruff format claw_r1/
```

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

```bash
pre-commit install
pre-commit run --all-files
```

## Areas for Contribution

### High Priority

- [ ] Complete black-box online mode endpoints in Gateway
- [ ] Add end-to-end integration tests
- [ ] Add more `AgentFlowBase` examples (tool-use agents, multi-modal agents)
- [ ] Improve reward model integration (RLHF reward models, LLM-as-judge)

### Documentation

- [ ] Add examples for specific use cases (OpenClaw, LangChain agents)
- [ ] Add performance benchmarks
- [ ] Chinese translation of docs

### Research

- [ ] Token-level GAE for multi-step trajectories
- [ ] Exploration bonuses for online training
- [ ] Curriculum learning for DataPool sampling

## Submitting a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and run `pre-commit run --all-files`
4. Push and open a PR against `main`
5. Fill in the PR description with a summary of your changes

## Building the Documentation Locally

```bash
pip install mkdocs-material pymdown-extensions
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000`.

## Contact

- GitHub Issues: [AgentR1/Claw-R1/issues](https://github.com/AgentR1/Claw-R1/issues)
- Team: Daoyu Wang, Jie Ouyang, Shuo Yu (USTC)

## Acknowledgements

Claw-R1 builds upon [Agent-R1](https://github.com/0russwest0/Agent-R1). We extend our gratitude to [MiniMax Forge](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) for architectural insights on the Middleware design, and to [rLLM](https://github.com/rllm-org/rllm) for pioneering work on RL framework design for language agents. We also thank [OpenClaw](https://github.com/openclaw/openclaw) for the modern agent paradigm that inspires our vision.
