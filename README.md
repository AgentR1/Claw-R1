<h1 align="center"> Claw-R1: The Data Foundation for <br> Agentic Reinforcement Learning </h1>

<p align="center">
  <a href="https://agentr1.github.io/"><img src="https://img.shields.io/badge/Project-Home-orange.svg" alt="Project Home"></a>
  <a href="https://github.com/AgentR1/Claw-R1/stargazers"><img src="https://img.shields.io/github/stars/AgentR1/Claw-R1" alt="GitHub Repo stars"></a>
  <a href="https://github.com/AgentR1/Claw-R1/network/members"><img src="https://img.shields.io/github/forks/AgentR1/Claw-R1" alt="GitHub forks"></a>
  <a href="https://agentr1.github.io/Claw-R1/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs"></a>
</p>

<p align="center"><img src="./assets/logo.jpeg" width="600px" alt="Claw-R1 Logo" /></p>

## News

- **[2026.04]** 🌲 **Prefix Tree Merge for Agentic RL Training.** A new algorithm that deduplicates shared prefix computation in multi-step agent training via prefix tree packing + FlexAttention. Currently under testing on the [`prefix-tree-merge`](https://github.com/AgentR1/Claw-R1/tree/prefix-tree-merge) branch. See [documentation](docs/components/prefix-tree-merge.md).

- **[2026.04]** 📚 **RL Training Internals Tutorial.** A comprehensive tutorial covering core RL concepts (Reward / Value / Advantage / Return / Loss), PPO & GRPO algorithms, and Claw-R1's step-level agentic RL design rationale. See [tutorial](docs/rl-training-internals/index.md).

- **[2026.03.06]** 📖 **Claw-R1 Documentation Released.** Project page and documentation are now available at [Claw-R1 Project Page](https://agentr1.github.io/) and [Claw-R1 docs](https://agentr1.github.io/Claw-R1/).

- **[2026.03.03]** 🚧 **Claw-R1 Project Init.** We are actively developing the framework. Stay tuned for more features and documentation.

## Overview

The **Agentic RL** ecosystem is thriving — frameworks like [verl](https://github.com/volcengine/verl), [Agent-R1](https://github.com/0russwest0/Agent-R1), and [MiniMax Forge](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) have made remarkable progress in RL runtime and training algorithms. Meanwhile, **General Agents** (e.g., [OpenClaw](https://github.com/openclaw/openclaw), Claude Code, Open Code) are producing interaction data that is far richer and more complex than traditional ReAct trajectories.

As agents grow more capable, a critical question emerges: **How do we systematically collect, evaluate, and curate high-quality training data from diverse agent interactions?** This is a relatively under-explored yet important direction — especially when human feedback is available as a natural quality signal.

**Claw-R1** provides the **data foundation** for Agentic RL. It introduces a Middleware Layer (Gateway + DataPool) between the Agent Side and the Training Side, focusing on data collection, evaluation, and curation rather than training algorithms themselves.

<p align="center"><img src="./assets/framework.png" width="800px" alt="Claw-R1 Framework" /></p>

## Key Features

- **Universal Data Collection**: White-box agents submit Steps via API; black-box agents integrate by simply pointing `base_url` to the Gateway (zero code changes); online services collect data from live user interactions in real-time.

- **Data Evaluation & Curation**: Multi-dimensional reward system (rule-based / discriminative RM / generative RM), human feedback signal integration, policy version tracking for freshness-aware curation, and channel-based data partitioning.

- **Flexible Data Serving**: Pluggable `TrainingBackend` to convert curated data into any training engine's native format, with GRPO-aware grouping, train/val channel isolation, and real-time monitoring.

## Get Started

- 📖 **[Full Documentation](https://agentr1.github.io/Claw-R1/)**
- 🚀 [Installation Guide](https://agentr1.github.io/Claw-R1/getting-started/installation/)
- 🛠️ [Architecture Overview](https://agentr1.github.io/Claw-R1/components/)

## Roadmap

- [ ] **Data Quality Dashboard**: Visual monitoring of data quality metrics, reward distributions, and collection statistics.
- [ ] **Human Feedback Pipeline**: Structured pipeline for capturing and integrating explicit and implicit human feedback signals from online agent services.
- [ ] **Dataset Export & Versioning**: Export curated datasets with full provenance tracking for reproducibility and sharing.
- [ ] **Extended TrainingBackend Support**: Native adapters for additional RL frameworks beyond verl.

## Contributors

**Team Members**: Daoyu Wang*, Qingchuan Li*, Jie Ouyang, Shuo Yu

**Supervisors**: Qi Liu, Mingyue Cheng

**Affiliation**: State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China

## Acknowledgements

We extend our gratitude to [Agent-R1](https://github.com/0russwest0/Agent-R1), [MiniMax Forge](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm), [verl](https://github.com/volcengine/verl), and [rLLM](https://github.com/rllm-org/rllm) for their pioneering work on Agentic RL training infrastructure. We also thank [OpenClaw](https://github.com/openclaw/openclaw) for their remarkable work on personal AI assistants. We are grateful to the broader Agentic RL community and all contributors for their support.

## Citation

```bibtex
@misc{clawr1-2026,
  title={Claw-R1: The Data Foundation for Agentic Reinforcement Learning},
  author={Wang, Daoyu and Li, Qingchuan and Ouyang, Jie and Yu, Shuo and Cheng, Mingyue and Liu, Qi},
  year={2025},
  howpublished={\url{https://github.com/AgentR1/Claw-R1}},
  note={GitHub repository}
}
```
