# Claw-R1

**The data foundation for Agentic Reinforcement Learning.**

Claw-R1 provides the data layer between agents and RL training systems. It collects step-level interactions, evaluates data quality, supports human or automated curation, and serves curated batches to training backends.

<div class="grid cards" markdown>

-   **Universal Data Collection**

    ---

    Connect white-box agents, black-box agents, and online services. OpenAI-compatible agents can route through the Gateway by changing `base_url`.

    [Base URL Integration](concepts/base-url-integration.md)

-   **Data Middleware Layer**

    ---

    Gateway captures interactions while DataPool stores, indexes, partitions, and serves step-level training data.

    [Middleware Layer](concepts/middleware-layer.md)

-   **Agentic RL Data Lifestyle Management**

    ---

    The dashboard tracks collection, representation, curation, optimization, and training consumption from real Ray actors.

    [Dashboard](dashboard.md)

-   **Data Evaluation and Curation**

    ---

    Combine rule rewards, reward models, human feedback, policy version freshness, and channel-based partitioning.

    [Reward System](components/reward-system.md)

-   **Prefix Tree Merge**

    ---

    Preview and develop shared-prefix packing for multi-step agent training to reduce redundant prefix computation.

    [Prefix Tree Merge](components/prefix-tree-merge.md)

</div>

## Why Claw-R1?

Agentic RL frameworks have made rapid progress on rollout runtime and policy optimization. As agents become more general, the harder missing layer is often data: how to collect real interactions, evaluate their quality, preserve provenance, and feed useful data back into training.

Claw-R1 focuses on that data layer.

| Area | Typical Agentic RL Stack | Claw-R1 |
|---|---|---|
| Core focus | Training algorithm and runtime | Data collection, quality, curation, and serving |
| Agent integration | Framework-specific APIs | OpenAI-compatible `base_url` routing or explicit Python APIs |
| Data source | Mostly offline rollouts | Offline datasets plus live agent interactions |
| Quality control | Limited or task-specific | Reward signals, human feedback, freshness, and curation metadata |
| Training engine | Usually tightly coupled | Pluggable backend boundary |

## Quick Start

```bash
git clone https://github.com/AgentR1/Claw-R1
cd Claw-R1

conda activate steppo
sh example/test_async_blackbox.sh
```

Start the dashboard after the Ray actors are running:

```bash
sh example/start_dashboard.sh
```

## Project Status

| Capability | Status |
|---|---|
| White-box agent data collection | Available |
| Black-box agent data collection | Available |
| Async training data serving | Available |
| Live dashboard | Available |
| Prefix-tree merge preview | Available in dashboard |
| Human feedback pipeline | Planned |

## Team

State Key Laboratory of Cognitive Intelligence, USTC
