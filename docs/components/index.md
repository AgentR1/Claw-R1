# Components

Claw-R1 components are organized around the data flow from agent interaction to training consumption. HTTP handles agent-facing traffic, while Ray actors handle stateful data and training coordination.

<div class="grid cards" markdown>

-   **Gateway Server**

    ---

    FastAPI service that receives white-box step submissions and black-box OpenAI-compatible chat traffic, then submits normalized `Step` records to DataPool.

    [Gateway Server](gateway.md)

-   **DataPool**

    ---

    Ray actor that stores, indexes, partitions, curates, and serves step-level data by channel and prompt group.

    [DataPool](datapool.md)

-   **Dashboard**

    ---

    Live UI for Agentic RL data lifestyle management: collection, representation, curation, optimization preview, and training consumption.

    [Dashboard](../dashboard.md)

-   **Reward System**

    ---

    Reward workers compute or attach quality signals from rule checks, reward models, generative judges, and human feedback.

    [Reward System](reward-system.md)

-   **Agent Flow**

    ---

    Agent execution lifecycle. White-box flows submit steps explicitly; black-box flows wrap agents that only know an OpenAI-compatible `base_url`.

    [Agent Flow](agent-flow.md)

-   **Async Training**

    ---

    Separate Ray actors for rollout generation and policy training, coordinated through DataPool and parameter synchronization.

    [Async Training](async-training.md)

-   **Prefix Tree Merge**

    ---

    Shared-prefix packing for multi-step agent training. The dashboard can preview prefix-tree structure from real DataPool steps.

    [Prefix Tree Merge](prefix-tree-merge.md)

</div>

## Data Flow

```text
Black-box Agent         White-box Agent
      |                       |
      v                       v
   Gateway Server  <---- explicit Step APIs
      |
      v
   DataPool  <---- Dashboard reads stats, steps, events, curation, and prefix-tree previews
      |
      v
   Async Trainer ---- Parameter Synchronizer ---- Async Rollouter / vLLM
```

## Lifecycle Mapping

| Lifecycle stage | Primary component | Dashboard view |
|---|---|---|
| Collect interactions | Gateway, Agent Flow | Collection |
| Store representation | DataPool | Representation |
| Evaluate quality | Reward System | Curation Signals |
| Curate candidates | DataPool curation APIs | Curation |
| Optimize shared context | Prefix Tree Merge preview | Optimization |
| Serve training data | DataPool, Async Trainer | Consumption |
