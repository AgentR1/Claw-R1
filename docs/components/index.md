# Components

Claw-R1 is composed of five independently runnable components that communicate via HTTP and Ray RPC.

<div class="grid cards" markdown>

-   **Gateway Server**

    ---

    FastAPI HTTP service. The network-layer entry point for all agent LLM calls. Manages load balancing across vLLM servers and submits steps to DataPool.

    [:octicons-arrow-right-24: Gateway Server](gateway.md)

-   **DataPool**

    ---

    Ray Actor. The central trajectory buffer between Agent Side and Training Side. Supports asynchronous writes from the Gateway and batch reads from the Trainer.

    [:octicons-arrow-right-24: DataPool](datapool.md)

-   **Agent Flow**

    ---

    Python framework for white-box agents. Manages chat templates, tokenization, multimodal data processing, and HTTP communication with the Gateway.

    [:octicons-arrow-right-24: Agent Flow](agent-flow.md)

-   **Async Training**

    ---

    `AsyncTrainer` and `AsyncRollouter` Ray Actors. Continuous, non-blocking training loop with parameter synchronization.

    [:octicons-arrow-right-24: Async Training](async-training.md)

-   **Reward System**

    ---

    `RewardLoopWorker` Ray Actor. Computes step-level rewards from rule-based, discriminative, or generative reward models.

    [:octicons-arrow-right-24: Reward System](reward-system.md)

</div>

## Component Interaction Map

```
                      ┌─────────────────────────────────────────┐
  Black-box Agent ───►│                                         │
  (base_url only)     │         GATEWAY SERVER                  │
                      │         (FastAPI, port 8000)            │
  White-box Agent ───►│                                         │
  (AgentFlow)         └────────────┬────────────────────────────┘
                                   │ Ray RPC (submit_step)
                                   ▼
                      ┌─────────────────────────────────────────┐
                      │         DATAPOOL                         │
                      │         (Ray Actor)                      │
                      └──────────────────┬──────────────────────┘
                                         │ fetch_batch()
                                         ▼
                      ┌─────────────────────────────────────────┐
                      │         ASYNC TRAINER                    │
                      │         (Ray Actor)                      │
                      │   ┌─────────────────────────────────┐   │
                      │   │  Actor │ Critic │ RefPolicy      │   │
                      │   └────────────────────────────────-┘   │
                      └────────────────┬────────────────────────┘
                                       │ weight sync (NCCL)
                                       ▼
                      ┌─────────────────────────────────────────┐
                      │         ASYNC ROLLOUTER                  │
                      │         (Ray Actor, rollout GPU pool)    │
                      │         vLLM servers                     │
                      └─────────────────────────────────────────┘
```
