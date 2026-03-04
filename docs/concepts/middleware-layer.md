# Middleware Layer

## The Problem with Synchronous Training

Traditional RLVR training uses a **synchronous loop**:

```
Generate batch of trajectories (Rollout)
  → Compute rewards
  → Update model weights
  → Generate next batch ...
```

This works well in research settings where the environment is a dataset and can be paused at will. But in production, the synchronous loop creates fundamental problems:

- **Rollout blocks training**: the model cannot update while serving requests
- **Training blocks service**: gradient updates stall the agent
- **Data waste**: real user interactions that fall outside the "current batch window" are discarded

## Architecture: Gateway + DataPool

The Middleware Layer consists of two components that together form the **only bridge** between the Agent Side and Training Side — the two sides never communicate directly.

```
┌──────────────────────────────────────────────────────┐
│          Agent Side                                  │
│  User request → Agent → Gateway (HTTP) → LLM resp.  │
│                             │                        │
│              [Gateway submits Step to DataPool]      │
└─────────────────────────────┬────────────────────────┘
                               │  Async, non-blocking
                               │  (Ray RPC)
┌─────────────────────────────▼────────────────────────┐
│          Training Side (runs independently)          │
│  Trainer ◄── DataPool.fetch_batch() ◄── Reward       │
│      │                                               │
│      └── weight update → sync to rollout vLLM        │
└──────────────────────────────────────────────────────┘
```

### Gateway Server

The Gateway is a **FastAPI HTTP service** (independent OS process, not a Ray actor). This design is deliberate:

- As a plain HTTP process it can be restarted without touching the Ray cluster
- It introduces no Ray scheduling overhead on the critical request-handling path
- It is the sole point of contact for agents — agents need only an HTTP address, nothing else

The Gateway has three responsibilities:

1. **Forward** LLM completion requests to the vLLM rollout servers (load-balanced)
2. **Tokenize** responses and construct `Step` objects
3. **Submit** steps to DataPool via Ray RPC (fire-and-forget, non-blocking)

### DataPool

DataPool is a **Ray Actor** that acts as a timestamped trajectory queue. It decouples write speed (determined by agent request rate) from read speed (determined by training throughput).

Key characteristics:

| Property | Behavior |
|---|---|
| Write | Asynchronous — Gateway submits steps without waiting for confirmation |
| Read | Async pull — Trainer fetches batches at its own pace, independent of rollout timing |
| Ordering | FIFO by `prompt_uid` group |
| Capacity | Configurable `max_queue_size`; oldest groups dropped when full |
| Persistence | Survives agent restarts (Ray actor keeps state in memory) |
| Mixed policy | Trainer can consume both on-policy (latest) and off-policy (historical) trajectories |

## Data Model: The Step

Every LLM call produces one `Step` — the atomic unit of RL data:

```python
@dataclass
class Step:
    prompt_ids:     list[int]  # full context token IDs (state)
    response_ids:   list[int]  # LLM-generated tokens (action)
    reward:         float      # immediate reward for this step
    trajectory_uid: str        # shared across all steps in one conversation
    prompt_uid:     str        # groups rollouts from the same prompt (GRPO)
    step_index:     int        # position within the trajectory
    policy_version: int        # for off-policy staleness detection
    is_last:        bool       # marks the final step of a trajectory
    metadata:       dict       # dataset fields, data source, etc.
```

Trajectories are stored **step-by-step** (not episode-by-episode), which allows the trainer to start computing advantages as soon as individual steps arrive, without waiting for the full episode to complete.

## Comparison with rLLM's DataPool

rLLM also uses a DataPool abstraction, but serves a different use case:

| Dimension | rLLM DataPool | Claw-R1 DataPool |
|---|---|---|
| Data source | Batch rollout engine (offline generation) | Real user requests (live service) |
| Data nature | Pre-defined synthetic trajectories | Authentic user interaction trajectories |
| Agent status during training | Not serving | Continuously serving |
| Reward type | Verifiable task result | Process reward + environment feedback |

rLLM's DataPool accelerates offline batch training. Claw-R1's DataPool makes the **production service itself the training data source**.

## Reward Annotation

DataPool stores not just raw trajectories but reward-annotated steps. Claw-R1 uses a `RewardLoopWorker` to score trajectories:

```
Trajectory:   [user msg] → [agent think] → [tool call] → [tool result] → [final reply]
Reward:            0.3           0.7            0.9            0.8
```

- **White-box offline mode**: AgentFlow builds `Step` objects and submits them via Gateway `/submit_steps`; the Trainer calls `/compute_reward` on its side
- **Black-box online mode** (reserved): The agent side computes rewards and passes them via Gateway `/complete_trajectory`

Reward computation never blocks the agent service. The Trainer fetches fully-annotated batches from `DataPool.fetch_batch()` at training time.
