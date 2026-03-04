# DataPool

DataPool is a **Ray Actor** that serves as the central trajectory buffer — the sole point of contact between the Agent Side (Gateway) and the Training Side (Trainer).

## Role in the Architecture

```
Gateway ──► DataPool.submit_step()      (async, fire-and-forget)
Trainer ◄── DataPool.fetch_batch()      (blocking pull, batch-ready groups)
```

DataPool completely decouples write speed (driven by agent request rate) from read speed (driven by training throughput). Neither side waits for the other.

## Data Storage Model

DataPool stores trajectories at **step granularity**. Each step is an `(s, a, r)` tuple:

```python
@dataclass
class Step:
    prompt_ids:     list[int]   # state: full context token IDs
    response_ids:   list[int]   # action: LLM-generated token IDs
    reward:         float       # immediate reward for this step
    trajectory_uid: str         # groups steps in the same conversation
    prompt_uid:     str         # groups rollouts from same prompt (for GRPO)
    step_index:     int         # position within trajectory (0-indexed)
    policy_version: int         # policy version when step was generated
    is_last:        bool        # True if this is the final step of the trajectory
    metadata:       dict        # auxiliary data (dataset fields, source info, etc.)
```

### Internal Indices

| Index | Type | Purpose |
|---|---|---|
| `trajectory_index` | `dict[str, list[int]]` | Maps `trajectory_uid` → list of step indices |
| `trajectory_complete` | `dict[str, bool]` | Tracks whether a trajectory has received its `is_last` step |
| `prompt_groups` | `dict[str, list[str]]` | Maps `prompt_uid` → list of `trajectory_uid`s |

## Producer API

### `submit_step(step: Step)`

Adds a single step to the pool. Called by the Gateway via Ray RPC (fire-and-forget from the Gateway's perspective).

### `submit_steps(steps: list[Step])`

Batch submission of multiple steps. More efficient than calling `submit_step` in a loop.

### `complete_trajectory(trajectory_uid: str)`

Marks a trajectory as complete (equivalent to submitting a step with `is_last=True`). Used in black-box online mode where the agent signals trajectory end separately.

## Consumer API

### `fetch_batch() → list[Step] | None`

FIFO pull of the next ready `prompt_uid` group. A group is "ready" when all its trajectories have received their `is_last` step.

Returns `None` if no complete group is available yet.

```python
# Trainer side (simplified)
while True:
    batch = await data_pool.fetch_batch.remote()
    if batch is not None:
        train_on_batch(batch)
```

## Capacity Management

When `max_queue_size` is set, DataPool drops the oldest ready group when the queue is full. This prevents unbounded memory growth during periods when the Trainer is slower than the agent.

```yaml
# async_ppo_trainer.yaml
data_pool:
  max_queue_size: null   # null = unlimited
```

## Memory Optimization

DataPool performs **lazy cleanup**: consumed entries are not immediately removed. Instead, when more than 50% of stored steps have been consumed, the DataPool compacts its internal arrays to reclaim memory.

## Off-policy Support

The Trainer can request batches that include historical (off-policy) data by configuring the staleness threshold:

```yaml
async_trainer:
  staleness_threshold: 0.1   # steps with policy_version lag > threshold are off-policy
```

Off-policy steps are still included in the batch but are down-weighted in the loss computation based on importance sampling.
