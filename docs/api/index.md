# API Reference

This section documents the HTTP and Python APIs exposed by Claw-R1's components.

<div class="grid cards" markdown>

-   **Gateway HTTP API**

    ---

    REST endpoints for agent integration and step submission.

    [:octicons-arrow-right-24: Gateway API](gateway.md)

</div>

## Python Interfaces

### DataPool (Ray Actor)

```python
import ray
from claw_r1.data_pool import DataPool
from claw_r1.data_pool.training_backend import VerlBackend

# Initialize
data_pool = DataPool.options(name="data_pool").remote(
    backend=VerlBackend(tokenizer_path="/path/to/model"),
    max_queue_size=None,
)

# Producer (called by Gateway internally)
ray.get(data_pool.submit_step.remote(step))
ray.get(data_pool.submit_steps.remote(steps))

# Consumer (called by Trainer)
batch = ray.get(data_pool.fetch_batch.remote())  # returns list[Step] or None
```

### RewardLoopWorker (Ray Actor)

```python
from claw_r1.reward_loop import RewardLoopWorker

reward_worker = RewardLoopWorker.remote(config=reward_config)
rewards = ray.get(reward_worker.compute_score_batch.remote(steps))
```

### AgentFlowBase (Python class)

```python
from claw_r1.agent_flow import SingleStepSingleTurnAgentFlow

class MyFlow(SingleStepSingleTurnAgentFlow):
    async def run(self, sample: dict) -> None:
        text, response_ids, prompt_ids = await self.gateway_generate(
            trajectory_uid=sample["traj_uid"],
            prompt_uid=sample["prompt_uid"],
            messages=[{"role": "user", "content": sample["question"]}],
        )
        # build Step and submit ...
```
