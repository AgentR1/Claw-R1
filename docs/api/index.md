# API Reference

本节文档化 Claw-R1 各组件暴露的 HTTP 和 Python API。

<div class="grid cards" markdown>

-   **Gateway HTTP API**

    ---

    REST 端点，用于 Agent 集成和 Step 提交。包括白盒端点（`/generate`、`/submit_steps`）和黑盒端点（`{base_url}/v1/chat/completions`）。

    [:octicons-arrow-right-24: Gateway API](gateway.md)

</div>

## Python 接口

### DataPool (Ray Actor)

```python
import ray
from claw_r1.data_pool import DataPool

data_pool = ray.get_actor("data_pool")

# Producer（由 Gateway 内部调用）
ray.get(data_pool.submit_step.remote(step, channel="train"))
ray.get(data_pool.submit_steps.remote(steps, channel="train"))
ray.get(data_pool.complete_trajectory.remote(trajectory_uid, channel="train"))

# Consumer（由 Trainer 调用）
batch = ray.get(data_pool.fetch_batch.remote(n_rollouts=5, channel="train"))
```

### RewardLoopWorker (Ray Actor)

```python
from claw_r1.reward_loop import RewardLoopWorker

reward_worker = ray.get_actor("reward_loop_worker")
rewards = ray.get(reward_worker.compute_score_batch.remote(steps))
```

### AgentFlowBase (Python class)

```python
from claw_r1.agent_flow import SingleStepSingleTurnAgentFlow

class MyFlow(SingleStepSingleTurnAgentFlow):
    async def run(self, sampling_params, **kwargs) -> int:
        text, response_ids, prompt_ids = await self.gateway_generate(
            trajectory_uid=kwargs["trajectory_uid"],
            prompt_uid=kwargs["prompt_uid"],
            messages=[{"role": "user", "content": kwargs["question"]}],
        )
        # 构建 Step 并提交 ...
        return 1
```

### BlackBoxAgentFlowBase (Python class)

```python
from claw_r1.agent_flow.agent_flow import register
from claw_r1.blackbox_agent.blackbox_agent_flow import BlackBoxAgentFlowBase

@register("my_blackbox_agent")
class MyBlackBoxFlow(BlackBoxAgentFlowBase):
    async def _run_agent(self, base_url: str, kwargs: dict) -> int:
        # 创建 Agent，使用 base_url 作为 OpenAI API endpoint
        agent = MyAgent(base_url=base_url)
        return await agent.solve(task=kwargs["raw_prompt"])
```
