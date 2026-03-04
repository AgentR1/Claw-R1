# Agent Flow

Agent Flow is the Python framework for **white-box agents** in Claw-R1. It handles the full lifecycle of an agent turn: vision data processing, chat template application, LLM generation via Gateway, and step submission.

## Class Hierarchy

```
AgentFlowBase           (abstract base)
    │
    ├── SingleStepSingleTurnAgentFlow   (simplest: one prompt → one response)
    └── MultiStepAgentFlow              (multi-turn: tools, planning, etc.)
```

## AgentFlowBase

All agent flows inherit from `AgentFlowBase`, which provides:

- HTTP client management (shared `aiohttp.ClientSession` per process)
- Tokenizer and processor initialization
- Chat template application
- Vision / multimodal data extraction
- Gateway communication helpers

### Key Methods

#### `gateway_generate(trajectory_uid, prompt_uid, messages, **kwargs)`

Sends an async HTTP POST to `Gateway /generate` and returns the raw text and token IDs.

```python
text, response_ids, prompt_ids = await self.gateway_generate(
    trajectory_uid="traj-abc",
    prompt_uid="prompt-xyz",
    messages=[
        {"role": "user", "content": "Summarize this document."}
    ],
    max_tokens=512,
    temperature=0.8,
)
```

#### `gateway_submit_steps(steps: list[Step])`

Sends an async HTTP POST to `Gateway /submit_steps`. This is a fire-and-forget call — the agent flow does not wait for confirmation from the DataPool.

#### `apply_chat_template(messages, add_generation_prompt=True)`

Applies the model's chat template and tokenizes the message sequence.

#### `process_vision_info(messages)`

Extracts images and videos from messages for multimodal models.

## SingleStepSingleTurnAgentFlow

The simplest implementation: a single prompt produces a single response. Useful for datasets where each example is a self-contained question-answer pair.

```python
from claw_r1.agent_flow import SingleStepSingleTurnAgentFlow

class MyAgentFlow(SingleStepSingleTurnAgentFlow):
    async def run(self, sample: dict) -> None:
        messages = [{"role": "user", "content": sample["question"]}]

        # Generate response via Gateway
        text, response_ids, prompt_ids = await self.gateway_generate(
            trajectory_uid=sample["trajectory_uid"],
            prompt_uid=sample["prompt_uid"],
            messages=messages,
        )

        # Build and submit the step
        step = Step(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            reward=0.0,           # reward filled in by Trainer via /compute_reward
            trajectory_uid=sample["trajectory_uid"],
            prompt_uid=sample["prompt_uid"],
            step_index=0,
            policy_version=self.policy_version,
            is_last=True,
            metadata=sample.get("metadata", {}),
        )
        await self.gateway_submit_steps([step])
```

## MultiStepAgentFlow

For complex agents that call tools, plan across multiple turns, or use a scratchpad. Each turn produces one `Step`, and steps are chained by `trajectory_uid`.

```python
from claw_r1.agent_flow import MultiStepAgentFlow

class ToolAgentFlow(MultiStepAgentFlow):
    async def run(self, sample: dict) -> None:
        messages = [{"role": "user", "content": sample["task"]}]
        step_index = 0

        while True:
            text, response_ids, prompt_ids = await self.gateway_generate(
                trajectory_uid=sample["trajectory_uid"],
                prompt_uid=sample["prompt_uid"],
                messages=messages,
            )

            is_last = self.is_terminal(text)

            step = Step(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                reward=0.0,
                trajectory_uid=sample["trajectory_uid"],
                prompt_uid=sample["prompt_uid"],
                step_index=step_index,
                policy_version=self.policy_version,
                is_last=is_last,
            )
            await self.gateway_submit_steps([step])

            if is_last:
                break

            # Append response and tool result for next turn
            messages.append({"role": "assistant", "content": text})
            tool_result = await self.execute_tool(text)
            messages.append({"role": "tool", "content": tool_result})
            step_index += 1
```

## Configuration

Agent flows are configured via `AgentFlowConfig`:

```python
@dataclass
class AgentFlowConfig:
    num_workers: int = 8                               # parallel agent workers
    default_agent_flow: str = "single_step_single_turn_agent"
```

Or in YAML (via Hydra):

```yaml
# overrides/rollout.yaml
agent_flow:
  num_workers: 8
  default_agent_flow: single_step_single_turn_agent
```
