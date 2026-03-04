"""Multi-step agent flow — simplest multi-turn example for pipeline testing.

The agent performs ``max_turns`` rounds of generation. After each non-final
turn the response is appended as an assistant message, followed by a fixed
user feedback prompt, forming the context for the next turn.

Each turn produces one :class:`Step`. Only the last Step has ``is_last=True``.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, register
from agent_r1.data_pool.data_model import Step

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FEEDBACK_PROMPT = "Please verify your answer step by step and provide the final answer again in the format #### <number>."


@register("multi_step_agent")
class MultiStepAgentFlow(AgentFlowBase):
    """Agent flow that performs multiple turns of generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_turns = 2

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> int:
        trajectory_uid = uuid4().hex
        prompt_uid = str(kwargs.get("uid", uuid4().hex))
        messages = list(kwargs["raw_prompt"])
        metadata = {k: v for k, v in kwargs.items() if k != "agent_name"}

        steps: list[Step] = []

        for turn in range(self.max_turns):
            prompt_ids = await self.apply_chat_template(messages)
            prompt_ids = prompt_ids[-self.prompt_length :]

            gen_result = await self.gateway_generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            response_ids = gen_result["token_ids"][: self.response_length]
            log_probs = gen_result.get("log_probs")
            if log_probs:
                log_probs = log_probs[: self.response_length]

            is_last = turn == self.max_turns - 1

            step = Step(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                rollout_log_probs=log_probs,
                trajectory_uid=trajectory_uid,
                prompt_uid=prompt_uid,
                step_index=turn,
                is_last=is_last,
                metadata=metadata,
            )

            if not is_last:
                await self.gateway_submit_steps([step])
                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True,
                )
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": FEEDBACK_PROMPT})
            else:
                dataset_fields = {k: v for k, v in kwargs.items() if k != "agent_name"}
                reward_result = await self.gateway_compute_reward(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    num_turns=len(messages),
                    dataset_fields=dataset_fields,
                )
                step.reward = reward_result["reward_score"]
                await self.gateway_submit_steps([step])

        return self.max_turns
