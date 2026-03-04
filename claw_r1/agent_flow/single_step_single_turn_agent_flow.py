# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-step, single-turn agent flow — the simplest possible agent."""

import logging
import os
from typing import Any
from uuid import uuid4

from claw_r1.agent_flow.agent_flow import AgentFlowBase, register
from claw_r1.data_pool.data_model import Step

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_step_single_turn_agent")
class SingleStepSingleTurnAgentFlow(AgentFlowBase):
    """Agent flow that performs a single chat-completion turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> int:
        messages = list(kwargs["raw_prompt"])

        # 1. Extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. Apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. Generate via Gateway
        gen_result = await self.gateway_generate(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        response_ids = gen_result["token_ids"][: self.response_length]
        log_probs = gen_result.get("log_probs")
        if log_probs:
            log_probs = log_probs[: self.response_length]

        # 4. Build Step and submit (reward is computed later by the Trainer)
        trajectory_uid = uuid4().hex
        prompt_uid = str(kwargs.get("uid", uuid4().hex))

        metadata = {k: v for k, v in kwargs.items() if k != "agent_name"}

        step = Step(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            rollout_log_probs=log_probs,
            trajectory_uid=trajectory_uid,
            prompt_uid=prompt_uid,
            step_index=0,
            is_last=True,
            metadata=metadata,
        )
        await self.gateway_submit_steps([step])
        return 1
