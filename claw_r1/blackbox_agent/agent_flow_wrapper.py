"""AgentFlow wrapper for offline black-box training.

This wrapper integrates the independent :class:`GSM8KAgent` into the
existing ``AgentFlowWorker`` scheduling system.  It generates a
``trajectory_uid``, constructs a ``base_url`` (which embeds both
``trajectory_uid`` and ``prompt_uid``), and hands it to the agent.

The agent itself is completely unaware of these UIDs — it just uses the
``base_url`` as a standard OpenAI API endpoint.
"""

import json
import logging
import os
from typing import Any
from uuid import uuid4

import httpx
import numpy as np

from claw_r1.agent_flow.agent_flow import AgentFlowBase, register
from claw_r1.blackbox_agent.gsm8k_agent import GSM8KAgent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to native Python types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


@register("blackbox_gsm8k_agent")
class BlackBoxGSM8KAgentFlow(AgentFlowBase):
    """Offline black-box wrapper that delegates to :class:`GSM8KAgent`."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> int:
        trajectory_uid = uuid4().hex
        prompt_uid = str(kwargs.get("uid", "1"))
        channel = kwargs.pop("channel", None)

        base_url = f"{self.gateway_url}/{trajectory_uid}/{prompt_uid}"

        # Collect dataset metadata (reward_model, data_source, etc.) for the
        # Step objects that Gateway will create on behalf of this trajectory.
        skip_keys = {"raw_prompt", "multi_modal_data", "channel", "agent_name"}
        metadata = {k: v for k, v in kwargs.items() if k not in skip_keys}

        # Pre-register trajectory with Gateway: channel routing + metadata.
        # Use a custom encoder to handle numpy scalar types that are not
        # natively JSON-serializable.
        reg_body: dict[str, Any] = {"trajectory_uid": trajectory_uid}
        if channel:
            reg_body["channel"] = channel
        if metadata:
            reg_body["metadata"] = metadata
        payload = json.dumps(reg_body, cls=_NumpyEncoder).encode()
        async with httpx.AsyncClient(timeout=30.0) as http:
            await http.post(
                f"{self.gateway_url}/register_trajectory",
                content=payload,
                headers={"content-type": "application/json"},
            )

        raw_prompt = kwargs.get("raw_prompt", [])
        if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
            last_user_msg = None
            for msg in raw_prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
            question = last_user_msg or str(raw_prompt)
        elif isinstance(raw_prompt, str):
            question = raw_prompt
        else:
            question = str(raw_prompt)

        reward_model = kwargs.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = str(reward_model.get("ground_truth", ""))
        else:
            ground_truth = str(getattr(reward_model, "ground_truth", ""))

        agent = GSM8KAgent(base_url=base_url)
        max_turns = self.config.actor_rollout_ref.rollout.get("max_turns", 3)

        num_turns = await agent.solve(
            question=question,
            ground_truth=ground_truth,
            max_turns=max_turns,
        )

        return num_turns
