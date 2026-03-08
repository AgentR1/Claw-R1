"""Black-box agent flow — base class.

BlackBoxAgentFlowBase handles the full protocol with Gateway (init_trajectory,
register_trajectory, complete) and delegates agent execution to subclasses via
_run_agent.  Subclasses only create and run the concrete Agent; they do not
touch Gateway or implement any task logic.  Concrete strategies live in
separate modules (e.g. gsm8k_agent_flow.py).
"""

import json
import logging
import os
from abc import abstractmethod
from typing import Any

import httpx
import numpy as np

from claw_r1.agent_flow.agent_flow import AgentFlowBase, register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_DEFAULT_SKIP_KEYS = frozenset({"raw_prompt", "multi_modal_data", "channel", "agent_name"})


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to native Python types for HTTP requests."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class BlackBoxAgentFlowBase(AgentFlowBase):
    """Base class for black-box agent flows.

    Handles generic parameter processing and the full Gateway protocol:
    init_trajectory (get base_url) -> register_trajectory (channel + metadata)
    -> call subclass _run_agent -> complete.  Subclasses only implement
    _run_agent to create and run the concrete Agent.
    """

    def _prepare_params(self, kwargs: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
        """Extract channel, prompt_uid, and metadata from kwargs."""
        channel = kwargs.pop("channel", None)
        prompt_uid = str(kwargs.get("uid", "1"))
        metadata = {k: v for k, v in kwargs.items() if k not in _DEFAULT_SKIP_KEYS}
        return channel, prompt_uid, metadata

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> int:
        channel, prompt_uid, metadata = self._prepare_params(kwargs)

        async with httpx.AsyncClient(timeout=30.0) as http:
            init_resp = await http.post(f"{self.gateway_url}/init_trajectory")
            init_resp.raise_for_status()
            init_data = init_resp.json()
            trajectory_uid = init_data["trajectory_uid"]
            base_url_from_init = init_data["base_url"]
            base_url = base_url_from_init.rsplit("/", 1)[0] + "/" + prompt_uid

            reg_body: dict[str, Any] = {"trajectory_uid": trajectory_uid}
            if channel:
                reg_body["channel"] = channel
            if metadata:
                reg_body["metadata"] = metadata
            payload = json.dumps(reg_body, cls=_NumpyEncoder).encode()
            await http.post(
                f"{self.gateway_url}/register_trajectory",
                content=payload,
                headers={"content-type": "application/json"},
            )

        try:
            num_turns = await self._run_agent(base_url, kwargs)
        finally:
            async with httpx.AsyncClient(timeout=30.0) as http:
                await http.post(f"{base_url}/v1/trajectory/complete")

        return num_turns

    @abstractmethod
    async def _run_agent(self, base_url: str, kwargs: dict[str, Any]) -> int:
        """Create and run the concrete Agent.  Subclasses implement this."""
        raise NotImplementedError
