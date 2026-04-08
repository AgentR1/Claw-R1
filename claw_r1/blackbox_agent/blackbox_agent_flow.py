"""Black-box agent flow — base class.

BlackBoxAgentFlowBase handles the full protocol with Gateway (init_trajectory,
register_trajectory, complete) and delegates agent execution to subclasses via
_run_agent.  Subclasses only create and run the concrete Agent; they do not
touch Gateway or implement any task logic.  Concrete strategies live in
separate modules (e.g. gsm8k_agent_flow.py).
"""

import asyncio
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

    async def _http_post_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        timeout: float = 600.0,
        **kwargs,
    ) -> httpx.Response:
        """POST with retry on transient connection errors."""
        last_exc = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as http:
                    resp = await http.post(url, **kwargs)
                    resp.raise_for_status()
                    return resp
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(
                        "HTTP POST %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        url, attempt + 1, max_retries, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("HTTP POST %s failed after %d attempts: %s", url, max_retries, exc)
        raise last_exc

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> int:
        channel, prompt_uid, metadata = self._prepare_params(kwargs)

        async with httpx.AsyncClient(timeout=30.0) as http:
            # 1. Allocate trajectory — get base_url with trajectory_uid embedded.
            init_resp = await http.post(f"{self.gateway_url}/init_trajectory")
            init_resp.raise_for_status()
            init_data = init_resp.json()
            base_url_from_init = init_data["base_url"]
            # base_url_from_init is http://host:port/{traj_uid}/{default_prompt_uid}/v1
            # Replace the default prompt_uid with the actual one.
            parts = base_url_from_init.rsplit("/", 2)  # [...base, prompt_uid, "v1"]
            base_url = f"{parts[0]}/{prompt_uid}/v1"

            # 2. Register channel + metadata via base_url.
            reg_body: dict[str, Any] = {}
            if channel:
                reg_body["channel"] = channel
            if metadata:
                reg_body["metadata"] = metadata
            payload = json.dumps(reg_body, cls=_NumpyEncoder).encode()
            await http.post(
                f"{base_url}/register_trajectory",
                content=payload,
                headers={"content-type": "application/json"},
            )

        # 3. Run the concrete agent (with retry on transient errors).
        reward = None
        num_turns = 0
        try:
            result = await self._run_agent(base_url, kwargs)
            # _run_agent may return int (num_turns) or tuple (num_turns, reward)
            if isinstance(result, tuple):
                num_turns, reward = result
            else:
                num_turns = result
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            # Agent failed due to transient connection error (e.g. vLLM weight update).
            # Log warning but still complete the trajectory with reward=0.
            logger.warning(
                "Agent run failed with transient error: %s. Completing trajectory with reward=0.",
                exc,
            )
            reward = 0.0
        finally:
            # 4. Mark trajectory complete, passing reward if available.
            # Use retry to handle transient Gateway connection issues.
            try:
                body: dict[str, Any] = {}
                if reward is not None:
                    body["reward"] = float(reward)
                if channel:
                    body["channel"] = channel
                if body:
                    await self._http_post_with_retry(
                        f"{base_url}/complete_trajectory",
                        json=body,
                    )
                else:
                    await self._http_post_with_retry(
                        f"{base_url}/complete_trajectory",
                    )
            except Exception as exc:
                logger.error(
                    "Failed to complete trajectory after retries: %s", exc,
                )

        return num_turns

    @abstractmethod
    async def _run_agent(self, base_url: str, kwargs: dict[str, Any]) -> int | tuple[int, float]:
        """Create and run the concrete Agent.  Subclasses implement this.

        Returns either:
        - ``int``: number of turns used (reward computed by Gateway)
        - ``tuple[int, float]``: (turns_used, reward) for direct reward passing
        """
        raise NotImplementedError
