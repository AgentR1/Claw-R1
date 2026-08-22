"""Gateway/DataPool orchestration shared by black-box recipes."""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any
from uuid import uuid4

import httpx

from claw_r1.agent_flow.agent_flow import AgentFlowBase
from .sandbox import create_sandbox_provider


class BlackBoxAgentFlowBase(AgentFlowBase):
    """Allocate/register/complete a Gateway trajectory around an external agent."""
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> int:
        channel = kwargs.pop("channel", None)
        prompt_uid = str(kwargs.get("uid", "1"))
        metadata = {k: v for k, v in kwargs.items() if k not in {"raw_prompt", "multi_modal_data", "agent_name"}}
        async with httpx.AsyncClient(timeout=600) as client:
            init = (await client.post(f"{self.gateway_url}/init_trajectory")).json()
            base_url = init["base_url"].rsplit("/", 2)[0] + f"/{prompt_uid}/v1"
            body = {"metadata": metadata}
            if channel: body["channel"] = channel
            response = await client.post(f"{base_url}/register_trajectory", json=body); response.raise_for_status()
            try:
                return await self._run_agent(base_url, kwargs, sampling_params, init["trajectory_uid"])
            finally:
                await client.post(f"{base_url}/complete_trajectory")

    @abstractmethod
    async def _run_agent(self, base_url: str, kwargs: dict[str, Any], sampling_params: dict[str, Any], trajectory_uid: str) -> int: ...


class BlackBoxCliAgentFlow(BlackBoxAgentFlowBase):
    adapter_path: str

    async def _run_agent(self, base_url: str, kwargs: dict[str, Any], sampling_params: dict[str, Any], trajectory_uid: str) -> int:
        metadata = dict(kwargs.get("metadata") or {}) if isinstance(kwargs.get("metadata"), dict) else {}
        # RL datasets commonly flatten metadata into non_tensor_batch.  Preserve
        # that form so recipes do not impose a dataset-specific wrapper.
        metadata.update({k: v for k, v in kwargs.items() if k not in {"raw_prompt", "agent_name", "channel", "uid", "policy_version", "reward_model", "metadata"}})
        provider = create_sandbox_provider(metadata.get("sandbox_provider"))
        lease = await provider.acquire(trajectory_uid=trajectory_uid, metadata=metadata)
        session_id = f"claw-{uuid4().hex}"
        try:
            async with httpx.AsyncClient(timeout=1900) as client:
                register = {"session_id": session_id, "gateway_url": base_url, "adapter": self.adapter_path, "workspace": metadata.get("workspace", lease.workspace), "backend_options": metadata.get("backend_options", {})}
                response = await client.post(f"{lease.endpoint}/v1/rollout/register", json=register); response.raise_for_status()
                if files := metadata.get("files"): await client.post(f"{lease.endpoint}/v1/sessions/{session_id}/files", json={"files": files})
                for cmd in metadata.get("before_agent_commands", []): await client.post(f"{lease.endpoint}/v1/sessions/{session_id}/execute_cmd", json={"cmd": cmd})
                prompt = kwargs.get("raw_prompt", [])
                messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
                result = await client.post(f"{lease.endpoint}/v1/sessions/{session_id}/messages", json={"turn_id": trajectory_uid, "messages": messages}); result.raise_for_status()
                for cmd in metadata.get("after_agent_commands", []): await client.post(f"{lease.endpoint}/v1/sessions/{session_id}/execute_cmd", json={"cmd": cmd})
                if grader := metadata.get("grader_command"):
                    graded = await client.post(f"{lease.endpoint}/v1/sessions/{session_id}/execute_cmd", json={"cmd": grader})
                    graded.raise_for_status()
                    try:
                        score = float(json.loads(graded.json()["stdout"])["overall_score"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        score = 0.0
                    await client.post(f"{base_url}/complete_trajectory", json={"reward": score})
                return 1
        finally:
            await provider.release(lease)
