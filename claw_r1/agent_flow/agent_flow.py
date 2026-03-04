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
"""Agent flow framework — refactored to use Gateway as the sole bridge to Training Side.

All LLM generation, reward computation, and trajectory submission go through
the Gateway HTTP service.  AgentFlowBase subclasses implement specific agent
strategies; they call Gateway helpers and construct / submit Step objects.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
import hydra
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from claw_r1.data_pool.data_model import Step
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ── Shared HTTP client (one per process) ──────────────────────────────────

_shared_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None:
        _shared_http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    return _shared_http_client


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of *value* to a JSON-serialisable form."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Quick round-trip check; drop anything non-serialisable.
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return None


# ── AgentFlowBase ─────────────────────────────────────────────────────────


class AgentFlowBase(ABC):
    """Base class for agent flows.

    An agent flow takes an input message, interacts with an LLM via the
    Gateway Server, optionally interacts with tool environments, then
    constructs and submits :class:`Step` objects to the DataPool (also via
    the Gateway).
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        gateway_url: str,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.gateway_url = gateway_url.rstrip("/")
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.apply_chat_template_kwargs = dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    # ── Vision helpers (unchanged) ────────────────────────────────────────

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages."""
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                config=self.dataset_config,
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple] = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        """Apply chat template to messages and return prompt token IDs."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        return prompt_ids

    # ── Gateway HTTP helpers ──────────────────────────────────────────────

    async def gateway_generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Call ``POST /generate`` on the Gateway.

        Returns a dict with ``token_ids``, ``log_probs``, and ``stop_reason``.
        """
        client = _get_http_client()
        resp = await client.post(
            f"{self.gateway_url}/generate",
            json={
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def gateway_compute_reward(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        num_turns: int = 2,
        extra_fields: dict[str, Any] | None = None,
        dataset_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call ``POST /compute_reward`` on the Gateway.

        Returns a dict with ``reward_score`` and ``reward_extra_info``.
        """
        client = _get_http_client()
        safe_dataset = {k: _json_safe(v) for k, v in (dataset_fields or {}).items()}
        safe_dataset = {k: v for k, v in safe_dataset.items() if v is not None}

        resp = await client.post(
            f"{self.gateway_url}/compute_reward",
            json={
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "num_turns": num_turns,
                "extra_fields": extra_fields or {},
                "dataset_fields": safe_dataset,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def gateway_submit_steps(
        self,
        steps: list[Step],
        channel: str | None = None,
    ) -> int:
        """Submit Steps to DataPool via ``POST /submit_steps``.

        Args:
            steps: Steps to submit.
            channel: DataPool channel name.  When *None* the Gateway uses
                its default (``"train"``).

        Returns the number of accepted steps.
        """
        client = _get_http_client()
        payloads = []
        for s in steps:
            payloads.append(
                {
                    "prompt_ids": s.prompt_ids,
                    "response_ids": s.response_ids,
                    "reward": s.reward,
                    "rollout_log_probs": s.rollout_log_probs,
                    "trajectory_uid": s.trajectory_uid,
                    "prompt_uid": s.prompt_uid,
                    "step_index": s.step_index,
                    "policy_version": s.policy_version,
                    "is_last": s.is_last,
                    "metadata": _json_safe(s.metadata),
                }
            )
        url = f"{self.gateway_url}/submit_steps"
        params = {"channel": channel} if channel else None
        resp = await client.post(url, json={"steps": payloads}, params=params)
        resp.raise_for_status()
        return resp.json()["accepted"]

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> int:
        """Run the agent flow for one trajectory.

        Implementations should:

        1. Generate LLM responses via :meth:`gateway_generate`.
        2. Construct :class:`Step` objects (``reward`` defaults to *None*;
           the Trainer computes it after ``fetch_batch``).
        3. Submit them via :meth:`gateway_submit_steps`.

        Optionally, reward can be pre-computed via
        :meth:`gateway_compute_reward` and set on the Step before
        submission.

        Args:
            sampling_params: LLM sampling parameters.
            **kwargs: Dataset fields from ``verl.utils.dataset.RLHFDataset``.

        Returns:
            Number of Steps submitted (0 on failure).
        """
        raise NotImplementedError


# ── Agent flow registry ───────────────────────────────────────────────────

_agent_flow_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register an AgentFlowBase subclass under *agent_name*."""

    def decorator(subclass: type[AgentFlowBase]) -> type[AgentFlowBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_flow_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


# ── Worker ────────────────────────────────────────────────────────────────


class AgentFlowWorkerBase:
    """Worker that takes a batch of prompts and runs each through an agent flow.

    Steps are submitted to DataPool via the Gateway during each ``run()``.
    ``generate_sequences`` returns a metadata dict (not a DataProto).
    """

    def __init__(self, config: DictConfig, gateway_url: str):
        self.config = config
        self.gateway_url = gateway_url

        self.dataset_cls = get_dataset_class(config.data)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_flow_config_path = config.actor_rollout_ref.rollout.agent.agent_flow_config_path
        if agent_flow_config_path:
            from verl.experimental.agent_loop.utils import resolve_config_path

            resolved_path = resolve_config_path(agent_flow_config_path)
            agent_flow_configs = OmegaConf.load(resolved_path)
            for afc in agent_flow_configs:
                _agent_flow_registry[afc.name] = afc

        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    async def generate_sequences(self, batch: DataProto) -> dict:
        """Run agent flows for every item in *batch*.

        Steps are submitted to DataPool via the Gateway during execution.

        Returns:
            dict with ``num_steps`` (list[int]) and ``timing`` (dict).
        """
        t_start = time.time()

        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_flow = config.agent.default_agent_flow
            batch.non_tensor_batch["agent_name"] = np.array(
                [default_agent_flow] * len(batch),
                dtype=object,
            )

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples is not None:
            unique_indices = np.unique(index)
            if max_samples < len(unique_indices):
                selected = set(np.random.choice(unique_indices, max_samples, replace=False).tolist())
                traced_indices = {i for i in range(len(batch)) if index[i] in selected}
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1),
            index.tolist(),
            batch.meta_info.get("validate", False),
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_flow(
                        sampling_params,
                        trajectory_info[i],
                        trace=i in traced_indices,
                        **kwargs,
                    )
                )
            )
        results = await asyncio.gather(*tasks)

        total_time = time.time() - t_start
        num_steps = list(results)

        return {
            "num_steps": num_steps,
            "timing": {"agent_flow/total_time": total_time},
        }

    async def _run_agent_flow(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> int:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_flow",
            trace=trace,
        ):
            assert agent_name in _agent_flow_registry, (
                f"Agent flow {agent_name} not registered, registered: {list(_agent_flow_registry.keys())}"
            )

            agent_flow_config = _agent_flow_registry[agent_name]
            agent_flow: AgentFlowBase = hydra.utils.instantiate(
                config=agent_flow_config,
                trainer_config=DictConfigWrap(config=self.config),
                gateway_url=self.gateway_url,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            return await agent_flow.run(sampling_params, **kwargs)


@ray.remote
class AgentFlowWorker(AgentFlowWorkerBase):
    """Ray-actor wrapper around :class:`AgentFlowWorkerBase`."""

    def __init__(self, config: DictConfig, gateway_url: str):
        super().__init__(config, gateway_url)


# ── Helper ────────────────────────────────────────────────────────────────


async def get_trajectory_info(step: int, index: list, validate: bool) -> list[dict]:
    """Build per-sample trajectory metadata for tracing."""
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


# ── Manager ───────────────────────────────────────────────────────────────


class AgentFlowManager:
    """Manages a group of :class:`AgentFlowWorker` actors.

    Unlike the pre-refactor version, this manager does **not** own the
    Rollout Engine or Reward Model.  Lifecycle management (wake_up / sleep)
    is handled by the Trainer directly.
    """

    def __init__(self, config: DictConfig, gateway_url: str):
        self.config = config
        self.gateway_url = gateway_url

        if not hasattr(self, "agent_flow_workers_class"):
            self.agent_flow_workers_class = AgentFlowWorker

        self._init_agent_flow_workers()

    def _init_agent_flow_workers(self):
        self.agent_flow_workers: list[ray.actor.ActorHandle] = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_flow_workers.append(
                self.agent_flow_workers_class.options(
                    name=f"agent_flow_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.gateway_url)
            )

    def generate_sequences(self, prompts: DataProto) -> dict:
        """Dispatch prompt batch across workers and collect metadata.

        Steps are submitted to DataPool via the Gateway during execution.

        Returns:
            dict with ``num_steps`` (list[int]) and ``timing`` (dict).
        """
        split_size = (len(prompts) - 1) // len(self.agent_flow_workers) + 1
        chunks = prompts.split(split_size)
        results = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_flow_workers, chunks, strict=True)
            ]
        )

        all_num_steps: list[int] = []
        combined_timing: dict[str, float] = {}
        for result in results:
            all_num_steps.extend(result["num_steps"])
            for k, v in result["timing"].items():
                if k not in combined_timing:
                    combined_timing[k] = v
                elif isinstance(v, (int, float)):
                    combined_timing[k] = max(combined_timing[k], v)

        return {
            "num_steps": all_num_steps,
            "timing": combined_timing,
        }
