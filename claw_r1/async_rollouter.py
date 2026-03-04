"""AsyncRollouter — continuous data generation for async training.

Owns the DataLoader, vLLM rollout replicas, AgentFlowManager, Gateway, and
RewardLoopWorker.  Runs as a Ray Actor on a dedicated GPU pool, continuously
generating trajectories and submitting them to the shared DataPool.

Based on ``verl/recipe/fully_async_policy/fully_async_rollouter.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from pprint import pformat
from typing import Any

import numpy as np
import ray
import torch
from ray import ObjectRef

from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote(num_cpus=10, max_concurrency=100)
class AsyncRollouter:
    """Continuous rollout generator for fully-async PPO training."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name or self.config.trainer.device

        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        # Worker groups
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Async config
        self.require_batches = config.async_training.require_batches
        self.required_samples = (
            config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        )
        self.staleness_threshold = config.async_training.get("staleness_threshold", 1)

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.global_steps = 1

        # Concurrency control
        self.paused = False
        self.running = True

        # Shared references (set externally)
        self._data_pool_name: str | None = None
        self._reward_worker_name: str | None = None
        self._gateway_url: str | None = None
        self._gateway_process = None

        # DataLoader — reuse the same creation logic as the synchronous Trainer
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        from torchdata.stateful_dataloader import StatefulDataLoader

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        num_workers = config.data.get("dataloader_num_workers", 0)

        self.train_dataset = train_dataset
        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = config.data.get("val_batch_size", None)
        if val_batch_size is None:
            val_batch_size = len(val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.total_rollout_steps = (
            len(self.train_dataloader) * config.trainer.total_epochs
        )
        rollout_limit = config.get("rollout", {}).get("total_rollout_steps")
        if rollout_limit:
            self.total_rollout_steps = min(self.total_rollout_steps, rollout_limit)

        gen_batch_size = config.data.get("gen_batch_size", config.data.train_batch_size)
        self.total_train_steps = int(
            self.total_rollout_steps * gen_batch_size / config.data.train_batch_size
        )

    # ── External wiring ──────────────────────────────────────────────────

    def set_data_pool_name(self, name: str):
        self._data_pool_name = name

    def set_reward_worker_name(self, name: str):
        self._reward_worker_name = name

    def get_rollout_wg(self):
        return self.rollout_wg

    def get_total_train_steps(self):
        return self.total_train_steps

    # ── Worker initialisation ────────────────────────────────────────────

    async def init_workers(self):
        """Initialise rollout worker group and vLLM replicas."""
        self._init_async_objects()
        self.resource_pool_manager.create_resource_pool()

        resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role=str(Role.Rollout),
        )
        resource_pool_to_cls[resource_pool][str(Role.Rollout)] = rollout_cls

        from verl.single_controller.ray.base import create_colocated_worker_cls
        for pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            self.rollout_wg = spawn_wg[str(Role.Rollout)]

        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

        await self._init_rollout_replicas()
        self._init_gateway()
        self._init_agent_flow_manager()

    def _init_async_objects(self):
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

    async def _init_rollout_replicas(self):
        """Create and start vLLM replicas on the rollout worker group."""
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        replica_cls = get_rollout_replica_class(rollout_config.name)

        rollout_world_size = (
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        world_size = self.rollout_wg.world_size
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            replica_cls(
                replica_rank=rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.get("rollout", {}).get(
                    "n_gpus_per_node", self.config.trainer.n_gpus_per_node,
                ),
            )
            for rank in range(num_replicas)
        ]

        await asyncio.gather(
            *[r.init_hybrid(self.rollout_wg) for r in self.rollout_replicas]
        )

        self._server_handles = [r._server_handle for r in self.rollout_replicas]
        self._server_addresses = [r._server_address for r in self.rollout_replicas]
        logger.info("Rollout replicas at %s", self._server_addresses)

    def _init_gateway(self):
        """Start the Gateway Server as a subprocess."""
        import atexit
        import subprocess

        import httpx

        gateway_port = self.config.trainer.get("gateway_port", 8100)

        ray_ctx = ray.get_runtime_context()
        ray_address = ray_ctx.gcs_address
        ray_namespace = ray_ctx.namespace

        cmd = [
            "python", "-m", "claw_r1.gateway.gateway",
            "--data-pool-name", self._data_pool_name,
            "--vllm-addresses", ",".join(self._server_addresses),
            "--tokenizer-path", self.config.actor_rollout_ref.model.path,
            "--prompt-length", str(self.config.actor_rollout_ref.rollout.prompt_length),
            "--response-length", str(self.config.actor_rollout_ref.rollout.response_length),
            "--reward-worker-name", self._reward_worker_name,
            "--ray-address", ray_address,
            "--ray-namespace", ray_namespace,
            "--port", str(gateway_port),
        ]

        self._gateway_process = subprocess.Popen(cmd)
        self._gateway_url = f"http://localhost:{gateway_port}"
        atexit.register(self._stop_gateway)

        for _ in range(120):
            try:
                resp = httpx.get(f"{self._gateway_url}/docs", timeout=2.0)
                if resp.status_code == 200:
                    logger.info("Gateway ready at %s", self._gateway_url)
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"Gateway did not start within 120s ({self._gateway_url})")

    def _stop_gateway(self):
        proc = getattr(self, "_gateway_process", None)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)

    def _init_agent_flow_manager(self):
        from claw_r1.agent_flow import AgentFlowManager
        self.async_rollout_manager = AgentFlowManager(
            config=self.config,
            gateway_url=self._gateway_url,
        )

    # ── Generation loop ──────────────────────────────────────────────────

    async def fit(self):
        """Main entry — runs generation and monitoring concurrently."""
        async with self.lock:
            self.paused = False
            self.running = True

        gen_task = asyncio.create_task(self._generation_main())
        monitor_task = asyncio.create_task(self._monitor_loop())

        try:
            await asyncio.gather(gen_task, monitor_task, return_exceptions=True)
        finally:
            for t in (gen_task, monitor_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(gen_task, monitor_task, return_exceptions=True)

        # Signal DataPool shutdown
        data_pool = ray.get_actor(self._data_pool_name)
        ray.get(data_pool.shutdown.remote())
        logger.info("Rollouter fit completed")

    async def _generation_main(self):
        """Iterate over epochs/batches, generate sequences, submit to DataPool."""
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # Respect pause
                async with self.lock:
                    while self.paused:
                        await self.condition.wait()

                batch = DataProto.from_single_dict(batch_dict)

                if "uid" not in batch.non_tensor_batch:
                    import uuid
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch))],
                        dtype=object,
                    )

                gen_batch = self._prepare_gen_batch(batch)
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "global_steps": self.global_steps,
                }

                self.async_rollout_manager.generate_sequences(gen_batch)

                self.total_generated_samples += len(gen_batch)
                self.global_steps += 1

                if self.global_steps > self.total_rollout_steps:
                    logger.info("Reached total_rollout_steps, stopping generation")
                    return

        logger.info("Finished all epochs")

    def _prepare_gen_batch(self, batch: DataProto) -> DataProto:
        """Repeat batch n times and add temperature / agent_name metadata."""
        n = self.config.actor_rollout_ref.rollout.n
        gen_batch = batch.repeat(repeat_times=n, interleave=True)

        rollout_cfg = self.config.actor_rollout_ref.rollout
        temperature = rollout_cfg.get("temperature", 1.0)
        gen_batch.non_tensor_batch["temperature"] = np.array(
            [temperature] * len(gen_batch), dtype=np.float32,
        )

        agent_cfg = rollout_cfg.get("agent", {})
        default_flow = agent_cfg.get("default_agent_flow", "single_step_single_turn_agent")
        gen_batch.non_tensor_batch["agent_name"] = np.array(
            [default_flow] * len(gen_batch), dtype=object,
        )

        return gen_batch

    # ── Pause / resume ───────────────────────────────────────────────────

    async def pause(self):
        """Pause generation (called by ParameterSynchronizer before weight sync)."""
        async with self.lock:
            self.paused = True

    async def resume(self, dependency_ref: ObjectRef = None):
        """Resume generation after weight sync."""
        if dependency_ref is not None:
            ray.get(dependency_ref)
        async with self.lock:
            self.paused = False
            self.condition.notify_all()

    async def update_param_version(
        self, version: int, validate: bool = False, global_steps: int = 0,
    ):
        """Update current parameter version; optionally run validation."""
        async with self.lock:
            self.current_param_version = version
            self.staleness_samples = 0

        val_metrics = None
        if validate and self.val_reward_fn is not None:
            val_metrics = self._validate()

        from ray import cloudpickle as ray_cloudpickle
        data_pool = ray.get_actor(self._data_pool_name)
        val_data = {
            "metrics": val_metrics,
            "param_version": version,
            "global_steps": global_steps,
        }
        ray.get(data_pool.put_validate.remote(ray_cloudpickle.dumps(val_data)))

    async def save_checkpoint(self, path: str):
        """Save dataloader state (checkpoint support)."""
        pass  # TODO: implement dataloader state saving

    # ── Validation ───────────────────────────────────────────────────────

    def _validate(self) -> dict | None:
        """Run validation and return metrics dict."""
        if self.val_dataloader is None:
            return None

        import uuid
        from verl.trainer.ppo.ray_trainer import compute_data_metrics

        all_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch))],
                    dtype=object,
                )

            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            gen_batch = self._prepare_gen_batch_for_val(test_batch)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            num_val_prompts = len(test_batch) // val_n if val_n > 1 else len(test_batch)

            gen_meta = self.async_rollout_manager.generate_sequences(gen_batch)

            data_pool = ray.get_actor(self._data_pool_name)
            output_batch = ray.get(
                data_pool.fetch_batch.remote(
                    batch_size=num_val_prompts, n_rollouts=val_n,
                )
            )

            if output_batch is None:
                continue

            output_batch.meta_info["validate"] = True

            result = self.val_reward_fn(output_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().numpy().tolist()
            all_scores.extend(scores)

        if not all_scores:
            return None

        return {
            "val/reward_mean": float(np.mean(all_scores)),
            "val/reward_std": float(np.std(all_scores)),
            "val/num_samples": len(all_scores),
        }

    def _prepare_gen_batch_for_val(self, batch: DataProto) -> DataProto:
        """Prepare a generation batch for validation (no repeat, already done)."""
        rollout_cfg = self.config.actor_rollout_ref.rollout
        temperature = rollout_cfg.val_kwargs.get("temperature", 1.0)
        batch.non_tensor_batch["temperature"] = np.array(
            [temperature] * len(batch), dtype=np.float32,
        )
        agent_cfg = rollout_cfg.get("agent", {})
        default_flow = agent_cfg.get("default_agent_flow", "single_step_single_turn_agent")
        batch.non_tensor_batch["agent_name"] = np.array(
            [default_flow] * len(batch), dtype=object,
        )
        return batch

    # ── Monitor loop ─────────────────────────────────────────────────────

    async def _monitor_loop(self):
        """Periodic statistics logging."""
        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(60)
            stats = self._get_stats()
            logger.info("[AsyncRollouter] %s", pformat(stats))

    def _get_stats(self) -> dict:
        return {
            "param_version": self.current_param_version,
            "total_generated": self.total_generated_samples,
            "global_steps": self.global_steps,
            "paused": self.paused,
        }
