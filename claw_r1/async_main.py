"""Async PPO training entry point -- AsyncTaskRunner.

Creates and coordinates AsyncRollouter (on rollout GPU pool) and AsyncTrainer
(on trainer GPU pool), connected through a shared DataPool and
ParameterSynchronizer.

Usage::

    python -m claw_r1.async_main ...hydra overrides...

Based on ``verl/recipe/fully_async_policy/fully_async_main.py``.
"""

from __future__ import annotations

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device

# -- Resource pool helpers ------------------------------------------------


def _create_resource_pool_manager(config, roles: list[Role]) -> ResourcePoolManager:
    """Build separate resource pools for trainer vs. rollout roles."""
    resource_pool_spec = {}
    mapping = {}

    trainer_roles = {Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel}
    if trainer_roles & set(roles):
        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool
        for role in trainer_roles:
            if role in roles:
                mapping[role] = "trainer_pool"

    if Role.Rollout in roles:
        rollout_cfg = config.get("rollout", config.trainer)
        rollout_pool = [rollout_cfg.n_gpus_per_node] * rollout_cfg.nnodes
        resource_pool_spec["rollout_pool"] = rollout_pool
        mapping[Role.Rollout] = "rollout_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def _create_role_worker_mapping(config):
    """Create mapping from Roles to worker classes for async mode."""
    from claw_r1.detach_workers import (
        CriticWorker,
        DetachActorWorker,
        DetachAsyncRolloutWorker,
    )
    from verl.single_controller.ray import RayWorkerGroup

    role_worker_mapping = {
        Role.Actor: ray.remote(DetachActorWorker),
        Role.Rollout: ray.remote(DetachAsyncRolloutWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, RayWorkerGroup


# -- AsyncTaskRunner ------------------------------------------------------


@ray.remote(num_cpus=1)
class AsyncTaskRunner:
    """Orchestrates async training: creates components, wires them, starts loops."""

    def __init__(self):
        self.components: dict = {}

    def run(self, config):
        print("[ASYNC] Starting fully-async PPO training...")
        self._initialize(config)
        self._run()

    # -- Setup ------------------------------------------------------------

    def _initialize(self, config):
        print(f"[ASYNC] host={socket.gethostname()} pid={os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        role_worker_mapping, ray_worker_group_cls = _create_role_worker_mapping(config)

        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(role_worker_mapping),
            use_critic=need_critic(config),
        )

        self.components.update(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        # -- DataPool -----------------------------------------------------
        from claw_r1.data_pool import DataPool, DataPoolConfig, VerlBackend

        enable_tree = config.async_training.get("enable_prefix_tree_merge", False)
        if enable_tree:
            from claw_r1.data_pool import TreeVerlBackend

            verl_backend = TreeVerlBackend(
                tokenizer=tokenizer,
                prompt_length=config.actor_rollout_ref.rollout.prompt_length,
                response_length=config.actor_rollout_ref.rollout.response_length,
            )
        else:
            verl_backend = VerlBackend(
                tokenizer=tokenizer,
                prompt_length=config.actor_rollout_ref.rollout.prompt_length,
                response_length=config.actor_rollout_ref.rollout.response_length,
            )
        data_pool_config = DataPoolConfig(
            n_rollouts=config.actor_rollout_ref.rollout.n,
        )
        max_queue_size = config.async_training.get("max_queue_size", None)
        data_pool_name = "data_pool"
        data_pool = DataPool.options(name=data_pool_name).remote(
            data_pool_config,
            verl_backend,
            max_queue_size=max_queue_size,
        )
        self.components["data_pool"] = data_pool
        self.components["data_pool_name"] = data_pool_name

        # -- RewardLoopWorker ---------------------------------------------
        # Always created so the Gateway can compute rule-based or model-based
        # rewards, consistent with the synchronous Trainer.
        from claw_r1.reward_loop import RewardLoopWorker

        reward_worker_name = "reward_loop_worker"
        reward_worker = RewardLoopWorker.options(
            name=reward_worker_name,
        ).remote(config, None)
        self.components["reward_worker"] = reward_worker
        self.components["reward_worker_name"] = reward_worker_name

        # -- Rollouter ----------------------------------------------------
        self._create_rollouter(config)

        # -- Trainer ------------------------------------------------------
        self._create_trainer(config)

        # -- Wire up ------------------------------------------------------
        rollouter = self.components["rollouter"]
        trainer = self.components["trainer"]

        ray.get(trainer.set_data_pool_name.remote(data_pool_name))

        total_train_steps = ray.get(rollouter.get_total_train_steps.remote())
        ray.get(trainer.set_total_train_steps.remote(total_train_steps))
        print(f"[ASYNC] total_train_steps = {total_train_steps}")

        # -- ParameterSynchronizer ----------------------------------------
        from claw_r1.param_sync import ParameterSynchronizer

        ps = ParameterSynchronizer.remote(
            config=config,
            trainer=trainer,
            rollouter=rollouter,
        )
        ray.get(trainer.set_parameter_synchronizer.remote(ps))
        self.components["param_synchronizer"] = ps

        val_before_train = config.trainer.get("val_before_train", False)
        ray.get(ps.sync_weights.remote(version=0, validate=val_before_train))
        ray.get(ps.wait_last_valid.remote())

        print("[ASYNC] All components initialized")

    def _create_rollouter(self, config):
        from claw_r1.async_rollouter import AsyncRollouter

        rollouter = AsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={
                Role.Rollout: self.components["role_worker_mapping"][Role.Rollout],
            },
            resource_pool_manager=_create_resource_pool_manager(config, [Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        # Gateway needs these before init_workers (which starts the Gateway subprocess)
        ray.get(rollouter.set_data_pool_name.remote(self.components["data_pool_name"]))
        ray.get(rollouter.set_reward_worker_name.remote(self.components["reward_worker_name"]))
        ray.get(rollouter.init_workers.remote())
        self.components["rollouter"] = rollouter
        print("[ASYNC] Rollouter created")

    def _create_trainer(self, config):
        from claw_r1.async_trainer import AsyncTrainer

        trainer_roles = [role for role in self.components["role_worker_mapping"] if role != Role.Rollout]
        trainer = AsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={
                role: cls for role, cls in self.components["role_worker_mapping"].items() if role != Role.Rollout
            },
            resource_pool_manager=_create_resource_pool_manager(config, trainer_roles),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC] Trainer created")

    # -- Main loop --------------------------------------------------------

    def _run(self):
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                done, remaining = ray.wait(futures, num_returns=1, timeout=None)
                for f in done:
                    try:
                        ray.get(f)
                        print("[ASYNC] Component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC] Component failed: {e}")
                        for r in remaining:
                            ray.cancel(r)
                        raise
                futures = remaining
        except Exception as e:
            print(f"[ASYNC] Training failed: {e}")
            for f in futures:
                ray.cancel(f)
            raise
        finally:
            print("[ASYNC] Training finished")


# -- Hydra entry point ----------------------------------------------------


@hydra.main(config_path="config", config_name="async_ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(namespace="claw_r1_async", **OmegaConf.to_container(ray_init_kwargs))

    if not hasattr(config, "async_training"):
        raise RuntimeError("async_training config section is required")

    runner = AsyncTaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
