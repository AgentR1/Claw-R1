"""AsyncTrainer — continuous PPO training for fully-async mode.

Owns Actor/Critic/RefPolicy worker groups on a dedicated GPU pool.
Continuously fetches batches from the shared DataPool and performs PPO
training steps. Triggers ParameterSynchronizer to push updated weights
to the AsyncRollouter after every N training steps.

Based on ``verl/recipe/fully_async_policy/fully_async_trainer.py``.
"""

from __future__ import annotations

import logging
import math
import os
from functools import reduce
from pprint import pprint

import numpy as np
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from claw_r1.ray_agent_trainer import get_valid_data
from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, compute_data_metrics, compute_response_mask
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote(num_cpus=10)
class AsyncTrainer:
    """Continuous PPO trainer for fully-async mode."""

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

        self.hybrid_engine = False
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        self.reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        self.val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                self.config.algorithm.kl_ctrl,
            )

        # Async config
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.use_rollout_log_probs = config.async_training.get("use_rollout_log_probs", True)

        # Statistics
        self.global_steps = 1
        self.local_trigger_step = 1
        self.current_param_version = 0
        self.total_train_steps: int | None = None
        self.processed_samples = 0

        # Worker groups (set during init_workers)
        self.actor_wg = None
        self.actor_rollout_wg = None
        self.critic_wg = None
        self.ref_policy_wg = None

        # External references (set by AsyncTaskRunner)
        self._data_pool_name: str | None = None
        self._param_synchronizer = None

    # ── External wiring ──────────────────────────────────────────────────

    def set_data_pool_name(self, name: str):
        self._data_pool_name = name

    def set_parameter_synchronizer(self, ps):
        self._param_synchronizer = ps

    def set_total_train_steps(self, steps: int):
        self.total_train_steps = steps

    def get_actor_wg(self):
        return self.actor_wg

    # ── Worker initialisation ────────────────────────────────────────────

    def init_workers(self):
        """Create Actor, Critic, RefPolicy worker groups."""
        self.resource_pool_manager.create_resource_pool()

        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Actor
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role=str(Role.Actor),
        )
        resource_pool_to_cls[resource_pool][str(Role.Actor)] = actor_cls

        # Critic
        if self.use_critic:
            from verl.utils.config import omega_conf_to_dataclass
            from verl.workers.config import CriticConfig

            crit_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic, CriticConfig)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=critic_cfg,
            )
            resource_pool_to_cls[crit_pool][str(Role.Critic)] = critic_cls

        # Reference policy
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            resource_pool_to_cls[ref_pool][str(Role.RefPolicy)] = ref_cls

        # Spawn worker groups
        all_wg = {}
        for pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Critic
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        # Reference policy
        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

        # Actor (last so vLLM estimate is not affected)
        self.actor_wg = all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_wg

    # ── Training loop ────────────────────────────────────────────────────

    def fit(self):
        """Main training loop — fetch batch from DataPool, train, sync."""
        if self._data_pool_name is None:
            raise ValueError("DataPool name not set")

        from verl.utils.tracking import Tracking

        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        progress_bar = tqdm(
            total=self.total_train_steps or 0,
            initial=0,
            desc="AsyncTrainer",
        )

        data_pool = ray.get_actor(self._data_pool_name)

        self._check_validation_data(tracking)

        while True:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                # Fetch batch from DataPool
                with marked_timer("fetch", timing_raw, color="red"):
                    batch = ray.get(
                        data_pool.fetch_batch.remote(
                            batch_size=self.config.data.train_batch_size,
                        )
                    )
                    if batch is None:
                        print("[AsyncTrainer] DataPool returned None, stopping")
                        break

                # Process batch
                batch = self._process_batch(batch, metrics, timing_raw)

            # Metrics
            self._collect_metrics(batch, metrics, timing_raw)
            tracking.log(data=metrics, step=self.global_steps)

            # Trigger parameter sync
            self._trigger_parameter_sync(self.global_steps)

            # Checkpoint
            self._maybe_save_checkpoint(timing_raw)

            # Check validation data from Rollouter
            self._check_validation_data(tracking)

            progress_bar.update(1)
            self.global_steps += 1

            if self.total_train_steps and self.global_steps >= self.total_train_steps:
                print(f"[AsyncTrainer] Reached total_train_steps={self.total_train_steps}")
                break

        progress_bar.close()

        # Final sync + validation (after progress bar so timing is accurate)
        if self._param_synchronizer:
            self._trigger_parameter_sync(
                self.global_steps,
                validate=True,
                force=True,
            )
            ray.get(self._param_synchronizer.wait_last_valid.remote())
            self._check_validation_data(tracking)

        print("[AsyncTrainer] Training finished")

    # ── Batch processing ─────────────────────────────────────────────────

    def _process_batch(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        """Run the full PPO pipeline on a single batch."""
        batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(dim=-1).tolist()

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        batch = self._pad_to_world_size(batch)

        # Reward
        with marked_timer("reward", timing_raw, color="yellow"):
            reward_tensor, reward_extra_infos_dict = self._compute_reward(batch)

        # Old log probs
        if self.use_rollout_log_probs and "rollout_log_probs" in batch.batch:
            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        else:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self._compute_old_log_prob(batch)
                batch = batch.union(old_log_prob)

        # Reference log probs
        if self.use_reference_policy:
            with marked_timer("ref_log_prob", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # Values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)

        # Advantage
        with marked_timer("adv", timing_raw, color="brown"):
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                from verl.trainer.ppo.ray_trainer import apply_kl_penalty

                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            from verl.trainer.ppo.ray_trainer import compute_advantage

            norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv,
                config=self.config.algorithm,
            )

        # Update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                import torch

                response_mask = batch.batch["response_mask"]
                value_mask = torch.zeros_like(response_mask)
                value_mask[:, 0] = 1
                batch.batch["response_mask"] = value_mask
                critic_output = self.critic_wg.update_critic(batch)
                batch.batch["response_mask"] = response_mask
                from verl.trainer.ppo.ray_trainer import reduce_metrics

                metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        # Update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self.actor_wg.update_actor(batch)
                from verl.trainer.ppo.ray_trainer import reduce_metrics

                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        return batch

    # ── Reward ────────────────────────────────────────────────────────────

    def _compute_reward(self, batch: DataProto):
        """Compute or extract reward for training."""
        if "rm_scores" in batch.batch:
            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return batch.batch["rm_scores"], reward_extra_infos_dict

        if self.reward_fn is not None:
            from verl.trainer.ppo.reward import compute_reward

            return compute_reward(batch, self.reward_fn)

        raise ValueError("No reward_fn and no pre-computed rm_scores in batch")

    # ── Compute helpers (delegate to worker groups) ──────────────────────

    def _compute_old_log_prob(self, batch: DataProto):
        return self.actor_wg.compute_log_prob(batch)

    def _compute_ref_log_prob(self, batch: DataProto):
        return self.ref_policy_wg.compute_ref_log_prob(batch)

    def _compute_values(self, batch: DataProto):
        return self.critic_wg.compute_values(batch)

    # ── Parameter sync ───────────────────────────────────────────────────

    def _trigger_parameter_sync(
        self,
        global_steps: int,
        validate: bool = False,
        force: bool = False,
    ):
        """Trigger weight sync after N local training steps."""
        if self._param_synchronizer is None:
            return

        if force or self.local_trigger_step >= self.trigger_parameter_sync_step:
            self.current_param_version += 1
            should_validate = validate or (
                self.config.trainer.test_freq > 0 and self.current_param_version % self.config.trainer.test_freq == 0
            )
            ray.get(
                self._param_synchronizer.sync_weights.remote(
                    version=self.current_param_version,
                    validate=should_validate,
                    global_steps=global_steps,
                )
            )
            self.local_trigger_step = 1
            print(f"[AsyncTrainer] Synced weights v{self.current_param_version} (validate={should_validate})")
        else:
            self.local_trigger_step += 1

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _maybe_save_checkpoint(self, timing_raw: dict):
        """Save checkpoint if conditions are met."""
        if self.config.trainer.save_freq <= 0:
            return

        is_last = self.total_train_steps is not None and self.global_steps >= self.total_train_steps
        if is_last or self.global_steps % self.config.trainer.save_freq == 0:
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()

    def _save_checkpoint(self):
        """Save actor checkpoint."""
        save_path = os.path.join(
            self.config.trainer.default_local_dir,
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            f"global_step_{self.global_steps}",
        )
        self.actor_wg.save_checkpoint(save_path)
        print(f"[AsyncTrainer] Checkpoint saved at {save_path}")

    # ── Validation data ──────────────────────────────────────────────────

    def _check_validation_data(self, tracking):
        """Check if the Rollouter pushed validation metrics to DataPool."""
        data_pool = ray.get_actor(self._data_pool_name)
        val_data = ray.get(data_pool.get_validate.remote())
        if val_data is None:
            return

        from ray import cloudpickle as ray_cloudpickle

        val_data = ray_cloudpickle.loads(val_data)
        val_metrics = val_data.get("metrics")
        if val_metrics:
            tracking.log(data=val_metrics, step=val_data.get("global_steps", self.global_steps))
            pprint(f"[AsyncTrainer] Validation: {val_metrics}")

    # ── Metrics ───────────────────────────────────────────────────────────

    def _collect_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Collect training metrics."""
        metrics["training/global_step"] = self.global_steps
        metrics["training/param_version"] = self.current_param_version

        valid_batch, _ = get_valid_data(batch)
        metrics.update(compute_data_metrics(batch=valid_batch, use_critic=self.use_critic))

        # DataPool statistics
        try:
            data_pool = ray.get_actor(self._data_pool_name)
            pool_stats = ray.get(data_pool.get_statistics.remote())
            for k, v in pool_stats.items():
                metrics[f"data_pool/{k}"] = v
        except Exception:
            pass

    # ── Padding ───────────────────────────────────────────────────────────

    def _pad_to_world_size(self, batch: DataProto) -> DataProto:
        """Pad batch to be divisible by all worker group world sizes."""
        world_sizes = []
        if self.use_critic and self.critic_wg and self.critic_wg.world_size:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg and self.ref_policy_wg.world_size:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.actor_wg and self.actor_wg.world_size:
            world_sizes.append(self.actor_wg.world_size)

        if not world_sizes:
            return batch

        ws = reduce(math.lcm, world_sizes)
        original_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, ws)
        batch.non_tensor_batch["is_pad"] = np.array(
            [False] * original_size + [True] * pad_size,
        )
        return batch
