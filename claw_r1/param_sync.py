"""ParameterSynchronizer — coordinates weight sync between Trainer and Rollouter.

Based on ``verl/recipe/fully_async_policy/param_sync.py``.

Flow::

    sync_weights(version)
      1. ray.get(rollouter.pause())           # stop generation, clear KV cache
      2. NCCL broadcast: Actor → vLLM          # via DetachActorWorker / DetachAsyncRolloutWorker
      3. rollouter.update_param_version(...)   # async — updates version, optionally validates
      4. rollouter.resume(...)                 # async — resumes generation
"""

import logging
import time

import ray
from ray.util.collective import collective

from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """Synchronizes model parameters between the actor (Trainer) and rollout (Rollouter).

    Creates an NCCL collective group spanning actor and rollout workers so
    that ``sync_weights`` can broadcast the latest actor parameters to the
    vLLM inference replicas.
    """

    def __init__(self, config, trainer, rollouter):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter

        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        self.weights_info = None
        self.sync_group_name = "actor_rollout"
        self.current_version = 0

        self._wait_last_update = None
        self._wait_last_resume = None

        self._init_weights_info()
        self._init_sync_group()

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        logger.info("Initializing NCCL sync group...")
        workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            workers,
            len(workers),
            list(range(len(workers))),
            backend=get_nccl_backend(),
            group_name=self.sync_group_name,
        )

    def get_current_param_version(self) -> int:
        return self.current_version

    def sync_weights(self, version: int, validate: bool = False, global_steps: int = 0):
        """Pause rollout, broadcast weights, then resume."""
        start = time.time()
        self.current_version = version

        ray.get(self.rollouter.pause.remote())
        pause_time = time.time()
        logger.info("Rollouter paused in %.2fs", pause_time - start)

        self.actor_wg.sync_rollout_weights(self.sync_group_name)
        ray.get(self.rollout_wg.sync_rollout_weights(self.sync_group_name))

        sync_time = time.time()
        logger.info(
            "sync_weights v%d done: pause=%.2fs sync=%.2fs total=%.2fs",
            version,
            pause_time - start,
            sync_time - pause_time,
            sync_time - start,
        )

        self._wait_last_update = self.rollouter.update_param_version.remote(
            version,
            validate,
            global_steps,
        )
        self._wait_last_resume = self.rollouter.resume.remote(self._wait_last_update)

    def wait_last_valid(self):
        """Block until the last sync + optional validation completes."""
        if self._wait_last_update:
            ray.get(self._wait_last_update)
        if self._wait_last_resume:
            ray.get(self._wait_last_resume)

    def rollouter_save_checkpoint(self, path: str):
        return ray.get(self.rollouter.save_checkpoint.remote(path))
