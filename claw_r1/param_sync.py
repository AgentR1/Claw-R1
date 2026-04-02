"""ParameterSynchronizer — coordinates weight sync between Trainer and Rollouter.

Based on ``verl/recipe/fully_async_policy/param_sync.py``.

Flow (NCCL mode — colocated or direct model access)::

    sync_weights(version)
      1. ray.get(rollouter.pause())           # stop generation, clear KV cache
      2. NCCL broadcast: Actor → vLLM          # via DetachActorWorker / DetachAsyncRolloutWorker
      3. rollouter.update_param_version(...)   # async — updates version, optionally validates
      4. rollouter.resume(...)                 # async — resumes generation

Flow (ServerAdapter mode — vLLM HTTP server with CUDA IPC)::

    sync_weights(version)
      1. ray.get(rollouter.pause())
      2. Actor extracts weights to CPU → Ray object store → Rollout worker
      3. Rollout worker pushes weights to vLLM server via ServerAdapter.update_weights()
      4. rollouter.update_param_version(...)
      5. rollouter.resume(...)
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

    For ServerAdapter rollouts (vLLM HTTP server mode), NCCL is not used for
    the rollout side. Instead, weights are extracted on the actor, transferred
    via Ray object store, and pushed to the vLLM server via CUDA IPC.
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

        # Detect if rollout uses ServerAdapter (no inference_engine)
        self._use_server_adapter = self._detect_server_adapter()

        self._init_weights_info()
        if not self._use_server_adapter:
            self._init_sync_group()
        else:
            logger.info("ServerAdapter detected — using Ray object store for weight sync (no NCCL group)")

    def _detect_server_adapter(self):
        """Check if the rollout workers use ServerAdapter."""
        try:
            # Ask the rollout worker if it has a ServerAdapter
            result = ray.get(self.rollout_wg.check_server_adapter())
            return any(result) if isinstance(result, list) else bool(result)
        except Exception:
            # If the method doesn't exist, assume ServerAdapter based on config
            # ServerAdapter is used when rollout is on a separate GPU pool (async mode)
            return True

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
        """Pause rollout, sync weights, then resume."""
        start = time.time()
        self.current_version = version

        ray.get(self.rollouter.pause.remote())
        pause_time = time.time()
        logger.info("Rollouter paused in %.2fs", pause_time - start)

        if self._use_server_adapter:
            self._sync_weights_via_ray_store()
        else:
            self._sync_weights_via_nccl()

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

    def _sync_weights_via_nccl(self):
        """Original NCCL broadcast path for colocated/direct model access."""
        self.actor_wg.sync_rollout_weights(self.sync_group_name)
        ray.get(self.rollout_wg.sync_rollout_weights(self.sync_group_name))

    def _sync_weights_via_ray_store(self):
        """ServerAdapter path: extract weights on actor, send via Ray, push to vLLM."""
        # Step 1: Extract weights from actor to CPU (returns dict via Ray object store)
        cpu_params_refs = self.actor_wg.extract_actor_weights()
        # actor_wg returns a list (one per worker), we need the first one
        cpu_params = ray.get(cpu_params_refs[0])

        # Step 2: Send to rollout worker which pushes to vLLM server
        ray.get(self.rollout_wg.receive_and_update_weights(cpu_params))

    def wait_last_valid(self):
        """Block until the last sync + optional validation completes."""
        if self._wait_last_update:
            ray.get(self._wait_last_update)
        if self._wait_last_resume:
            ray.get(self._wait_last_resume)

    def rollouter_save_checkpoint(self, path: str):
        return ray.get(self.rollouter.save_checkpoint.remote(path))
