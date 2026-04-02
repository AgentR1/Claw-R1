"""Detached actor and rollout workers for async training.

In async mode, Actor (training) and Rollout (inference) run on separate GPU
pools.  These workers extend verl's ``AsyncActorRolloutRefWorker`` with NCCL
weight synchronization so the ParameterSynchronizer can broadcast updated
actor weights to the rollout replicas.

When the rollout uses a ServerAdapter (vLLM HTTP server mode), the rollout
worker skips the NCCL broadcast (vLLM sleep mode consumes all GPU memory)
and instead receives weights via Ray object store, then pushes them to the
vLLM server via the ServerAdapter's CUDA IPC update_weights mechanism.

Based on ``verl/recipe/fully_async_policy/fsdp_workers.py``.
"""

import logging
import os

import torch
import torch.distributed
from omegaconf import DictConfig

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
    CriticWorker,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


def _get_inference_model(rollout):
    """Extract the underlying model from a vLLM/SGLang inference engine.

    Returns None for ServerAdapter (async HTTP server mode) which uses
    CUDA IPC/ZMQ for weight updates instead of direct model access.
    """
    if not hasattr(rollout, "inference_engine"):
        # ServerAdapter in async mode — no direct model access
        return None
    engine = rollout.inference_engine
    if hasattr(engine, "llm_engine"):
        return engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    if hasattr(engine, "worker"):
        return engine.worker.model_runner.model
    raise AttributeError(f"Unsupported inference_engine type: {type(engine)}")


def _is_server_adapter(rollout):
    """Check if the rollout is a ServerAdapter (async HTTP server mode)."""
    return not hasattr(rollout, "inference_engine")


class _DetachNcclSync(AsyncActorRolloutRefWorker):
    """Mixin adding NCCL-based weight synchronization between actor and rollout.

    For ServerAdapter rollouts, the NCCL broadcast is skipped on the rollout side
    because vLLM sleep mode consumes all GPU memory. Instead, the actor extracts
    weights and sends them via Ray object store to the rollout worker.
    """

    def _get_actor_params(self):
        raise NotImplementedError

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
        """Broadcast actor weights to rollout via NCCL collective.

        For ServerAdapter rollouts, this is a no-op on the rollout side.
        Weight transfer is handled separately via extract_and_send_weights
        on the actor side and receive_and_update_weights on the rollout side.
        """
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        # For ServerAdapter rollouts, skip NCCL broadcast entirely
        if self._is_rollout and _is_server_adapter(self.rollout):
            return

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        params = self._get_actor_params() if self._is_actor else None

        if self._is_rollout:
            inference_model = _get_inference_model(self.rollout)
            if inference_model is not None:
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)

        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            from ray.util.collective import collective

            collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)

            if self._is_rollout:
                inference_model.load_weights([(key, tensor)])

        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def extract_actor_weights(self):
        """Extract actor weights as a CPU state dict for ServerAdapter sync."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        params = self._get_actor_params()
        # Move to CPU to avoid GPU memory issues during transfer
        # Also convert DTensors (from FSDP2) to regular tensors to avoid
        # "mixed torch.Tensor and DTensor" errors in vLLM weight transfer
        cpu_params = {}
        for k, v in params.items():
            if hasattr(v, 'full_tensor'):
                v = v.full_tensor()
            elif hasattr(v, '_local_tensor'):
                v = v._local_tensor
            cpu_params[k] = v.cpu()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()

        return cpu_params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def receive_and_update_weights(self, cpu_params):
        """Receive weights from actor and push to vLLM server via ServerAdapter.

        Ray async actors use uvloop which doesn't support nested event loops.
        We run the async update_weights in a separate thread with its own
        asyncio event loop to avoid the 'event loop already running' error.
        """
        assert self._is_rollout

        import asyncio
        import concurrent.futures

        def _weight_generator():
            for key, tensor in cpu_params.items():
                # Ensure we have a plain torch.Tensor (not DTensor from FSDP)
                if hasattr(tensor, 'full_tensor'):
                    tensor = tensor.full_tensor()
                elif hasattr(tensor, '_local_tensor'):
                    tensor = tensor._local_tensor
                t = tensor.to(get_torch_device().current_device())
                # Final safety: if still a DTensor, extract the local data
                if not isinstance(t, torch.Tensor) or type(t).__name__ == 'DTensor':
                    t = t.data if hasattr(t, 'data') else t
                yield key, t

        def _run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(
                    self.rollout.update_weights(_weight_generator())
                )
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop)
            future.result()

        get_torch_device().empty_cache()


class DetachActorWorker(_DetachNcclSync):
    """Actor worker for async mode — training only, no rollout."""

    def _get_actor_params(self):
        assert self._is_actor
        params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params,
            getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp),
        )
        return params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        """Return a list of (name, shape, dtype) for all actor parameters."""
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info

        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        params = self._get_actor_params()
        self._weights_info = [(key, t.size(), t.dtype) for key, t in params.items()]
        return self._weights_info


class DetachAsyncRolloutWorker(_DetachNcclSync):
    """Rollout worker for async mode — inference only, no training."""

    def __init__(self, config: DictConfig, role: str):
        ActorRolloutRefWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        """Receive weights info from the actor side so sync can proceed."""
        assert self._is_rollout
        self._weights_info = weights_info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_server_adapter(self):
        """Return True if this rollout worker uses a ServerAdapter."""
        return _is_server_adapter(self.rollout)
