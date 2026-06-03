#!/usr/bin/env python3
"""Minimal vLLM rollout HTTP server check for Claw-R1 async rollout."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import ray
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _compose_config(args: argparse.Namespace):
    overrides = [
        f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.max_num_seqs}",
        f"trainer.n_gpus_per_node={args.trainer_gpus_per_node}",
        "trainer.nnodes=1",
        f"rollout.n_gpus_per_node={args.rollout_gpus_per_node}",
        "rollout.nnodes=1",
    ]
    overrides.extend(args.overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "claw_r1" / "config"), version_base=None):
        config = compose(config_name="async_ppo_trainer", overrides=overrides)
    OmegaConf.resolve(config)
    return config


async def _check_http(addresses: list[str], timeout_s: float) -> None:
    async with httpx.AsyncClient(timeout=2.0) as client:
        for address in addresses:
            base_url = f"http://{address}"
            print(f"[check] probing {base_url}")
            last_error = None
            for attempt in range(1, int(timeout_s) + 1):
                try:
                    health = await client.get(f"{base_url}/health")
                    models = await client.get(f"{base_url}/v1/models")
                    print(f"[check] attempt {attempt}: /health={health.status_code} /v1/models={models.status_code}")
                    health.raise_for_status()
                    models.raise_for_status()
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[check] attempt {attempt}: unreachable ({type(exc).__name__}: {exc})")
                    await asyncio.sleep(1)
            else:
                raise RuntimeError(f"{base_url} stayed unreachable; last error: {last_error}")


async def _main_async(args: argparse.Namespace) -> int:
    config = _compose_config(args)
    print("[check] effective rollout config:")
    print(
        OmegaConf.to_yaml(
            OmegaConf.masked_copy(
                config.actor_rollout_ref.rollout,
                [
                    "name",
                    "tensor_model_parallel_size",
                    "data_parallel_size",
                    "pipeline_model_parallel_size",
                    "gpu_memory_utilization",
                    "max_model_len",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "load_format",
                    "enable_sleep_mode",
                ],
            ),
            resolve=True,
        )
    )

    from claw_r1.async_main import _create_resource_pool_manager, _create_role_worker_mapping
    from claw_r1.vllm_server_patch import apply_verl_vllm_server_patches, patch_rollout_replicas
    from verl.single_controller.ray import RayClassWithInitArgs
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.ppo.utils import Role
    from verl.workers.rollout.replica import get_rollout_replica_class

    if not ray.is_initialized():
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        init_kwargs = OmegaConf.to_container(ray_init_kwargs, resolve=True)
        if args.ray_address:
            init_kwargs["address"] = args.ray_address
        ray.init(namespace=args.ray_namespace, **init_kwargs)

    role_worker_mapping, ray_worker_group_cls = _create_role_worker_mapping(config)
    resource_pool_manager = _create_resource_pool_manager(config, [Role.Rollout])
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.get_resource_pool(Role.Rollout)

    rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.Rollout],
        config=config.actor_rollout_ref,
        role=str(Role.Rollout),
    )

    from verl.single_controller.ray.base import create_colocated_worker_cls

    worker_dict_cls = create_colocated_worker_cls(class_dict={str(Role.Rollout): rollout_cls})
    wg_dict = ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=config.trainer.device,
    )
    rollout_wg = wg_dict.spawn(prefix_set={str(Role.Rollout)})[str(Role.Rollout)]

    print(f"[check] rollout worker group world_size={rollout_wg.world_size}")
    print("[check] calling rollout_wg.init_model()")
    rollout_wg.init_model()

    apply_verl_vllm_server_patches()
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    replica_cls = get_rollout_replica_class(rollout_config.name)
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = rollout_wg.world_size // rollout_world_size
    replicas = [
        replica_cls(
            replica_rank=rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.rollout.n_gpus_per_node,
        )
        for rank in range(num_replicas)
    ]
    patch_rollout_replicas(replicas)

    print(f"[check] launching {len(replicas)} rollout replica(s)")
    await asyncio.gather(*[replica.init_hybrid(rollout_wg) for replica in replicas])
    addresses = [replica._server_address for replica in replicas]
    print(f"[check] returned server addresses: {addresses}")
    await _check_http(addresses, args.timeout_s)
    print("[check] OK: rollout HTTP server is reachable")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/wdy/Downloads/models/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--rollout-gpus-per-node", type=int, default=1)
    parser.add_argument("--trainer-gpus-per-node", type=int, default=1)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-namespace", default="claw_r1_vllm_check")
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(_main_async(args))
    except Exception as exc:
        print(f"[check] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
