"""Gateway Server — standalone FastAPI process bridging Agent Side and Training Side.

Runs inside the Ray cluster.  Connects to DataPool (Ray Actor) for trajectory
storage and to vLLM servers for LLM generation via HTTP forwarding.

Start with::

    python -m claw_r1.gateway.gateway \\
        --data-pool-name data_pool \\
        --vllm-addresses http://host1:8000,http://host2:8000 \\
        --tokenizer-path /path/to/model \\
        --prompt-length 4096 \\
        --response-length 4096 \\
        --host 0.0.0.0 \\
        --port 8100
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from typing import Any, Optional

import httpx
import ray
import uvicorn
from fastapi import FastAPI, HTTPException

from claw_r1.data_pool.data_model import Step
from claw_r1.gateway.models import (
    CompleteTrajectoryRequest,
    ComputeRewardRequest,
    ComputeRewardResponse,
    GenerateRequest,
    GenerateResponse,
    StepPayload,
    SubmitStepsRequest,
    SubmitStepsResponse,
)

logger = logging.getLogger("claw_r1.gateway")
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

app = FastAPI(title="Agent-R1 Gateway")

# ── Global state (set during lifespan) ────────────────────────────────────

_data_pool: Any = None
_vllm_cycle: Any = None
_vllm_addresses: list[str] = []
_tokenizer: Any = None
_prompt_length: int = 0
_response_length: int = 0
_reward_worker: Any = None
_http_client: Optional[httpx.AsyncClient] = None


# ── Helpers ───────────────────────────────────────────────────────────────

def _step_payload_to_step(p: StepPayload) -> Step:
    return Step(
        prompt_ids=p.prompt_ids,
        response_ids=p.response_ids,
        multi_modal_data=p.multi_modal_data,
        reward=p.reward,
        rollout_log_probs=p.rollout_log_probs,
        routed_experts=p.routed_experts,
        trajectory_uid=p.trajectory_uid,
        prompt_uid=p.prompt_uid,
        step_index=p.step_index,
        policy_version=p.policy_version,
        is_last=p.is_last,
        metadata=p.metadata,
    )


def _normalize_address(addr: str) -> str:
    """Ensure an address has an http:// scheme prefix."""
    if not addr.startswith(("http://", "https://")):
        return f"http://{addr}"
    return addr


def _next_vllm_address() -> str:
    if _vllm_cycle is None:
        raise HTTPException(503, "No vLLM servers configured")
    return _normalize_address(next(_vllm_cycle))


# ── White-box endpoints (implemented) ────────────────────────────────────

@app.post("/submit_steps", response_model=SubmitStepsResponse)
async def submit_steps(req: SubmitStepsRequest):
    """Accept Steps from white-box agents and forward to DataPool."""
    if _data_pool is None:
        raise HTTPException(503, "DataPool not connected")

    steps = [_step_payload_to_step(p) for p in req.steps]
    ray.get(_data_pool.submit_steps.remote(steps))
    return SubmitStepsResponse(accepted=len(steps))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Forward a generation request to a vLLM server and return token-level results.

    Uses vLLM's ``/v1/completions`` endpoint with token-ID prompt, then
    tokenises the response text to recover ``response_ids``.
    """
    base_url = _next_vllm_address()
    url = f"{base_url}/v1/completions"

    max_tokens = req.sampling_params.get("max_tokens", _response_length)
    logprobs_n = 1 if req.sampling_params.get("logprobs", False) else None

    vllm_payload: dict[str, Any] = {
        "prompt": req.prompt_ids,
        "max_tokens": max_tokens,
        "temperature": req.sampling_params.get("temperature", 1.0),
        "top_p": req.sampling_params.get("top_p", 1.0),
        "repetition_penalty": req.sampling_params.get("repetition_penalty", 1.0),
        "logprobs": logprobs_n,
        "skip_special_tokens": False,
    }
    model = req.sampling_params.get("model")
    if model:
        vllm_payload["model"] = model

    try:
        resp = await _http_client.post(url, json=vllm_payload, timeout=600.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, f"vLLM error: {exc.response.text}")
    except httpx.RequestError as exc:
        raise HTTPException(502, f"vLLM unreachable: {exc}")

    data = resp.json()
    choice = data["choices"][0]
    response_text = choice.get("text", "")

    token_ids: list[int] = _tokenizer.encode(response_text, add_special_tokens=False)

    log_probs: Optional[list[float]] = None
    logprobs_obj = choice.get("logprobs")
    if logprobs_obj and "token_logprobs" in logprobs_obj:
        log_probs = logprobs_obj["token_logprobs"]

    stop_reason = choice.get("finish_reason")

    return GenerateResponse(
        token_ids=token_ids,
        log_probs=log_probs,
        stop_reason=stop_reason,
    )


@app.post("/compute_reward", response_model=ComputeRewardResponse)
async def compute_reward(req: ComputeRewardRequest):
    """Compute reward for a single sample via the RewardLoopWorker."""
    if _reward_worker is None:
        raise HTTPException(503, "RewardLoopWorker not available")

    import numpy as np
    import torch
    from tensordict import TensorDict

    from verl.protocol import DataProto
    from verl.utils.model import compute_position_id_with_mask

    _tokenizer.padding_side = "left"
    prompt_out = _tokenizer.pad(
        {"input_ids": req.prompt_ids},
        padding="max_length",
        max_length=_prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if prompt_out["input_ids"].dim() == 1:
        prompt_out["input_ids"] = prompt_out["input_ids"].unsqueeze(0)
        prompt_out["attention_mask"] = prompt_out["attention_mask"].unsqueeze(0)

    _tokenizer.padding_side = "right"
    response_out = _tokenizer.pad(
        {"input_ids": req.response_ids},
        padding="max_length",
        max_length=_response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if response_out["input_ids"].dim() == 1:
        response_out["input_ids"] = response_out["input_ids"].unsqueeze(0)
        response_out["attention_mask"] = response_out["attention_mask"].unsqueeze(0)

    attention_mask = torch.cat(
        [prompt_out["attention_mask"], response_out["attention_mask"]], dim=1,
    )
    input_ids = torch.cat(
        [prompt_out["input_ids"], response_out["input_ids"]], dim=1,
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    batch = TensorDict(
        {
            "prompts": prompt_out["input_ids"],
            "responses": response_out["input_ids"],
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "position_ids": position_ids,
        },
        batch_size=1,
    )
    non_tensor_batch: dict[str, Any] = {
        **{k: np.array([v]) for k, v in req.dataset_fields.items()},
        "__num_turns__": np.array([req.num_turns]),
        "tool_extra_fields": np.array([req.extra_fields], dtype=object),
    }

    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    result = await _reward_worker.compute_score.remote(data)

    return ComputeRewardResponse(
        reward_score=result["reward_score"],
        reward_extra_info=result.get("reward_extra_info", {}),
    )


# ── Black-box endpoints (stubs, to be implemented later) ─────────────────

@app.post("/complete_trajectory/{trajectory_uid}")
async def complete_trajectory(trajectory_uid: str, req: CompleteTrajectoryRequest = None):
    """Mark a black-box trajectory as complete. Stub — not yet implemented."""
    raise HTTPException(501, "Black-box trajectory completion is not yet implemented")


@app.api_route(
    "/{trajectory_uid}/{prompt_uid}/v1/chat/completions",
    methods=["POST"],
)
async def chat_completions_proxy(trajectory_uid: str, prompt_uid: str):
    """OpenAI-compatible reverse proxy with auto Step creation. Stub — not yet implemented."""
    raise HTTPException(501, "Black-box chat completions proxy is not yet implemented")


# ── Server bootstrap ─────────────────────────────────────────────────────

def init_gateway(
    *,
    data_pool_name: str,
    vllm_addresses: list[str],
    tokenizer_path: str,
    prompt_length: int,
    response_length: int,
    reward_worker_name: Optional[str] = None,
    ray_address: Optional[str] = None,
    ray_namespace: Optional[str] = None,
):
    """Initialise Gateway global state.  Called once before the server starts."""
    global _data_pool, _vllm_cycle, _vllm_addresses, _tokenizer
    global _prompt_length, _response_length, _reward_worker, _http_client

    ray.init(
        address=ray_address or "auto",
        namespace=ray_namespace,
        ignore_reinit_error=True,
    )

    _data_pool = ray.get_actor(data_pool_name)
    _vllm_addresses = vllm_addresses
    _vllm_cycle = itertools.cycle(vllm_addresses)

    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(tokenizer_path)
    _tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    _prompt_length = prompt_length
    _response_length = response_length

    if reward_worker_name:
        try:
            _reward_worker = ray.get_actor(reward_worker_name)
        except ValueError:
            logger.warning("RewardLoopWorker '%s' not found; /compute_reward will be unavailable", reward_worker_name)

    _http_client = httpx.AsyncClient()
    logger.info(
        "Gateway initialised: data_pool=%s, vllm=%s, tokenizer=%s",
        data_pool_name,
        vllm_addresses,
        tokenizer_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Agent-R1 Gateway Server")
    parser.add_argument("--data-pool-name", required=True)
    parser.add_argument("--vllm-addresses", required=True, help="Comma-separated list")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--reward-worker-name", default=None)
    parser.add_argument("--ray-address", default=None, help="Ray GCS address (e.g. ip:port)")
    parser.add_argument("--ray-namespace", default=None, help="Ray namespace for actor lookup")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_gateway(
        data_pool_name=args.data_pool_name,
        vllm_addresses=args.vllm_addresses.split(","),
        tokenizer_path=args.tokenizer_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        reward_worker_name=args.reward_worker_name,
        ray_address=args.ray_address,
        ray_namespace=args.ray_namespace,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
