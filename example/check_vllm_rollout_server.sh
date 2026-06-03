#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export VLLM_USE_V1=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy

PYTHON=${PYTHON:-/data/wdy/Softwares/miniconda3/envs/steppo/bin/python}
MODEL=${MODEL:-/data/wdy/Downloads/models/Qwen/Qwen2.5-0.5B-Instruct}

"$PYTHON" tools/check_vllm_rollout_server.py \
  --model "$MODEL" \
  --max-model-len "${MAX_MODEL_LEN:-1536}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.6}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --rollout-gpus-per-node "${ROLLOUT_GPUS_PER_NODE:-1}" \
  --trainer-gpus-per-node "${TRAINER_GPUS_PER_NODE:-1}" \
  "$@"
