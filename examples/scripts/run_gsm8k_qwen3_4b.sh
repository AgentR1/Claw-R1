#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_qwen3_4b_async_common.sh"
run_qwen3_4b_async "${TRAIN_FILE:-data/gsm8k/train.parquet}" "${VAL_FILE:-data/gsm8k/test.parquet}" gsm8k_agent recipes/gsm8k_agent/agent_flow_config.yaml gsm8k_qwen3_4b gsm8k_agent "$@"
