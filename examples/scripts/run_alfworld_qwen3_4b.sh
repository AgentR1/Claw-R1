#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_qwen3_4b_async_common.sh"
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}; export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
run_qwen3_4b_async "${TRAIN_FILE:-data/alfworld/train.parquet}" "${VAL_FILE:-data/alfworld/test.parquet}" alfworld_agent recipes/alfworld_agent/agent_flow_config.yaml alfworld_qwen3_4b alfworld_agent "$@"
