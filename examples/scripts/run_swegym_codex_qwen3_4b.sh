#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_qwen3_4b_async_common.sh"
export CLAW_R1_SANDBOX_PROVIDER=${CLAW_R1_SANDBOX_PROVIDER:-local_bwrap}; export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}; export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
run_qwen3_4b_async "${TRAIN_FILE:-data/swegym/train.parquet}" "${VAL_FILE:-data/swegym/test.parquet}" codex recipes/codex/agent_flow_config.yaml swegym_qwen3_4b swegym_codex "$@"
