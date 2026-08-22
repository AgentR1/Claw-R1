# Qwen3-4B training launches

All scripts source `_qwen3_4b_async_common.sh`, which preserves the complete
async GRPO/FSDP/vLLM/reference-policy/checkpoint/logging profile. Scenario
scripts only set prepared Parquet paths, the recipe flow, context limits, and
black-box sandbox requirements.

```bash
bash examples/scripts/run_gsm8k_qwen3_4b.sh
bash examples/scripts/run_alfworld_qwen3_4b.sh
bash examples/scripts/run_swegym_claude_code_qwen3_4b.sh
bash examples/scripts/run_swegym_codex_qwen3_4b.sh
bash examples/scripts/start_dashboard.sh
```

All resource and optimization values are overrideable through environment
variables or trailing Hydra overrides. `MODEL` defaults to `Qwen/Qwen3-4B`.
