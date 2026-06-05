# Quick Start

This guide runs Claw-R1 asynchronous training with a black-box GSM8K agent.

## Prerequisites

- Complete the [installation guide](installation.md).
- Prepare at least 3 GPUs for the small example.
- Place GSM8K parquet files where your script expects them.

## Run Black-Box Training

Black-box mode lets an agent use the standard OpenAI API while Claw-R1 captures the interaction through the Gateway.

```bash
conda activate steppo
export CUDA_VISIBLE_DEVICES=0,1,2

sh example/test_async_blackbox.sh
```

The script starts Ray, creates the DataPool actor, launches the trainer and rollout GPU pools, starts the Gateway on port `8100`, and runs the registered GSM8K black-box agent flow.

## Start the Dashboard

After the training actors are available, open a second shell:

```bash
conda activate steppo
sh example/start_dashboard.sh
```

Open `http://127.0.0.1:8120`.

The dashboard uses real Ray actors and shows collection events, step representation, curation state, prefix-tree preview, and training consumption.

## Key Overrides

```bash
trainer.n_gpus_per_node=2
rollout.n_gpus_per_node=1
actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent
actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml
async_training.trigger_parameter_sync_step=1
actor_rollout_ref.rollout.n=5
```

## White-Box Mode

White-box agents build and submit `Step` objects directly through Gateway endpoints:

```bash
conda activate steppo
export CUDA_VISIBLE_DEVICES=0,1,2

sh example/test_async.sh
```

## Next Steps

- [Components](../components/index.md)
- [Dashboard](../dashboard.md)
- [Configuration](../configuration/index.md)
- [Gateway API](../api/gateway.md)
