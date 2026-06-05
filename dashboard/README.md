# Claw-R1 Dashboard

Standalone dashboard for inspecting DataPool steps, curating candidates, previewing batch manifests, and visualizing prefix-tree packing. Dashboard code lives under `dashboard/`; training code is not changed beyond read-only DataPool query methods and a dashboard-only curation side table.

## Run

Use the project conda environment:

```bash
conda activate steppo
python -m dashboard.backend.server --config dashboard/config.example.yaml
```

Open `http://127.0.0.1:8120`.

## Configuration

The server accepts CLI flags, environment variables, or a YAML file:

- `ray_address`: Ray address, usually `auto`.
- `ray_namespace`: Ray namespace used by the training process, for example `claw_r1_async`.
- `actor_name`: named DataPool actor, default `data_pool`.
- `sync_actor_name`: named parameter synchronizer actor, default `parameter_synchronizer`.
- `channel`: DataPool channel, default `train`.
- `refresh_interval_ms`: frontend polling cadence.
- `host` / `port`: dashboard bind address.

## Views

- `Monitor`: event stream and high-level DataPool stats.
- `Steps`: filterable step table with trajectory, step index, state, reward, policy version, action summary, and metadata.
- `Curate`: batch update quality, trainability, reward status, tags, and notes. These markers are stored outside `Step` and do not affect `fetch_batch()`.
- `Batch`: generate a preview manifest for PPO/GRPO batch candidates.
- `Optimize`: prefix-tree visualization preview grouped by `prompt_uid`.

## Prefix Tree Preview

The preview groups steps by `prompt_uid` and treats `prompt_ids + response_ids` as a token sequence. A dashboard-side pure Python trie builds compressed nodes with `tokens`, `sequence_ids`, `start_pos`, `end_pos`, sequence paths, a step map, and a small attention-mask thumbnail.

This is visualization only. It does not mutate DataPool, does not alter training batches, and does not import the experimental `prefix-tree-merge` backend, FlexAttention patch, or logprob restoration code.
