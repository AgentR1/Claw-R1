# Dashboard

The Claw-R1 dashboard is the live control surface for Agentic RL data lifestyle management. It connects to the real Ray DataPool actor and parameter synchronizer, so every view reflects the running training system rather than sample data.

## What It Shows

- **Overview**: end-to-end collection, reward, curation, optimization, and consumption status.
- **Collection**: live DataPool events and source breakdowns across selected channels.
- **Representation**: step-level prompt/action token IDs, trajectory IDs, reward state, policy version, and metadata.
- **Curation**: batch quality labels, trainability switches, tags, and notes stored in the dashboard curation table.
- **Optimization**: prefix-tree preview built from real DataPool steps for a prompt group.
- **Consumption**: fetch-batch counters, consumed prompt groups, policy sync version, and last sync status.

## Start It

Run from the repository root after a training job has created the Ray actors:

```bash
conda activate steppo
sh example/start_dashboard.sh
```

The default server is `http://127.0.0.1:8120`.

You can override any server option:

```bash
sh example/start_dashboard.sh \
  --ray-address auto \
  --ray-namespace claw_r1_async \
  --actor-name data_pool \
  --sync-actor-name parameter_synchronizer \
  --channel train,val \
  --port 8120
```

## Configuration

The dashboard accepts CLI flags, environment variables, or `dashboard/config.example.yaml`.

| Field | Default | Purpose |
|---|---:|---|
| `ray_address` | `auto` | Ray cluster address. |
| `ray_namespace` | `null` | Ray namespace used by the training job. |
| `actor_name` | `data_pool` | Named DataPool Ray actor. |
| `sync_actor_name` | `parameter_synchronizer` | Named parameter synchronizer Ray actor. |
| `channel` | `train` | One or more comma-separated DataPool channels. |
| `refresh_interval_ms` | `2000` | Frontend polling interval. |
| `host` / `port` | `0.0.0.0` / `8120` | Dashboard bind address. |

The dashboard no longer ships a mock-data mode. If the DataPool actor is unavailable, API calls return a service-unavailable error so deployment problems are visible immediately.
