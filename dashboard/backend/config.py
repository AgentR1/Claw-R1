"""Configuration helpers for the standalone dashboard server."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DashboardConfig:
    ray_address: str | None = "auto"
    ray_namespace: str | None = None
    actor_name: str = "data_pool"
    sync_actor_name: str = "parameter_synchronizer"
    channel: str = "train"
    refresh_interval_ms: int = 2000
    host: str = "0.0.0.0"
    port: int = 8120


def load_config(argv: list[str] | None = None) -> DashboardConfig:
    parser = argparse.ArgumentParser(description="Claw-R1 dashboard server")
    parser.add_argument("--config", default=None, help="Optional YAML config file")
    parser.add_argument("--ray-address", default=os.getenv("CLAW_DASHBOARD_RAY_ADDRESS"))
    parser.add_argument("--ray-namespace", default=os.getenv("CLAW_DASHBOARD_RAY_NAMESPACE"))
    parser.add_argument("--actor-name", default=os.getenv("CLAW_DASHBOARD_ACTOR_NAME"))
    parser.add_argument("--sync-actor-name", default=os.getenv("CLAW_DASHBOARD_SYNC_ACTOR_NAME"))
    parser.add_argument("--channel", default=os.getenv("CLAW_DASHBOARD_CHANNEL"))
    parser.add_argument("--refresh-interval-ms", type=int, default=None)
    parser.add_argument("--host", default=os.getenv("CLAW_DASHBOARD_HOST"))
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    data: dict[str, Any] = {}
    if args.config:
        data.update(_read_yaml(Path(args.config)))

    cfg = DashboardConfig(**{k: v for k, v in data.items() if hasattr(DashboardConfig, k)})
    for key in ("ray_address", "ray_namespace", "actor_name", "sync_actor_name", "channel", "host"):
        value = getattr(args, key)
        if value:
            setattr(cfg, key, value)
    if args.refresh_interval_ms is not None:
        cfg.refresh_interval_ms = args.refresh_interval_ms
    if args.port is not None:
        cfg.port = args.port
    return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Dashboard config must be a mapping: {path}")
    return loaded
