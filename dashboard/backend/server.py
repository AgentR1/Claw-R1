"""Standalone FastAPI server for the Claw-R1 dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.backend.config import DashboardConfig, load_config
from dashboard.backend.prefix_tree_preview import build_prefix_tree_preview


class CurationUpdate(BaseModel):
    step_key: str | None = None
    trajectory_uid: str | None = None
    step_index: int | None = None
    quality: str | None = None
    trainable: bool | None = None
    reward_status: str | None = None
    tags: list[str] | None = None
    note: str | None = None


class CurationRequest(BaseModel):
    updates: list[CurationUpdate]
    channel: str | None = None


class BatchPreviewRequest(BaseModel):
    algorithm: str = "PPO"
    batch_size: int = 4
    n_rollouts: int = 1
    channel: str | None = None
    max_policy_staleness: int | None = None
    trainable_only: bool = True
    quality: str | None = None


def create_app(config: DashboardConfig) -> FastAPI:
    app = FastAPI(title="Claw-R1 Dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    client = RayDataPoolClient(config)
    static_dir = Path(__file__).resolve().parents[1] / "frontend"

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {
            "actor_name": config.actor_name,
            "sync_actor_name": config.sync_actor_name,
            "channel": config.channel,
            "refresh_interval_ms": config.refresh_interval_ms,
        }

    @app.get("/api/stats")
    def stats(channel: str | None = None) -> dict[str, Any]:
        return _aggregate_stats(client, _channels(channel or config.channel))

    @app.get("/api/sync")
    def sync() -> dict[str, Any]:
        return client.sync_stats()

    @app.get("/api/steps")
    def steps(
        channel: str | None = None,
        prompt_uid: str | None = None,
        trajectory_uid: str | None = None,
        agent: str | None = None,
        quality: str | None = None,
        trainable: bool | None = None,
        reward_state: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        policy_version: int | None = None,
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        channels = _channels(channel or config.channel)
        rows: list[dict[str, Any]] = []
        total = 0
        for ch in channels:
            data = client.list_steps(
                ch,
                prompt_uid=prompt_uid,
                trajectory_uid=trajectory_uid,
                agent=agent,
                quality=quality,
                trainable=trainable,
                reward_state=reward_state,
                min_reward=min_reward,
                max_reward=max_reward,
                policy_version=policy_version,
                limit=limit,
                offset=offset,
            )
            rows.extend(data["steps"])
            total += data.get("total", len(data["steps"]))
        rows.sort(key=lambda r: (r.get("channel", ""), r["prompt_uid"], r["trajectory_uid"], r["step_index"]))
        return {
            "channel": ",".join(channels),
            "channels": channels,
            "total": total,
            "steps": rows[:limit],
        }

    @app.get("/api/trajectories")
    def trajectories(
        channel: str | None = None,
        prompt_uid: str | None = None,
        trajectory_uid: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        return client.list_trajectories(
            channel or config.channel,
            prompt_uid=prompt_uid,
            trajectory_uid=trajectory_uid,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/events")
    def events(
        channel: str | None = None,
        cursor: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ) -> dict[str, Any]:
        channels = _channels(channel or config.channel)
        events_out: list[dict[str, Any]] = []
        next_cursor = cursor
        for ch in channels:
            data = client.get_step_events(ch, cursor=cursor, limit=limit)
            for event in data["events"]:
                event = dict(event)
                event.setdefault("channel", ch)
                events_out.append(event)
            next_cursor = max(next_cursor, data.get("cursor", cursor))
        events_out.sort(key=lambda e: (e.get("cursor", 0), e.get("channel", "")))
        return {
            "channel": ",".join(channels),
            "channels": channels,
            "cursor": next_cursor,
            "events": events_out[-limit:],
            "has_more": False,
        }

    @app.post("/api/curation")
    def curation(req: CurationRequest) -> dict[str, Any]:
        channels = _channels(req.channel or config.channel)
        results = [client.update_step_curation([_model_to_dict(u) for u in req.updates], ch) for ch in channels]
        return {
            "channel": ",".join(channels),
            "channels": channels,
            "updated": [item for result in results for item in result.get("updated", [])],
            "updated_count": sum(result.get("updated_count", 0) for result in results),
        }

    @app.post("/api/batch-preview")
    def batch_preview(req: BatchPreviewRequest) -> dict[str, Any]:
        channels = _channels(req.channel or config.channel)
        steps = []
        for ch in channels:
            data = client.list_steps(
                ch,
                trainable=True if req.trainable_only else None,
                quality=req.quality,
                limit=1000,
            )
            steps.extend(data["steps"])
        if req.max_policy_staleness is not None and steps:
            current = max(step["policy_version"] for step in steps)
            steps = [s for s in steps if current - s["policy_version"] <= req.max_policy_staleness]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for step in steps:
            grouped.setdefault(step["prompt_uid"], []).append(step)
        selected_prompts = list(grouped.keys())[: req.batch_size]
        selected_steps = [step for uid in selected_prompts for step in grouped[uid]]
        prompt_groups = []
        for uid in selected_prompts:
            group_steps = grouped[uid]
            rewards = [s["reward"] for s in group_steps if s.get("reward") is not None]
            quality_counts: dict[str, int] = {}
            for step in group_steps:
                quality = (step.get("curation") or {}).get("quality", "unreviewed")
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            prompt_groups.append(
                {
                    "prompt_uid": uid,
                    "trajectory_count": len({s["trajectory_uid"] for s in group_steps}),
                    "step_count": len(group_steps),
                    "token_count": sum(s["token_count"] for s in group_steps),
                    "reward_mean": sum(rewards) / len(rewards) if rewards else None,
                    "policy_versions": sorted({s["policy_version"] for s in group_steps}),
                    "quality_counts": quality_counts,
                }
            )
        return {
            "mode": "preview_only",
            "algorithm": req.algorithm,
            "channel": ",".join(channels),
            "channels": channels,
            "batch_size": req.batch_size,
            "n_rollouts": req.n_rollouts,
            "max_policy_staleness": req.max_policy_staleness,
            "prompt_uids": selected_prompts,
            "prompt_groups": prompt_groups,
            "step_keys": [s["step_key"] for s in selected_steps],
            "step_count": len(selected_steps),
            "token_count": sum(s["token_count"] for s in selected_steps),
            "curation_summary": {
                "trainable_only": req.trainable_only,
                "quality_filter": req.quality,
                "selected_prompt_groups": len(selected_prompts),
                "selected_trajectories": len({s["trajectory_uid"] for s in selected_steps}),
            },
            "manifest": {
                "algorithm": req.algorithm,
                "channel": ",".join(channels),
                "prompt_uids": selected_prompts,
                "step_keys": [s["step_key"] for s in selected_steps],
                "filters": {
                    "trainable_only": req.trainable_only,
                    "quality": req.quality,
                    "max_policy_staleness": req.max_policy_staleness,
                },
            },
        }

    @app.get("/api/prefix-tree-preview")
    def prefix_tree_preview(
        channel: str | None = None,
        prompt_uid: str | None = None,
        limit: int = Query(200, ge=1, le=1000),
    ) -> dict[str, Any]:
        channels = _channels(channel or config.channel)
        steps = []
        for ch in channels:
            data = client.list_steps(ch, prompt_uid=prompt_uid, limit=limit)
            steps.extend(data["steps"])
        if not prompt_uid and steps:
            first_prompt = steps[0]["prompt_uid"]
            steps = [step for step in steps if step["prompt_uid"] == first_prompt]
            prompt_uid = first_prompt
        return {
            "channel": ",".join(channels),
            "channels": channels,
            "prompt_uid": prompt_uid,
            **build_prefix_tree_preview(steps),
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def _channels(value: str | None) -> list[str]:
    channels = [part.strip() for part in (value or "train").split(",") if part.strip()]
    return channels or ["train"]


def _aggregate_stats(client: Any, channels: list[str]) -> dict[str, Any]:
    stats_by_channel = {channel: client.stats(channel) for channel in channels}
    totals: dict[str, Any] = {
        "channel": ",".join(channels),
        "channels": channels,
        "by_channel": stats_by_channel,
    }
    numeric_keys = {
        key
        for stats in stats_by_channel.values()
        for key, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for key in numeric_keys:
        totals[key] = sum(stats.get(key, 0) for stats in stats_by_channel.values())
    last_fetches = [
        stats.get("last_fetch")
        for stats in stats_by_channel.values()
        if stats.get("last_fetch")
    ]
    totals["last_fetch"] = max(last_fetches, key=lambda item: item.get("fetched_at", 0)) if last_fetches else None
    return totals


class RayDataPoolClient:
    def __init__(self, config: DashboardConfig):
        self.config = config
        self._actor = None
        self._sync_actor = None

    def _get_actor(self):
        if self._actor is None:
            try:
                import ray

                ray.init(
                    address=self.config.ray_address or "auto",
                    namespace=self.config.ray_namespace,
                    ignore_reinit_error=True,
                )
                self._actor = ray.get_actor(self.config.actor_name)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"DataPool actor unavailable: {exc}") from exc
        return self._actor

    def stats(self, channel: str) -> dict[str, Any]:
        return self._call("stats", channel)

    def list_steps(self, channel: str, **kwargs: Any) -> dict[str, Any]:
        return self._call("list_steps", channel, **kwargs)

    def list_trajectories(self, channel: str, **kwargs: Any) -> dict[str, Any]:
        return self._call("list_trajectories", channel, **kwargs)

    def get_step_events(self, channel: str, **kwargs: Any) -> dict[str, Any]:
        return self._call("get_step_events", channel, **kwargs)

    def update_step_curation(self, updates: list[dict[str, Any]], channel: str) -> dict[str, Any]:
        return self._call("update_step_curation", updates, channel)

    def sync_stats(self) -> dict[str, Any]:
        try:
            import ray

            if self._sync_actor is None:
                self._get_actor()
                self._sync_actor = ray.get_actor(self.config.sync_actor_name)
            return ray.get(self._sync_actor.get_statistics.remote())
        except Exception as exc:
            return {
                "available": False,
                "current_version": None,
                "policy_version": None,
                "sync_count": 0,
                "last_sync": None,
                "error": str(exc),
            }

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        import ray

        actor = self._get_actor()
        remote_method = getattr(actor, method)
        return ray.get(remote_method.remote(*args, **kwargs))


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def main() -> None:
    config = load_config()
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
