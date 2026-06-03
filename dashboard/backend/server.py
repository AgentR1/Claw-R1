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
    client = MockDataPoolClient() if config.mock else RayDataPoolClient(config)
    static_dir = Path(__file__).resolve().parents[1] / "frontend"

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {
            "actor_name": config.actor_name,
            "channel": config.channel,
            "refresh_interval_ms": config.refresh_interval_ms,
            "mock": config.mock,
        }

    @app.get("/api/stats")
    def stats(channel: str | None = None) -> dict[str, Any]:
        return client.stats(channel or config.channel)

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
        return client.list_steps(
            channel or config.channel,
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
        return client.get_step_events(channel or config.channel, cursor=cursor, limit=limit)

    @app.post("/api/curation")
    def curation(req: CurationRequest) -> dict[str, Any]:
        return client.update_step_curation(
            [_model_to_dict(u) for u in req.updates],
            req.channel or config.channel,
        )

    @app.post("/api/batch-preview")
    def batch_preview(req: BatchPreviewRequest) -> dict[str, Any]:
        channel = req.channel or config.channel
        data = client.list_steps(
            channel,
            trainable=True if req.trainable_only else None,
            quality=req.quality,
            limit=1000,
        )
        steps = data["steps"]
        if req.max_policy_staleness is not None and steps:
            current = max(step["policy_version"] for step in steps)
            steps = [s for s in steps if current - s["policy_version"] <= req.max_policy_staleness]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for step in steps:
            grouped.setdefault(step["prompt_uid"], []).append(step)
        selected_prompts = list(grouped.keys())[: req.batch_size]
        selected_steps = [step for uid in selected_prompts for step in grouped[uid]]
        return {
            "mode": "preview_only",
            "algorithm": req.algorithm,
            "channel": channel,
            "batch_size": req.batch_size,
            "n_rollouts": req.n_rollouts,
            "max_policy_staleness": req.max_policy_staleness,
            "prompt_uids": selected_prompts,
            "step_keys": [s["step_key"] for s in selected_steps],
            "step_count": len(selected_steps),
            "token_count": sum(s["token_count"] for s in selected_steps),
            "manifest": {
                "algorithm": req.algorithm,
                "channel": channel,
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
        data = client.list_steps(channel or config.channel, prompt_uid=prompt_uid, limit=limit)
        steps = data["steps"]
        if not prompt_uid and steps:
            first_prompt = steps[0]["prompt_uid"]
            steps = [step for step in steps if step["prompt_uid"] == first_prompt]
            prompt_uid = first_prompt
        return {"channel": channel or config.channel, "prompt_uid": prompt_uid, **build_prefix_tree_preview(steps)}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


class RayDataPoolClient:
    def __init__(self, config: DashboardConfig):
        self.config = config
        self._actor = None

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

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        import ray

        actor = self._get_actor()
        remote_method = getattr(actor, method)
        return ray.get(remote_method.remote(*args, **kwargs))


class MockDataPoolClient:
    def __init__(self):
        self.steps = _mock_steps()
        self.curation: dict[str, dict[str, Any]] = {}
        self.events = [
            {"cursor": i + 1, "type": "step_submitted", "step_key": step["step_key"], "payload": {}}
            for i, step in enumerate(self.steps)
        ]

    def stats(self, channel: str) -> dict[str, Any]:
        trajectories = {s["trajectory_uid"] for s in self.steps}
        prompts = {s["prompt_uid"] for s in self.steps}
        return {
            "total_steps": len(self.steps),
            "total_trajectories": len(trajectories),
            "complete_trajectories": len(trajectories),
            "total_prompt_groups": len(prompts),
            "pending_prompt_groups": len(prompts),
            "ready_prompt_groups": len(prompts),
            "curated_steps": len(self.curation),
            "event_cursor": len(self.events),
        }

    def list_steps(self, channel: str, **kwargs: Any) -> dict[str, Any]:
        rows = [self._with_curation(s, channel) for s in self.steps]
        for key in ("prompt_uid", "trajectory_uid", "agent", "quality", "reward_state", "policy_version"):
            value = kwargs.get(key)
            if value is None:
                continue
            if key == "quality":
                rows = [r for r in rows if r["curation"].get("quality") == value]
            else:
                rows = [r for r in rows if r.get(key) == value]
        if kwargs.get("trainable") is not None:
            rows = [r for r in rows if r["curation"].get("trainable", True) == kwargs["trainable"]]
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 200)
        return {"channel": channel, "total": len(rows), "steps": rows[offset : offset + limit]}

    def list_trajectories(self, channel: str, **kwargs: Any) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for step in self.list_steps(channel, limit=1000)["steps"]:
            grouped.setdefault(step["trajectory_uid"], []).append(step)
        rows = []
        for uid, steps in grouped.items():
            first = steps[0]
            if kwargs.get("prompt_uid") and first["prompt_uid"] != kwargs["prompt_uid"]:
                continue
            rows.append(
                {
                    "trajectory_uid": uid,
                    "prompt_uid": first["prompt_uid"],
                    "step_count": len(steps),
                    "complete": True,
                    "reward_sum": sum(s["reward"] or 0 for s in steps),
                    "policy_versions": sorted({s["policy_version"] for s in steps}),
                    "agents": sorted({s["agent"] for s in steps if s["agent"]}),
                    "metadata": first["metadata"],
                }
            )
        return {"channel": channel, "total": len(rows), "trajectories": rows}

    def get_step_events(self, channel: str, **kwargs: Any) -> dict[str, Any]:
        cursor = kwargs.get("cursor", 0)
        events = [e for e in self.events if e["cursor"] > cursor][: kwargs.get("limit", 200)]
        return {
            "channel": channel,
            "cursor": events[-1]["cursor"] if events else cursor,
            "events": events,
            "has_more": False,
        }

    def update_step_curation(self, updates: list[dict[str, Any]], channel: str) -> dict[str, Any]:
        updated = []
        for update in updates:
            key = update.get("step_key") or f"{update.get('trajectory_uid')}:{update.get('step_index')}"
            existing = dict(self.curation.get(key, {}))
            existing.update(
                {
                    k: v
                    for k, v in update.items()
                    if k in {"quality", "trainable", "reward_status", "tags", "note"}
                }
            )
            self.curation[key] = existing
            self.events.append(
                {
                    "cursor": len(self.events) + 1,
                    "type": "curation_updated",
                    "step_key": key,
                    "payload": existing,
                }
            )
            updated.append({"step_key": key, "curation": existing})
        return {"channel": channel, "updated": updated, "updated_count": len(updated)}

    def _with_curation(self, step: dict[str, Any], channel: str) -> dict[str, Any]:
        row = dict(step)
        row["channel"] = channel
        row["curation"] = {"quality": "unreviewed", "trainable": True, "reward_status": row["reward_state"], "tags": []}
        row["curation"].update(self.curation.get(row["step_key"], {}))
        return row


def _mock_steps() -> list[dict[str, Any]]:
    rows = []
    base_prompt = [1001, 1002, 1003, 1101, 1102, 1201, 1202, 1203, 1301, 1302, 1303, 1304]
    rubric = [1401, 1402, 1403, 1404]
    task = [1501, 1502, 1503, 1504, 1505]
    shared_context = base_prompt + rubric + task
    samples = [
        ("prompt-a", "traj-a-0", 0, shared_context, [2101, 2102, 2103, 2104, 2105], 0.32),
        (
            "prompt-a",
            "traj-a-0",
            1,
            shared_context + [2101, 2102, 2103, 2104, 2105, 3001, 3002, 3003],
            [4101, 4102, 4103, 4104],
            0.74,
        ),
        (
            "prompt-a",
            "traj-a-0",
            2,
            shared_context + [2101, 2102, 2103, 2104, 2105, 3001, 3002, 3003, 4101, 4102, 4103, 4104, 5001],
            [6101, 6102, 6103],
            0.91,
        ),
        ("prompt-a", "traj-a-1", 0, shared_context, [2101, 2102, 2103, 2104, 2115], 0.28),
        (
            "prompt-a",
            "traj-a-1",
            1,
            shared_context + [2101, 2102, 2103, 2104, 2115, 3001, 3002, 3014],
            [4101, 4102, 4203],
            0.63,
        ),
        ("prompt-a", "traj-a-2", 0, shared_context, [2201, 2202, 2203, 2204], 0.18),
        (
            "prompt-a",
            "traj-a-2",
            1,
            shared_context + [2201, 2202, 2203, 2204, 3301, 3302],
            [4301, 4302, 4303, 4304],
            0.52,
        ),
        ("prompt-a", "traj-a-3", 0, shared_context, [2101, 2102, 2103, 2104, 2105], 0.35),
        (
            "prompt-a",
            "traj-a-3",
            1,
            shared_context + [2101, 2102, 2103, 2104, 2105, 3001, 3002, 3003],
            [4101, 4102, 4103, 4114],
            0.81,
        ),
        ("prompt-b", "traj-b-0", 0, [7001, 7002, 7003, 7004], [7101, 7102, 7103], None),
        ("prompt-b", "traj-b-1", 0, [7001, 7002, 7003, 7004], [7201, 7202], 0.11),
    ]
    last_step_by_trajectory = {}
    for _prompt_uid, traj_uid, step_index, _prompt_ids, _response_ids, _reward in samples:
        last_step_by_trajectory[traj_uid] = max(last_step_by_trajectory.get(traj_uid, 0), step_index)

    for prompt_uid, traj_uid, step_index, prompt_ids, response_ids, reward in samples:
        rows.append(
            {
                "step_key": f"{traj_uid}:{step_index}",
                "storage_index": len(rows),
                "trajectory_uid": traj_uid,
                "prompt_uid": prompt_uid,
                "step_index": step_index,
                "is_last": step_index == last_step_by_trajectory[traj_uid],
                "complete": True,
                "reward": reward,
                "reward_state": "present" if reward is not None else "missing",
                "policy_version": 5 if prompt_uid == "prompt-a" else 4,
                "prompt_len": len(prompt_ids),
                "response_len": len(response_ids),
                "token_count": len(prompt_ids) + len(response_ids),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "action_summary": str(response_ids),
                "agent": "mock-agent",
                "task": "prefix-tree-demo" if prompt_uid == "prompt-a" else "baseline-demo",
                "metadata": {
                    "agent": "mock-agent",
                    "task": "prefix-tree-demo" if prompt_uid == "prompt-a" else "baseline-demo",
                    "demo": "multi-branch-shared-prefix" if prompt_uid == "prompt-a" else "small-control",
                },
            }
        )
    return rows


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def main() -> None:
    config = load_config()
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
