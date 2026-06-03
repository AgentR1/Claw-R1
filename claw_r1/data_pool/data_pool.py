"""DataPool — the central trajectory buffer between rollout and training.

DataPool is a Ray Actor that stores rollout trajectories at Step granularity
and serves training-ready batches via a pluggable TrainingBackend.  It is
completely agnostic to how trajectories are generated (AgentFlow, external
agents, etc.) and how training is performed (verl PPO, GRPO, etc.).

Data is partitioned by **channel** (e.g. ``"train"`` / ``"val"``), so
training and validation flows can share the same DataPool without mixing.

Responsibilities:
- Store Steps (raw, unpadded).
- Track trajectory completeness via ``is_last`` flags.
- Track prompt-group readiness (all n rollouts complete).
- Serve batches in FIFO order through ``fetch_batch``.
- Delegate format conversion to a ``TrainingBackend`` instance.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import ray

from claw_r1.data_pool.data_model import DataPoolConfig, Step
from claw_r1.data_pool.training_backend import TrainingBackend

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_CHANNEL = "train"


@dataclass
class _PromptGroup:
    """Internal bookkeeping for all trajectories originating from one prompt."""

    prompt_uid: str
    """The prompt identifier shared by all trajectories in this group."""

    trajectory_uids: set[str] = field(default_factory=set)
    """Trajectory UIDs that belong to this group."""

    complete_trajectories: set[str] = field(default_factory=set)
    """Subset of trajectory_uids whose trajectories are complete (received is_last)."""

    step_indices: list[int] = field(default_factory=list)
    """Indices into DataPool._steps for all steps in this group."""

    def is_ready(self, n_rollouts: int) -> bool:
        """Whether all expected trajectories are complete.

        Args:
            n_rollouts (int): Expected number of rollout trajectories per prompt.

        Returns:
            bool: True if all n_rollouts trajectories have been fully received.
        """
        return len(self.complete_trajectories) >= n_rollouts and len(self.trajectory_uids) >= n_rollouts


class _ChannelState:
    """Independent per-channel state (storage, indices, FIFO, statistics)."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.trajectory_index: dict[str, list[int]] = {}
        self.trajectory_complete: dict[str, bool] = {}
        self.prompt_groups: dict[str, _PromptGroup] = {}
        self.fifo_queue: list[str] = []
        self.fifo_set: set[str] = set()
        self.consume_cursor: int = 0
        self.ready_event: asyncio.Event = asyncio.Event()
        self.shutdown: bool = False
        self.total_produced: int = 0
        self.total_consumed: int = 0
        self.total_dropped: int = 0
        self.curation: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.event_cursor: int = 0


@ray.remote
class DataPool:
    """Central async buffer between rollout generation and training.

    Data is partitioned by *channel*.  The default channel ``"train"`` is
    used for the training data flow.  Validation (or any other independent
    flow) can use a different channel name (e.g. ``"val"``) to keep its
    data completely isolated.
    """

    def __init__(
        self,
        config: DataPoolConfig,
        training_backend: TrainingBackend,
        max_queue_size: int | None = None,
    ):
        self._config = config
        self._training_backend = training_backend
        self._max_queue_size = max_queue_size

        self._channels: dict[str, _ChannelState] = {}

        # Validation metrics channel (independent of data channels)
        self._val_queue: list[Any] = []

    # ── Channel helper ────────────────────────────────────────────────────

    def _ch(self, channel: str) -> _ChannelState:
        """Get or create the state for *channel*."""
        if channel not in self._channels:
            self._channels[channel] = _ChannelState()
        return self._channels[channel]

    # ── Producer API ──────────────────────────────────────────────────────

    def submit_step(self, step: Step, channel: str = DEFAULT_CHANNEL) -> None:
        """Submit a single step to the pool.

        Args:
            step: The step to store.
            channel: Data channel name (default ``"train"``).
        """
        ch = self._ch(channel)
        idx = len(ch.steps)
        ch.steps.append(step)

        traj_uid = step.trajectory_uid
        if traj_uid not in ch.trajectory_index:
            ch.trajectory_index[traj_uid] = []
            ch.trajectory_complete[traj_uid] = False
        ch.trajectory_index[traj_uid].append(idx)

        if step.is_last:
            ch.trajectory_complete[traj_uid] = True

        prompt_uid = step.prompt_uid
        if prompt_uid not in ch.prompt_groups:
            ch.prompt_groups[prompt_uid] = _PromptGroup(prompt_uid=prompt_uid)
            ch.fifo_queue.append(prompt_uid)
            ch.fifo_set.add(prompt_uid)

        group = ch.prompt_groups[prompt_uid]
        group.trajectory_uids.add(traj_uid)
        group.step_indices.append(idx)

        if ch.trajectory_complete.get(traj_uid, False):
            group.complete_trajectories.add(traj_uid)

        ch.total_produced += 1
        self._append_event(
            ch,
            "step_submitted",
            step,
            {"channel": channel, "storage_index": idx},
        )

        if self._max_queue_size is not None and channel == DEFAULT_CHANNEL:
            unconsumed = len(ch.fifo_queue) - ch.consume_cursor
            while unconsumed > self._max_queue_size:
                self._drop_oldest_group(ch)
                unconsumed = len(ch.fifo_queue) - ch.consume_cursor

        self._check_and_signal(ch)

    def submit_steps(self, steps: list[Step], channel: str = DEFAULT_CHANNEL) -> None:
        """Submit multiple steps at once."""
        for step in steps:
            self.submit_step(step, channel=channel)

    # ── Consumer API ──────────────────────────────────────────────────────

    async def fetch_batch(
        self,
        batch_size: int,
        n_rollouts: int | None = None,
        channel: str = DEFAULT_CHANNEL,
    ) -> Any:
        """Fetch the next training batch in FIFO order.

        Blocks until ``batch_size`` prompt groups at the head of the FIFO
        queue are ready, then converts via ``TrainingBackend``.

        Returns *None* when the channel has been shut down and no data is
        available.

        Args:
            batch_size: Number of prompt groups to include.
            n_rollouts: Expected rollouts per prompt group.
            channel: Data channel to fetch from (default ``"train"``).
        """
        ch = self._ch(channel)
        effective_n = n_rollouts if n_rollouts is not None else self._config.n_rollouts

        while not self._has_ready_batch(ch, batch_size, effective_n):
            if ch.shutdown:
                return None
            ch.ready_event.clear()
            await ch.ready_event.wait()

        consumed_uids = ch.fifo_queue[ch.consume_cursor : ch.consume_cursor + batch_size]

        batch_steps: list[Step] = []
        for prompt_uid in consumed_uids:
            group = ch.prompt_groups[prompt_uid]
            group_steps = [ch.steps[i] for i in group.step_indices]
            group_steps.sort(key=lambda s: (s.trajectory_uid, s.step_index))
            batch_steps.extend(group_steps)

        result = self._training_backend.convert(batch_steps)

        ch.consume_cursor += batch_size
        ch.total_consumed += batch_size
        self._cleanup(ch, consumed_uids)

        return result

    # ── Trajectory management ────────────────────────────────────────────

    def complete_trajectory(
        self,
        trajectory_uid: str,
        reward: float | None = None,
        channel: str = DEFAULT_CHANNEL,
    ) -> bool:
        """Mark the last Step of a trajectory as ``is_last=True``."""
        ch = self._ch(channel)
        idx_list = ch.trajectory_index.get(trajectory_uid)
        if not idx_list:
            return False

        last_idx = idx_list[-1]
        last_step = ch.steps[last_idx]
        last_step.is_last = True
        if reward is not None:
            last_step.reward = reward

        ch.trajectory_complete[trajectory_uid] = True

        prompt_uid = last_step.prompt_uid
        group = ch.prompt_groups.get(prompt_uid)
        if group:
            group.complete_trajectories.add(trajectory_uid)
            self._check_and_signal(ch)

        self._append_event(
            ch,
            "trajectory_completed",
            last_step,
            {"channel": channel, "reward": reward},
        )
        return True

    # ── Dashboard read/curation API ───────────────────────────────────────

    def list_steps(
        self,
        channel: str = DEFAULT_CHANNEL,
        *,
        prompt_uid: str | None = None,
        trajectory_uid: str | None = None,
        agent: str | None = None,
        quality: str | None = None,
        trainable: bool | None = None,
        reward_state: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        policy_version: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return serialized Step rows for dashboard inspection.

        Curation fields are read from an independent side table keyed by
        ``trajectory_uid`` and ``step_index``.  The stored Step objects are not
        modified by curation updates, so training batch conversion is unchanged.
        """
        ch = self._ch(channel)
        rows = [self._serialize_step(ch, step, idx, channel) for idx, step in enumerate(ch.steps)]
        rows = [
            row
            for row in rows
            if self._matches_step_filters(
                row,
                prompt_uid=prompt_uid,
                trajectory_uid=trajectory_uid,
                agent=agent,
                quality=quality,
                trainable=trainable,
                reward_state=reward_state,
                min_reward=min_reward,
                max_reward=max_reward,
                policy_version=policy_version,
            )
        ]
        rows.sort(key=lambda r: (r["prompt_uid"], r["trajectory_uid"], r["step_index"]))
        total = len(rows)
        end = max(offset, 0) + max(limit, 0)
        return {"channel": channel, "total": total, "steps": rows[max(offset, 0) : end]}

    def list_trajectories(
        self,
        channel: str = DEFAULT_CHANNEL,
        *,
        prompt_uid: str | None = None,
        trajectory_uid: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return trajectory summaries assembled from current channel steps."""
        ch = self._ch(channel)
        summaries: list[dict[str, Any]] = []
        for uid, indices in ch.trajectory_index.items():
            if trajectory_uid and uid != trajectory_uid:
                continue
            steps = [ch.steps[i] for i in indices]
            if not steps:
                continue
            first = steps[0]
            if prompt_uid and first.prompt_uid != prompt_uid:
                continue
            rewards = [s.reward for s in steps if s.reward is not None]
            curated = [self._curation_for_step(ch, s) for s in steps]
            summaries.append(
                {
                    "trajectory_uid": uid,
                    "prompt_uid": first.prompt_uid,
                    "step_count": len(steps),
                    "complete": ch.trajectory_complete.get(uid, False),
                    "reward_sum": sum(rewards) if rewards else None,
                    "reward_min": min(rewards) if rewards else None,
                    "reward_max": max(rewards) if rewards else None,
                    "policy_versions": sorted({s.policy_version for s in steps}),
                    "agents": sorted(
                        {
                            str((s.metadata or {}).get("agent"))
                            for s in steps
                            if (s.metadata or {}).get("agent") is not None
                        }
                    ),
                    "quality_counts": self._count_values(c.get("quality") for c in curated),
                    "trainable_steps": sum(1 for c in curated if c.get("trainable", True)),
                    "metadata": first.metadata or {},
                }
            )
        summaries.sort(key=lambda r: (r["prompt_uid"], r["trajectory_uid"]))
        total = len(summaries)
        end = max(offset, 0) + max(limit, 0)
        return {"channel": channel, "total": total, "trajectories": summaries[max(offset, 0) : end]}

    def get_step_events(
        self,
        channel: str = DEFAULT_CHANNEL,
        *,
        cursor: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Return step/curation events after a monotonic cursor."""
        ch = self._ch(channel)
        events = [event for event in ch.events if event["cursor"] > cursor]
        events = events[: max(limit, 0)]
        next_cursor = events[-1]["cursor"] if events else cursor
        return {
            "channel": channel,
            "cursor": next_cursor,
            "events": events,
            "has_more": bool(ch.events and next_cursor < ch.events[-1]["cursor"]),
        }

    def update_step_curation(
        self,
        updates: list[dict[str, Any]],
        channel: str = DEFAULT_CHANNEL,
    ) -> dict[str, Any]:
        """Update dashboard-only curation fields for one or more steps."""
        ch = self._ch(channel)
        updated: list[dict[str, Any]] = []
        allowed = {"quality", "trainable", "reward_status", "tags", "note"}
        for item in updates:
            step = self._find_step_for_curation(ch, item)
            if step is None:
                continue
            key = self._step_key(step)
            existing = dict(ch.curation.get(key, {}))
            patch = {k: item[k] for k in allowed if k in item}
            if "tags" in patch and patch["tags"] is None:
                patch["tags"] = []
            existing.update(patch)
            ch.curation[key] = existing
            self._append_event(ch, "curation_updated", step, {"curation": existing, "patch": patch})
            updated.append({"step_key": key, "curation": existing})
        return {"channel": channel, "updated": updated, "updated_count": len(updated)}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def shutdown(self, channel: str | None = None) -> None:
        """Signal channel(s) to stop serving batches.

        If *channel* is None, shuts down all channels.
        """
        targets = [self._ch(channel)] if channel else list(self._channels.values())
        for ch in targets:
            ch.shutdown = True
            ch.ready_event.set()

    # ── Validation metrics channel ────────────────────────────────────────

    def put_validate(self, data: Any) -> None:
        """Push validation metrics/data for the Trainer to consume."""
        self._val_queue.append(data)

    def get_validate(self) -> Any | None:
        """Pop the oldest validation entry, or *None* if empty."""
        if self._val_queue:
            return self._val_queue.pop(0)
        return None

    # ── Observability ─────────────────────────────────────────────────────

    def get_statistics(self, channel: str = DEFAULT_CHANNEL) -> dict:
        """Return detailed pool statistics for a channel."""
        ch = self._ch(channel)
        unconsumed = len(ch.fifo_queue) - ch.consume_cursor
        ready = sum(
            1 for uid in ch.fifo_queue[ch.consume_cursor :] if ch.prompt_groups[uid].is_ready(self._config.n_rollouts)
        )
        return {
            "total_steps": len(ch.steps),
            "total_produced": ch.total_produced,
            "total_consumed": ch.total_consumed,
            "total_dropped": ch.total_dropped,
            "queue_size": unconsumed,
            "ready_prompt_groups": ready,
            "max_queue_size": self._max_queue_size,
            "shutdown": ch.shutdown,
            "curated_steps": len(ch.curation),
            "event_cursor": ch.event_cursor,
        }

    def stats(self, channel: str = DEFAULT_CHANNEL) -> dict:
        """Return pool statistics for monitoring."""
        ch = self._ch(channel)
        total_groups = len(ch.fifo_queue)
        consumed = ch.consume_cursor
        pending = total_groups - consumed
        ready = sum(
            1 for uid in ch.fifo_queue[ch.consume_cursor :] if ch.prompt_groups[uid].is_ready(self._config.n_rollouts)
        )
        return {
            "total_steps": len(ch.steps),
            "total_trajectories": len(ch.trajectory_index),
            "complete_trajectories": sum(1 for v in ch.trajectory_complete.values() if v),
            "total_prompt_groups": total_groups,
            "consumed_prompt_groups": consumed,
            "pending_prompt_groups": pending,
            "ready_prompt_groups": ready,
            "curated_steps": len(ch.curation),
            "event_cursor": ch.event_cursor,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _drop_oldest_group(self, ch: _ChannelState) -> None:
        if ch.consume_cursor >= len(ch.fifo_queue):
            return
        uid = ch.fifo_queue[ch.consume_cursor]
        group = ch.prompt_groups.pop(uid, None)
        if group:
            for traj_uid in group.trajectory_uids:
                ch.trajectory_index.pop(traj_uid, None)
                ch.trajectory_complete.pop(traj_uid, None)
            ch.fifo_set.discard(uid)
        ch.consume_cursor += 1
        ch.total_dropped += 1
        logger.debug("DataPool dropped oldest group %s (total dropped: %d)", uid, ch.total_dropped)

    def _has_ready_batch(self, ch: _ChannelState, batch_size: int, n_rollouts: int) -> bool:
        end = ch.consume_cursor + batch_size
        if end > len(ch.fifo_queue):
            return False
        return all(ch.prompt_groups[uid].is_ready(n_rollouts) for uid in ch.fifo_queue[ch.consume_cursor : end])

    def _check_and_signal(self, ch: _ChannelState) -> None:
        if ch.consume_cursor < len(ch.fifo_queue):
            uid = ch.fifo_queue[ch.consume_cursor]
            if ch.prompt_groups[uid].is_ready(self._config.n_rollouts):
                ch.ready_event.set()

    def _cleanup(self, ch: _ChannelState, consumed_uids: list[str]) -> None:
        for prompt_uid in consumed_uids:
            group = ch.prompt_groups.pop(prompt_uid, None)
            if group is None:
                continue
            for traj_uid in group.trajectory_uids:
                ch.trajectory_index.pop(traj_uid, None)
                ch.trajectory_complete.pop(traj_uid, None)
            ch.fifo_set.discard(prompt_uid)

        total = len(ch.steps)
        if total > 0 and ch.consume_cursor > 0:
            remaining_indices = set()
            for group in ch.prompt_groups.values():
                remaining_indices.update(group.step_indices)

            if len(remaining_indices) < total * 0.5:
                self._compact(ch, remaining_indices)

    def _compact(self, ch: _ChannelState, remaining_indices: set[int]) -> None:
        old_to_new: dict[int, int] = {}
        new_steps: list[Step] = []
        for old_idx in sorted(remaining_indices):
            old_to_new[old_idx] = len(new_steps)
            new_steps.append(ch.steps[old_idx])

        ch.steps = new_steps

        for traj_uid, idx_list in ch.trajectory_index.items():
            ch.trajectory_index[traj_uid] = [old_to_new[i] for i in idx_list if i in old_to_new]

        for group in ch.prompt_groups.values():
            group.step_indices = [old_to_new[i] for i in group.step_indices if i in old_to_new]

        logger.debug(
            "DataPool compacted: %d -> %d steps",
            len(remaining_indices) + ch.consume_cursor,
            len(new_steps),
        )

    def _step_key(self, step: Step) -> str:
        return f"{step.trajectory_uid}:{step.step_index}"

    def _curation_for_step(self, ch: _ChannelState, step: Step) -> dict[str, Any]:
        curation = dict(ch.curation.get(self._step_key(step), {}))
        curation.setdefault("quality", "unreviewed")
        curation.setdefault("trainable", True)
        curation.setdefault("reward_status", "present" if step.reward is not None else "missing")
        curation.setdefault("tags", [])
        return curation

    def _serialize_step(self, ch: _ChannelState, step: Step, storage_index: int, channel: str) -> dict[str, Any]:
        curation = self._curation_for_step(ch, step)
        prompt_len = len(step.prompt_ids or [])
        response_len = len(step.response_ids or [])
        metadata = step.metadata or {}
        return {
            "step_key": self._step_key(step),
            "storage_index": storage_index,
            "channel": channel,
            "trajectory_uid": step.trajectory_uid,
            "prompt_uid": step.prompt_uid,
            "step_index": step.step_index,
            "is_last": step.is_last,
            "complete": ch.trajectory_complete.get(step.trajectory_uid, False),
            "reward": step.reward,
            "reward_state": "present" if step.reward is not None else "missing",
            "policy_version": step.policy_version,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "token_count": prompt_len + response_len,
            "prompt_ids": list(step.prompt_ids or []),
            "response_ids": list(step.response_ids or []),
            "action_summary": self._summarize_tokens(step.response_ids),
            "agent": metadata.get("agent"),
            "task": metadata.get("task") or metadata.get("data_source") or metadata.get("dataset"),
            "metadata": metadata,
            "curation": curation,
        }

    def _matches_step_filters(self, row: dict[str, Any], **filters: Any) -> bool:
        if filters["prompt_uid"] and row["prompt_uid"] != filters["prompt_uid"]:
            return False
        if filters["trajectory_uid"] and row["trajectory_uid"] != filters["trajectory_uid"]:
            return False
        if filters["agent"] and row["agent"] != filters["agent"]:
            return False
        if filters["quality"] and row["curation"].get("quality") != filters["quality"]:
            return False
        if filters["trainable"] is not None and row["curation"].get("trainable", True) != filters["trainable"]:
            return False
        if filters["reward_state"] and row["reward_state"] != filters["reward_state"]:
            return False
        if filters["policy_version"] is not None and row["policy_version"] != filters["policy_version"]:
            return False
        reward = row["reward"]
        if filters["min_reward"] is not None and (reward is None or reward < filters["min_reward"]):
            return False
        if filters["max_reward"] is not None and (reward is None or reward > filters["max_reward"]):
            return False
        return True

    def _find_step_for_curation(self, ch: _ChannelState, item: dict[str, Any]) -> Step | None:
        step_key = item.get("step_key")
        for step in ch.steps:
            if step_key and self._step_key(step) == step_key:
                return step
            if (
                item.get("trajectory_uid") == step.trajectory_uid
                and item.get("step_index") == step.step_index
            ):
                return step
        return None

    def _append_event(
        self,
        ch: _ChannelState,
        event_type: str,
        step: Step,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ch.event_cursor += 1
        ch.events.append(
            {
                "cursor": ch.event_cursor,
                "type": event_type,
                "step_key": self._step_key(step),
                "trajectory_uid": step.trajectory_uid,
                "prompt_uid": step.prompt_uid,
                "step_index": step.step_index,
                "policy_version": step.policy_version,
                "reward": step.reward,
                "payload": payload or {},
            }
        )
        if len(ch.events) > 5000:
            ch.events = ch.events[-5000:]

    def _summarize_tokens(self, tokens: list[int] | None, max_items: int = 12) -> str:
        tokens = tokens or []
        shown = tokens[:max_items]
        suffix = " ..." if len(tokens) > max_items else ""
        return "[" + ", ".join(str(t) for t in shown) + suffix + "]"

    def _count_values(self, values: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            key = str(value or "unreviewed")
            counts[key] = counts.get(key, 0) + 1
        return counts
