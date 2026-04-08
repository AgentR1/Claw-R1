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

        return True

    def get_last_step(
        self,
        trajectory_uid: str,
        channel: str = DEFAULT_CHANNEL,
    ) -> Step | None:
        """Return the last Step of a trajectory, or None if not found."""
        ch = self._ch(channel)
        idx_list = ch.trajectory_index.get(trajectory_uid)
        if not idx_list:
            return None
        return ch.steps[idx_list[-1]]

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
            "max_queue_size": self._max_queue_size if self._max_queue_size is not None else -1,
            "shutdown": ch.shutdown,
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
