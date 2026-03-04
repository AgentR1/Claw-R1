"""DataPool — the central trajectory buffer between rollout and training.

DataPool is a Ray Actor that stores rollout trajectories at Step granularity
and serves training-ready batches via a pluggable TrainingBackend.  It is
completely agnostic to how trajectories are generated (AgentFlow, external
agents, etc.) and how training is performed (verl PPO, GRPO, etc.).

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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import ray

from agent_r1.data_pool.data_model import DataPoolConfig, Step
from agent_r1.data_pool.training_backend import TrainingBackend

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
        return (
            len(self.complete_trajectories) >= n_rollouts
            and len(self.trajectory_uids) >= n_rollouts
        )


@ray.remote
class DataPool:
    """Central async buffer between rollout generation and training.

    Stores Steps in a flat list, maintains indices for trajectory and
    prompt-group membership, and serves batches in FIFO order through
    ``fetch_batch``.  Format conversion is delegated to a pluggable
    ``TrainingBackend``.
    """

    def __init__(
        self,
        config: DataPoolConfig,
        training_backend: TrainingBackend,
        max_queue_size: int | None = None,
    ):
        """Initialize the DataPool.

        Args:
            config (DataPoolConfig): Pool configuration.
            training_backend (TrainingBackend): Backend that converts
                ``list[Step]`` into the training engine's native format.
            max_queue_size (int | None): Maximum number of unconsumed prompt
                groups to keep.  When exceeded the oldest groups are dropped.
                *None* means unlimited (suitable for synchronous mode).
        """
        self._config = config
        self._training_backend = training_backend
        self._max_queue_size = max_queue_size

        # ── Storage ──
        self._steps: list[Step] = []

        # ── Trajectory index: trajectory_uid → list of indices into _steps ──
        self._trajectory_index: dict[str, list[int]] = {}

        # ── Trajectory completeness: trajectory_uid → True if is_last received ──
        self._trajectory_complete: dict[str, bool] = {}

        # ── Prompt group index: prompt_uid → _PromptGroup ──
        self._prompt_groups: dict[str, _PromptGroup] = {}

        # ── FIFO queue of prompt_uids, in the order they first appeared ──
        self._fifo_queue: list[str] = []
        self._fifo_set: set[str] = set()

        # ── Consumed cursor: how many prompt groups have been consumed ──
        self._consume_cursor: int = 0

        # ── Async signalling ──
        self._ready_event = asyncio.Event()
        self._shutdown = False

        # ── Statistics ──
        self._total_produced: int = 0
        self._total_consumed: int = 0
        self._total_dropped: int = 0

        # ── Validation channel ──
        self._val_queue: list[Any] = []

    # ── Producer API ──────────────────────────────────────────────────────

    def submit_step(self, step: Step) -> None:
        """Submit a single step to the pool.

        Updates internal indices and, when ``step.is_last`` is True, marks the
        trajectory as complete.  When all trajectories in a prompt group are
        complete the group becomes ready for consumption.

        Args:
            step (Step): The step to store.
        """
        idx = len(self._steps)
        self._steps.append(step)

        # Update trajectory index
        traj_uid = step.trajectory_uid
        if traj_uid not in self._trajectory_index:
            self._trajectory_index[traj_uid] = []
            self._trajectory_complete[traj_uid] = False
        self._trajectory_index[traj_uid].append(idx)

        if step.is_last:
            self._trajectory_complete[traj_uid] = True

        # Update prompt group
        prompt_uid = step.prompt_uid
        if prompt_uid not in self._prompt_groups:
            self._prompt_groups[prompt_uid] = _PromptGroup(prompt_uid=prompt_uid)
            self._fifo_queue.append(prompt_uid)
            self._fifo_set.add(prompt_uid)

        group = self._prompt_groups[prompt_uid]
        group.trajectory_uids.add(traj_uid)
        group.step_indices.append(idx)

        if self._trajectory_complete.get(traj_uid, False):
            group.complete_trajectories.add(traj_uid)

        self._total_produced += 1

        # Enforce queue capacity: drop oldest unconsumed groups when full
        if self._max_queue_size is not None:
            unconsumed = len(self._fifo_queue) - self._consume_cursor
            while unconsumed > self._max_queue_size:
                self._drop_oldest_group()
                unconsumed = len(self._fifo_queue) - self._consume_cursor

        # Signal if enough prompt groups at the head of the FIFO are ready
        self._check_and_signal()

    def submit_steps(self, steps: list[Step]) -> None:
        """Submit multiple steps at once (convenience wrapper).

        Args:
            steps (list[Step]): Steps to store.
        """
        for step in steps:
            self.submit_step(step)

    # ── Consumer API ──────────────────────────────────────────────────────

    async def fetch_batch(self, batch_size: int, n_rollouts: int | None = None) -> Any:
        """Fetch the next training batch in FIFO order.

        Blocks until ``batch_size`` prompt groups at the head of the FIFO queue
        are ready (all their trajectories are complete), then collects the
        corresponding steps, converts them via the ``TrainingBackend``, removes
        the consumed data from internal storage, and returns the result.

        Returns *None* when the pool has been shut down and no more data is
        available.

        Args:
            batch_size (int): Number of prompt groups to include in the batch.
            n_rollouts (int | None): Expected rollouts per prompt group.
                Defaults to ``DataPoolConfig.n_rollouts`` when *None*.

        Returns:
            Any: Training-ready batch produced by ``TrainingBackend.convert()``,
                or *None* if the pool has been shut down.
        """
        effective_n = n_rollouts if n_rollouts is not None else self._config.n_rollouts
        while not self._has_ready_batch(batch_size, effective_n):
            if self._shutdown:
                return None
            self._ready_event.clear()
            await self._ready_event.wait()

        # Collect the first batch_size ready prompt groups
        consumed_uids = self._fifo_queue[self._consume_cursor : self._consume_cursor + batch_size]

        # Gather all steps for these prompt groups, ordered by trajectory
        # then by step_index so that each trajectory's steps are contiguous.
        batch_steps: list[Step] = []
        for prompt_uid in consumed_uids:
            group = self._prompt_groups[prompt_uid]
            group_steps = [self._steps[i] for i in group.step_indices]
            group_steps.sort(key=lambda s: (s.trajectory_uid, s.step_index))
            batch_steps.extend(group_steps)

        # Convert via training backend
        result = self._training_backend.convert(batch_steps)

        # Advance cursor and clean up consumed data
        self._consume_cursor += batch_size
        self._total_consumed += batch_size
        self._cleanup(consumed_uids)

        return result

    # ── Trajectory management ────────────────────────────────────────────

    def complete_trajectory(self, trajectory_uid: str, reward: float | None = None) -> bool:
        """Mark the last Step of a trajectory as ``is_last=True``.

        Used by the Gateway when a black-box trajectory finishes.  Optionally
        sets the reward on the last Step as well.

        Args:
            trajectory_uid: UID of the trajectory to complete.
            reward: Optional reward to write onto the last Step.

        Returns:
            True if the trajectory was found and marked; False otherwise.
        """
        idx_list = self._trajectory_index.get(trajectory_uid)
        if not idx_list:
            return False

        last_idx = idx_list[-1]
        last_step = self._steps[last_idx]
        last_step.is_last = True
        if reward is not None:
            last_step.reward = reward

        self._trajectory_complete[trajectory_uid] = True

        prompt_uid = last_step.prompt_uid
        group = self._prompt_groups.get(prompt_uid)
        if group:
            group.complete_trajectories.add(trajectory_uid)
            self._check_and_signal()

        return True

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Signal the pool to stop serving batches.

        Any blocked ``fetch_batch`` call will return *None*.
        """
        self._shutdown = True
        self._ready_event.set()

    # ── Validation channel ────────────────────────────────────────────────

    def put_validate(self, data: Any) -> None:
        """Push validation metrics/data for the Trainer to consume."""
        self._val_queue.append(data)

    def get_validate(self) -> Any | None:
        """Pop the oldest validation entry, or *None* if empty."""
        if self._val_queue:
            return self._val_queue.pop(0)
        return None

    # ── Observability ─────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Return detailed pool statistics for async monitoring."""
        unconsumed = len(self._fifo_queue) - self._consume_cursor
        ready = sum(
            1
            for uid in self._fifo_queue[self._consume_cursor:]
            if self._prompt_groups[uid].is_ready(self._config.n_rollouts)
        )
        return {
            "total_steps": len(self._steps),
            "total_produced": self._total_produced,
            "total_consumed": self._total_consumed,
            "total_dropped": self._total_dropped,
            "queue_size": unconsumed,
            "ready_prompt_groups": ready,
            "max_queue_size": self._max_queue_size,
            "shutdown": self._shutdown,
        }

    def stats(self) -> dict:
        """Return pool statistics for monitoring.

        Returns:
            dict: Statistics including pool size, pending/complete counts, etc.
        """
        total_groups = len(self._fifo_queue)
        consumed = self._consume_cursor
        pending = total_groups - consumed
        ready = sum(
            1
            for uid in self._fifo_queue[self._consume_cursor :]
            if self._prompt_groups[uid].is_ready(self._config.n_rollouts)
        )
        return {
            "total_steps": len(self._steps),
            "total_trajectories": len(self._trajectory_index),
            "complete_trajectories": sum(1 for v in self._trajectory_complete.values() if v),
            "total_prompt_groups": total_groups,
            "consumed_prompt_groups": consumed,
            "pending_prompt_groups": pending,
            "ready_prompt_groups": ready,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _drop_oldest_group(self) -> None:
        """Drop the oldest unconsumed prompt group to enforce queue capacity."""
        if self._consume_cursor >= len(self._fifo_queue):
            return
        uid = self._fifo_queue[self._consume_cursor]
        group = self._prompt_groups.pop(uid, None)
        if group:
            for traj_uid in group.trajectory_uids:
                self._trajectory_index.pop(traj_uid, None)
                self._trajectory_complete.pop(traj_uid, None)
            self._fifo_set.discard(uid)
        self._consume_cursor += 1
        self._total_dropped += 1
        logger.debug("DataPool dropped oldest group %s (total dropped: %d)", uid, self._total_dropped)

    def _has_ready_batch(self, batch_size: int, n_rollouts: int | None = None) -> bool:
        """Check if batch_size consecutive prompt groups from the cursor are ready."""
        effective_n = n_rollouts if n_rollouts is not None else self._config.n_rollouts
        end = self._consume_cursor + batch_size
        if end > len(self._fifo_queue):
            return False
        return all(
            self._prompt_groups[uid].is_ready(effective_n)
            for uid in self._fifo_queue[self._consume_cursor : end]
        )

    def _check_and_signal(self) -> None:
        """Signal the ready event if any waiting fetch_batch can proceed."""
        # We don't know the requested batch_size here, so signal optimistically
        # whenever any unconsumed prompt group is ready; fetch_batch re-checks.
        if self._consume_cursor < len(self._fifo_queue):
            uid = self._fifo_queue[self._consume_cursor]
            if self._prompt_groups[uid].is_ready(self._config.n_rollouts):
                self._ready_event.set()

    def _cleanup(self, consumed_uids: list[str]) -> None:
        """Remove consumed prompt groups and their data from internal storage.

        To avoid O(n) list compaction on every fetch, we use lazy cleanup:
        consumed entries are removed from indices but the underlying _steps
        list is left intact (entries are referenced by index).  A full
        compaction runs when the consumed fraction exceeds a threshold.

        Args:
            consumed_uids (list[str]): Prompt UIDs to clean up.
        """
        for prompt_uid in consumed_uids:
            group = self._prompt_groups.pop(prompt_uid, None)
            if group is None:
                continue
            for traj_uid in group.trajectory_uids:
                self._trajectory_index.pop(traj_uid, None)
                self._trajectory_complete.pop(traj_uid, None)
            self._fifo_set.discard(prompt_uid)

        # Compact storage when >50% of stored steps have been consumed
        total = len(self._steps)
        if total > 0 and self._consume_cursor > 0:
            remaining_indices = set()
            for group in self._prompt_groups.values():
                remaining_indices.update(group.step_indices)

            if len(remaining_indices) < total * 0.5:
                self._compact(remaining_indices)

    def _compact(self, remaining_indices: set[int]) -> None:
        """Rebuild internal storage keeping only steps at remaining_indices.

        Args:
            remaining_indices (set[int]): Step indices to keep.
        """
        old_to_new: dict[int, int] = {}
        new_steps: list[Step] = []
        for old_idx in sorted(remaining_indices):
            old_to_new[old_idx] = len(new_steps)
            new_steps.append(self._steps[old_idx])

        self._steps = new_steps

        # Re-index trajectories
        for traj_uid, idx_list in self._trajectory_index.items():
            self._trajectory_index[traj_uid] = [
                old_to_new[i] for i in idx_list if i in old_to_new
            ]

        # Re-index prompt groups
        for group in self._prompt_groups.values():
            group.step_indices = [
                old_to_new[i] for i in group.step_indices if i in old_to_new
            ]

        logger.debug("DataPool compacted: %d -> %d steps", len(remaining_indices) + self._consume_cursor, len(new_steps))
