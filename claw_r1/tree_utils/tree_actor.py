"""Tree-aware actor wrapper for prefix-tree-packed forward passes.

Provides :class:`TreeDataParallelPPOActor`, a subclass of verl's
:class:`DataParallelPPOActor` that intercepts tree-packed batches
(those carrying ``tree_attention_masks`` in ``non_tensor_batch``) and
routes them through a FlexAttention-based forward path using
:func:`tree_attention_context`.

Standard (non-tree) batches are handled by the parent class unchanged.

Design notes
~~~~~~~~~~~~
verl's ``compute_log_prob`` and ``update_policy`` call ``data.select()``
which strips unrecognised ``non_tensor_batch`` keys — including
``tree_attention_masks``.  To avoid modifying verl, this class:

1. Pre-extracts the per-row dense masks before ``super()`` is called.
2. Stores them on ``self._active_tree_masks`` (a list aligned with the
   batch dimension).
3. Overrides ``_forward_micro_batch`` to check this list and activate
   tree attention context when needed.

The instance variable ``_active_tree_masks`` is set/cleared around each
``compute_log_prob`` / ``update_policy`` call and is therefore safe
across sequential calls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor

from claw_r1.tree_utils.attention_patch import (
    create_block_mask_from_dense,
    tree_attention_context,
)

logger = logging.getLogger(__name__)


def _resolve_mask(mask_val: Any) -> torch.Tensor | None:
    """Unwrap a per-row mask value from numpy / tensor / None."""
    if mask_val is None:
        return None
    if isinstance(mask_val, np.ndarray):
        inner = mask_val.item() if mask_val.ndim == 0 else mask_val
        if inner is None:
            return None
        if isinstance(inner, torch.Tensor):
            return inner
        return torch.as_tensor(inner, dtype=torch.bool)
    if isinstance(mask_val, torch.Tensor):
        return mask_val
    return None


class TreeDataParallelPPOActor(DataParallelPPOActor):
    """DataParallelPPOActor with tree-packed forward support.

    Overrides ``_forward_micro_batch`` to detect tree rows (via
    ``self._active_tree_masks``) and run the model inside a
    ``tree_attention_context`` with the corresponding ``BlockMask``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_tree_masks: list[torch.Tensor | None] | None = None
        self._active_row_cursor: int = 0
        self._active_total_rows: int = 0

    # ------------------------------------------------------------------
    # Internal: tree-aware micro-batch forward
    # ------------------------------------------------------------------

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, Any],
        temperature: float,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        masks = self._active_tree_masks
        if masks is None:
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)

        input_ids = micro_batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Reset cursor when a new pass through the data starts
        if self._active_row_cursor >= self._active_total_rows:
            self._active_row_cursor = 0

        cursor = self._active_row_cursor
        row_masks = masks[cursor : cursor + batch_size]
        self._active_row_cursor = cursor + batch_size

        has_tree = any(m is not None for m in row_masks)
        if not has_tree:
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)

        # Process row-by-row because each tree has its own BlockMask.
        log_probs_parts: list[torch.Tensor] = []
        entropy_parts: list[torch.Tensor] = []

        for i in range(batch_size):
            dense_mask = row_masks[i]
            row_input_ids = input_ids[i : i + 1]
            row_position_ids = micro_batch["position_ids"][i : i + 1]
            row_responses = micro_batch["responses"][i : i + 1]
            response_length = row_responses.size(-1)

            if dense_mask is None:
                single = {k: v[i : i + 1] if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()}
                out = super()._forward_micro_batch(single, temperature, calculate_entropy)
                log_probs_parts.append(out["log_probs"])
                if calculate_entropy and "entropys" in out:
                    entropy_parts.append(out["entropys"])
                continue

            seq_len = dense_mask.shape[-1]
            device = row_input_ids.device
            block_mask = create_block_mask_from_dense(dense_mask, seq_len, device)

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                with tree_attention_context(block_mask):
                    output = self.actor_module(
                        input_ids=row_input_ids,
                        attention_mask=None,
                        position_ids=row_position_ids,
                        use_cache=False,
                    )

                logits = output.logits
                logits.div_(temperature)

                logits_for_lp = logits[:, -response_length - 1 : -1, :]
                lp = logprobs_from_logits(logits_for_lp, row_responses)
                log_probs_parts.append(lp)

                if calculate_entropy:
                    ent = verl_F.entropy_from_logits(logits_for_lp)
                    entropy_parts.append(ent)

        result: dict[str, torch.Tensor] = {
            "log_probs": torch.cat(log_probs_parts, dim=0),
        }
        if calculate_entropy and entropy_parts:
            result["entropys"] = torch.cat(entropy_parts, dim=0)
        return result

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    def compute_log_prob(
        self, data: DataProto, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        masks = self._extract_tree_masks(data)
        if masks is None:
            return super().compute_log_prob(data, calculate_entropy=calculate_entropy)

        self._active_tree_masks = masks
        self._active_row_cursor = 0
        self._active_total_rows = len(masks)
        try:
            return super().compute_log_prob(data, calculate_entropy=calculate_entropy)
        finally:
            self._active_tree_masks = None

    def update_policy(self, data: DataProto):
        masks = self._extract_tree_masks(data)
        if masks is None:
            return super().update_policy(data)

        self._active_tree_masks = masks
        self._active_row_cursor = 0
        self._active_total_rows = len(masks)
        try:
            return super().update_policy(data)
        finally:
            self._active_tree_masks = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tree_masks(data: DataProto) -> list[torch.Tensor | None] | None:
        """Extract per-row dense tree masks from DataProto, or None if absent."""
        if "tree_attention_masks" not in data.non_tensor_batch:
            return None
        raw = data.non_tensor_batch["tree_attention_masks"]
        resolved = [_resolve_mask(m) for m in raw]
        if all(m is None for m in resolved):
            return None
        return resolved
