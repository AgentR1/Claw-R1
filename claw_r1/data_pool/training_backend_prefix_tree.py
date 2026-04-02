"""TreeVerlBackend — prefix-tree-aware variant of VerlBackend.

This is a *sister class* to ``VerlBackend``.  It shares the same
``TrainingBackend`` interface but groups Steps from the same prompt-group
into a prefix tree before converting them into a ``DataProto``.

The tree-packed representation deduplicates shared prefix tokens so the
downstream actor only needs to compute a single forward pass over a
shorter, merged sequence.  The resulting ``DataProto`` carries extra
metadata (the ``PrefixTree`` object and per-step mapping info) in
``non_tensor_batch`` so that log-probability restoration can be performed
after the forward pass.

**Phase 1** — correctness mode:
  Each tree is expanded back into independent sequences using a full 2-D
  attention mask (no tree-aware attention kernel).  This lets us validate
  that the logprob restoration logic is numerically equivalent to the
  standard ``VerlBackend`` path.

**Phase 2** (future) — performance mode:
  A tree-structured attention kernel (FlexAttention / FA2 with custom mask)
  will allow a single forward pass over the packed sequence with genuine
  computation savings.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from claw_r1.data_pool.data_model import Step
from claw_r1.data_pool.training_backend import TrainingBackend
from claw_r1.tree_utils.prefix_tree import PrefixTree
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class TreeVerlBackend(TrainingBackend):
    """Prefix-tree-aware training backend.

    Steps belonging to the same ``prompt_uid`` group are merged into a
    ``PrefixTree``.  When the tree shrinks the total token count below the
    sum of individual sequence lengths, the forward pass benefits from
    reduced redundant computation.

    The output ``DataProto`` has the following additions compared to the
    standard ``VerlBackend`` output:

    * ``non_tensor_batch["prefix_trees"]`` — array of ``PrefixTree`` objects,
      one per tree-packed row in the batch (a row may merge several Steps).
    * ``non_tensor_batch["tree_step_maps"]`` — array of
      ``list[dict]`` per row, mapping each original Step back to its
      position within the tree.

    For rows where no meaningful prefix sharing exists (single step, or all
    sequences are unique), the backend falls back to the standard padded
    layout identical to ``VerlBackend``.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_length: int,
        response_length: int,
        max_tree_tokens: int | None = None,
        processor: Optional[AutoProcessor] = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (used for pad token).
            prompt_length: Max prompt token length (used for fallback padding).
            response_length: Max response token length (used for fallback padding).
            max_tree_tokens: Hard cap on packed tree size.  Groups whose tree
                would exceed this are split into standard padded rows instead.
                Defaults to ``prompt_length + response_length``.
            processor: Optional multi-modal processor (currently unused in
                tree mode; reserved for future compatibility).
        """
        self._tokenizer = tokenizer
        self._processor = processor
        self._prompt_length = prompt_length
        self._response_length = response_length
        self._max_tree_tokens = max_tree_tokens or (prompt_length + response_length)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, steps: list[Step]) -> DataProto:
        """Convert a list of Steps into a ``DataProto`` with prefix-tree packing.

        Steps are first grouped by ``prompt_uid``.  Within each group,
        full sequences (``prompt_ids + response_ids``) are built and fed
        into ``PrefixTree.build()``.

        The resulting packed tensors are stacked into a batch where each
        *row* corresponds to one prompt group's merged tree (or falls back
        to per-step padding if tree building is not beneficial).

        Args:
            steps: Raw Steps from the DataPool (already sorted by
                ``(trajectory_uid, step_index)`` within each prompt group).

        Returns:
            ``DataProto`` ready for the actor forward pass.
        """
        if not steps:
            raise ValueError("Cannot convert an empty step list.")

        # Group steps by prompt_uid, preserving insertion order
        groups: dict[str, list[Step]] = defaultdict(list)
        for step in steps:
            groups[step.prompt_uid].append(step)

        pad_token_id = self._tokenizer.pad_token_id or 0
        total_seq_len = self._prompt_length + self._response_length

        # Accumulators for the batch
        input_ids_rows: list[torch.Tensor] = []
        attention_mask_rows: list[torch.Tensor] = []
        position_ids_rows: list[torch.Tensor] = []
        prompt_ids_rows: list[torch.Tensor] = []
        response_ids_rows: list[torch.Tensor] = []
        response_mask_rows: list[torch.Tensor] = []

        reward_rows: list[torch.Tensor] = []
        has_reward = any(s.reward is not None for s in steps)

        trees: list[Any] = []
        step_maps: list[Any] = []

        # Per-step metadata (one entry per *original* step, in batch order)
        meta_prompt_uids: list[str] = []
        meta_traj_uids: list[str] = []
        meta_step_indices: list[int] = []
        meta_extra_keys: set[str] = set()
        for s in steps:
            if s.metadata:
                meta_extra_keys.update(s.metadata.keys())

        global_step_idx = 0  # index into flattened steps list

        for prompt_uid, group_steps in groups.items():
            # Build full sequences for the tree
            sequences: list[list[int]] = []
            seq_prompt_lens: list[int] = []
            seq_response_lens: list[int] = []

            for s in group_steps:
                full_seq = list(s.prompt_ids) + list(s.response_ids)
                sequences.append(full_seq)
                seq_prompt_lens.append(len(s.prompt_ids))
                seq_response_lens.append(len(s.response_ids))

            tree = PrefixTree.build(sequences)

            if tree.total_tokens > self._max_tree_tokens or len(sequences) < 2:
                # Fallback: emit each step as a standard padded row
                for i, s in enumerate(group_steps):
                    row = self._pad_single_step(s, pad_token_id, total_seq_len)
                    input_ids_rows.append(row["input_ids"])
                    attention_mask_rows.append(row["attention_mask"])
                    position_ids_rows.append(row["position_ids"])
                    prompt_ids_rows.append(row["prompt_ids"])
                    response_ids_rows.append(row["response_ids"])
                    response_mask_rows.append(row["response_mask"])
                    trees.append(None)
                    step_maps.append(None)

                    if has_reward:
                        reward_rows.append(row["reward"])

                    meta_prompt_uids.append(s.prompt_uid)
                    meta_traj_uids.append(s.trajectory_uid)
                    meta_step_indices.append(s.step_index)
                    global_step_idx += 1
                continue

            # Pack the tree into a single row
            packed = tree.pack(pad_to=total_seq_len)
            packed_input_ids = packed["packed_input_ids"]       # (1, T)
            packed_pos_ids = packed["packed_position_ids"]      # (1, T)
            packed_attn_mask = packed["packed_attention_mask"]  # (T, T)

            input_ids_rows.append(packed_input_ids)
            attention_mask_rows.append(packed_attn_mask.unsqueeze(0))  # (1, T, T)
            position_ids_rows.append(packed_pos_ids)

            # For the tree-packed row, prompt/response/response_mask
            # record the *tree-level* layout.  The downstream logprob
            # restoration will use the PrefixTree to recover per-step values.
            prompt_ids_rows.append(packed_input_ids.clone())
            response_ids_rows.append(
                torch.full((1, self._response_length), pad_token_id, dtype=torch.long)
            )

            # Response mask: mark all positions that belong to any response
            resp_mask = torch.zeros(1, total_seq_len, dtype=torch.long)
            for sid, s in enumerate(group_steps):
                path_positions = tree.get_sequence_positions(sid)
                plen = len(s.prompt_ids)
                for pos_idx, packed_pos in enumerate(path_positions):
                    if pos_idx >= plen:
                        if packed_pos < total_seq_len:
                            resp_mask[0, packed_pos] = 1
            response_mask_rows.append(resp_mask)

            # Build per-step mapping for logprob restoration
            step_map_list: list[dict[str, Any]] = []
            for sid, s in enumerate(group_steps):
                path_positions = tree.get_sequence_positions(sid)
                plen = len(s.prompt_ids)
                rlen = len(s.response_ids)
                step_map_list.append({
                    "seq_id": sid,
                    "prompt_len": plen,
                    "response_len": rlen,
                    "path_positions": path_positions,
                })
            trees.append(tree)
            step_maps.append(step_map_list)

            if has_reward:
                r = torch.zeros(1, total_seq_len, dtype=torch.float32)
                for sid, s in enumerate(group_steps):
                    if s.reward is not None:
                        path_positions = tree.get_sequence_positions(sid)
                        plen = len(s.prompt_ids)
                        resp_positions = [p for j, p in enumerate(path_positions) if j >= plen]
                        if resp_positions:
                            r[0, resp_positions[-1]] = float(s.reward)
                reward_rows.append(r)

            # Metadata: tree-packed rows represent multiple steps, but the
            # DataProto batch dim is 1 row.  We record the first step's
            # metadata for the row; detailed per-step info is in step_maps.
            first = group_steps[0]
            meta_prompt_uids.append(first.prompt_uid)
            meta_traj_uids.append(first.trajectory_uid)
            meta_step_indices.append(first.step_index)
            global_step_idx += len(group_steps)

        return self._assemble_batch(
            input_ids_rows=input_ids_rows,
            attention_mask_rows=attention_mask_rows,
            position_ids_rows=position_ids_rows,
            prompt_ids_rows=prompt_ids_rows,
            response_ids_rows=response_ids_rows,
            response_mask_rows=response_mask_rows,
            reward_rows=reward_rows if has_reward else None,
            trees=trees,
            step_maps=step_maps,
            meta_prompt_uids=meta_prompt_uids,
            meta_traj_uids=meta_traj_uids,
            meta_step_indices=meta_step_indices,
            meta_extra_keys=meta_extra_keys,
            steps=steps,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_single_step(
        self,
        step: Step,
        pad_token_id: int,
        total_seq_len: int,
    ) -> dict[str, torch.Tensor]:
        """Pad a single step into the standard verl layout (fallback path)."""
        pids = step.prompt_ids if step.prompt_ids else [pad_token_id]
        rids = step.response_ids if step.response_ids else [pad_token_id]

        # Left-pad prompt
        if len(pids) >= self._prompt_length:
            padded_prompt = pids[-self._prompt_length:]
            prompt_mask = [1] * self._prompt_length
        else:
            pad_len = self._prompt_length - len(pids)
            padded_prompt = [pad_token_id] * pad_len + pids
            prompt_mask = [0] * pad_len + [1] * len(pids)

        # Right-pad response
        if len(rids) >= self._response_length:
            padded_response = rids[:self._response_length]
            response_mask = [1] * self._response_length
        else:
            pad_len = self._response_length - len(rids)
            padded_response = rids + [pad_token_id] * pad_len
            response_mask = [1] * len(rids) + [0] * pad_len

        input_ids = torch.tensor([padded_prompt + padded_response], dtype=torch.long)
        attention_mask = torch.tensor([prompt_mask + [1] * len(rids) + [0] * (self._response_length - len(rids))], dtype=torch.long)
        resp_mask_t = torch.tensor([response_mask], dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_ids": torch.tensor([padded_prompt], dtype=torch.long),
            "response_ids": torch.tensor([padded_response], dtype=torch.long),
            "response_mask": resp_mask_t,
        }

        if step.reward is not None:
            reward_t = torch.zeros(1, total_seq_len, dtype=torch.float32)
            valid_len = int(resp_mask_t.sum().item())
            if valid_len > 0:
                reward_t[0, self._prompt_length + valid_len - 1] = float(step.reward)
            result["reward"] = reward_t

        return result

    def _assemble_batch(
        self,
        *,
        input_ids_rows: list[torch.Tensor],
        attention_mask_rows: list[torch.Tensor],
        position_ids_rows: list[torch.Tensor],
        prompt_ids_rows: list[torch.Tensor],
        response_ids_rows: list[torch.Tensor],
        response_mask_rows: list[torch.Tensor],
        reward_rows: list[torch.Tensor] | None,
        trees: list[Any],
        step_maps: list[Any],
        meta_prompt_uids: list[str],
        meta_traj_uids: list[str],
        meta_step_indices: list[int],
        meta_extra_keys: set[str],
        steps: list[Step],
    ) -> DataProto:
        """Stack row-level tensors into the final DataProto."""
        batch_size = len(input_ids_rows)
        total_seq_len = self._prompt_length + self._response_length

        input_ids = torch.cat(input_ids_rows, dim=0)
        position_ids = torch.cat(position_ids_rows, dim=0)

        # Attention mask may be 1-D (standard) or 2-D (tree) per row.
        # We always store a 1-D mask in the batch tensor; the 2-D tree mask
        # is carried separately in non_tensor_batch.
        # For standard rows: (1, T) -> use directly
        # For tree rows: the 1-D summary is "1 where any sequence can attend"
        attn_mask_1d = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
        tree_attn_masks: list[Any] = []
        for i, mask in enumerate(attention_mask_rows):
            if mask.dim() == 2:
                # Standard 1-D row
                attn_mask_1d[i] = mask.squeeze(0)
                tree_attn_masks.append(None)
            elif mask.dim() == 3:
                # Tree 2-D mask (1, T, T) — flatten to 1-D (any col is attended)
                attn_mask_1d[i] = (mask.squeeze(0).any(dim=0)).long()
                tree_attn_masks.append(mask.squeeze(0))  # (T, T)
            else:
                attn_mask_1d[i] = mask.view(-1)[:total_seq_len]
                tree_attn_masks.append(None)

        prompts = torch.cat(prompt_ids_rows, dim=0)
        responses = torch.cat(response_ids_rows, dim=0)
        resp_mask = torch.cat(response_mask_rows, dim=0)

        optional: dict[str, torch.Tensor] = {}
        if reward_rows is not None:
            optional["rm_scores"] = torch.cat(reward_rows, dim=0)

        # Ensure response_mask matches response length
        if resp_mask.size(-1) != total_seq_len:
            padded_resp_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
            padded_resp_mask[:, :resp_mask.size(-1)] = resp_mask
            resp_mask = padded_resp_mask

        batch = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "response_mask": resp_mask,
                "attention_mask": attn_mask_1d,
                "input_ids": input_ids,
                "position_ids": position_ids,
                **optional,
            },
            batch_size=batch_size,
        )

        non_tensor_batch: dict[str, np.ndarray] = {
            "prompt_uids": np.array(meta_prompt_uids, dtype=object),
            "trajectory_uids": np.array(meta_traj_uids, dtype=object),
            "step_indices": np.array(meta_step_indices, dtype=np.int32),
            "prefix_trees": np.array(trees, dtype=object),
            "tree_step_maps": np.array(step_maps, dtype=object),
            "tree_attention_masks": np.array(tree_attn_masks, dtype=object),
        }

        # Expand Step.metadata fields
        for key in sorted(meta_extra_keys):
            vals: list[Any] = []
            step_cursor = 0
            for tree_obj, smap in zip(trees, step_maps):
                if tree_obj is None:
                    s = steps[step_cursor]
                    vals.append(s.metadata.get(key) if s.metadata else None)
                    step_cursor += 1
                else:
                    first_step = steps[step_cursor]
                    vals.append(first_step.metadata.get(key) if first_step.metadata else None)
                    step_cursor += len(smap)
            non_tensor_batch[key] = np.array(vals, dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
