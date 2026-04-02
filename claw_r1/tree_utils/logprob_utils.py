"""Logprob and entropy restoration from tree-packed model outputs.

After a forward pass on a tree-packed sequence, the logits correspond to the
packed (deduplicated) layout.  This module recovers per-sequence log
probabilities and (optionally) entropy by walking each sequence's path through
the prefix tree and gathering the appropriate logit slices.

Node-level caching is used so that computations for shared prefix nodes are
performed only once and reused across all sequences that pass through them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from claw_r1.tree_utils.prefix_tree import PrefixTree


# ============================================================================
# Public API
# ============================================================================


def gather_logprobs_from_packed(
    logits: torch.Tensor,
    tree: PrefixTree,
    packed_input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> dict[int, torch.Tensor]:
    """Restore per-sequence log probabilities from tree-packed logits.

    For each original sequence *s*, its tokens live at various positions in the
    packed sequence (shared prefix positions are deduplicated).  This function
    walks the node path of each sequence, computes log P(next_token | context)
    at every position, and concatenates the results.

    Args:
        logits: ``(T, V)`` model output logits for the packed sequence.
        tree: The ``PrefixTree`` that produced the packing.
        packed_input_ids: ``(T,)`` or ``(1, T)`` packed token IDs.
        temperature: Softmax temperature (default 1.0).
        chunk_size: Process logits in chunks of this size to limit peak memory.

    Returns:
        ``dict[seq_id, tensor]`` where each tensor has shape ``(L_s - 1,)``
        containing log P(token_{i+1} | token_{0..i}) for every position *i*
        in original sequence *s* (length ``L_s``).
    """
    return _gather_impl(
        logits, tree, packed_input_ids, temperature, chunk_size,
        compute_entropy=False,
    )[0]


def gather_logprobs_entropy_from_packed(
    logits: torch.Tensor,
    tree: PrefixTree,
    packed_input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Like :func:`gather_logprobs_from_packed` but also computes entropy.

    Returns:
        ``(logprobs_dict, entropy_dict)`` — both map ``seq_id`` to a tensor
        of shape ``(L_s - 1,)``.
    """
    return _gather_impl(
        logits, tree, packed_input_ids, temperature, chunk_size,
        compute_entropy=True,
    )


# ============================================================================
# Internal implementation
# ============================================================================


def _gather_impl(
    logits: torch.Tensor,
    tree: PrefixTree,
    packed_input_ids: torch.Tensor,
    temperature: float,
    chunk_size: int,
    compute_entropy: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    device = logits.device
    dtype = torch.float32

    if packed_input_ids.dim() == 2:
        packed_input_ids = packed_input_ids.squeeze(0)

    # Caches keyed by (start_pos, end_pos) of the node
    node_lp_cache: dict[tuple[int, int], torch.Tensor] = {}
    node_ent_cache: dict[tuple[int, int], torch.Tensor] = {}
    # Transition cache keyed by (pred_pos, label_pos)
    trans_lp_cache: dict[tuple[int, int], torch.Tensor] = {}
    trans_ent_cache: dict[tuple[int, int], torch.Tensor] = {}

    logprobs_out: dict[int, torch.Tensor] = {}
    entropy_out: dict[int, torch.Tensor] = {}

    for sid in tree.sequence_ids:
        path = tree.get_sequence_path(sid)
        if not path:
            logprobs_out[sid] = torch.empty(0, device=device, dtype=dtype)
            entropy_out[sid] = torch.empty(0, device=device, dtype=dtype)
            continue

        lp_parts: list[torch.Tensor] = []
        ent_parts: list[torch.Tensor] = []

        for node_idx, node in enumerate(path):
            s, e = node.start_pos, node.end_pos
            nkey = (s, e)

            # --- internal logprobs within the node ---
            if nkey not in node_lp_cache:
                lp, ent = _compute_internal(
                    logits, packed_input_ids, s, e,
                    temperature, chunk_size, compute_entropy,
                )
                node_lp_cache[nkey] = lp
                if compute_entropy:
                    node_ent_cache[nkey] = ent

            if node_lp_cache[nkey].numel() > 0:
                lp_parts.append(node_lp_cache[nkey])
                if compute_entropy:
                    ent_parts.append(node_ent_cache[nkey])

            # --- transition logprob to the next node ---
            if node_idx + 1 < len(path):
                next_node = path[node_idx + 1]
                pred_pos = e
                label_pos = next_node.start_pos
                tkey = (pred_pos, label_pos)

                if tkey not in trans_lp_cache:
                    tlp, tent = _compute_transition(
                        logits, packed_input_ids, pred_pos, label_pos,
                        temperature, compute_entropy,
                    )
                    trans_lp_cache[tkey] = tlp
                    if compute_entropy:
                        trans_ent_cache[tkey] = tent

                lp_parts.append(trans_lp_cache[tkey].unsqueeze(0))
                if compute_entropy:
                    ent_parts.append(trans_ent_cache[tkey].unsqueeze(0))

        if lp_parts:
            logprobs_out[sid] = torch.cat(lp_parts, dim=0)
        else:
            logprobs_out[sid] = torch.empty(0, device=device, dtype=dtype)

        if compute_entropy:
            if ent_parts:
                entropy_out[sid] = torch.cat(ent_parts, dim=0)
            else:
                entropy_out[sid] = torch.empty(0, device=device, dtype=dtype)

    return logprobs_out, entropy_out


# ============================================================================
# Low-level logprob / entropy helpers (pure PyTorch, no external deps)
# ============================================================================


def _compute_internal(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start: int,
    end: int,
    temperature: float,
    chunk_size: int,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logprobs (and optionally entropy) for tokens *within* a node.

    For a node spanning packed positions [start, end], the internal
    predictions are: logits[start] predicts input_ids[start+1], ...
    logits[end-1] predicts input_ids[end].  That gives (end - start) values.
    """
    n = end - start
    if n <= 0:
        empty = torch.empty(0, device=logits.device, dtype=torch.float32)
        return empty, empty

    pred_logits = logits[start:end]         # (n, V)
    labels = input_ids[start + 1: end + 1]  # (n,)

    return _logprobs_from_logits(pred_logits, labels, temperature, chunk_size, compute_entropy)


def _compute_transition(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pred_pos: int,
    label_pos: int,
    temperature: float,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logprob (and optionally entropy) for the transition between two nodes."""
    pred_logit = logits[pred_pos: pred_pos + 1]  # (1, V)
    label = input_ids[label_pos: label_pos + 1]  # (1,)

    lp, ent = _logprobs_from_logits(pred_logit, label, temperature, 1, compute_entropy)
    return lp.squeeze(0), ent.squeeze(0) if compute_entropy else ent


def _logprobs_from_logits(
    pred_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    chunk_size: int,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core helper: compute gathered logprobs and optional entropy.

    Args:
        pred_logits: ``(N, V)`` prediction logits.
        labels: ``(N,)`` target token IDs.
        temperature: Softmax temperature.
        chunk_size: Chunk size for memory-efficient processing.
        compute_entropy: Whether to compute entropy.

    Returns:
        ``(logprobs, entropy)`` each of shape ``(N,)``.
        If *compute_entropy* is False, *entropy* is an empty tensor.
    """
    N = pred_logits.size(0)
    device = pred_logits.device

    all_lp: list[torch.Tensor] = []
    all_ent: list[torch.Tensor] = []

    for i in range(0, N, chunk_size):
        chunk_logits = pred_logits[i: i + chunk_size]  # (C, V)
        chunk_labels = labels[i: i + chunk_size]        # (C,)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        log_probs = F.log_softmax(chunk_logits.float(), dim=-1)
        gathered = log_probs.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)
        all_lp.append(gathered)

        if compute_entropy:
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1)
            all_ent.append(ent)

    logprobs = torch.cat(all_lp, dim=0)
    entropy = torch.cat(all_ent, dim=0) if compute_entropy else torch.empty(0, device=device, dtype=torch.float32)
    return logprobs, entropy
