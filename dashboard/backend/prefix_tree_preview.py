"""Lightweight prefix-tree preview for the dashboard.

This mirrors the display shape of the experimental ``prefix-tree-merge``
branch without importing its training backend, attention patching, or torch
code.  It is intentionally pure Python and only produces visualization data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreviewNode:
    node_id: int
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, PreviewNode] = field(default_factory=dict)
    parent: PreviewNode | None = field(default=None, repr=False)
    start_pos: int = -1
    end_pos: int = -1

    @property
    def is_root(self) -> bool:
        return self.parent is None and not self.tokens


class _RawNode:
    __slots__ = ("token_id", "children", "sequence_ids", "is_end")

    def __init__(self, token_id: int):
        self.token_id = token_id
        self.children: dict[int, _RawNode] = {}
        self.sequence_ids: list[int] = []
        self.is_end = False


def build_prefix_tree_preview(
    steps: list[dict[str, Any]],
    *,
    max_matrix_tokens: int = 64,
) -> dict[str, Any]:
    """Build a compressed trie preview from serialized DataPool steps."""
    sequences: list[list[int]] = []
    sequence_meta: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        prompt_ids = list(step.get("prompt_ids") or [])
        response_ids = list(step.get("response_ids") or [])
        seq = prompt_ids + response_ids
        if not seq:
            continue
        sequences.append(seq)
        sequence_meta.append(
            {
                "sequence_id": idx,
                "step_key": step.get("step_key"),
                "prompt_uid": step.get("prompt_uid"),
                "trajectory_uid": step.get("trajectory_uid"),
                "step_index": step.get("step_index"),
                "prompt_len": len(prompt_ids),
                "response_len": len(response_ids),
                "token_count": len(seq),
            }
        )

    if not sequences:
        return _empty_preview()

    root = _build_tree(sequences)
    nodes: list[PreviewNode] = []
    _assign_positions(root, nodes)
    sequence_ids = list(range(len(sequences)))
    original_tokens = sum(len(seq) for seq in sequences)
    packed_tokens = sum(len(node.tokens) for node in nodes)
    packed_sequence: list[int] = []
    for node in nodes:
        packed_sequence.extend(node.tokens)

    sequence_paths = []
    step_map = []
    for meta in sequence_meta:
        sid = meta["sequence_id"]
        path = [node for node in nodes if sid in node.sequence_ids]
        positions = _positions_for_path(path)
        sequence_paths.append(
            {
                **meta,
                "node_ids": [node.node_id for node in path],
                "node_ranges": [[node.start_pos, node.end_pos] for node in path],
                "packed_positions": positions,
            }
        )
        for source_pos, packed_pos in enumerate(positions):
            step_map.append(
                {
                    "sequence_id": sid,
                    "step_key": meta.get("step_key"),
                    "source_pos": source_pos,
                    "packed_pos": packed_pos,
                }
            )

    return {
        "mode": "visualization_only",
        "notice": "This preview does not mutate DataPool and is not used by fetch_batch().",
        "sequence_count": len(sequences),
        "original_tokens": original_tokens,
        "packed_tokens": packed_tokens,
        "saved_tokens": original_tokens - packed_tokens,
        "token_ratio": packed_tokens / original_tokens if original_tokens else 1.0,
        "packed_sequence": packed_sequence,
        "nodes": [_node_to_dict(node) for node in nodes],
        "sequence_paths": sequence_paths,
        "step_map": step_map,
        "attention_mask": _attention_mask_thumbnail(nodes, sequence_ids, max_matrix_tokens),
    }


def _empty_preview() -> dict[str, Any]:
    return {
        "mode": "visualization_only",
        "notice": "This preview does not mutate DataPool and is not used by fetch_batch().",
        "sequence_count": 0,
        "original_tokens": 0,
        "packed_tokens": 0,
        "saved_tokens": 0,
        "token_ratio": 1.0,
        "packed_sequence": [],
        "nodes": [],
        "sequence_paths": [],
        "step_map": [],
        "attention_mask": {"size": 0, "truncated": False, "matrix": []},
    }


def _build_tree(sequences: list[list[int]]) -> PreviewNode:
    raw_root = _RawNode(-1)
    for sid, seq in enumerate(sequences):
        cur = raw_root
        for token in seq:
            if token not in cur.children:
                cur.children[token] = _RawNode(token)
            child = cur.children[token]
            child.sequence_ids.append(sid)
            cur = child
        cur.is_end = True

    root = PreviewNode(node_id=-1)
    counter = [0]
    _compress(raw_root, root, counter)
    return root


def _compress(raw_node: _RawNode, parent: PreviewNode, counter: list[int]) -> None:
    for token, child in sorted(raw_node.children.items()):
        tokens = []
        cur = child
        while True:
            tokens.append(cur.token_id)
            if len(cur.children) != 1 or cur.is_end:
                break
            next_child = next(iter(cur.children.values()))
            if set(cur.sequence_ids) != set(next_child.sequence_ids):
                break
            cur = next_child

        node = PreviewNode(
            node_id=counter[0],
            tokens=tokens,
            sequence_ids=sorted(set(cur.sequence_ids)),
            parent=parent,
        )
        counter[0] += 1
        parent.children[token] = node
        _compress(cur, node, counter)


def _assign_positions(root: PreviewNode, nodes: list[PreviewNode]) -> None:
    ordered: list[PreviewNode] = []
    _preorder(root, ordered)
    cursor = 0
    for node in ordered:
        node.start_pos = cursor
        node.end_pos = cursor + len(node.tokens) - 1
        cursor += len(node.tokens)
    nodes.extend(ordered)


def _preorder(node: PreviewNode, out: list[PreviewNode]) -> None:
    for _token, child in sorted(node.children.items()):
        out.append(child)
        _preorder(child, out)


def _positions_for_path(path: list[PreviewNode]) -> list[int]:
    positions: list[int] = []
    for node in path:
        positions.extend(range(node.start_pos, node.end_pos + 1))
    return positions


def _node_to_dict(node: PreviewNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "tokens": node.tokens,
        "sequence_ids": node.sequence_ids,
        "start_pos": node.start_pos,
        "end_pos": node.end_pos,
        "num_tokens": len(node.tokens),
        "parent_id": node.parent.node_id if node.parent and not node.parent.is_root else None,
        "child_ids": [child.node_id for child in node.children.values()],
        "is_leaf": not node.children,
    }


def _attention_mask_thumbnail(
    nodes: list[PreviewNode],
    sequence_ids: list[int],
    max_matrix_tokens: int,
) -> dict[str, Any]:
    total_tokens = sum(len(node.tokens) for node in nodes)
    size = min(total_tokens, max_matrix_tokens)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for sid in sequence_ids:
        positions = _positions_for_path([node for node in nodes if sid in node.sequence_ids])
        positions = [pos for pos in positions if pos < size]
        for qi, qpos in enumerate(positions):
            for kpos in positions[: qi + 1]:
                matrix[qpos][kpos] = 1
    return {
        "size": size,
        "truncated": total_tokens > size,
        "matrix": matrix,
    }
