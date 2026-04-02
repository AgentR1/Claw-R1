"""Prefix tree (trie) data structures and algorithms for sequence packing.

Builds a compressed prefix tree from multiple token sequences that may share
common prefixes.  The tree is then flattened into a packed representation
suitable for model forward passes with tree-structured causal attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class PrefixNode:
    """A node in the compressed prefix tree.

    Each node stores a contiguous run of tokens shared by exactly the same
    set of sequences.  When sequences diverge, child nodes are created.

    Attributes:
        node_id: Globally unique identifier within the tree.
        tokens: Token IDs stored in this node.
        sequence_ids: IDs of sequences that pass through this node.
        children: Child nodes keyed by the first diverging token.
        parent: Parent node (``None`` for root).
        start_pos: Start position in the packed (flattened) sequence.
        end_pos: End position (inclusive) in the packed sequence.
    """

    node_id: int
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, PrefixNode] = field(default_factory=dict)
    parent: PrefixNode | None = field(default=None, repr=False)
    start_pos: int = -1
    end_pos: int = -1

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def is_root(self) -> bool:
        return self.parent is None and not self.tokens

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ancestors(self) -> list[PrefixNode]:
        """Return list of ancestor nodes from root (exclusive) to parent."""
        result: list[PrefixNode] = []
        node = self.parent
        while node is not None and not node.is_root:
            result.append(node)
            node = node.parent
        result.reverse()
        return result


class PrefixTree:
    """Compressed prefix tree built from a set of token sequences.

    The tree discovers shared prefixes by token-level comparison, making it
    agnostic to metadata — it works correctly regardless of context management
    strategies (truncation, compression, etc.).

    Example usage::

        sequences = [
            [1, 2, 3, 4, 5],       # seq 0
            [1, 2, 3, 6, 7],       # seq 1 — shares [1,2,3] with seq 0
            [1, 2, 3, 6, 7, 8, 9], # seq 2 — shares [1,2,3,6,7] with seq 1
        ]
        tree = PrefixTree.build(sequences)
        packed = tree.pack()
    """

    def __init__(
        self,
        root: PrefixNode,
        nodes: list[PrefixNode],
        sequence_ids: list[int],
        total_tokens: int,
    ):
        self.root = root
        self.nodes = nodes
        self.sequence_ids = sequence_ids
        self.total_tokens = total_tokens

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def build(
        sequences: list[list[int]],
        sequence_ids: list[int] | None = None,
    ) -> PrefixTree:
        """Build a compressed prefix tree from token sequences.

        Args:
            sequences: List of token-ID lists.
            sequence_ids: Optional explicit IDs for each sequence.
                Defaults to ``range(len(sequences))``.

        Returns:
            A fully built and position-assigned ``PrefixTree``.
        """
        if sequence_ids is None:
            sequence_ids = list(range(len(sequences)))
        if len(sequences) != len(sequence_ids):
            raise ValueError("sequences and sequence_ids must have the same length")

        # Phase 1: insert sequences into an uncompressed per-token trie
        raw_root = _RawNode(token_id=-1)
        all_raw: list[_RawNode] = []
        for seq, sid in zip(sequences, sequence_ids):
            _insert_raw(raw_root, all_raw, seq, sid)

        # Phase 2: compress linear chains
        root = PrefixNode(node_id=-1)
        compressed_nodes: list[PrefixNode] = []
        _compress(raw_root, root, compressed_nodes, counter=[0])

        # Phase 3: assign packed positions via pre-order traversal
        total = _assign_positions(root, compressed_nodes)

        return PrefixTree(
            root=root,
            nodes=compressed_nodes,
            sequence_ids=sorted(set(sequence_ids)),
            total_tokens=total,
        )

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    def pack(self, pad_to: int | None = None) -> dict[str, Any]:
        """Flatten the tree into packed tensors for model forward.

        Args:
            pad_to: If given, pad the packed sequence to this length.
                Must be >= ``self.total_tokens``.

        Returns:
            Dictionary with:
            - ``packed_input_ids``:  ``(1, T)`` int64
            - ``packed_position_ids``: ``(1, T)`` int64
            - ``packed_attention_mask``: ``(T, T)`` bool
            - ``prefix_tree``: reference to ``self``
        """
        T = pad_to if pad_to is not None else self.total_tokens
        if T < self.total_tokens:
            raise ValueError(
                f"pad_to ({T}) < total_tokens ({self.total_tokens})"
            )

        packed_ids = torch.zeros(T, dtype=torch.long)
        position_ids = torch.zeros(T, dtype=torch.long)
        attn_mask = torch.zeros(T, T, dtype=torch.bool)

        # Fill tokens and build per-sequence paths for attention mask
        for node in self.nodes:
            s, e = node.start_pos, node.end_pos
            packed_ids[s: e + 1] = torch.tensor(node.tokens, dtype=torch.long)

        # Position IDs: depth in the path (count of ancestor tokens + offset)
        for node in self.nodes:
            ancestor_len = sum(a.num_tokens for a in node.ancestors())
            for offset in range(node.num_tokens):
                position_ids[node.start_pos + offset] = ancestor_len + offset

        # Attention mask: for each sequence, all tokens on its path are
        # mutually visible under causal (lower-triangular) constraint.
        for sid in self.sequence_ids:
            path_positions = self.get_sequence_positions(sid)
            n = len(path_positions)
            for qi in range(n):
                for ki in range(qi + 1):
                    attn_mask[path_positions[qi], path_positions[ki]] = True

        return {
            "packed_input_ids": packed_ids.unsqueeze(0),
            "packed_position_ids": position_ids.unsqueeze(0),
            "packed_attention_mask": attn_mask,
            "prefix_tree": self,
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_sequence_path(self, seq_id: int) -> list[PrefixNode]:
        """Return the ordered list of nodes that *seq_id* passes through."""
        return [n for n in self.nodes if seq_id in n.sequence_ids]

    def get_sequence_positions(self, seq_id: int) -> list[int]:
        """Return all packed-sequence positions belonging to *seq_id*."""
        positions: list[int] = []
        for node in self.get_sequence_path(seq_id):
            positions.extend(range(node.start_pos, node.end_pos + 1))
        return positions

    def get_sequence_node_ranges(self, seq_id: int) -> list[tuple[int, int]]:
        """Return ``(start_pos, end_pos)`` for each node on *seq_id*'s path."""
        return [
            (n.start_pos, n.end_pos)
            for n in self.get_sequence_path(seq_id)
        ]

    def token_ratio(self) -> float:
        """Ratio of packed tokens to sum of original sequence lengths.

        Values < 1.0 indicate prefix sharing savings.
        """
        original = sum(
            sum(n.num_tokens for n in self.get_sequence_path(sid))
            for sid in self.sequence_ids
        )
        if original == 0:
            return 1.0
        return self.total_tokens / original

    def pretty_print(self, max_tokens: int = 8) -> str:
        """Return a human-readable tree representation for debugging."""
        lines: list[str] = []
        _pretty(self.root, lines, indent=0, max_tokens=max_tokens)
        return "\n".join(lines)


# ============================================================================
# Internal helpers — raw (uncompressed) trie
# ============================================================================


class _RawNode:
    """Lightweight per-token node used only during trie construction."""

    __slots__ = ("token_id", "children", "sequence_ids", "is_end")

    def __init__(self, token_id: int):
        self.token_id = token_id
        self.children: dict[int, _RawNode] = {}
        self.sequence_ids: list[int] = []
        self.is_end: bool = False


def _insert_raw(
    root: _RawNode,
    all_nodes: list[_RawNode],
    sequence: list[int],
    seq_id: int,
) -> None:
    """Insert a token sequence into the raw trie."""
    cur = root
    for token in sequence:
        if token not in cur.children:
            node = _RawNode(token)
            cur.children[token] = node
            all_nodes.append(node)
        child = cur.children[token]
        child.sequence_ids.append(seq_id)
        cur = child
    cur.is_end = True


# ============================================================================
# Compression — merge single-child chains
# ============================================================================


def _compress(
    raw_node: _RawNode,
    parent: PrefixNode,
    out_nodes: list[PrefixNode],
    counter: list[int],
) -> None:
    """Recursively compress linear chains in the raw trie into PrefixNodes."""
    for _tok, child in sorted(raw_node.children.items()):
        tokens: list[int] = []
        cur = child

        # Follow single-child chains
        while True:
            tokens.append(cur.token_id)
            if len(cur.children) != 1 or cur.is_end:
                break
            next_child = next(iter(cur.children.values()))
            if set(cur.sequence_ids) != set(next_child.sequence_ids):
                break
            cur = next_child

        node = PrefixNode(
            node_id=counter[0],
            tokens=tokens,
            sequence_ids=sorted(set(cur.sequence_ids)),
            parent=parent,
        )
        counter[0] += 1
        out_nodes.append(node)
        parent.children[tokens[0]] = node

        # Recurse into remaining children
        _compress(cur, node, out_nodes, counter)


# ============================================================================
# Position assignment — pre-order DFS
# ============================================================================


def _assign_positions(
    root: PrefixNode,
    nodes: list[PrefixNode],
) -> int:
    """Assign ``start_pos`` / ``end_pos`` to every node via pre-order DFS.

    Returns the total number of tokens in the packed sequence.
    """
    cursor = 0
    # nodes list is already in insertion order which follows the DFS of
    # _compress, but we re-traverse to guarantee pre-order.
    ordered: list[PrefixNode] = []
    _preorder(root, ordered)

    for node in ordered:
        node.start_pos = cursor
        node.end_pos = cursor + node.num_tokens - 1
        cursor += node.num_tokens

    # Replace the caller's list contents with the correct pre-order
    nodes.clear()
    nodes.extend(ordered)

    return cursor


def _preorder(node: PrefixNode, out: list[PrefixNode]) -> None:
    """Collect non-root nodes in pre-order."""
    for _tok, child in sorted(node.children.items()):
        out.append(child)
        _preorder(child, out)


# ============================================================================
# Debug helper
# ============================================================================


def _pretty(
    node: PrefixNode,
    lines: list[str],
    indent: int,
    max_tokens: int,
) -> None:
    prefix = "  " * indent
    if node.is_root:
        lines.append(f"{prefix}(root)")
    else:
        tok_str = str(node.tokens[:max_tokens])
        if len(node.tokens) > max_tokens:
            tok_str = tok_str[:-1] + ", ...]"
        lines.append(
            f"{prefix}Node {node.node_id}: {tok_str}  "
            f"seqs={node.sequence_ids}  "
            f"pos=[{node.start_pos},{node.end_pos}]"
        )
    for _tok, child in sorted(node.children.items()):
        _pretty(child, lines, indent + 1, max_tokens)
