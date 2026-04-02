"""Prefix tree merging utilities for efficient RL training.

Provides data structures and algorithms for building prefix trees from
Step sequences, packing them into tree-structured inputs, and restoring
per-sequence log probabilities from packed model outputs.
"""

from claw_r1.tree_utils.prefix_tree import PrefixNode, PrefixTree

__all__ = ["PrefixNode", "PrefixTree"]
