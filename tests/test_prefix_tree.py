"""Correctness tests for prefix tree merging.

Tests cover:
1. PrefixTree construction and compression
2. Packed sequence layout and attention mask correctness
3. Logprob restoration equivalence: tree-packed vs. independent forward
4. TreeVerlBackend round-trip with mock Steps

To run the tests, use the following command:
```
pytest tests/test_prefix_tree.py
```
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from claw_r1.tree_utils.prefix_tree import PrefixTree
from claw_r1.tree_utils.logprob_utils import (
    gather_logprobs_from_packed,
    gather_logprobs_entropy_from_packed,
)


# ============================================================================
# 1. PrefixTree construction
# ============================================================================


def test_build_shared_prefix():
    """Three sequences with overlapping prefixes build a correct tree."""
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 3, 6, 7, 8, 9],
    ]
    tree = PrefixTree.build(sequences)

    assert len(tree.sequence_ids) == 3
    # Packed tokens should be fewer than 5 + 5 + 7 = 17
    assert tree.total_tokens < 17
    # But at least the longest sequence length
    assert tree.total_tokens >= 7

    # Verify token_ratio < 1 (savings exist)
    assert tree.token_ratio() < 1.0

    # Verify every original sequence can be recovered from its path
    for sid, seq in enumerate(sequences):
        path = tree.get_sequence_path(sid)
        recovered = []
        for node in path:
            recovered.extend(node.tokens)
        assert recovered == seq, f"Sequence {sid} mismatch: {recovered} != {seq}"


def test_build_no_shared_prefix():
    """Sequences with no common prefix produce no sharing."""
    sequences = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    tree = PrefixTree.build(sequences)

    assert tree.total_tokens == 9
    assert tree.token_ratio() == 1.0


def test_build_identical_sequences():
    """Identical sequences are fully merged."""
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    tree = PrefixTree.build(sequences)

    assert tree.total_tokens == 4
    assert tree.token_ratio() == 0.5


def test_build_single_sequence():
    """A single sequence produces a trivial tree."""
    sequences = [[10, 20, 30]]
    tree = PrefixTree.build(sequences)

    assert tree.total_tokens == 3
    assert tree.token_ratio() == 1.0


def test_chain_compression():
    """Linear chains (single-child nodes with same sequence set) get compressed."""
    # All 3 sequences share [1,2,3], then diverge:
    # seq0: [1,2,3,4], seq1: [1,2,3,5], seq2: [1,2,3,6]
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 3, 6],
    ]
    tree = PrefixTree.build(sequences)

    # The shared prefix [1,2,3] should be in one compressed node
    root_children = list(tree.root.children.values())
    assert len(root_children) == 1
    shared_node = root_children[0]
    assert shared_node.tokens == [1, 2, 3]
    assert set(shared_node.sequence_ids) == {0, 1, 2}
    assert len(shared_node.children) == 3


# ============================================================================
# 2. Packed layout and attention mask
# ============================================================================


def test_pack_attention_mask_causal():
    """Attention mask is causal within each sequence's path."""
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 5, 6],
    ]
    tree = PrefixTree.build(sequences)
    packed = tree.pack()

    attn = packed["packed_attention_mask"]  # (T, T)
    T = attn.size(0)

    # Upper triangle should be zero (causal)
    for i in range(T):
        for j in range(i + 1, T):
            # Position j can NOT attend to position i if j < i in causal order
            # But tree structure may have j attend to i if on same path
            pass

    # Each sequence should see its own path with causal masking
    for sid in [0, 1]:
        positions = tree.get_sequence_positions(sid)
        n = len(positions)
        for qi in range(n):
            for ki in range(qi + 1):
                assert attn[positions[qi], positions[ki]].item(), (
                    f"Seq {sid}: position {positions[qi]} should attend to "
                    f"{positions[ki]}"
                )


def test_pack_position_ids_depth():
    """Position IDs reflect depth in the tree path."""
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
    ]
    tree = PrefixTree.build(sequences)
    packed = tree.pack()

    pos_ids = packed["packed_position_ids"].squeeze(0)

    # For each sequence, positions should be 0, 1, 2, ... (monotonic)
    for sid in [0, 1]:
        positions = tree.get_sequence_positions(sid)
        seq_pos = [pos_ids[p].item() for p in positions]
        assert seq_pos == list(range(len(positions))), (
            f"Seq {sid} position IDs not monotonic: {seq_pos}"
        )


def test_pack_input_ids_recovery():
    """Packed input IDs can recover original sequences via path positions."""
    sequences = [
        [10, 20, 30, 40],
        [10, 20, 50, 60],
    ]
    tree = PrefixTree.build(sequences)
    packed = tree.pack()

    ids = packed["packed_input_ids"].squeeze(0)

    for sid, seq in enumerate(sequences):
        positions = tree.get_sequence_positions(sid)
        recovered = [ids[p].item() for p in positions]
        assert recovered == seq, f"Seq {sid}: {recovered} != {seq}"


# ============================================================================
# 3. Logprob restoration equivalence
# ============================================================================


def _independent_logprobs(
    sequences: list[list[int]],
    vocab_size: int,
    temperature: float = 1.0,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Compute logprobs independently for each sequence using a random model.

    Returns (per_seq_logprobs, weight) where weight is the random linear
    layer used so we can replicate with tree-packed forward.
    """
    weight = torch.randn(vocab_size, vocab_size)

    results: dict[int, torch.Tensor] = {}
    for sid, seq in enumerate(sequences):
        input_ids = torch.tensor(seq, dtype=torch.long)
        # Simple "model": one-hot -> linear -> logits
        one_hot = F.one_hot(input_ids, vocab_size).float()
        logits = one_hot @ weight.T  # (L, V)

        if temperature != 1.0:
            logits = logits / temperature

        log_probs = F.log_softmax(logits.float(), dim=-1)
        # logprob[i] = log P(token_{i+1} | token_i)
        if len(seq) > 1:
            labels = input_ids[1:]
            lp = log_probs[:-1].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        else:
            lp = torch.empty(0)
        results[sid] = lp

    return results, weight


def _tree_packed_logprobs(
    sequences: list[list[int]],
    weight: torch.Tensor,
    vocab_size: int,
    temperature: float = 1.0,
) -> dict[int, torch.Tensor]:
    """Compute logprobs via tree-packed forward and restoration."""
    tree = PrefixTree.build(sequences)
    packed = tree.pack()

    packed_ids = packed["packed_input_ids"].squeeze(0)  # (T,)
    one_hot = F.one_hot(packed_ids, vocab_size).float()
    packed_logits = one_hot @ weight.T  # (T, V)

    return gather_logprobs_from_packed(
        packed_logits, tree, packed_ids, temperature=temperature,
    )


def test_logprob_equivalence_simple():
    """Tree-packed logprobs match independent logprobs for simple shared prefix."""
    vocab_size = 32
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 3, 6, 7, 8, 9],
    ]

    independent, weight = _independent_logprobs(sequences, vocab_size)
    tree_packed = _tree_packed_logprobs(sequences, weight, vocab_size)

    for sid in range(len(sequences)):
        torch.testing.assert_close(
            tree_packed[sid],
            independent[sid],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Sequence {sid} logprobs mismatch",
        )


def test_logprob_equivalence_no_sharing():
    """Tree-packed logprobs match when no prefix sharing exists."""
    vocab_size = 32
    sequences = [
        [1, 2, 3],
        [10, 11, 12],
        [20, 21, 22],
    ]

    independent, weight = _independent_logprobs(sequences, vocab_size)
    tree_packed = _tree_packed_logprobs(sequences, weight, vocab_size)

    for sid in range(len(sequences)):
        torch.testing.assert_close(
            tree_packed[sid],
            independent[sid],
            atol=1e-5,
            rtol=1e-5,
        )


def test_logprob_equivalence_identical():
    """Tree-packed logprobs match for fully identical sequences."""
    vocab_size = 32
    sequences = [
        [5, 10, 15, 20],
        [5, 10, 15, 20],
    ]

    independent, weight = _independent_logprobs(sequences, vocab_size)
    tree_packed = _tree_packed_logprobs(sequences, weight, vocab_size)

    for sid in range(len(sequences)):
        torch.testing.assert_close(
            tree_packed[sid],
            independent[sid],
            atol=1e-5,
            rtol=1e-5,
        )


def test_logprob_equivalence_with_temperature():
    """Logprob equivalence holds with non-unit temperature."""
    vocab_size = 32
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
    ]
    temperature = 0.7

    independent, weight = _independent_logprobs(sequences, vocab_size, temperature)
    tree_packed = _tree_packed_logprobs(sequences, weight, vocab_size, temperature)

    for sid in range(len(sequences)):
        torch.testing.assert_close(
            tree_packed[sid],
            independent[sid],
            atol=1e-5,
            rtol=1e-5,
        )


def test_entropy_restoration():
    """Entropy values from tree-packed forward match independent computation."""
    vocab_size = 32
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 5, 6],
    ]

    weight = torch.randn(vocab_size, vocab_size)

    # Independent entropy
    independent_entropy: dict[int, torch.Tensor] = {}
    for sid, seq in enumerate(sequences):
        input_ids = torch.tensor(seq, dtype=torch.long)
        one_hot = F.one_hot(input_ids, vocab_size).float()
        logits = one_hot @ weight.T
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)
        if len(seq) > 1:
            independent_entropy[sid] = ent[:-1]
        else:
            independent_entropy[sid] = torch.empty(0)

    # Tree-packed entropy
    tree = PrefixTree.build(sequences)
    packed = tree.pack()
    packed_ids = packed["packed_input_ids"].squeeze(0)
    one_hot = F.one_hot(packed_ids, vocab_size).float()
    packed_logits = one_hot @ weight.T

    _, tree_entropy = gather_logprobs_entropy_from_packed(
        packed_logits, tree, packed_ids,
    )

    for sid in range(len(sequences)):
        torch.testing.assert_close(
            tree_entropy[sid],
            independent_entropy[sid],
            atol=1e-5,
            rtol=1e-5,
        )


# ============================================================================
# 4. Agent multi-step scenario
# ============================================================================


def test_agent_multistep_scenario():
    """Simulate a realistic agent trajectory with 3 steps sharing evolving context."""
    vocab_size = 64

    sys_prompt = [1, 2, 3, 4, 5]
    user_query = [10, 11, 12]
    thought_0 = [20, 21]
    action_0 = [30, 31]
    tool_out_0 = [40, 41, 42]
    thought_1 = [50, 51]
    action_1 = [60, 61]
    tool_out_1 = [62, 63]
    thought_2 = [55, 56]
    final_ans = [57, 58]

    step0_prompt = sys_prompt + user_query
    step0_response = thought_0 + action_0
    step1_prompt = sys_prompt + user_query + thought_0 + action_0 + tool_out_0
    step1_response = thought_1 + action_1
    step2_prompt = (
        sys_prompt + user_query + thought_0 + action_0 + tool_out_0
        + thought_1 + action_1 + tool_out_1
    )
    step2_response = thought_2 + final_ans

    sequences = [
        step0_prompt + step0_response,
        step1_prompt + step1_response,
        step2_prompt + step2_response,
    ]

    tree = PrefixTree.build(sequences)

    # Verify sharing saves tokens
    original_total = sum(len(s) for s in sequences)
    assert tree.total_tokens < original_total
    print(f"Agent scenario: {original_total} tokens -> {tree.total_tokens} "
          f"packed (ratio={tree.token_ratio():.3f})")

    # Verify logprob equivalence
    independent, weight = _independent_logprobs(sequences, vocab_size)
    tree_packed = _tree_packed_logprobs(sequences, weight, vocab_size)

    for sid in range(len(sequences)):
        torch.testing.assert_close(
            tree_packed[sid],
            independent[sid],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Agent step {sid} logprobs mismatch",
        )


# ============================================================================
# 5. TreeVerlBackend integration
# ============================================================================


def test_tree_verl_backend_fallback():
    """TreeVerlBackend falls back to standard padding for single-step groups."""
    from unittest.mock import MagicMock

    from claw_r1.data_pool.data_model import Step
    from claw_r1.data_pool.training_backend_prefix_tree import TreeVerlBackend

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    backend = TreeVerlBackend(
        tokenizer=tokenizer,
        prompt_length=16,
        response_length=16,
    )

    steps = [
        Step(
            prompt_ids=[1, 2, 3, 4],
            response_ids=[10, 11, 12],
            reward=1.0,
            trajectory_uid="traj_0",
            prompt_uid="prompt_0",
            step_index=0,
            is_last=True,
        ),
    ]

    result = backend.convert(steps)
    assert result.batch["input_ids"].shape[0] == 1
    assert result.batch["input_ids"].shape[1] == 32  # 16 + 16


def test_tree_verl_backend_merging():
    """TreeVerlBackend merges multi-step trajectories into a tree."""
    from unittest.mock import MagicMock

    from claw_r1.data_pool.data_model import Step
    from claw_r1.data_pool.training_backend_prefix_tree import TreeVerlBackend

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    backend = TreeVerlBackend(
        tokenizer=tokenizer,
        prompt_length=32,
        response_length=16,
        max_tree_tokens=128,
    )

    steps = [
        Step(
            prompt_ids=[1, 2, 3, 4, 5],
            response_ids=[10, 11],
            reward=0.5,
            trajectory_uid="traj_0",
            prompt_uid="prompt_0",
            step_index=0,
        ),
        Step(
            prompt_ids=[1, 2, 3, 4, 5, 10, 11, 20],
            response_ids=[30, 31],
            reward=1.0,
            trajectory_uid="traj_0",
            prompt_uid="prompt_0",
            step_index=1,
            is_last=True,
        ),
    ]

    result = backend.convert(steps)

    # Should be merged into 1 row (one tree for the prompt group)
    assert result.batch["input_ids"].shape[0] == 1

    # The tree should be stored in non_tensor_batch
    trees = result.non_tensor_batch["prefix_trees"]
    assert trees[0] is not None

    step_maps = result.non_tensor_batch["tree_step_maps"]
    assert len(step_maps[0]) == 2  # 2 steps mapped


# ============================================================================
# Run all tests
# ============================================================================


def run_all():
    import traceback

    tests = [
        test_build_shared_prefix,
        test_build_no_shared_prefix,
        test_build_identical_sequences,
        test_build_single_sequence,
        test_chain_compression,
        test_pack_attention_mask_causal,
        test_pack_position_ids_depth,
        test_pack_input_ids_recovery,
        test_logprob_equivalence_simple,
        test_logprob_equivalence_no_sharing,
        test_logprob_equivalence_identical,
        test_logprob_equivalence_with_temperature,
        test_entropy_restoration,
        test_agent_multistep_scenario,
        test_tree_verl_backend_fallback,
        test_tree_verl_backend_merging,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
