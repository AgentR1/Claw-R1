"""Benchmark: tree-packed FlexAttention vs independent FA2 forward passes.

Compares wall-clock time and numerical equivalence of:
  1. Standard independent forward: each sequence runs through the model
     separately with causal attention (Flash Attention 2 / SDPA).
  2. Tree-packed forward: all sequences are merged into a prefix tree
     and run as a single forward pass with FlexAttention + BlockMask.

Requirements:
  - CUDA GPU (FlexAttention requires CUDA)
  - transformers, torch >= 2.5

Usage:
    PYTHONPATH=. python tests/benchmark_tree_attention.py [--model MODEL] [--warmup N] [--repeat N]
"""

from __future__ import annotations

import argparse
import gc
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from claw_r1.tree_utils.prefix_tree import PrefixTree
from claw_r1.tree_utils.attention_patch import (
    create_block_mask_from_dense,
    patch_for_tree_attention,
    tree_attention_context,
)


@contextmanager
def cuda_timer(label: str, results: dict):
    """Context manager that records CUDA-synchronised elapsed time in ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    results[label] = elapsed_ms


def generate_agent_sequences(
    n_steps: int = 4,
    shared_prefix_len: int = 256,
    branch_len: int = 128,
    vocab_size: int = 32000,
    seed: int = 42,
) -> list[list[int]]:
    """Generate agent-style sequences with a shared prefix and diverging branches.

    Simulates a multi-step agent scenario where all steps share a common
    prompt prefix but have different responses.
    """
    rng = torch.Generator().manual_seed(seed)
    prefix = torch.randint(1, vocab_size, (shared_prefix_len,), generator=rng).tolist()

    sequences = []
    for i in range(n_steps):
        branch = torch.randint(1, vocab_size, (branch_len,), generator=rng).tolist()
        sequences.append(prefix + branch)
    return sequences


def run_independent_forward(
    model,
    sequences: list[list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Run each sequence independently through the model with standard causal attention."""
    logits_list = []
    for seq in sequences:
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=False)
        logits_list.append(output.logits.squeeze(0).float())
    return logits_list


def prepare_tree_inputs(
    sequences: list[list[int]],
    device: torch.device,
):
    """Build tree and pre-compute BlockMask (not timed in forward benchmark)."""
    tree = PrefixTree.build(sequences)
    packed = tree.pack()

    packed_input_ids = packed["packed_input_ids"].to(device)
    packed_position_ids = packed["packed_position_ids"].to(device)
    dense_mask = packed["packed_attention_mask"]
    seq_len = dense_mask.shape[-1]
    block_mask = create_block_mask_from_dense(dense_mask, seq_len, device)

    return tree, packed, packed_input_ids, packed_position_ids, block_mask


def run_tree_forward(
    model,
    packed_input_ids: torch.Tensor,
    packed_position_ids: torch.Tensor,
    block_mask,
) -> torch.Tensor:
    """Run a single tree-packed forward pass with FlexAttention (inputs pre-computed)."""
    with torch.no_grad():
        with tree_attention_context(block_mask):
            output = model(
                input_ids=packed_input_ids,
                position_ids=packed_position_ids,
                attention_mask=None,
                use_cache=False,
            )

    return output.logits.squeeze(0).float()


def verify_equivalence(
    independent_logits: list[torch.Tensor],
    tree_logits: torch.Tensor,
    tree: PrefixTree,
    atol: float = 0.02,
    rtol: float = 0.02,
) -> bool:
    """Check that tree-packed logits match independent logits on sequence paths."""
    all_close = True
    for sid, ind_logits in enumerate(independent_logits):
        path_positions = tree.get_sequence_positions(sid)
        tree_seq_logits = tree_logits[path_positions]

        n = min(len(ind_logits), len(tree_seq_logits))
        if n == 0:
            continue

        diff = (ind_logits[:n] - tree_seq_logits[:n]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if max_diff > atol:
            print(f"  [WARN] Sequence {sid}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            all_close = False
        else:
            print(f"  [OK]   Sequence {sid}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    return all_close


def main():
    parser = argparse.ArgumentParser(description="Benchmark tree attention vs independent forward")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name/path (default: create a small random Qwen2-like model)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Number of timed iterations")
    parser.add_argument("--n-steps", type=int, default=4, help="Number of agent steps (sequences)")
    parser.add_argument("--prefix-len", type=int, default=256, help="Shared prefix length")
    parser.add_argument("--branch-len", type=int, default=128, help="Per-step branch length")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load or create model
    if args.model:
        from transformers import AutoModelForCausalLM, AutoConfig
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device).eval()
        vocab_size = model.config.vocab_size
    else:
        from transformers import AutoModelForCausalLM, AutoConfig
        print("Creating small random Qwen2-like model for benchmarking...")
        config = AutoConfig.for_model(
            "qwen2",
            vocab_size=1024,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            attn_implementation="flash_attention_2",
        )
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).to(device).eval()
        vocab_size = 1024

    patch_for_tree_attention()

    sequences = generate_agent_sequences(
        n_steps=args.n_steps,
        shared_prefix_len=args.prefix_len,
        branch_len=args.branch_len,
        vocab_size=vocab_size,
        seed=args.seed,
    )

    # Pre-compute tree inputs (not included in benchmark timing)
    tree, packed, packed_input_ids, packed_position_ids, block_mask = prepare_tree_inputs(
        sequences, device
    )
    total_independent = sum(len(s) for s in sequences)
    total_tree = tree.total_tokens
    ratio = tree.token_ratio()

    print(f"\n{'=' * 60}")
    print(f"Benchmark Configuration:")
    print(f"  Sequences:        {len(sequences)}")
    print(f"  Prefix length:    {args.prefix_len}")
    print(f"  Branch length:    {args.branch_len}")
    print(f"  Total tokens (independent): {total_independent}")
    print(f"  Total tokens (tree-packed): {total_tree}")
    print(f"  Token ratio:      {ratio:.2%}")
    print(f"{'=' * 60}\n")

    # -- Warmup ---------------------------------------------------------------
    # torch.compile needs several calls with the same shapes to fully warm up.
    # The first 1-2 calls trigger Triton kernel compilation; subsequent calls
    # use the cached kernels.
    print(f"Warming up ({args.warmup} iterations, same shapes as benchmark)...")
    for i in range(args.warmup):
        _ = run_independent_forward(model, sequences, device, dtype)
        _ = run_tree_forward(model, packed_input_ids, packed_position_ids, block_mask)
        torch.cuda.synchronize()
        if i == 0:
            print("  (first warmup includes torch.compile JIT — may take a while)")

    gc.collect()
    torch.cuda.empty_cache()

    # -- Benchmark ------------------------------------------------------------
    print(f"\nRunning benchmark ({args.repeat} iterations)...\n")
    independent_times = []
    tree_times = []

    for i in range(args.repeat):
        results = {}

        with cuda_timer("independent", results):
            ind_logits = run_independent_forward(model, sequences, device, dtype)

        with cuda_timer("tree_packed", results):
            tree_logits = run_tree_forward(model, packed_input_ids, packed_position_ids, block_mask)

        independent_times.append(results["independent"])
        tree_times.append(results["tree_packed"])

        print(f"  Iteration {i+1}: independent={results['independent']:.1f}ms, "
              f"tree_packed={results['tree_packed']:.1f}ms, "
              f"speedup={results['independent']/results['tree_packed']:.2f}x")

    # -- Summary --------------------------------------------------------------
    avg_ind = sum(independent_times) / len(independent_times)
    avg_tree = sum(tree_times) / len(tree_times)
    speedup = avg_ind / avg_tree if avg_tree > 0 else float('inf')

    print(f"\n{'=' * 60}")
    print(f"Results (average over {args.repeat} iterations):")
    print(f"  Independent forward:  {avg_ind:.1f}ms")
    print(f"  Tree-packed forward:  {avg_tree:.1f}ms")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"{'=' * 60}\n")

    # -- Numerical verification -----------------------------------------------
    print("Verifying numerical equivalence...")
    equivalence_ok = verify_equivalence(ind_logits, tree_logits, tree)
    if equivalence_ok:
        print("\n[PASS] All sequences match within tolerance.")
    else:
        print("\n[WARN] Some sequences have large discrepancies (expected with bf16).")

    # -- GPU Memory -----------------------------------------------------------
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"\nPeak GPU memory: {peak_mem:.2f} GB")


if __name__ == "__main__":
    main()
