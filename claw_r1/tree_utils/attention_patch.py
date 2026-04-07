"""Tree attention monkey patch for FlexAttention integration.

Provides a wrapper that sits on top of verl's already-patched
``_flash_attention_forward`` to intercept calls that carry a
``tree_block_mask`` kwarg (or find one in the thread-local context)
and route them through PyTorch's FlexAttention instead of FA2.

When no ``BlockMask`` is active, the wrapper is completely transparent
and delegates to verl's original (possibly Ulysses-wrapped) implementation.

Two mechanisms for passing the ``BlockMask`` to the attention layer:

1. **kwargs forwarding** — if the HuggingFace model forwards ``**kwargs``
   all the way from ``model.forward()`` to the attention function (verified
   for Qwen2, Llama, etc.), pass ``tree_block_mask=mask`` as a model kwarg.

2. **Context manager** — use :func:`tree_attention_context` to set a
   thread-local ``BlockMask`` that the patched function picks up
   automatically.  This is the recommended approach when the caller
   cannot inject kwargs into the model forward call (e.g. verl's
   ``dp_actor._forward_micro_batch``).

Usage::

    # MUST be called AFTER verl's apply_monkey_patch()
    from claw_r1.tree_utils.attention_patch import (
        patch_for_tree_attention,
        tree_attention_context,
        create_block_mask_from_dense,
    )
    patch_for_tree_attention()

    # At forward time:
    block_mask = create_block_mask_from_dense(dense_mask, seq_len, device)
    with tree_attention_context(block_mask):
        output = model(input_ids=..., position_ids=..., attention_mask=None)
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Optional

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compile flex_attention once at module level
# ---------------------------------------------------------------------------

_FLEX_DYNAMIC = os.environ.get("CLAW_DISABLE_FLEX_ATTENTION_DYNAMIC", "0") != "1"

_TORCH_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": not _FLEX_DYNAMIC,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

_compiled_flex_attention = torch.compile(
    flex_attention,
    dynamic=_FLEX_DYNAMIC,
    options=_TORCH_COMPILE_OPTIONS,
)

BLOCK_SIZE = int(os.environ.get("CLAW_FLEX_ATTENTION_BLOCK_SIZE", "128"))

# ---------------------------------------------------------------------------
# Thread-local context for passing BlockMask without kwargs
# ---------------------------------------------------------------------------

_thread_local = threading.local()


@contextmanager
def tree_attention_context(block_mask: BlockMask | None):
    """Context manager that makes *block_mask* visible to the patched attention.

    All attention calls within this context will use the supplied
    ``BlockMask`` instead of the standard causal/FA2 path.  Nesting is
    **not** supported — the innermost context wins.

    Args:
        block_mask: A ``BlockMask`` created by :func:`create_block_mask_from_dense`,
            or ``None`` to explicitly disable tree attention.
    """
    prev = getattr(_thread_local, "block_mask", None)
    _thread_local.block_mask = block_mask
    try:
        yield
    finally:
        _thread_local.block_mask = prev


def _get_active_block_mask() -> BlockMask | None:
    """Return the thread-local ``BlockMask`` if one is set."""
    return getattr(_thread_local, "block_mask", None)


# ---------------------------------------------------------------------------
# BlockMask creation helper (called during data preparation, not forward)
# ---------------------------------------------------------------------------


def create_block_mask_from_dense(
    dense_mask: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> BlockMask:
    """Convert a dense ``(T, T)`` bool attention mask to a ``BlockMask``.

    Should be called just before the forward pass on the target GPU so the
    dense tensor can be released afterwards.

    Args:
        dense_mask: ``(seq_len, seq_len)`` bool tensor.
        seq_len: Sequence length (must match dense_mask dimensions).
        device: Target device for the block mask.

    Returns:
        A ``BlockMask`` for use with ``flex_attention``.
    """
    dense_on_device = dense_mask.to(device=device, dtype=torch.bool)

    def mask_fn(
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
    ):
        return dense_on_device[q_idx, k_idx]

    return create_block_mask(
        mask_fn,
        B=1,
        H=1,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        device=device,
        _compile=False,
    )


# ---------------------------------------------------------------------------
# FlexAttention forward (tree path)
# ---------------------------------------------------------------------------


def _flex_tree_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: BlockMask,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run FlexAttention with a tree-structured ``BlockMask``.

    Expects tensors in HuggingFace layout ``(B, S, H, D)`` and converts
    to ``(B, H, S, D)`` for ``flex_attention``.

    Args:
        query: ``(B, S, num_heads, head_dim)``
        key: ``(B, S, num_kv_heads, head_dim)``
        value: ``(B, S, num_kv_heads, head_dim)``
        block_mask: Pre-computed ``BlockMask`` from the prefix tree.
        softmax_scale: Optional scaling factor for QK^T.

    Returns:
        ``(B, S, num_heads, head_dim)`` attention output.
    """
    # [B, S, H, D] -> [B, H, S, D]
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    enable_gqa = query.shape[1] != key.shape[1]

    output = _compiled_flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        score_mod=None,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
    )

    # [B, H, S, D] -> [B, S, H, D]
    return output.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Monkey patch injection
# ---------------------------------------------------------------------------

_ORIGINAL_FN = None
_PATCHED = False


def patch_for_tree_attention() -> None:
    """Wrap the current ``_flash_attention_forward`` to support tree attention.

    This function **must** be called *after* verl's ``apply_monkey_patch``
    so that the wrapped function already contains verl's Ulysses logic.

    The patched function checks for a ``BlockMask`` in two places
    (in priority order):

    1. ``kwargs["tree_block_mask"]`` — explicit kwarg forwarding.
    2. Thread-local context set via :func:`tree_attention_context`.

    If neither is found, the call falls through to verl's original path.
    """
    global _ORIGINAL_FN, _PATCHED
    if _PATCHED:
        logger.warning("Tree attention patch is already applied.")
        return

    from transformers.integrations import flash_attention

    _ORIGINAL_FN = flash_attention._flash_attention_forward

    def tree_aware_flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        *args,
        **kwargs,
    ):
        block_mask = kwargs.pop("tree_block_mask", None)
        if block_mask is None:
            block_mask = _get_active_block_mask()

        if block_mask is not None and isinstance(block_mask, BlockMask):
            softmax_scale = kwargs.get("softmax_scale", None)
            return _flex_tree_attention_forward(
                query_states,
                key_states,
                value_states,
                block_mask,
                softmax_scale=softmax_scale,
            )

        return _ORIGINAL_FN(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            *args,
            **kwargs,
        )

    flash_attention._flash_attention_forward = tree_aware_flash_attention_forward
    _PATCHED = True
    logger.info(
        "Patched _flash_attention_forward with tree-aware wrapper "
        "(FlexAttention + BlockMask, block_size=%d, dynamic=%s).",
        BLOCK_SIZE,
        _FLEX_DYNAMIC,
    )


def restore_tree_attention_patch() -> None:
    """Restore the original ``_flash_attention_forward``. For testing only."""
    global _ORIGINAL_FN, _PATCHED
    if not _PATCHED or _ORIGINAL_FN is None:
        logger.warning("Tree attention patch was not applied or already restored.")
        return

    from transformers.integrations import flash_attention

    flash_attention._flash_attention_forward = _ORIGINAL_FN
    _ORIGINAL_FN = None
    _PATCHED = False
    logger.info("Restored original _flash_attention_forward.")
