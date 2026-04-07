"""Factory for creating tree-attention-aware worker subclasses.

Provides :func:`make_tree_aware_worker_cls`, which dynamically generates
a thin subclass of any verl worker class.  The subclass applies the tree
attention monkey patch and swaps in ``TreeDataParallelPPOActor`` after
model initialization.
"""

from __future__ import annotations

import logging

from verl.single_controller.base.decorator import Dispatch, register

logger = logging.getLogger(__name__)


def make_tree_aware_worker_cls(base_cls: type) -> type:
    """Return a subclass of *base_cls* that applies tree attention after init.

    The returned class overrides ``init_model`` to call the base
    implementation first, then:

    1. Apply :func:`patch_for_tree_attention` (FlexAttention wrapper).
    2. Replace ``self.actor`` with :class:`TreeDataParallelPPOActor`.
    """

    class _TreeAwareWorker(base_cls):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            from claw_r1.tree_utils.attention_patch import patch_for_tree_attention

            patch_for_tree_attention()

            if self._is_actor and hasattr(self, "actor"):
                from claw_r1.tree_utils.tree_actor import TreeDataParallelPPOActor

                self.actor = TreeDataParallelPPOActor(
                    config=self.actor.config,
                    actor_module=self.actor.actor_module,
                    actor_optimizer=self.actor.actor_optimizer,
                )

    _TreeAwareWorker.__name__ = f"TreeAware{base_cls.__name__}"
    _TreeAwareWorker.__qualname__ = f"TreeAware{base_cls.__qualname__}"
    return _TreeAwareWorker
