"""DataPool — trajectory storage and training batch serving.

The DataPool stores rollout trajectories at Step granularity and serves
training-ready batches via a pluggable TrainingBackend interface.
"""

from claw_r1.data_pool.data_model import DataPoolConfig, Step
from claw_r1.data_pool.data_pool import DataPool
from claw_r1.data_pool.training_backend import TrainingBackend, VerlBackend
from claw_r1.data_pool.training_backend_prefix_tree import TreeVerlBackend

__all__ = [
    "DataPool",
    "DataPoolConfig",
    "Step",
    "TrainingBackend",
    "TreeVerlBackend",
    "VerlBackend",
]
