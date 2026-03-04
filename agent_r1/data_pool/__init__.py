"""DataPool — trajectory storage and training batch serving.

The DataPool stores rollout trajectories at Step granularity and serves
training-ready batches via a pluggable TrainingBackend interface.
"""

from agent_r1.data_pool.data_model import DataPoolConfig, Step
from agent_r1.data_pool.data_pool import DataPool
from agent_r1.data_pool.training_backend import TrainingBackend, VerlBackend

__all__ = [
    "DataPool",
    "DataPoolConfig",
    "Step",
    "TrainingBackend",
    "VerlBackend",
]
