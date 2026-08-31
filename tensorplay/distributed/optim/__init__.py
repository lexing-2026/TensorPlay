"""Distributed optimizer utilities.
:mod:`tensorplay.distributed.optim` exposes distributed optimizer utilities
"""

import warnings

from tensorplay import optim

from .apply_optimizer_in_backward import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)
from .named_optimizer import _NamedOptimizer
from .utils import as_functional_optim


# DistributedOptimizer imports tensorplay.distributed.rpc names, so gate availability
try:
    from .optimizer import DistributedOptimizer
except RuntimeError:  # pragma: no cover - rpc gate raises lazily instead
    DistributedOptimizer = None

from .post_localSGD_optimizer import PostLocalSGDOptimizer


from .zero_redundancy_optimizer import ZeroRedundancyOptimizer

__all__ = [
    "as_functional_optim",
    "DistributedOptimizer",
    "PostLocalSGDOptimizer",
    "ZeroRedundancyOptimizer",
]
