"""Making batch normalization safe to run under function transforms."""

from __future__ import annotations

import tensorplay.nn as nn

from .utils import exposed_in


def batch_norm_without_running_stats(module: "nn.Module") -> None:
    """Drops the running statistics of one batch-normalization module.

    Running statistics are updated in place on every forward pass.  Under a
    transform that evaluates the module more than once -- mapping over an
    ensemble, or differentiating through it -- those updates would be applied
    repeatedly and in an order the caller never asked for, so the module is
    switched to normalizing with the batch's own statistics instead.
    """
    if (
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ):
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        module.track_running_stats = False


@exposed_in("tensorplay.func")
def replace_all_batch_norm_modules_(root: "nn.Module") -> "nn.Module":
    """Drops the running statistics of every batch-normalization module in
    ``root``, in place, and returns ``root``."""
    # Covers ``root`` itself, which ``modules()`` also yields, plus every child.
    batch_norm_without_running_stats(root)
    for obj in root.modules():
        batch_norm_without_running_stats(obj)
    return root
