"""Mean squared log error metric."""

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape

__all__ = ["mean_squared_log_error"]


def _mean_squared_log_error_update(preds: Tensor, target: Tensor) -> tuple[Tensor, int]:
    """Return variables required to compute the mean squared log error."""
    _check_same_shape(preds, target)
    sum_squared_log_error = tp.sum(tp.pow(tp.log1p(preds) - tp.log1p(target), 2))
    return sum_squared_log_error, target.numel()


def mean_squared_log_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the mean squared log error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Returns:
        scalar mean squared log error
    """
    sum_squared_log_error, num_obs = _mean_squared_log_error_update(preds, target)
    return sum_squared_log_error / num_obs
