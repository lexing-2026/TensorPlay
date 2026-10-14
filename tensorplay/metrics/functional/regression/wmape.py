"""Functional interface for the weighted mean absolute percentage error metric."""

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape


def _weighted_mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Update and return variables required to compute the weighted absolute percentage error.

    Check for same shape of input tensors.

    Args:
        preds: predicted tensor
        target: ground truth tensor

    """
    _check_same_shape(preds, target)

    sum_abs_error = (preds - target).abs().sum()
    sum_scale = target.abs().sum()

    return sum_abs_error, sum_scale


def _weighted_mean_absolute_percentage_error_compute(
    sum_abs_error: Tensor,
    sum_scale: Tensor,
    epsilon: float = 1.17e-06,
) -> Tensor:
    """Compute the weighted absolute percentage error.

    Args:
        sum_abs_error: scalar with sum of absolute errors
        sum_scale: scalar with sum of target values
        epsilon: small float to prevent division by zero

    """
    return sum_abs_error / tp.clamp(sum_scale, min=epsilon)


def weighted_mean_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the weighted mean absolute percentage error (WMAPE).

    The output of the metric is a non-negative floating point value, where the optimal value is 0.
    It is computed as the sum of absolute errors divided by the sum of the absolute target values.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with WMAPE

    """
    sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(preds, target)
    return _weighted_mean_absolute_percentage_error_compute(sum_abs_error, sum_scale)