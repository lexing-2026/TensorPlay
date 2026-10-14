"""Functional symmetric mean absolute percentage error."""

from typing import Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape


def _symmetric_mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
    epsilon: float = 1.17e-06,
) -> tuple[Tensor, int]:
    """Update and return variables required to compute Symmetric Mean Absolute Percentage Error."""
    _check_same_shape(preds, target)

    abs_diff = tensorplay.abs(preds - target)
    abs_per_error = abs_diff / tensorplay.clamp(tensorplay.abs(target) + tensorplay.abs(preds), min=epsilon)

    sum_abs_per_error = 2 * tensorplay.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _symmetric_mean_absolute_percentage_error_compute(
    sum_abs_per_error: Tensor, num_obs: Union[int, Tensor]
) -> Tensor:
    """Compute Symmetric Mean Absolute Percentage Error.

    Args:
        sum_abs_per_error: Sum of values of symmetric absolute percentage errors over all observations
            ``(symmetric absolute percentage error = 2 * |target - prediction| / (target + prediction))``
        num_obs: Number of predictions or observations

    """
    return sum_abs_per_error / num_obs


def symmetric_mean_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute symmetric mean absolute percentage error (SMAPE).

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with SMAPE

    """
    sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(preds, target)
    return _symmetric_mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)