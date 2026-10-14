"""Functional interface for the mean absolute percentage error metric."""

from typing import Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape


def _mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
    epsilon: float = 1.17e-06,
) -> tuple[Tensor, int]:
    """Update and return variables required to compute the mean absolute percentage error.

    Check for same shape of input tensors.

    Args:
        preds: predicted tensor
        target: ground truth tensor
        epsilon: specifies the lower bound for target values. Any target value below epsilon
            is set to epsilon (avoids ``ZeroDivisionError``).

    """
    _check_same_shape(preds, target)

    abs_diff = tp.abs(preds - target)
    abs_per_error = abs_diff / tp.clamp(tp.abs(target), min=epsilon)

    sum_abs_per_error = tp.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _mean_absolute_percentage_error_compute(sum_abs_per_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    """Compute the mean absolute percentage error.

    Args:
        sum_abs_per_error: sum of absolute value of percentage errors over all observations
            (``percentage error = (target - prediction) / target``)
        num_obs: number of predictions or observations

    """
    return sum_abs_per_error / num_obs


def mean_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the mean absolute percentage error (MAPE).

    The output is a non-negative floating point value, where the optimal value is 0. Note that bad
    predictions can lead to arbitrarily large values, especially when some ``target`` values are
    close to zero. This implementation returns a very large number instead of ``inf``.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAPE

    """
    sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)
    return _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)