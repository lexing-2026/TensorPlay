"""Functional mean absolute error."""

from typing import Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape


def _mean_absolute_error_update(preds: Tensor, target: Tensor, num_outputs: int) -> tuple[Tensor, int]:
    """Update and return variables required to compute Mean Absolute Error."""
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    preds = preds if tensorplay.is_floating_point(preds) else preds.float()
    target = target if tensorplay.is_floating_point(target) else target.float()
    if preds.device.type == "cuda":
        # The fused dual-input reduction currently ships for CPU only; CUDA
        # inputs fall back to the composed kernels.
        sum_abs_error = tensorplay.sum(tensorplay.abs(preds - target), dim=0)
    else:
        # A single fused pass computes sum(|preds - target|) without
        # materialising the difference tensor.
        sum_abs_error = tensorplay.sum_abs_error(preds, target, dim=[0])
    return sum_abs_error, target.shape[0]


def _mean_absolute_error_compute(sum_abs_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    """Compute Mean Absolute Error.

    Args:
        sum_abs_error: Sum of absolute value of errors over all observations
        num_obs: Number of predictions or observations

    """
    return sum_abs_error / num_obs


def mean_absolute_error(preds: Tensor, target: Tensor, num_outputs: int = 1) -> Tensor:
    """Compute mean absolute error.

    Args:
        preds: estimated labels
        target: ground truth labels
        num_outputs: Number of outputs in multioutput setting

    Return:
        Tensor with MAE

    """
    sum_abs_error, num_obs = _mean_absolute_error_update(preds, target, num_outputs=num_outputs)
    return _mean_absolute_error_compute(sum_abs_error, num_obs)