"""Functional mean squared error."""

from typing import Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape


def _mean_squared_error_update(preds: Tensor, target: Tensor, num_outputs: int) -> tuple[Tensor, int]:
    """Update and return variables required to compute Mean Squared Error."""
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    if preds.device.type == "cuda":
        # The fused dual-input reduction currently ships for CPU only; CUDA
        # inputs fall back to the composed kernels.
        diff = preds - target
        sum_squared_error = tensorplay.sum(diff * diff, dim=0)
    else:
        # A single fused pass computes sum((preds - target)^2) without
        # materialising the difference/power tensors.
        sum_squared_error = tensorplay.sum_squared_error(preds, target, dim=[0])
    return sum_squared_error, target.shape[0]


def _mean_squared_error_compute(sum_squared_error: Tensor, num_obs: Union[int, Tensor], squared: bool = True) -> Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        num_obs: Number of predictions or observations
        squared: Returns RMSE value if set to ``False``.

    """
    return sum_squared_error / num_obs if squared else tensorplay.sqrt(sum_squared_error / num_obs)


def mean_squared_error(preds: Tensor, target: Tensor, squared: bool = True, num_outputs: int = 1) -> Tensor:
    """Compute mean squared error.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to ``False``
        num_outputs: Number of outputs in multioutput setting

    Return:
        Tensor with MSE

    """
    sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs=num_outputs)
    return _mean_squared_error_compute(sum_squared_error, num_obs, squared=squared)