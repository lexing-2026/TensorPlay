"""Normalized root mean squared error metric."""

from typing import Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.mse import _mean_squared_error_update

from typing import Literal

__all__ = ["normalized_root_mean_squared_error"]


def _normalized_root_mean_squared_error_update(
    preds: Tensor, target: Tensor, num_outputs: int, normalization: Literal["mean", "range", "std", "l2"] = "mean"
) -> tuple[Tensor, int, Tensor]:
    """Return the sum of squared errors, observation count and normalization denominator."""
    sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs)

    target = target.view(-1) if num_outputs == 1 else target
    if normalization == "mean":
        denom = tp.mean(target, dim=0)
    elif normalization == "range":
        denom = tp.max(target, dim=0).values - tp.min(target, dim=0).values
    elif normalization == "std":
        denom = tp.std(target, correction=0, dim=0)
    elif normalization == "l2":
        denom = tp.norm(target, p=2, dim=0)
    else:
        raise ValueError(
            f"Argument `normalization` should be either 'mean', 'range', 'std' or 'l2' but got {normalization}"
        )
    return sum_squared_error, num_obs, denom


def normalized_root_mean_squared_error(
    preds: Tensor,
    target: Tensor,
    normalization: Literal["mean", "range", "std", "l2"] = "mean",
    num_outputs: int = 1,
) -> Tensor:
    """Compute the normalized root mean squared error (NRMSE).

    Args:
        preds: estimated labels
        target: ground truth labels
        normalization: one of ``'mean'``, ``'range'``, ``'std'`` or ``'l2'``
        num_outputs: number of outputs in the multioutput setting

    Returns:
        NRMSE value (scalar, or per-output tensor in multioutput mode)
    """
    sum_squared_error, num_obs, denom = _normalized_root_mean_squared_error_update(
        preds, target, num_outputs=num_outputs, normalization=normalization
    )
    return _normalized_root_mean_squared_error_compute(sum_squared_error, num_obs, denom)


def _normalized_root_mean_squared_error_compute(sum_squared_error: Tensor, num_obs: int, denom: Tensor) -> Tensor:
    """Compute the NRMSE from the accumulated error and the normalization denominator."""
    rmse = tp.sqrt(sum_squared_error / num_obs)
    return rmse / denom
