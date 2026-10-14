"""Functional interface for the log cosh error metric."""

import math

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.utils import _check_data_shape_to_num_outputs
from tensorplay.metrics.utilities.checks import _check_same_shape


def _unsqueeze_tensors(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if preds.ndim == 2:
        return preds, target
    return preds.unsqueeze(1), target.unsqueeze(1)


def _log_cosh_error_update(preds: Tensor, target: Tensor, num_outputs: int) -> tuple[Tensor, Tensor]:
    """Update and return variables required to compute the log cosh error.

    Check for same shape of input tensors.

    Args:
        preds: predicted tensor
        target: ground truth tensor
        num_outputs: number of outputs in multioutput setting

    Return:
        Sum of log cosh error over examples, and total number of examples

    """
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    if preds.is_complex() or target.is_complex():
        raise ValueError("Expected `preds` and `target` to be real tensors.")

    preds, target = _unsqueeze_tensors(preds, target)
    if preds.dtype == tp.float16 or preds.dtype == tp.bfloat16:
        preds = preds.float()
    if target.dtype == tp.float16 or target.dtype == tp.bfloat16:
        target = target.float()
    diff = preds - target
    if not diff.is_floating_point():
        diff = diff.float()
    sum_log_cosh_error = (tp.logaddexp(diff, -diff) - math.log(2.0)).sum(0).squeeze()
    num_obs = tp.tensor(target.shape[0], device=preds.device)
    return sum_log_cosh_error, num_obs


def _log_cosh_error_compute(sum_log_cosh_error: Tensor, num_obs: Tensor) -> Tensor:
    """Compute the mean log cosh error.

    Args:
        sum_log_cosh_error: sum of log cosh errors over all observations
        num_obs: number of predictions or observations

    """
    return (sum_log_cosh_error / num_obs).squeeze()


def log_cosh_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the log cosh error.

    The log cosh error is defined as the logarithm of the mean of the exponentials of the
    positive and negative prediction errors, which behaves similarly to the mean absolute error
    for large errors and to the mean squared error divided by two for small errors.

    Args:
        preds: estimated labels with shape ``(batch_size,)`` or ``(batch_size, num_outputs)``
        target: ground truth labels with shape ``(batch_size,)`` or ``(batch_size, num_outputs)``

    Return:
        Tensor with the log cosh error

    """
    sum_log_cosh_error, num_obs = _log_cosh_error_update(
        preds, target, num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    return _log_cosh_error_compute(sum_log_cosh_error, num_obs)