"""Functional explained variance."""

from collections.abc import Sequence
from typing import Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape

ALLOWED_MULTIOUTPUT = ("raw_values", "uniform_average", "variance_weighted")


def _explained_variance_update(preds: Tensor, target: Tensor) -> tuple[int, Tensor, Tensor, Tensor, Tensor]:
    """Update and return variables required to compute Explained Variance."""
    _check_same_shape(preds, target)

    num_obs = preds.size(0)
    sum_error = tensorplay.sum(target - preds, dim=0)
    diff = target - preds
    sum_squared_error = tensorplay.sum(diff * diff, dim=0)

    sum_target = tensorplay.sum(target, dim=0)
    sum_squared_target = tensorplay.sum(target * target, dim=0)

    return num_obs, sum_error, sum_squared_error, sum_target, sum_squared_target


def _explained_variance_compute(
    num_obs: Union[int, Tensor],
    sum_error: Tensor,
    sum_squared_error: Tensor,
    sum_target: Tensor,
    sum_squared_target: Tensor,
    multioutput: str = "uniform_average",
) -> Tensor:
    """Compute Explained Variance.

    Args:
        num_obs: Number of predictions or observations
        sum_error: Sum of errors over all observations
        sum_squared_error: Sum of square of errors over all observations
        sum_target: Sum of target values
        sum_squared_target: Sum of squares of target values
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    """
    diff_avg = sum_error / num_obs
    numerator = sum_squared_error / num_obs - (diff_avg * diff_avg)

    target_avg = sum_target / num_obs
    denominator = sum_squared_target / num_obs - (target_avg * target_avg)

    # Take care of division by zero
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = tensorplay.ones_like(diff_avg)
    output_scores[valid_score] = 1.0 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    # Decide what to do in multioutput case
    if multioutput == "raw_values":
        return output_scores
    if multioutput == "uniform_average":
        return tensorplay.mean(output_scores)
    denom_sum = tensorplay.sum(denominator)
    return tensorplay.sum(denominator / denom_sum * output_scores)


def explained_variance(
    preds: Tensor,
    target: Tensor,
    multioutput: str = "uniform_average",
) -> Tensor:
    """Compute explained variance.

    Args:
        preds: estimated labels
        target: ground truth labels
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Raises:
        ValueError:
            If ``preds`` and ``target`` have different shapes.
        ValueError:
            If ``multioutput`` is not one of ``raw_values``, ``uniform_average`` or ``variance_weighted``.

    Return:
        Tensor with explained variance

    """
    if multioutput not in ALLOWED_MULTIOUTPUT:
        raise ValueError(
            f"Invalid input to argument `multioutput`. Choose one of the following: {ALLOWED_MULTIOUTPUT}"
        )
    num_obs, sum_error, sum_squared_error, sum_target, sum_squared_target = _explained_variance_update(preds, target)
    return _explained_variance_compute(
        num_obs, sum_error, sum_squared_error, sum_target, sum_squared_target, multioutput
    )