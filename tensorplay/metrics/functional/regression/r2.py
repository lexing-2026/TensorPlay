"""Functional R2 score."""

from typing import Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities import rank_zero_warn
from tensorplay.metrics.utilities.checks import _check_same_shape


def _r2_score_update(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    """Update and return variables required to compute R2 score."""
    _check_same_shape(preds, target)
    if preds.ndim > 2:
        raise ValueError(
            "Expected both prediction and target to be 1D or 2D tensors,"
            f" but received tensors with dimension {preds.shape}"
        )

    sum_obs = tensorplay.sum(target, dim=0)
    sum_squared_obs = tensorplay.sum(target * target, dim=0)
    if preds.device.type == "cuda":
        # The fused dual-input reduction currently ships for CPU only; CUDA
        # inputs fall back to the composed kernels.
        residual = target - preds
        rss = tensorplay.sum(residual * residual, dim=0)
    else:
        # One fused pass computes sum((target - preds)^2) directly.
        rss = tensorplay.sum_squared_error(preds, target, dim=[0])
    return sum_squared_obs, sum_obs, rss, target.size(0)


def _r2_score_compute(
    sum_squared_obs: Tensor,
    sum_obs: Tensor,
    rss: Tensor,
    num_obs: Union[int, Tensor],
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> Tensor:
    """Compute R2 score.

    Args:
        sum_squared_obs: Sum of square of all observations
        sum_obs: Sum of all observations
        rss: Residual sum of squares
        num_obs: Number of predictions or observations
        adjusted: number of independent regressors for calculating adjusted r2 score
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    """
    if num_obs < 2:
        raise ValueError("Needs at least two samples to calculate r2 score.")

    mean_obs = sum_obs / num_obs
    tss = sum_squared_obs - sum_obs * mean_obs

    # Account for near constant targets
    cond_rss = ~tensorplay.isclose(rss, tensorplay.zeros_like(rss), atol=1e-4)
    cond_tss = ~tensorplay.isclose(tss, tensorplay.zeros_like(tss), atol=1e-4)
    cond = cond_rss & cond_tss

    raw_scores = tensorplay.ones_like(rss)
    raw_scores[cond] = 1 - (rss[cond] / tss[cond])
    raw_scores[cond_rss & ~cond_tss] = 0.0

    if multioutput == "raw_values":
        r2 = raw_scores
    elif multioutput == "uniform_average":
        r2 = tensorplay.mean(raw_scores)
    elif multioutput == "variance_weighted":
        tss_sum = tensorplay.sum(tss)
        r2 = tensorplay.sum(tss / tss_sum * raw_scores)
    else:
        raise ValueError(
            "Argument `multioutput` must be either `raw_values`,"
            f" `uniform_average` or `variance_weighted`. Received {multioutput}."
        )

    if adjusted < 0 or not isinstance(adjusted, int):
        raise ValueError("`adjusted` parameter should be an integer larger or equal to 0.")

    if adjusted != 0:
        if adjusted > num_obs - 1:
            rank_zero_warn(
                "More independent regressions than data points in adjusted r2 score. Falls back to standard r2 score.",
                UserWarning,
            )
        elif adjusted == num_obs - 1:
            rank_zero_warn("Division by zero in adjusted r2 score. Falls back to standard r2 score.", UserWarning)
        else:
            return 1 - (1 - r2) * (num_obs - 1) / (num_obs - adjusted - 1)
    return r2


def r2_score(
    preds: Tensor,
    target: Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> Tensor:
    r"""Compute r2 score also known as the coefficient of determination.

    .. math:: R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\sum_i (y_i - \bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the ``adjusted`` argument.

    Args:
        preds: estimated labels
        target: ground truth labels
        adjusted: number of independent regressors for calculating adjusted r2 score
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Raises:
        ValueError:
            If ``preds`` and ``target`` are of different shapes, or ``preds`` and ``target``
            are not 1D or 2D tensors.
        ValueError:
            If ``adjusted`` is not an integer larger or equal to 0.
        ValueError:
            If ``multioutput`` is not one of ``raw_values``, ``uniform_average`` or ``variance_weighted``.
        ValueError:
            If less than two samples are provided.

    Return:
        Tensor with R2 score

    """
    sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
    return _r2_score_compute(sum_squared_obs, sum_obs, rss, num_obs, adjusted, multioutput)