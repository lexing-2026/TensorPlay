"""Relative squared error metric."""

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.r2 import _r2_score_update

__all__ = ["relative_squared_error"]


def _relative_squared_error_compute(
    sum_squared_obs: Tensor,
    sum_obs: Tensor,
    sum_squared_error: Tensor,
    num_obs: int,
    squared: bool = True,
) -> Tensor:
    """Compute the relative squared error."""
    epsilon = tp.finfo(sum_squared_error.dtype).eps
    rse = sum_squared_error / tp.clamp(sum_squared_obs - sum_obs * sum_obs / num_obs, min=epsilon)
    if not squared:
        rse = tp.sqrt(rse)
    return tp.mean(rse)


def relative_squared_error(preds: Tensor, target: Tensor, squared: bool = True) -> Tensor:
    r"""Compute the relative squared error (RSE).

    .. math:: \text{RSE} = \frac{\sum_i^N(y_i - \hat{y_i})^2}{\sum_i^N(y_i - \overline{y})^2}

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: if ``False`` returns the root relative squared error

    Returns:
        scalar relative squared error
    """
    sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
    return _relative_squared_error_compute(sum_squared_obs, sum_obs, rss, num_obs, squared=squared)
