"""Tweedie deviance score metric."""

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.compute import _safe_xlogy

__all__ = ["tweedie_deviance_score"]


def _tweedie_deviance_score_update(preds: Tensor, targets: Tensor, power: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return the sum of deviance scores and the observation count."""
    _check_same_shape(preds, targets)

    zero_tensor = tp.zeros(preds.shape, device=preds.device)

    if 0 < power < 1:
        raise ValueError(f"Deviance Score is not defined for power={power}.")

    if power == 0:
        deviance_score = tp.pow(targets - preds, exponent=2)
    elif power == 1:
        if tp.any(preds <= 0) or tp.any(targets < 0):
            raise ValueError(
                f"For power={power}, 'preds' has to be strictly positive and 'targets' cannot be negative."
            )
        deviance_score = 2 * (_safe_xlogy(targets, targets / preds) + preds - targets)
    elif power == 2:
        if tp.any(preds <= 0) or tp.any(targets <= 0):
            raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")
        deviance_score = 2 * (tp.log(preds / targets) + (targets / preds) - 1)
    else:
        if power < 0:
            if tp.any(preds <= 0):
                raise ValueError(f"For power={power}, 'preds' has to be strictly positive.")
        elif 1 < power < 2:
            if tp.any(preds <= 0) or tp.any(targets < 0):
                raise ValueError(
                    f"For power={power}, 'targets' has to be strictly positive and 'preds' cannot be negative."
                )
        else:
            if tp.any(preds <= 0) or tp.any(targets <= 0):
                raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")

        term_1 = tp.pow(tp.max(targets, zero_tensor), 2 - power) / ((1 - power) * (2 - power))
        term_2 = targets * tp.pow(preds, 1 - power) / (1 - power)
        term_3 = tp.pow(preds, 2 - power) / (2 - power)
        deviance_score = 2 * (term_1 - term_2 + term_3)

    sum_deviance_score = tp.sum(deviance_score)
    num_observations = tp.tensor(tp.numel(deviance_score), device=preds.device)

    return sum_deviance_score, num_observations


def tweedie_deviance_score(preds: Tensor, targets: Tensor, power: float = 0.0) -> Tensor:
    r"""Compute the Tweedie deviance score.

    .. math::
        deviance\_score(\hat{y},y) =
        \begin{cases}
        (\hat{y} - y)^2, & \text{for }p=0\\
        2 * (y * log(\frac{y}{\hat{y}}) + \hat{y} - y),  & \text{for }p=1\\
        2 * (log(\frac{\hat{y}}{y}) + \frac{y}{\hat{y}} - 1),  & \text{for }p=2\\
        2 * (\frac{(max(y,0))^{2 - p}}{(1 - p)(2 - p)} - \frac{y(\hat{y})^{1 - p}}{1 - p} + \frac{(\hat{y})^{2 - p}}{2 - p}), & \text{otherwise}
        \end{cases}

    Args:
        preds: predicted tensor with shape ``(N, ...)``
        targets: ground truth tensor with shape ``(N, ...)``
        power: distribution power parameter (0, 1, 2 or any other valid value)

    Returns:
        scalar Tweedie deviance score
    """
    sum_deviance_score, num_observations = _tweedie_deviance_score_update(preds, targets, power=power)
    return sum_deviance_score / num_observations
