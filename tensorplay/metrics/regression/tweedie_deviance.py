"""Tweedie deviance score metric."""

from typing import Any, Optional, Union
from collections.abc import Sequence

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.tweedie_deviance import _tweedie_deviance_score_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = ["TweedieDevianceScore"]


class TweedieDevianceScore(Metric):
    """Compute the Tweedie deviance score.

    Args:
        power: distribution power parameter (0, 1, 2 or any other valid value).
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    sum_deviance_score: Tensor
    total: Tensor

    def __init__(self, power: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.power = power
        self.add_state("sum_deviance_score", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_deviance_score, num_obs = _tweedie_deviance_score_update(preds, target, self.power)
        self.sum_deviance_score += sum_deviance_score
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_deviance_score / self.total

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot metric values."""
        return self._plot(val, ax)
