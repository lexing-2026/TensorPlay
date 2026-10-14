"""Weighted mean absolute percentage error metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.wmape import (
    _weighted_mean_absolute_percentage_error_compute,
    _weighted_mean_absolute_percentage_error_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class WeightedMeanAbsolutePercentageError(Metric):
    """Compute the weighted mean absolute percentage error (WMAPE).

    The output of the metric is a non-negative floating point value, where the optimal value is 0.
    It is computed as the sum of absolute errors divided by the sum of the absolute target values.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predictions from model
    - ``target`` (float tensor): ground truth values

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    sum_abs_error: Tensor
    sum_scale: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_scale", default=tp.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(preds, target)
        self.sum_abs_error += sum_abs_error
        self.sum_scale += sum_scale

    def compute(self) -> Tensor:
        """Compute the weighted mean absolute percentage error over state."""
        return _weighted_mean_absolute_percentage_error_compute(self.sum_abs_error, self.sum_scale)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        """
        return self._plot(val, ax)