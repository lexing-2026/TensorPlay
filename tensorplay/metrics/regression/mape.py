"""Mean absolute percentage error metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor, tensor

from tensorplay.metrics.functional.regression.mape import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class MeanAbsolutePercentageError(Metric):
    """Compute the mean absolute percentage error (MAPE).

    The output is a non-negative floating point value. Best result is ``0.0``, but bad predictions can lead to
    arbitrarily large values, especially when some ``target`` values are close to zero. This implementation
    returns a very large number instead of ``inf``.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predictions from model
    - ``target`` (float tensor): ground truth values

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    sum_abs_per_error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)
        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute the mean absolute percentage error over state."""
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

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