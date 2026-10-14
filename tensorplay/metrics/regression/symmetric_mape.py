"""Class-based symmetric mean absolute percentage error."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.symmetric_mape import (
    _symmetric_mean_absolute_percentage_error_compute,
    _symmetric_mean_absolute_percentage_error_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class SymmetricMeanAbsolutePercentageError(Metric):
    """Compute symmetric mean absolute percentage error (SMAPE).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (float tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the SMAPE.

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute percentage error over state."""
        return _symmetric_mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)