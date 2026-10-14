"""Class-based mean absolute error."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.mae import (
    _mean_absolute_error_compute,
    _mean_absolute_error_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class MeanAbsoluteError(Metric):
    """Compute mean absolute error.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (float tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the mean absolute error.

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_abs_error", default=tensorplay.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensorplay.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_error, num_obs = _mean_absolute_error_update(preds, target, num_outputs=self.num_outputs)

        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)