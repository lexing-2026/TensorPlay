"""Class-based mean squared error."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.mse import (
    _mean_squared_error_compute,
    _mean_squared_error_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class MeanSquaredError(Metric):
    """Compute mean squared error.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (float tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the mean squared error. If ``squared`` is set to
    ``False`` the root mean squared error is returned instead.

    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        squared: bool = True,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(f"Expected argument `squared` to be a boolean but got {squared}")
        self.squared = squared

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=tensorplay.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensorplay.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs=self.num_outputs)

        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)