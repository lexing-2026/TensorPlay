"""Mean squared log error metric."""

from typing import Any, Optional, Union
from collections.abc import Sequence

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.log_mse import _mean_squared_log_error_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = ["MeanSquaredLogError"]


class MeanSquaredLogError(Metric):
    """Compute the mean squared log error.

    Args:
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    sum_squared_log_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_squared_log_error", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_log_error, num_obs = _mean_squared_log_error_update(preds, target)
        self.sum_squared_log_error += sum_squared_log_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_squared_log_error / self.total

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot metric values."""
        return self._plot(val, ax)
