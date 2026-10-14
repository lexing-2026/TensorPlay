"""Relative squared error metric."""

from typing import Any, Optional, Union
from collections.abc import Sequence

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.r2 import _r2_score_update
from tensorplay.metrics.functional.regression.rse import _relative_squared_error_compute
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = ["RelativeSquaredError"]


class RelativeSquaredError(Metric):
    """Compute the relative squared error.

    Args:
        squared: if ``False`` returns the root relative squared error.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    sum_squared_obs: Tensor
    sum_obs: Tensor
    sum_squared_error: Tensor
    total: Tensor

    def __init__(self, squared: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.squared = squared
        self.add_state("sum_squared_obs", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_obs", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_error", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_obs, sum_obs, sum_squared_error, num_obs = _r2_score_update(preds, target)
        self.sum_squared_obs += sum_squared_obs
        self.sum_obs += sum_obs
        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute the metric."""
        return _relative_squared_error_compute(
            self.sum_squared_obs, self.sum_obs, self.sum_squared_error, self.total, squared=self.squared
        )

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot metric values."""
        return self._plot(val, ax)
