"""Class-based explained variance."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.explained_variance import (
    ALLOWED_MULTIOUTPUT,
    _explained_variance_compute,
    _explained_variance_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class ExplainedVariance(Metric):
    """Compute explained variance.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)`` or ``(N, M, ...)``
    - ``target`` (float tensor): same shape as ``preds``

    As output to ``forward`` and ``compute`` the metric returns the explained variance.

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        multioutput: str = "uniform_average",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if multioutput not in ALLOWED_MULTIOUTPUT:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {ALLOWED_MULTIOUTPUT}"
            )
        self.multioutput = multioutput
        self.add_state("sum_error", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_error", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_target", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_target", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        num_obs, sum_error, sum_squared_error, sum_target, sum_squared_target = _explained_variance_update(
            preds, target
        )
        self.num_obs = self.num_obs + num_obs
        self.sum_error = self.sum_error + sum_error
        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.sum_target = self.sum_target + sum_target
        self.sum_squared_target = self.sum_squared_target + sum_squared_target

    def compute(self) -> Union[Tensor, Sequence[Tensor]]:
        """Compute explained variance over state."""
        return _explained_variance_compute(
            self.num_obs,
            self.sum_error,
            self.sum_squared_error,
            self.sum_target,
            self.sum_squared_target,
            self.multioutput,
        )

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)