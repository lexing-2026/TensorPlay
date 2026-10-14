"""Class-based R2 score."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.r2 import _r2_score_compute, _r2_score_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class R2Score(Metric):
    """Compute R2 score (coefficient of determination).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)`` or ``(N, M, ...)``
    - ``target`` (float tensor): same shape as ``preds``

    As output to ``forward`` and ``compute`` the metric returns the R2 score.

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        adjusted: int = 0,
        multioutput: str = "uniform_average",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if adjusted < 0 or not isinstance(adjusted, int):
            raise ValueError("`adjusted` parameter should be an integer larger or equal to 0.")
        self.adjusted = adjusted

        allowed_multioutput = ("raw_values", "uniform_average", "variance_weighted")
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}"
            )
        self.multioutput = multioutput

        self.add_state("sum_squared_error", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_error", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("residual", default=tensorplay.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensorplay.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, sum_error, residual, total = _r2_score_update(preds, target)

        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.sum_error = self.sum_error + sum_error
        self.residual = self.residual + residual
        self.total = self.total + total

    def compute(self) -> Tensor:
        """Compute r2 score over the metric states."""
        return _r2_score_compute(
            self.sum_squared_error, self.sum_error, self.residual, self.total, self.adjusted, self.multioutput
        )

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)