"""Normalized root mean squared error metric."""

from typing import Any, Optional, Union
from collections.abc import Sequence

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.mse import _mean_squared_error_update
from tensorplay.metrics.functional.regression.nrmse import _normalized_root_mean_squared_error_compute
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from typing import Literal

__all__ = ["NormalizedRootMeanSquaredError"]


class NormalizedRootMeanSquaredError(Metric):
    """Compute the normalized root mean squared error (NRMSE).

    Args:
        normalization: one of ``'mean'``, ``'range'``, ``'std'`` or ``'l2'``.
        num_outputs: number of outputs in the multioutput setting.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    sum_squared_error: Tensor
    total: Tensor
    min_val: Tensor
    max_val: Tensor
    mean_val: Tensor
    var_val: Tensor
    target_squared: Tensor

    def __init__(
        self,
        normalization: Literal["mean", "range", "std", "l2"] = "mean",
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if normalization not in ("mean", "range", "std", "l2"):
            raise ValueError(
                f"Argument `normalization` should be either 'mean', 'range', 'std' or 'l2', but got {normalization}"
            )
        self.normalization = normalization

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=tp.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tp.zeros(num_outputs), dist_reduce_fx=None)
        self.add_state("min_val", default=float("Inf") * tp.ones(self.num_outputs), dist_reduce_fx=None)
        self.add_state("max_val", default=-float("Inf") * tp.ones(self.num_outputs), dist_reduce_fx=None)
        self.add_state("mean_val", default=tp.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_val", default=tp.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("target_squared", default=tp.zeros(self.num_outputs), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, num_obs = _mean_squared_error_update(preds, target, self.num_outputs)
        self.sum_squared_error += sum_squared_error
        target = target.view(-1) if self.num_outputs == 1 else target

        # Update min and max and target squared
        self.min_val = tp.minimum(target.min(dim=0).values, self.min_val)
        self.max_val = tp.maximum(target.max(dim=0).values, self.max_val)
        self.target_squared += (target**2).sum(dim=0)

        # Update mean and variance
        new_mean = (self.total * self.mean_val + target.sum(dim=0)) / (self.total + num_obs)
        self.total += num_obs
        new_var = ((target - new_mean) * (target - self.mean_val)).sum(dim=0)
        self.mean_val = new_mean
        self.var_val += new_var

    def compute(self) -> Tensor:
        """Compute the NRMSE over the accumulated state."""
        if self.normalization == "mean":
            denom = self.mean_val
        elif self.normalization == "range":
            denom = self.max_val - self.min_val
        elif self.normalization == "std":
            denom = tp.sqrt(self.var_val / self.total)
        else:
            denom = tp.sqrt(self.target_squared)
        return _normalized_root_mean_squared_error_compute(self.sum_squared_error, self.total, denom)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot metric values."""
        return self._plot(val, ax)
