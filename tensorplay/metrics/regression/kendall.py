"""Kendall rank correlation coefficient metric."""

from typing import Any, Optional, Union
from collections.abc import Sequence

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.kendall import kendall_rank_corrcoef
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from typing import Literal

__all__ = ["KendallRankCorrCoef"]


class KendallRankCorrCoef(Metric):
    """Compute the Kendall rank correlation coefficient.

    Args:
        variant: which variant of Kendall's tau to use (``'a'``, ``'b'`` or ``'c'``).
        t_test: whether to run a t-test and return the p-value.
        alternative: alternative hypothesis for the t-test.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: Optional[float] = -1.0
    plot_upper_bound: Optional[float] = 1.0

    sum_tau: Tensor
    total: Tensor

    def __init__(
        self,
        variant: Literal["a", "b", "c"] = "b",
        t_test: bool = False,
        alternative: Optional[Literal["two-sided", "less", "greater"]] = "two-sided",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.variant = variant
        self.t_test = t_test
        self.alternative = alternative
        self.add_state("sum_tau", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        tau = kendall_rank_corrcoef(preds, target, variant=self.variant, t_test=self.t_test, alternative=self.alternative)
        if isinstance(tau, tuple):
            tau = tau[0]
        self.sum_tau += tau.sum()
        self.total += tau.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_tau / self.total

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot metric values."""
        return self._plot(val, ax)
