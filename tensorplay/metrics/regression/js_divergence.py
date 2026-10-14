"""Jensen-Shannon divergence metric."""

from collections.abc import Sequence
from math import log
from typing import Any, List, Optional, Union, cast

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.js_divergence import _jsd_compute, _jsd_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class JensenShannonDivergence(Metric):
    """Compute the Jensen-Shannon divergence.

    The Jensen-Shannon divergence is a symmetrical measure of similarity between two probability
    distributions, defined as the average of the KL divergence of each distribution to their mean.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``p`` (float tensor): a data distribution with shape ``(N, d)``
    - ``q`` (float tensor): a prior or approximate distribution with shape ``(N, d)``

    As output of ``forward`` and ``compute`` the metric returns a tensor with the Jensen-Shannon divergence.

    Args:
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributions sum to 1
        reduction: determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: averages score across samples
            - ``'sum'``: sums score across samples
            - ``'none'`` or ``None``: returns score per sample

    Raises:
        TypeError: if ``log_prob`` is not a ``bool``
        ValueError: if ``reduction`` is not one of ``'mean'``, ``'sum'``, ``'none'`` or ``None``

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = log(2)
    measures: Union[Tensor, List[Tensor]]
    total: Tensor

    def __init__(
        self,
        log_prob: bool = False,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(log_prob, bool):
            raise TypeError(f"Expected argument `log_prob` to be bool but got {log_prob}")
        self.log_prob = log_prob
        allowed_reduction = ["mean", "sum", "none", None]
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}")
        self.reduction = reduction
        if self.reduction in ["mean", "sum"]:
            self.add_state("measures", tp.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("measures", [], dist_reduce_fx="cat")
        self.add_state("total", tp.tensor(0), dist_reduce_fx="sum")

    def update(self, p: Tensor, q: Tensor) -> None:
        """Update the metric state."""
        measures, total = _jsd_update(p, q, self.log_prob)
        if self.reduction is None or self.reduction == "none":
            cast(List[Tensor], self.measures).append(measures)
        else:
            self.measures = cast(Tensor, self.measures) + measures.sum()
            self.total += total

    def compute(self) -> Tensor:
        """Compute the Jensen-Shannon divergence."""
        measures: Tensor = (
            dim_zero_cat(cast(List[Tensor], self.measures))
            if self.reduction in ["none", None]
            else cast(Tensor, self.measures)
        )
        return _jsd_compute(measures, self.total, self.reduction)

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