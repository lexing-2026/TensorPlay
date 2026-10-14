"""Class-based Pearson correlation coefficient."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.pearson import (
    _pearson_corrcoef_compute,
    _pearson_corrcoef_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


def _final_aggregation(
    means_x: Tensor,
    means_y: Tensor,
    maxs_abs_x: Tensor,
    maxs_abs_y: Tensor,
    vars_x: Tensor,
    vars_y: Tensor,
    corrs_xy: Tensor,
    nbs: Tensor,
    eps: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from the parallel algorithm for calculating variance. We use ``eps`` to avoid division by zero
    when ``n1`` and ``n2`` are both zero.
    """
    if len(means_x) == 1:
        return means_x[0], means_y[0], maxs_abs_x[0], maxs_abs_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    mx1 = means_x[0]
    my1 = means_y[0]
    max1 = maxs_abs_x[0]
    may1 = maxs_abs_y[0]
    vx1 = vars_x[0]
    vy1 = vars_y[0]
    cxy1 = corrs_xy[0]
    n1 = nbs[0]
    for i in range(1, len(means_x)):
        mx2 = means_x[i]
        my2 = means_y[i]
        max2 = maxs_abs_x[i]
        may2 = maxs_abs_y[i]
        vx2 = vars_x[i]
        vy2 = vars_y[i]
        cxy2 = corrs_xy[i]
        n2 = nbs[i]
        # count
        nb = tensorplay.where(tensorplay.logical_or(n1, n2), n1 + n2, eps)
        # mean_x
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        # mean_y
        mean_y = (n1 * my1 + n2 * my2) / nb
        # intermediates for running variances
        n12_b = n1 * n2 / nb
        delta_x = mx2 - mx1
        delta_y = my2 - my1
        # var_x
        var_x = vx1 + vx2 + n12_b * delta_x**2
        # var_y
        var_y = vy1 + vy2 + n12_b * delta_y**2
        # corr_xy
        corr_xy = cxy1 + cxy2 + n12_b * delta_x * delta_y
        max_abs_dev_x = tensorplay.maximum(max1, max2)
        max_abs_dev_y = tensorplay.maximum(may1, may2)

        mx1 = mean_x
        my1 = mean_y
        max1 = max_abs_dev_x
        may1 = max_abs_dev_y
        vx1 = var_x
        vy1 = var_y
        cxy1 = corr_xy
        n1 = nb
    return mean_x, mean_y, max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, nb


class PearsonCorrCoef(Metric):
    """Compute Pearson correlation coefficient.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)`` or ``(N, M, ...)``
    - ``target`` (float tensor): same shape as ``preds``

    As output to ``forward`` and ``compute`` the metric returns the Pearson correlation coefficient.

    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None  # both -1 and 1 are optimal
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError("Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("mean_x", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("mean_y", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("max_abs_dev_x", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("max_abs_dev_y", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_x", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_y", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("corr_xy", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("n_total", default=tensorplay.zeros(self.num_outputs), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        (
            self.mean_x,
            self.mean_y,
            self.max_abs_dev_x,
            self.max_abs_dev_y,
            self.var_x,
            self.var_y,
            self.corr_xy,
            self.n_total,
        ) = _pearson_corrcoef_update(
            preds=preds,
            target=target,
            mean_x=self.mean_x,
            mean_y=self.mean_y,
            max_abs_dev_x=self.max_abs_dev_x,
            max_abs_dev_y=self.max_abs_dev_y,
            var_x=self.var_x,
            var_y=self.var_y,
            corr_xy=self.corr_xy,
            num_prior=self.n_total,
            num_outputs=self.num_outputs,
        )

    def compute(self) -> Tensor:
        """Compute pearson correlation coefficient over state."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            # multiple devices, need further reduction
            _, _, max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, n_total = _final_aggregation(
                means_x=self.mean_x,
                means_y=self.mean_y,
                maxs_abs_x=self.max_abs_dev_x,
                maxs_abs_y=self.max_abs_dev_y,
                vars_x=self.var_x,
                vars_y=self.var_y,
                corrs_xy=self.corr_xy,
                nbs=self.n_total,
            )
        else:
            max_abs_dev_x = self.max_abs_dev_x
            max_abs_dev_y = self.max_abs_dev_y
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _pearson_corrcoef_compute(max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, n_total)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)