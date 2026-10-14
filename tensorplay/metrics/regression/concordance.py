"""Concordance correlation coefficient metric."""

from collections.abc import Sequence
from typing import Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.functional.regression.concordance import _concordance_corrcoef_compute
from tensorplay.metrics.regression.pearson import PearsonCorrCoef, _final_aggregation
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class ConcordanceCorrCoef(PearsonCorrCoef):
    """Compute the concordance correlation coefficient that measures the agreement between two variables.

    The concordance correlation coefficient combines measures of both precision and accuracy to assess the
    agreement between two variables.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): either a single output float tensor with shape ``(N,)`` or a multioutput
      float tensor of shape ``(N, d)``
    - ``target`` (float tensor): either a single output float tensor with shape ``(N,)`` or a multioutput
      float tensor of shape ``(N, d)``

    As output of ``forward`` and ``compute`` the metric returns a scalar float tensor with the concordance
    coefficient for non-multioutput input or a float tensor with shape ``(d,)`` for multioutput input.

    Args:
        num_outputs: number of outputs in multioutput setting

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute the final concordance correlation coefficient over metric states."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            mean_x, mean_y, max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, n_total = _final_aggregation(
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
            mean_x = self.mean_x
            mean_y = self.mean_y
            max_abs_dev_x = self.max_abs_dev_x
            max_abs_dev_y = self.max_abs_dev_y
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _concordance_corrcoef_compute(
            max_abs_dev_x, max_abs_dev_y, mean_x, mean_y, var_x, var_y, corr_xy, n_total
        )

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