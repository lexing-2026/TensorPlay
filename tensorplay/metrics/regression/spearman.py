"""Class-based Spearman rank correlation coefficient."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.spearman import (
    _spearman_corrcoef_compute,
    _spearman_corrcoef_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.utilities.prints import rank_zero_warn


class SpearmanCorrCoef(Metric):
    """Compute Spearman's rank correlation coefficient.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): ``(N, ...)`` or ``(N, M, ...)``
    - ``target`` (float tensor): same shape as ``preds``

    As output to ``forward`` and ``compute`` the metric returns the Spearman correlation coefficient.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpearmanCorrCoef` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to large memory footprint."
        )
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError(f"Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _spearman_corrcoef_update(preds, target, num_outputs=self.num_outputs)
        self.preds.append(preds.to(self.dtype))
        self.target.append(target.to(self.dtype))

    def compute(self) -> Tensor:
        """Compute Spearman's correlation coefficient."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _spearman_corrcoef_compute(preds, target)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)