"""Minkowski distance metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor, tensor

from tensorplay.metrics.functional.regression.minkowski import _minkowski_distance_compute, _minkowski_distance_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.exceptions import MetricsUserError
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class MinkowskiDistance(Metric):
    """Compute the Minkowski distance.

    The Minkowski distance is a generalization of the Euclidean distance (``p=2``) and the Manhattan
    distance (``p=1``).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predictions from model
    - ``targets`` (float tensor): ground truth values

    Args:
        p: int or float larger than 1, exponent to which the difference between preds and target is to be raised

    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    plot_lower_bound: float = 0.0
    minkowski_dist_sum: Tensor

    def __init__(self, p: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not (isinstance(p, (float, int)) and p >= 1):
            raise MetricsUserError(f"Argument ``p`` must be a float or int greater than 1, but got {p}")
        self.p = p
        self.add_state("minkowski_dist_sum", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update state with predictions and targets."""
        minkowski_dist_sum = _minkowski_distance_update(preds, targets, self.p)
        self.minkowski_dist_sum += minkowski_dist_sum

    def compute(self) -> Tensor:
        """Compute the Minkowski distance."""
        return _minkowski_distance_compute(self.minkowski_dist_sum, self.p)

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