"""Log cosh error metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.log_cosh import _log_cosh_error_compute, _log_cosh_error_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class LogCoshError(Metric):
    """Compute the log cosh error.

    The log cosh error behaves similarly to the mean absolute error for large errors and to the mean
    squared error divided by two for small errors.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): estimated labels with shape ``(batch_size,)`` or ``(batch_size, num_outputs)``
    - ``target`` (float tensor): ground truth labels with shape ``(batch_size,)`` or ``(batch_size, num_outputs)``

    As output of ``forward`` and ``compute`` the metric returns a tensor with the log cosh error.

    Args:
        num_outputs: number of outputs in multioutput setting

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    sum_log_cosh_error: Tensor
    total: Tensor

    def __init__(self, num_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError(f"Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs
        self.add_state("sum_log_cosh_error", default=tp.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Raises:
            ValueError: if ``preds`` or ``target`` has multiple outputs when ``num_outputs=1``

        """
        sum_log_cosh_error, num_obs = _log_cosh_error_update(preds, target, self.num_outputs)
        self.sum_log_cosh_error += sum_log_cosh_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute the log cosh error over state."""
        return _log_cosh_error_compute(self.sum_log_cosh_error, self.total)

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