"""Running window wrapper for metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.wrappers.abstract import WrapperMetric


class Running(WrapperMetric):
    """Running wrapper for metrics.

    Using this wrapper allows for calculating metrics over a running window of values, instead of the whole history
    of values. This is beneficial when you want to get a better estimate of the metric during training and don't want
    to wait for the whole training to finish to get epoch level estimates.

    The running window is defined by the ``window`` argument. The window is a fixed size and this wrapper will store
    a duplicate of the underlying metric state for each value in the window. Thus memory usage will increase linearly
    with window size. Use accordingly. Also note that the running only works with metrics that have the
    ``full_state_update`` set to ``False``.

    Importantly, the wrapper does not alter the value of the ``forward`` method of the underlying metric. Thus,
    forward will still return the value on the current batch. To get the running value call ``compute`` instead.

    Args:
        base_metric: the metric to wrap
        window: the size of the running window

    """

    def __init__(self, base_metric: Metric, window: int = 5) -> None:
        super().__init__()
        if not isinstance(base_metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `Metric` but got {base_metric}")
        if not (isinstance(window, int) and window > 0):
            raise ValueError(f"Expected argument `window` to be a positive integer but got {window}")
        self.base_metric = base_metric
        self.window = window

        if base_metric.full_state_update is not False:
            raise ValueError(
                f"Expected attribute `full_state_update` set to `False` but got {base_metric.full_state_update}"
            )
        self._num_vals_seen = 0

        for key in base_metric._defaults:
            for i in range(window):
                self.add_state(
                    name=key + f"_{i}", default=base_metric._defaults[key], dist_reduce_fx=base_metric._reductions[key]
                )

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the underlying metric and save state afterwards."""
        val = self._num_vals_seen % self.window
        self.base_metric.update(*args, **kwargs)
        for key in self.base_metric._defaults:
            setattr(self, key + f"_{val}", getattr(self.base_metric, key))
        self.base_metric.reset()
        self._num_vals_seen += 1

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward input to the underlying metric and save state afterwards."""
        val = self._num_vals_seen % self.window
        res = self.base_metric.forward(*args, **kwargs)
        for key in self.base_metric._defaults:
            setattr(self, key + f"_{val}", getattr(self.base_metric, key))
        self.base_metric.reset()
        self._num_vals_seen += 1
        self._computed = None
        return res

    def compute(self) -> Any:
        """Compute the metric over the running window."""
        for i in range(self.window):
            self.base_metric._reduce_states({key: getattr(self, key + f"_{i}") for key in self.base_metric._defaults})
        self.base_metric._update_count = self._num_vals_seen
        val = self.base_metric.compute()
        self.base_metric.reset()
        return val

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self._num_vals_seen = 0

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