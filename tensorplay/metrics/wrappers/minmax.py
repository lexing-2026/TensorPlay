"""Min/max tracking wrapper metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.wrappers.abstract import WrapperMetric


class MinMaxMetric(WrapperMetric):
    """Wrapper metric that tracks both the minimum and maximum of a scalar/tensor across an experiment.

    The min/max value will be updated each time ``.compute`` is called.

    Args:
        base_metric: the metric of which you want to keep track of its maximum and minimum values

    Raises:
        ValueError: if ``base_metric`` argument is not a subclass instance of ``Metric``

    """

    full_state_update: Optional[bool] = True
    min_val: Tensor
    max_val: Tensor

    def __init__(
        self,
        base_metric: Metric,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected base metric to be an instance of `Metric` but received {base_metric}"
            )
        self._base_metric = base_metric
        self.add_state("min_val", default=tp.tensor(float("inf")), dist_reduce_fx="min", persistent=True)
        self.add_state("max_val", default=tp.tensor(float("-inf")), dist_reduce_fx="max", persistent=True)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the underlying metric."""
        self._base_metric.update(*args, **kwargs)

    def compute(self) -> dict[str, Tensor]:
        """Compute the underlying metric as well as max and min values for this metric.

        Returns a dictionary that consists of the computed value (``raw``), as well as the minimum (``min``)
        and maximum (``max``) values.

        """
        val = self._base_metric.compute()
        if not self._is_suitable_val(val):
            raise RuntimeError(f"Returned value from base metric should be a float or scalar tensor, but got {val}.")
        self.max_val = val if self.max_val.to(val.device) < val else self.max_val.to(val.device)
        self.min_val = val if self.min_val.to(val.device) > val else self.min_val.to(val.device)
        return {"raw": val, "max": self.max_val, "min": self.min_val}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the original forward method of the base metric class."""
        return super(WrapperMetric, self).forward(*args, **kwargs)

    def reset(self) -> None:
        """Set ``max_val`` and ``min_val`` to the initialization bounds and reset the base metric.

        In contrast to the base ``Metric.reset()``, this method preserves the accumulated
        ``min_val`` and ``max_val`` across resets so that they survive checkpoint save/load
        boundaries. Only the wrapped base metric is fully reset.

        """
        # Save current min/max values before the base class resets them to defaults
        _min = self.min_val.clone()
        _max = self.max_val.clone()
        super().reset()
        # Restore the accumulated min/max so they persist across epochs
        self.min_val = _min
        self.max_val = _max
        self._base_metric.reset()

    @staticmethod
    def _is_suitable_val(val: Union[float, Tensor]) -> bool:
        """Check whether min/max is a scalar value."""
        if isinstance(val, (int, float)):
            return True
        if isinstance(val, Tensor):
            return val.numel() == 1
        return False

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