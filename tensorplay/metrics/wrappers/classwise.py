"""Classwise output wrapper metric."""

import typing
from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.wrappers.abstract import WrapperMetric

if typing.TYPE_CHECKING:
    from tensorplay.nn import Module


class ClasswiseWrapper(WrapperMetric):
    """Wrapper metric for altering the output of classification metrics.

    This metric works together with classification metrics that return multiple values (one value per class)
    such that label information can be automatically included in the output.

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
        labels: list of strings indicating the different classes
        prefix: string that is prepended to the metric names
        postfix: string that is appended to the metric names

    """

    metric: Metric
    labels: Optional[list[str]]

    def __init__(
        self,
        metric: Metric,
        labels: Optional[list[str]] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `Metric` but got {metric}")
        self.metric = metric
        if labels is not None and not (isinstance(labels, list) and all(isinstance(lab, str) for lab in labels)):
            raise ValueError(f"Expected argument `labels` to either be `None` or a list of strings but got {labels}")
        self.labels = labels
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(f"Expected argument `prefix` to either be `None` or a string but got {prefix}")
        self._prefix = prefix
        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(f"Expected argument `postfix` to either be `None` or a string but got {postfix}")
        self._postfix = postfix
        self._update_count = 1

    @property
    def higher_is_better(self) -> Optional[bool]:  # type: ignore
        """Return if the metric is higher the better."""
        return self.metric.higher_is_better

    def _filter_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Filter kwargs for the metric."""
        return self.metric._filter_kwargs(**kwargs)

    def _convert_output(self, x: Tensor) -> dict[str, Any]:
        """Convert output to dictionary with labels as keys."""
        # Will set the class name as prefix if neither prefix nor postfix is given
        if not self._prefix and not self._postfix:
            prefix = f"{self.metric.__class__.__name__.lower()}_"
            postfix = ""
        else:
            prefix = self._prefix or ""
            postfix = self._postfix or ""
        if self.labels is None:
            return {f"{prefix}{i}{postfix}": val for i, val in enumerate(x)}
        return {f"{prefix}{lab}{postfix}": val for lab, val in zip(self.labels, x)}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate on batch and accumulate to global state."""
        return self._convert_output(self.metric(*args, **kwargs))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update state."""
        self.metric.update(*args, **kwargs)

    def compute(self) -> dict[str, Tensor]:
        """Compute metric."""
        return self._convert_output(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()

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

    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        """Get attribute from classwise wrapper."""
        if name == "metric" or (name in self.__dict__ and name not in self.metric.__dict__):
            # we need this to prevent from infinite getattribute loop.
            return super().__getattr__(name)
        return getattr(self.metric, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute to classwise wrapper."""
        if hasattr(self, "metric") and name in self.metric._defaults:
            setattr(self.metric, name, value)
        else:
            super().__setattr__(name, value)
            if name == "metric":
                self._defaults = self.metric._defaults
                self._persistent = self.metric._persistent
                self._reductions = self.metric._reductions