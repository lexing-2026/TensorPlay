"""Aggregation metrics that reduce a stream of values."""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.utilities.prints import rank_zero_warn


class BaseAggregator(Metric):
    """Base class for aggregation metrics.

    Args:
        fn: string specifying the reduction function
        default_value: default tensor value to use for the metric state
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a RuntimeError
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: if a float is provided will impute any ``nan`` values with this value

        state_name: name of the metric state
        kwargs: additional keyword arguments for the ``Metric`` base class

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore``, ``disable`` or a float

    """

    is_differentiable = None
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        fn: Union[Callable, str],
        default_value: Union[Tensor, list],
        nan_strategy: Union[str, float] = "error",
        state_name: str = "value",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore", "disable")
        if nan_strategy not in allowed_nan_strategy and not isinstance(nan_strategy, float):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy} but got {nan_strategy}."
            )

        self.nan_strategy = nan_strategy
        self.add_state(state_name, default=default_value, dist_reduce_fx=fn)
        self.state_name = state_name

    def _cast_and_nan_check_input(
        self, x: Union[float, Tensor], weight: Optional[Union[float, Tensor]] = None
    ) -> tuple[Tensor, Tensor]:
        """Convert input ``x`` to a tensor and check for Nans."""
        if not isinstance(x, Tensor):
            x = tensorplay.as_tensor(x, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, Tensor):
            weight = tensorplay.as_tensor(weight, dtype=self.dtype, device=self.device)

        if self.nan_strategy != "disable":
            nans = tensorplay.isnan(x)
            if weight is not None:
                nans_weight = tensorplay.isnan(weight)
            else:
                nans_weight = tensorplay.zeros_like(nans).bool()
                weight = tensorplay.ones_like(x)
            if nans.any() or nans_weight.any():
                if self.nan_strategy == "error":
                    raise RuntimeError("Encountered `nan` values in tensor")
                if self.nan_strategy in ("ignore", "warn"):
                    if self.nan_strategy == "warn":
                        rank_zero_warn("Encountered `nan` values in tensor. Will be removed.", UserWarning)
                    x = x[~(nans | nans_weight)]
                    weight = weight[~(nans | nans_weight)]
                else:
                    if not isinstance(self.nan_strategy, float):
                        raise ValueError(f"`nan_strategy` shall be float but you pass {self.nan_strategy}")
                    x[nans | nans_weight] = self.nan_strategy
                    weight[nans | nans_weight] = 1
        else:
            weight = tensorplay.ones_like(x)
        return x.to(self.dtype), weight.to(self.dtype)

    def update(self, value: Union[float, Tensor]) -> None:
        """Overwrite in child class."""

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return getattr(self, self.state_name)


class MaxMetric(BaseAggregator):
    """Aggregate a stream of values into their maximum value.

    As input to ``forward`` and ``update`` the metric accepts a single float or a tensor of float values with
    arbitrary shape ``(...,)``. As output of ``forward`` and ``compute`` the metric returns a scalar float tensor
    with the aggregated maximum value over all inputs received.

    Args:
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a RuntimeError
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: if a float is provided will impute any ``nan`` values with this value

        kwargs: additional keyword arguments for the ``Metric`` base class

    """

    full_state_update: bool = True
    max_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "max",
            -tensorplay.tensor(float("inf"), dtype=tensorplay.get_default_dtype()),
            nan_strategy,
            state_name="max_value",
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():  # make sure tensor not empty
            self.max_value = tensorplay.max(self.max_value, value.max())


class MinMetric(BaseAggregator):
    """Aggregate a stream of values into their minimum value.

    As input to ``forward`` and ``update`` the metric accepts a single float or a tensor of float values with
    arbitrary shape ``(...,)``. As output of ``forward`` and ``compute`` the metric returns a scalar float tensor
    with the aggregated minimum value over all inputs received.

    Args:
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a RuntimeError
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: if a float is provided will impute any ``nan`` values with this value

        kwargs: additional keyword arguments for the ``Metric`` base class

    """

    full_state_update: bool = True
    min_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "min",
            tensorplay.tensor(float("inf"), dtype=tensorplay.get_default_dtype()),
            nan_strategy,
            state_name="min_value",
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():  # make sure tensor not empty
            self.min_value = tensorplay.min(self.min_value, value.min())


class SumMetric(BaseAggregator):
    """Aggregate a stream of values into their sum.

    As input to ``forward`` and ``update`` the metric accepts a single float or a tensor of float values with
    arbitrary shape ``(...,)``. As output of ``forward`` and ``compute`` the metric returns a scalar float tensor
    with the aggregated sum over all inputs received.

    Args:
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a RuntimeError
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: if a float is provided will impute any ``nan`` values with this value

        kwargs: additional keyword arguments for the ``Metric`` base class

    """

    sum_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            tensorplay.tensor(0.0, dtype=tensorplay.get_default_dtype()),
            nan_strategy,
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.sum_value += value.sum()

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)


class CatMetric(BaseAggregator):
    """Concatenate a stream of values.

    As input to ``forward`` and ``update`` the metric accepts a single float or a tensor of float values with
    arbitrary shape ``(...,)``. As output of ``forward`` and ``compute`` the metric returns a tensor with the
    concatenated values over all inputs received.

    Args:
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a RuntimeError
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: if a float is provided will impute any ``nan`` values with this value

        kwargs: additional keyword arguments for the ``Metric`` base class

    """

    value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__("cat", [], nan_strategy, **kwargs)

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.value.append(value)

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return dim_zero_cat(self.value)
        return self.value


class MeanMetric(BaseAggregator):
    """Aggregate a stream of values into their mean value.

    As input to ``forward`` and ``update`` the metric accepts a single float or a tensor of float values with
    arbitrary shape ``(...,)``, and optionally a weight tensor broadcastable to the value shape. As output of
    ``forward`` and ``compute`` the metric returns a scalar float tensor with the aggregated (weighted) mean over
    all inputs received.

    Args:
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a RuntimeError
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: if a float is provided will impute any ``nan`` values with this value

        kwargs: additional keyword arguments for the ``Metric`` base class

    """

    mean_value: Tensor
    weight: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            tensorplay.tensor(0.0, dtype=tensorplay.get_default_dtype()),
            nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state("weight", default=tensorplay.tensor(0.0, dtype=tensorplay.get_default_dtype()), dist_reduce_fx="sum")

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor, None] = None) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of ``value``. Default to ``None`` corresponding to simple
                harmonic average.

        """
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = tensorplay.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is None:
            weight = tensorplay.ones_like(value)
        elif not isinstance(weight, Tensor):
            weight = tensorplay.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = tensorplay.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        if value.numel() == 0:
            return
        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)