"""Signal-to-distortion ratio metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.audio.sdr import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
    source_aggregated_signal_distortion_ratio,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = [
    "SignalDistortionRatio",
    "ScaleInvariantSignalDistortionRatio",
    "SourceAggregatedSignalDistortionRatio",
]


class SignalDistortionRatio(Metric):
    """Calculate the signal-to-distortion ratio (SDR).

    Args:
        use_cg_iter: if provided, conjugate gradient descent is used to solve
            for the distortion filter coefficients (requires ``fast-bss-eval``).
        filter_length: length of the distortion filter.
        zero_mean: whether to subtract the mean of all signals first.
        load_diag: if provided, this small value is added to the diagonal
            coefficients of the system matrix when solving for the filter
            coefficients.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    sum_sdr: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(
        self,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.use_cg_iter = use_cg_iter
        self.filter_length = filter_length
        self.zero_mean = zero_mean
        self.load_diag = load_diag

        self.add_state("sum_sdr", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sdr_batch = signal_distortion_ratio(
            preds, target, self.use_cg_iter, self.filter_length, self.zero_mean, self.load_diag
        )

        self.sum_sdr += sdr_batch.sum()
        self.total += sdr_batch.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_sdr / self.total

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or
                ``metric.compute`` or a list of these results. If no value is
                provided, ``metric.compute`` is called and that result plotted.
            ax: a matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object
        """
        return self._plot(val, ax)


class ScaleInvariantSignalDistortionRatio(Metric):
    """Calculate the scale-invariant signal-to-distortion ratio (SI-SDR).

    Args:
        zero_mean: whether to subtract the mean of each signal first.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    sum_sdr: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, zero_mean: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.zero_mean = zero_mean
        self.add_state("sum_sdr", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sdr_batch = scale_invariant_signal_distortion_ratio(preds, target, self.zero_mean)

        self.sum_sdr += sdr_batch.sum()
        self.total += sdr_batch.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_sdr / self.total

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or
                ``metric.compute`` or a list of these results. If no value is
                provided, ``metric.compute`` is called and that result plotted.
            ax: a matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object
        """
        return self._plot(val, ax)


class SourceAggregatedSignalDistortionRatio(Metric):
    """Calculate the source-aggregated signal-to-distortion ratio (SA-SDR).

    Args:
        scale_invariant: whether to scale the targets of different speakers
            with the same factor.
        zero_mean: whether to subtract the mean of each signal first.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    sum_sdr: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, scale_invariant: bool = True, zero_mean: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.scale_invariant = scale_invariant
        self.zero_mean = zero_mean
        self.add_state("sum_sdr", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sdr_batch = source_aggregated_signal_distortion_ratio(preds, target, self.scale_invariant, self.zero_mean)

        self.sum_sdr += sdr_batch.sum()
        self.total += sdr_batch.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_sdr / self.total

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or
                ``metric.compute`` or a list of these results. If no value is
                provided, ``metric.compute`` is called and that result plotted.
            ax: a matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object
        """
        return self._plot(val, ax)