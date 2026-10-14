"""Signal-to-noise ratio metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.audio.snr import (
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = ["SignalNoiseRatio", "ScaleInvariantSignalNoiseRatio"]


class SignalNoiseRatio(Metric):
    """Calculate the signal-to-noise ratio (SNR).

    Args:
        zero_mean: whether to subtract the mean of each signal first.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    sum_snr: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, zero_mean: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.zero_mean = zero_mean
        self.add_state("sum_snr", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        snr_batch = signal_noise_ratio(preds, target, self.zero_mean)

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_snr / self.total

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


class ScaleInvariantSignalNoiseRatio(Metric):
    """Calculate the scale-invariant signal-to-noise ratio (SI-SNR).

    Args:
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    sum_snr: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_snr", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        snr_batch = scale_invariant_signal_noise_ratio(preds, target)

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_snr / self.total

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