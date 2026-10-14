"""Peak signal-to-noise ratio metric."""

from collections.abc import Sequence
from functools import partial
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.image.psnr import _psnr_compute, _psnr_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities import rank_zero_warn
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class PeakSignalNoiseRatio(Metric):
    """Compute the peak signal-to-noise ratio (PSNR).

    PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting
    noise, expressed in a logarithmic scale.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predictions from model of shape ``(N, C, H, W)``
    - ``target`` (float tensor): ground truth values of shape ``(N, C, H, W)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``psnr`` (float tensor): if ``reduction!='none'`` returns a float scalar tensor with average PSNR value
      over sample else returns tensor of shape ``(N,)`` with PSNR values per sample

    Args:
        data_range: the range of the data. If a tuple is provided, then the range is calculated as the
            difference and input is clamped between the values.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        dim: dimensions to reduce PSNR scores over, provided as either an integer or a list of integers.
            Default is ``None`` meaning scores will be reduced across all dimensions and all batches.

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    data_range: Tensor

    def __init__(
        self,
        data_range: Union[float, tuple[float, float]],
        base: float = 10.0,
        reduction: str = "elementwise_mean",
        dim: Optional[Union[int, tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

        if dim is None:
            self.add_state("sum_squared_error", default=tp.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
            self.add_state("total", default=[], dist_reduce_fx="cat")

        self.clamping_fn = None
        if isinstance(data_range, tuple):
            self.add_state("data_range", default=tp.tensor(data_range[1] - data_range[0]), dist_reduce_fx="mean")
            self.clamping_fn = partial(tp.clamp, min=data_range[0], max=data_range[1])
        else:
            self.add_state("data_range", default=tp.tensor(float(data_range)), dist_reduce_fx="mean")

        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.clamping_fn is not None:
            preds = self.clamping_fn(preds)
            target = self.clamping_fn(target)

        sum_squared_error, num_obs = _psnr_update(preds, target, dim=self.dim)
        if self.dim is None:
            if not isinstance(self.sum_squared_error, Tensor):
                raise TypeError(
                    f"Expected `self.sum_squared_error` to be a Tensor, but got {type(self.sum_squared_error)}"
                )
            if not isinstance(self.total, Tensor):
                raise TypeError(f"Expected `self.total` to be a Tensor, but got {type(self.total)}")

            self.sum_squared_error += sum_squared_error
            self.total += num_obs
        else:
            if not isinstance(self.sum_squared_error, list):
                raise TypeError(
                    f"Expected `self.sum_squared_error` to be a list, but got {type(self.sum_squared_error)}"
                )
            if not isinstance(self.total, list):
                raise TypeError(f"Expected `self.total` to be a list, but got {type(self.total)}")
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(num_obs)

    def compute(self) -> Tensor:
        """Compute the peak signal-to-noise ratio over state."""
        if isinstance(self.sum_squared_error, Tensor):
            sum_squared_error = self.sum_squared_error
        elif isinstance(self.sum_squared_error, list):
            sum_squared_error = tp.cat([value.flatten() for value in self.sum_squared_error])
        else:
            raise TypeError("Expected sum_squared_error to be a Tensor or a list of Tensors")

        if isinstance(self.total, Tensor):
            total = self.total
        elif isinstance(self.total, list):
            total = tp.cat([value.flatten() for value in self.total])
        else:
            raise TypeError("Expected total to be a Tensor or a list of Tensors")

        return _psnr_compute(sum_squared_error, total, self.data_range, base=self.base, reduction=self.reduction)

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