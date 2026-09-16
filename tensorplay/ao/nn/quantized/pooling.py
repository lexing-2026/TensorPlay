"""Quantized pooling modules.

The window maximum is order-preserving on the affine grid, so the output
inherits the input scale and zero point.
"""

from __future__ import annotations

from tensorplay import nn
from tensorplay._C import (
    quantized_max_pool1d as _quantized_max_pool1d,
    quantized_max_pool2d as _quantized_max_pool2d,
    quantized_max_pool3d as _quantized_max_pool3d,
)

__all__ = ["MaxPool1d", "MaxPool2d", "MaxPool3d"]


def _pair_or_more(x, rank):
    if isinstance(x, (tuple, list)):
        if len(x) != rank:
            raise ValueError(f"expected {rank} values, got {len(x)}")
        return tuple(int(v) for v in x)
    return (int(x),) * rank


class _MaxPoolNd(nn.Module):
    _RANK = None
    _KERNEL = None

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = _pair_or_more(kernel_size, self._RANK)
        self.stride = _pair_or_more(stride if stride is not None else kernel_size,
                                    self._RANK)
        self.padding = _pair_or_more(padding, self._RANK)
        self.dilation = _pair_or_more(dilation, self._RANK)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        if return_indices:
            raise ValueError(
                f"{type(self).__name__}: return_indices is not supported")

    def forward(self, input):
        return self._KERNEL(
            input, list(self.kernel_size), list(self.stride),
            list(self.padding), list(self.dilation), self.ceil_mode)

    def extra_repr(self):
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"ceil_mode={self.ceil_mode}")


class MaxPool1d(_MaxPoolNd):
    """Quantized 1d max pooling."""

    _RANK = 1
    _KERNEL = staticmethod(_quantized_max_pool1d)

    def _get_name(self):
        return "QuantizedMaxPool1d"


class MaxPool2d(_MaxPoolNd):
    """Quantized 2d max pooling."""

    _RANK = 2
    _KERNEL = staticmethod(_quantized_max_pool2d)

    def _get_name(self):
        return "QuantizedMaxPool2d"


class MaxPool3d(_MaxPoolNd):
    """Quantized 3d max pooling."""

    _RANK = 3
    _KERNEL = staticmethod(_quantized_max_pool3d)

    def _get_name(self):
        return "QuantizedMaxPool3d"
