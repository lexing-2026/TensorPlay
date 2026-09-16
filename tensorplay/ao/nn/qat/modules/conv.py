"""Quantization-aware convolution modules.

Same contract as the linear QAT module: float weights stay trainable while a
straight-through fake-quantize module simulates the quantized grid on every
forward pass.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay.nn import functional as F

from ....quantization.fake_quantize import PerChannelFakeQuantize

__all__ = ["Conv1d", "Conv2d", "Conv3d"]


class _ConvNd(nn.Module):
    _BASE = None
    _FUNC = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, qconfig=None):
        super().__init__()
        self.base = self._BASE(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight_fake_quant = (
            PerChannelFakeQuantize(ch_axis=0) if qconfig is None else qconfig.weight())
        self.qconfig = qconfig

    @property
    def weight(self):
        return self.base.weight

    @weight.setter
    def weight(self, value):
        self.base.weight = value

    @property
    def bias(self):
        return self.base.bias

    @bias.setter
    def bias(self, value):
        self.base.bias = value

    def forward(self, input):
        return self._FUNC(
            input, self.weight_fake_quant(self.weight), self.bias,
            stride=self.base.stride, padding=self.base.padding,
            dilation=self.base.dilation, groups=self.base.groups)

    def _get_name(self):
        return f"QAT{self._BASE.__name__}"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if not isinstance(mod, cls._BASE):
            raise TypeError(
                f"from_float(): expected a {cls._BASE.__name__} module")
        qconfig = getattr(mod, "qconfig", None)
        qat_conv = cls(
            mod.in_channels, mod.out_channels, mod.kernel_size,
            stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
            groups=mod.groups, bias=mod.bias is not None, qconfig=qconfig)
        qat_conv.weight = mod.weight
        if mod.bias is not None:
            qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        float_conv = self._BASE(
            self.base.in_channels, self.base.out_channels,
            self.base.kernel_size, stride=self.base.stride,
            padding=self.base.padding, dilation=self.base.dilation,
            groups=self.base.groups, bias=self.base.bias is not None)
        float_conv.weight = nn.Parameter(self.weight.detach())
        if self.base.bias is not None:
            float_conv.bias = nn.Parameter(self.bias.detach())
        float_conv.train(self.training)
        return float_conv


class Conv1d(_ConvNd):
    """Quantization-aware 1d convolution."""

    _BASE = nn.Conv1d
    _FUNC = staticmethod(F.conv1d)


class Conv2d(_ConvNd):
    """Quantization-aware 2d convolution."""

    _BASE = nn.Conv2d
    _FUNC = staticmethod(F.conv2d)


class Conv3d(_ConvNd):
    """Quantization-aware 3d convolution."""

    _BASE = nn.Conv3d
    _FUNC = staticmethod(F.conv3d)
