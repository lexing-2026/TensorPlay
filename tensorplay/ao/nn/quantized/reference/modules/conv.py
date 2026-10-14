"""Reference quantized convolution modules with float weights."""

from typing import Any

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from .utils import ReferenceQuantizedModule

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


def _pair(value, rank):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * rank


class _ConvNd(ReferenceQuantizedModule):
    _RANK = None
    _TRANSPOSE = False

    def _init_conv(self, in_channels, out_channels, kernel_size, stride=1,
                   padding=0, dilation=1, groups=1, bias=True,
                   padding_mode="zeros", device=None, dtype=None,
                   weight_qparams=None):
        nn.Module.__init__(self)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size, self._RANK)
        self.stride = _pair(stride, self._RANK)
        self.padding = _pair(padding, self._RANK)
        self.dilation = _pair(dilation, self._RANK)
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = tuple(
            p for p in reversed(self.padding) for _ in range(2)
        )
        self.weight = nn.Parameter(
            tp.empty(
                (out_channels, in_channels // groups) + self.kernel_size
                if not self._TRANSPOSE
                else (in_channels, out_channels // groups) + self.kernel_size,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias = nn.Parameter(tp.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self._init_weight_qparams(weight_qparams, device)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.kernel_size[0]
            for k in self.kernel_size[1:]:
                fan_in *= k
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_name(self):
        return f"Quantized{type(self).__name__}(Reference)"

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        weight_quant_dequant = self.get_weight()
        conv_fn = getattr(F, f"conv{self._RANK}d")
        if self._TRANSPOSE:
            return conv_fn(
                x,
                weight_quant_dequant,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
        return conv_fn(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        args = (
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,
            float_conv.stride,
            float_conv.padding,
            float_conv.dilation,
            float_conv.groups,
            float_conv.bias is not None,
            float_conv.padding_mode,
        )
        kwargs = {
            "device": float_conv.weight.device,
            "dtype": float_conv.weight.dtype,
            "weight_qparams": weight_qparams,
        }
        if cls._TRANSPOSE:
            args = args + (float_conv.output_padding,)
        qref_conv = cls(*args, **kwargs)
        qref_conv.weight = nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = nn.Parameter(float_conv.bias.detach())
        return qref_conv


class Conv1d(_ConvNd, nn.Conv1d):
    _RANK = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None, weight_qparams: dict[str, Any] | None = None):
        self._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                        dilation, groups, bias, padding_mode, device, dtype,
                        weight_qparams)


class Conv2d(_ConvNd, nn.Conv2d):
    _RANK = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None, weight_qparams: dict[str, Any] | None = None):
        self._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                        dilation, groups, bias, padding_mode, device, dtype,
                        weight_qparams)


class Conv3d(_ConvNd, nn.Conv3d):
    _RANK = 3

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None, weight_qparams: dict[str, Any] | None = None):
        self._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                        dilation, groups, bias, padding_mode, device, dtype,
                        weight_qparams)


class _ConvTransposeNd(_ConvNd):
    _TRANSPOSE = True

    def _init_conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   output_padding=0, dilation=1, groups=1, bias=True,
                   padding_mode="zeros", device=None, dtype=None, weight_qparams=None):
        self.output_padding = _pair(output_padding, self._RANK)
        super()._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                           dilation, groups, bias, padding_mode, device, dtype,
                           weight_qparams)


class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    _RANK = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", device=None, dtype=None,
                 weight_qparams: dict[str, Any] | None = None):
        self._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                        output_padding, dilation, groups, bias, padding_mode,
                        device, dtype, weight_qparams)


class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    _RANK = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", device=None, dtype=None,
                 weight_qparams: dict[str, Any] | None = None):
        self._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                        output_padding, dilation, groups, bias, padding_mode,
                        device, dtype, weight_qparams)


class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    _RANK = 3

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", device=None, dtype=None,
                 weight_qparams: dict[str, Any] | None = None):
        self._init_conv(in_channels, out_channels, kernel_size, stride, padding,
                        output_padding, dilation, groups, bias, padding_mode,
                        device, dtype, weight_qparams)