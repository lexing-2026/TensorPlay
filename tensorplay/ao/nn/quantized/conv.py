"""Quantized convolution modules.

Weights are stored as per-tensor affine QInt8 buffers; activation qparams
arrive from the incoming quantized tensor (or from module attributes for raw
code tensors), and the output activation qparams are declared on the module.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay._C import (
    quantize_per_tensor as _quantize_per_tensor,
    quantized_conv1d as _quantized_conv1d,
    quantized_conv2d as _quantized_conv2d,
    quantized_conv3d as _quantized_conv3d,
)

__all__ = ["Conv1d", "Conv2d", "Conv3d"]


def _pair(x, rank):
    if isinstance(x, (tuple, list)):
        if len(x) != rank:
            raise ValueError(f"expected {rank} values, got {len(x)}")
        return tuple(int(v) for v in x)
    return (int(x),) * rank


class _ConvNd(nn.Module):
    _RANK = None
    _KERNEL = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 weight_scale=1.0, weight_zero_point=0,
                 input_scale=None, input_zero_point=None,
                 out_scale=1.0, out_zero_point=0):
        super().__init__()
        if groups != 1:
            raise ValueError(
                f"{type(self).__name__}: the native kernel supports groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size, self._RANK)
        self.stride = _pair(stride, self._RANK)
        self.padding = _pair(padding, self._RANK)
        self.dilation = _pair(dilation, self._RANK)
        self.groups = int(groups)
        self.input_scale = input_scale
        self.input_zero_point = input_zero_point
        self.out_scale = float(out_scale)
        self.out_zero_point = int(out_zero_point)
        empty = (0,) * self._RANK
        shape = (self.out_channels, self.in_channels) + self.kernel_size
        qweight = _quantize_per_tensor(
            self=tensorplay.zeros(shape, dtype=tensorplay.float32),
            scale=float(weight_scale),
            zero_point=int(weight_zero_point),
            dtype=tensorplay.qint8)
        self.register_buffer("weight", qweight)
        if bias:
            self.register_buffer(
                "bias", tensorplay.zeros(self.out_channels, dtype=tensorplay.float32))
        else:
            self.bias = None

    @property
    def weight_scale(self):
        return self.weight.q_scale()

    @property
    def weight_zero_point(self):
        return self.weight.q_zero_point()

    def _resolve_input_qparams(self, x):
        if x.is_quantized():
            return x.q_scale(), x.q_zero_point()
        if x.dtype == tensorplay.int8:
            if self.input_scale is None:
                raise TypeError(
                    f"{type(self).__name__}: a raw Int8 input requires "
                    "input_scale and input_zero_point on the module")
            return float(self.input_scale), int(self.input_zero_point)
        raise TypeError(
            f"{type(self).__name__} expects a quantized activation tensor; "
            "run it through QuantStub (or quantize_per_tensor) first")

    def forward(self, x):
        input_scale, input_zero_point = self._resolve_input_qparams(x)
        return self._KERNEL(
            x, self.weight, self.bias,
            input_scale=input_scale, input_zero_point=input_zero_point,
            weight_scale=self.weight_scale,
            weight_zero_point=self.weight_zero_point,
            out_scale=self.out_scale, out_zero_point=self.out_zero_point,
            stride=list(self.stride), padding=list(self.padding),
            dilation=list(self.dilation), groups=self.groups)

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"groups={self.groups}, out_scale={self.out_scale}, "
                f"out_zero_point={self.out_zero_point}")

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        """Quantize a calibrated float convolution module.

        The weight is quantized per tensor from the module's weight observer
        (or the module's input activation observer supplies the raw weight
        range when no weight observer is attached).  Output activation
        qparams come from the module's output observer; input qparams come
        from its input observer and are stored for raw-code inputs.
        """
        if type(mod) not in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
            raise TypeError(
                f"{cls.__name__}.from_float(): expected a float Conv module, "
                f"got {type(mod).__name__}")
        weight = mod.weight.detach()
        min_vals, max_vals = tensorplay.aminmax(weight)
        s, z = _range_qparams(float(min_vals), float(max_vals))
        qweight = _quantize_per_tensor(
            self=weight, scale=s, zero_point=z, dtype=tensorplay.qint8)
        input_scale = input_zero_point = None
        input_observer = getattr(mod, "input_activation_post_process", None)
        if input_observer is not None:
            input_scale, input_zero_point = input_observer.calculate_qparams()
        out_scale, out_zero_point = 1.0, 0
        output_observer = getattr(mod, "activation_post_process", None)
        if output_observer is not None:
            out_scale, out_zero_point = output_observer.calculate_qparams()
            out_scale, out_zero_point = float(out_scale), int(out_zero_point)
        qconv = cls(
            mod.in_channels, mod.out_channels, mod.kernel_size,
            stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
            groups=mod.groups, bias=mod.bias is not None,
            input_scale=float(input_scale) if input_scale is not None else None,
            input_zero_point=int(input_zero_point) if input_zero_point is not None else None,
            out_scale=out_scale, out_zero_point=out_zero_point)
        qconv.weight = qweight.to(weight.device)
        if mod.bias is not None:
            qconv.bias = mod.bias.detach().to(tensorplay.float32)
        return qconv


def _range_qparams(min_val, max_val):
    """Derive affine qparams covering [min_val, max_val] on the Int8 grid."""
    from ...quantization.observer import ObserverBase

    return ObserverBase._calculate_qparams(min_val, max_val)


class Conv1d(_ConvNd):
    """Quantized 1d convolution over a quantized input.

    out[o] = requantize( conv(dequant(x), dequant(w)) + bias[o] )
    """

    _RANK = 1
    _KERNEL = staticmethod(_quantized_conv1d)

    def _get_name(self):
        return "QuantizedConv1d"


class Conv2d(_ConvNd):
    """Quantized 2d convolution over a quantized input.

    out[o] = requantize( conv(dequant(x), dequant(w)) + bias[o] )
    """

    _RANK = 2
    _KERNEL = staticmethod(_quantized_conv2d)

    def _get_name(self):
        return "QuantizedConv2d"


class Conv3d(_ConvNd):
    """Quantized 3d convolution over a quantized input.

    out[o] = requantize( conv(dequant(x), dequant(w)) + bias[o] )
    """

    _RANK = 3
    _KERNEL = staticmethod(_quantized_conv3d)

    def _get_name(self):
        return "QuantizedConv3d"
