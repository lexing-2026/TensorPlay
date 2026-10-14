"""Quantized 2d convolution with a residual add."""

from __future__ import annotations

import tensorplay as tp

from ... import ConvAdd2d as _FloatConvAdd2d
from ... import ConvAddReLU2d as _FloatConvAddReLU2d
from ....quantized.conv import Conv2d

__all__ = ["ConvAdd2d", "ConvAddReLU2d"]


class ConvAdd2d(Conv2d):
    """A fused quantized Conv2d + add module."""

    _FLOAT_MODULE = _FloatConvAdd2d

    def forward(self, input, extra_input):
        if input.dim() != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if input.dtype == tp.int8:
            input = tp._C._make_per_tensor_quantized_tensor(input, self.input_scale, self.input_zero_point)
        if input.dtype not in (tp.qint8, tp.quint8):
            raise TypeError('Quantized convolution requires a quantized input')
        padding = self.padding
        explicit = self._explicit_padding
        if self.padding_mode != 'zeros' and explicit is None:
            explicit = tuple(value for p in reversed(padding) for value in (p, p))
        if explicit is not None:
            mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
            codes = tp.nn.functional.pad(input.int_repr(), explicit, mode=mode,
                                         value=input.q_zero_point() if mode == 'constant' else None)
            input = tp._C._make_per_tensor_quantized_tensor(codes, input.q_scale(), input.q_zero_point())
            padding = (0,)
        return self._packed_params.run(input, self.out_scale, self.out_zero_point,
                                      self.stride, padding, self.dilation,
                                      self.groups, residual=extra_input)

    def _get_name(self):
        return "QuantizedConvAdd2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)


class ConvAddReLU2d(Conv2d):
    """A fused quantized Conv2d + add + ReLU module."""

    _FLOAT_MODULE = _FloatConvAddReLU2d
    _FUSED_RELU = True

    def forward(self, input, extra_input):
        if input.dim() != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if input.dtype == tp.int8:
            input = tp._C._make_per_tensor_quantized_tensor(input, self.input_scale, self.input_zero_point)
        if input.dtype not in (tp.qint8, tp.quint8):
            raise TypeError('Quantized convolution requires a quantized input')
        padding = self.padding
        explicit = self._explicit_padding
        if self.padding_mode != 'zeros' and explicit is None:
            explicit = tuple(value for p in reversed(padding) for value in (p, p))
        if explicit is not None:
            mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
            codes = tp.nn.functional.pad(input.int_repr(), explicit, mode=mode,
                                         value=input.q_zero_point() if mode == 'constant' else None)
            input = tp._C._make_per_tensor_quantized_tensor(codes, input.q_scale(), input.q_zero_point())
            padding = (0,)
        return self._packed_params.run(input, self.out_scale, self.out_zero_point,
                                       self.stride, padding, self.dilation,
                                       self.groups, relu=True, residual=extra_input)

    def _get_name(self):
        return "QuantizedConvAddReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)