"""Quantized linear transformations fused with an activation."""

from __future__ import annotations

import tensorplay as tp

from ... import LinearReLU as _FloatLinearReLU
from ... import LinearLeakyReLU as _FloatLinearLeakyReLU
from ... import LinearTanh as _FloatLinearTanh
from ....quantized.linear import QuantizedLinear
from ....quantized.packed import quantize_module_weight, conversion_source

__all__ = ["LinearReLU", "LinearLeakyReLU", "LinearTanh"]


class LinearReLU(QuantizedLinear):
    """A LinearReLU module fused from Linear and ReLU modules."""

    _FLOAT_MODULE = _FloatLinearReLU
    _FUSED_RELU = True

    def _get_name(self):
        return "QuantizedLinearReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)


class LinearLeakyReLU(QuantizedLinear):
    """A LinearLeakyReLU module fused from Linear and LeakyReLU modules."""

    _FLOAT_MODULE = _FloatLinearLeakyReLU

    def __init__(self, in_features, out_features, negative_slope, bias=True,
                 dtype=tp.qint8):
        bias_tensor = tp.zeros(out_features, dtype=tp.float32) if bias else None
        super().__init__(in_features, out_features, bias=bias_tensor)
        self.negative_slope = negative_slope

    def forward(self, x):
        if x.dtype == tp.int8:
            x = tp._C._make_per_tensor_quantized_tensor(x, self.input_scale, self.input_zero_point)
        if x.dtype not in (tp.qint8, tp.quint8):
            raise TypeError('Quantized linear requires signed or unsigned 8-bit quantized input')
        return self._packed_params.run(x, self.out_scale, self.out_zero_point,
                                       activation=1, negative_slope=self.negative_slope)

    def _get_name(self):
        return "QuantizedLinearLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        source = conversion_source(mod[0])
        if not hasattr(mod, 'activation_post_process'):
            raise ValueError('Linear conversion requires output calibration or explicit parameters')
        out_scale, out_zero_point = mod.activation_post_process.calculate_qparams()
        qweight = quantize_module_weight(source, use_precomputed_fake_quant, weight=source.weight)
        result = cls(source.in_features, source.out_features, mod[1].negative_slope,
                     bias=source.bias is not None)
        result.set_weight_bias(qweight, source.bias.detach() if source.bias is not None else None)
        result.out_scale = float(out_scale)
        result.out_zero_point = int(out_zero_point)
        result.train(mod.training)
        return result


class LinearTanh(QuantizedLinear):
    """A LinearTanh module fused from Linear and Tanh modules."""

    _FLOAT_MODULE = _FloatLinearTanh

    def __init__(self, in_features, out_features, bias=True, dtype=tp.qint8):
        bias_tensor = tp.zeros(out_features, dtype=tp.float32) if bias else None
        super().__init__(in_features, out_features, bias=bias_tensor)

    def forward(self, x):
        if x.dtype == tp.int8:
            x = tp._C._make_per_tensor_quantized_tensor(x, self.input_scale, self.input_zero_point)
        if x.dtype not in (tp.qint8, tp.quint8):
            raise TypeError('Quantized linear requires signed or unsigned 8-bit quantized input')
        return self._packed_params.run(x, self.out_scale, self.out_zero_point, activation=2)

    def _get_name(self):
        return "QuantizedLinearTanh"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        source = conversion_source(mod[0])
        if not hasattr(mod, 'activation_post_process'):
            raise ValueError('Linear conversion requires output calibration or explicit parameters')
        out_scale, out_zero_point = mod.activation_post_process.calculate_qparams()
        qweight = quantize_module_weight(source, use_precomputed_fake_quant, weight=source.weight)
        result = cls(source.in_features, source.out_features, bias=source.bias is not None)
        result.set_weight_bias(qweight, source.bias.detach() if source.bias is not None else None)
        result.out_scale = float(out_scale)
        result.out_zero_point = int(out_zero_point)
        result.train(mod.training)
        return result