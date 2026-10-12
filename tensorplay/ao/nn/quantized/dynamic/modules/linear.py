"""Dynamically quantized linear module.

Weights use the configured tensor or output-channel grid. Activations use
one affine grid over the complete input at each call. The output stays in
the float domain.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn

__all__ = ["Linear"]


class Linear(nn.Module):
    _FUSED_RELU = False
    def __init__(self, in_features, out_features, bias_=True, dtype=tensorplay.qint8):
        super().__init__()
        if dtype != tensorplay.qint8:
            raise ValueError('Dynamic linear weights require signed 8-bit storage')
        from ...packed import PackedWeight
        self.in_features, self.out_features = int(in_features), int(out_features)
        weight = tensorplay._C.quantize_per_tensor(tensorplay.zeros(out_features, in_features), 1., 0, dtype)
        self._packed_params = PackedWeight(weight, tensorplay.zeros(out_features) if bias_ else None)
        self.reduce_range = True

    @property
    def weight(self):
        return self._packed_params.weight

    @property
    def bias(self):
        return self._packed_params.bias

    @property
    def weight_scales(self):
        return self._packed_params.scales

    @property
    def weight_zero_points(self):
        return self._packed_params.zero_points

    def set_weight_bias(self, weight, bias):
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError('Dynamic linear weight dimensions do not match the module')
        self._packed_params.set_weight_bias(weight, bias)

    def forward(self, x):
        if x.is_quantized():
            x = x.dequantize()
        quantized = (tensorplay._C.quantize_per_tensor_dynamic(x, tensorplay.quint8, self.reduce_range)
                     if x.numel() else tensorplay._C.quantize_per_tensor(x, .1, 0, tensorplay.quint8))
        return self._packed_params.run(quantized, relu=self._FUSED_RELU, output_float=True).to(x.dtype)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        from ...packed import quantize_module_weight, conversion_source
        source = conversion_source(mod)
        if not isinstance(source, nn.Linear):
            raise TypeError('Dynamic linear conversion requires a floating linear module')
        result = cls(source.in_features, source.out_features, source.bias is not None)
        weight = quantize_module_weight(source, use_precomputed_fake_quant)
        result.set_weight_bias(weight, source.bias.detach() if source.bias is not None else None)
        return result

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
