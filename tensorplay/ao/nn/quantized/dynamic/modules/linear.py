"""Dynamically quantized linear module.

Weights are quantized per output channel at conversion time; activations are
quantized per row at inference from their own observed range inside the
fused kernel, so no calibration pass is required. The output stays in the
float domain.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay._C import (
    quantize_per_channel as _quantize_per_channel,
    quantized_linear_dynamic as _quantized_linear_dynamic,
)

__all__ = ["Linear"]


class Linear(nn.Module):
    """A linear module that quantizes its activations at runtime.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias_: whether to learn an additive bias
        dtype: quantized weight dtype; only ``tensorplay.qint8`` is supported

    Attributes:
        weight: the quantized weight of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`
        bias: the float bias of shape :math:`(\\text{out\\_features})`
    """

    def __init__(self, in_features, out_features, bias_=True, dtype=tensorplay.qint8):
        super().__init__()
        if dtype != tensorplay.qint8:
            raise ValueError("dynamic Linear: only qint8 weights are supported")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer(
            "weight",
            tensorplay.zeros((self.out_features, self.in_features),
                             dtype=tensorplay.float32))
        self.register_buffer("weight_scales",
                             tensorplay.ones(self.out_features,
                                             dtype=tensorplay.float32))
        self.register_buffer("weight_zero_points",
                             tensorplay.zeros(self.out_features,
                                              dtype=tensorplay.int64))
        if bias_:
            self.register_buffer(
                "bias", tensorplay.zeros(self.out_features,
                                         dtype=tensorplay.float32))
        else:
            self.bias = None

    def forward(self, x):
        if x.is_quantized():
            x = x.dequantize()
        return _quantized_linear_dynamic(
            x, self.weight,
            weight_scales=self.weight_scales,
            weight_zero_points=self.weight_zero_points,
            bias=self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None}")

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        """Quantize a float Linear's weights per output channel."""
        if not isinstance(mod, nn.Linear):
            raise TypeError("from_float(): expected a Linear module")
        weight = mod.weight.detach()
        out_features, _ = weight.shape
        min_vals, max_vals = tensorplay.aminmax(weight, dim=list(range(1, weight.dim())), keepdim=False)
        from tensorplay.ao.quantization.observer import ObserverBase

        scales, zero_points = [], []
        for n in range(out_features):
            s, z = ObserverBase._calculate_qparams(float(min_vals[n]),
                                                   float(max_vals[n]))
            scales.append(s)
            zero_points.append(z)
        scales_t = tensorplay.as_tensor(scales, dtype=tensorplay.float32).to(weight.device)
        zero_points_t = tensorplay.as_tensor(zero_points,
                                             dtype=tensorplay.int64).to(weight.device)
        qweight = _quantize_per_channel(
            self=weight, scales=scales_t, zero_points=zero_points_t, axis=0,
            dtype=tensorplay.qint8)
        qlinear = cls(mod.in_features, mod.out_features,
                      bias_=mod.bias is not None)
        qlinear.weight = qweight
        qlinear.weight_scales = scales_t
        qlinear.weight_zero_points = zero_points_t
        if mod.bias is not None:
            qlinear.bias = mod.bias.detach().to(tensorplay.float32)
        return qlinear
