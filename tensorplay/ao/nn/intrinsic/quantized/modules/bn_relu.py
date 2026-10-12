"""Quantized batch norms fused with a rectifier.

Normalization, output quantization and rectification share one native kernel.
"""

from __future__ import annotations

import tensorplay as tp

from ... import BNReLU2d as _FloatBNReLU2d
from ... import BNReLU3d as _FloatBNReLU3d
from ....quantized.batchnorm import BatchNorm2d, BatchNorm3d

__all__ = ["BNReLU2d", "BNReLU3d"]


class BNReLU2d(BatchNorm2d):
    """A BNReLU2d module fused from BatchNorm2d and ReLU modules."""

    _FLOAT_MODULE = _FloatBNReLU2d

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError('Expected a four dimensional input')
        return tp._C.quantized_batch_norm_relu(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, float(self.scale), int(self.zero_point))

    def _get_name(self):
        return "QuantizedBNReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        base = getattr(mod, 'bn', mod)
        if hasattr(mod, 'activation_post_process'):
            base.activation_post_process = mod.activation_post_process
        return super().from_float(base)


class BNReLU3d(BatchNorm3d):
    """A BNReLU3d module fused from BatchNorm3d and ReLU modules."""

    _FLOAT_MODULE = _FloatBNReLU3d

    def forward(self, input):
        if input.dim() != 5:
            raise ValueError('Expected a five dimensional input')
        return tp._C.quantized_batch_norm_relu(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, float(self.scale), int(self.zero_point))

    def _get_name(self):
        return "QuantizedBNReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        base = getattr(mod, 'bn', mod)
        if hasattr(mod, 'activation_post_process'):
            base.activation_post_process = mod.activation_post_process
        return super().from_float(base)
