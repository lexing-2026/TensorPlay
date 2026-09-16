"""Quantized batch norms fused with a rectifier.

The base batch norm emits a quantized output; the rectifier runs in the
integer domain and keeps the same affine parameters.
"""

from __future__ import annotations

from ... import BNReLU2d as _FloatBNReLU2d
from ... import BNReLU3d as _FloatBNReLU3d
from ....quantized.batchnorm import BatchNorm2d, BatchNorm3d

__all__ = ["BNReLU2d", "BNReLU3d"]


class BNReLU2d(BatchNorm2d):
    """A BNReLU2d module fused from BatchNorm2d and ReLU modules."""

    _FLOAT_MODULE = _FloatBNReLU2d

    def forward(self, input):
        out = BatchNorm2d.forward(self, input)
        return out.quantized_relu() if out.is_quantized() else out.relu()

    def _get_name(self):
        return "QuantizedBNReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(getattr(mod, "bn", mod))


class BNReLU3d(BatchNorm3d):
    """A BNReLU3d module fused from BatchNorm3d and ReLU modules."""

    _FLOAT_MODULE = _FloatBNReLU3d

    def forward(self, input):
        out = BatchNorm3d.forward(self, input)
        return out.quantized_relu() if out.is_quantized() else out.relu()

    def _get_name(self):
        return "QuantizedBNReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(getattr(mod, "bn", mod))
