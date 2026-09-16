"""A quantized linear transformation fused with a rectifier."""

from __future__ import annotations

from ... import LinearReLU as _FloatLinearReLU
from ....quantized.linear import QuantizedLinear

__all__ = ["LinearReLU"]


class LinearReLU(QuantizedLinear):
    """A LinearReLU module fused from Linear and ReLU modules.

    Interface is identical to :class:`tensorplay.ao.nn.quantized.Linear`;
    the output is the rectified linear transformation.
    """

    _FLOAT_MODULE = _FloatLinearReLU

    def forward(self, x):
        return QuantizedLinear.forward(self, x).relu()

    def _get_name(self):
        return "QuantizedLinearReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        base = getattr(mod, "op", mod)
        return super().from_float(base, None, None)
