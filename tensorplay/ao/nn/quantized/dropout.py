"""Quantized dropout placeholder.

The operation is a no-op on quantized tensors: the affine grid is already
discrete, so the float dropout mask would not map cleanly onto it.  The
module exists so models containing dropout can be converted unchanged.
"""

from __future__ import annotations

from tensorplay import nn

__all__ = ["Dropout"]


class Dropout(nn.Dropout):
    """Identity module accepting the float dropout interface."""

    def forward(self, input):
        return input

    def _get_name(self):
        return "QuantizedDropout"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return cls(mod.p, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.p, mod.inplace)