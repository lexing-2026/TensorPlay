"""Quantized convolutions fused with a rectifier.

Integer convolution, output quantization and rectification share a native
compute epilogue.
"""

from __future__ import annotations

import tensorplay as tp

from ... import ConvReLU1d as _FloatConvReLU1d
from ... import ConvReLU2d as _FloatConvReLU2d
from ... import ConvReLU3d as _FloatConvReLU3d
from ....quantized.conv import Conv1d, Conv2d, Conv3d

__all__ = ["ConvReLU1d", "ConvReLU2d", "ConvReLU3d"]


class ConvReLU1d(Conv1d):
    """A ConvReLU1d module fused from Conv1d and ReLU modules."""

    _FLOAT_MODULE = _FloatConvReLU1d

    _FUSED_RELU = True

    def _get_name(self):
        return "QuantizedConvReLU1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)


class ConvReLU2d(Conv2d):
    """A ConvReLU2d module fused from Conv2d and ReLU modules."""

    _FLOAT_MODULE = _FloatConvReLU2d

    _FUSED_RELU = True

    def _get_name(self):
        return "QuantizedConvReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)


class ConvReLU3d(Conv3d):
    """A ConvReLU3d module fused from Conv3d and ReLU modules."""

    _FLOAT_MODULE = _FloatConvReLU3d

    _FUSED_RELU = True

    def _get_name(self):
        return "QuantizedConvReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
