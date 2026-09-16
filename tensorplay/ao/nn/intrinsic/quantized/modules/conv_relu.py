"""Quantized convolutions fused with a rectifier.

The base convolution kernel emits a quantized output; the rectifier runs in
the integer domain and keeps the same affine parameters.
"""

from __future__ import annotations

import tensorplay as tp

from ... import ConvReLU1d as _FloatConvReLU1d
from ... import ConvReLU2d as _FloatConvReLU2d
from ... import ConvReLU3d as _FloatConvReLU3d
from ....quantized.conv import Conv1d, Conv2d, Conv3d

__all__ = ["ConvReLU1d", "ConvReLU2d", "ConvReLU3d"]


def _relu_forward(base_cls, self, x):
    out = base_cls.forward(self, x)
    return tp._C.quantized_relu(out) if out.is_quantized() else out.relu()


class ConvReLU1d(Conv1d):
    """A ConvReLU1d module fused from Conv1d and ReLU modules."""

    _FLOAT_MODULE = _FloatConvReLU1d

    def forward(self, x):
        return _relu_forward(Conv1d, self, x)

    def _get_name(self):
        return "QuantizedConvReLU1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(getattr(mod, "op", mod))


class ConvReLU2d(Conv2d):
    """A ConvReLU2d module fused from Conv2d and ReLU modules."""

    _FLOAT_MODULE = _FloatConvReLU2d

    def forward(self, x):
        return _relu_forward(Conv2d, self, x)

    def _get_name(self):
        return "QuantizedConvReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(getattr(mod, "op", mod))


class ConvReLU3d(Conv3d):
    """A ConvReLU3d module fused from Conv3d and ReLU modules."""

    _FLOAT_MODULE = _FloatConvReLU3d

    def forward(self, x):
        return _relu_forward(Conv3d, self, x)

    def _get_name(self):
        return "QuantizedConvReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(getattr(mod, "op", mod))
