"""Quantized fused modules produced by converting their float counterparts.

Each class pairs a native quantized operator with the rectifier that
immediately follows it, sharing the output affine parameters of the base
operator.
"""

from .modules import *  # noqa: F403
from . import dynamic

__all__ = [
    "BNReLU2d",
    "BNReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearLeakyReLU",
    "LinearReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]
