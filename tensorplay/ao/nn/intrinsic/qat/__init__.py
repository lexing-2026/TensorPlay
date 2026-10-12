"""Quantization-aware fused modules.

Float fused operators (operator followed by a rectifier) with simulated
weight quantization during training.
"""

from __future__ import annotations

import tensorplay.nn.functional as F

from ...qat.modules.conv import Conv1d, Conv2d, Conv3d
from ...qat.modules.linear import Linear

__all__ = ["LinearReLU", "ConvReLU1d", "ConvReLU2d", "ConvReLU3d"]


class LinearReLU(Linear):
    """A linear transformation followed by a rectifier, trained with
    simulated weight quantization."""

    def forward(self, input):
        return F.relu(super().forward(input))

    def _get_name(self):
        return "QATLinearReLU"


class ConvReLU1d(Conv1d):
    """A 1d convolution followed by a rectifier, trained with simulated
    weight quantization."""

    def forward(self, input):
        return F.relu(super().forward(input))

    def _get_name(self):
        return "QATConvReLU1d"


class ConvReLU2d(Conv2d):
    """A 2d convolution followed by a rectifier, trained with simulated
    weight quantization."""

    def forward(self, input):
        return F.relu(super().forward(input))

    def _get_name(self):
        return "QATConvReLU2d"


class ConvReLU3d(Conv3d):
    """A 3d convolution followed by a rectifier, trained with simulated
    weight quantization."""

    def forward(self, input):
        return F.relu(super().forward(input))

    def _get_name(self):
        return "QATConvReLU3d"


from .. import LinearReLU as _FloatLinearReLU, ConvReLU1d as _FloatConvReLU1d, ConvReLU2d as _FloatConvReLU2d, ConvReLU3d as _FloatConvReLU3d

def _fused_to_float(self):
    base = Linear.to_float(self) if isinstance(self, Linear) else self._BASE_TO_FLOAT(self)
    result = self._FLOAT_FUSED.from_float(base)
    result.train(self.training)
    return result

LinearReLU._FLOAT_FUSED = _FloatLinearReLU
LinearReLU.to_float = _fused_to_float
for _qat, _base, _float in [(ConvReLU1d, Conv1d, _FloatConvReLU1d), (ConvReLU2d, Conv2d, _FloatConvReLU2d), (ConvReLU3d, Conv3d, _FloatConvReLU3d)]:
    _qat._FLOAT_FUSED = _float
    _qat._BASE_TO_FLOAT = staticmethod(_base.to_float)
    _qat.to_float = _fused_to_float

from .conv_bn import ConvBn1d, ConvBn2d, ConvBn3d, ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, freeze_bn_stats, update_bn_stats
from .linear_bn import LinearBn1d
__all__ += ['ConvBn1d', 'ConvBn2d', 'ConvBn3d', 'ConvBnReLU1d', 'ConvBnReLU2d', 'ConvBnReLU3d', 'LinearBn1d', 'freeze_bn_stats', 'update_bn_stats']
