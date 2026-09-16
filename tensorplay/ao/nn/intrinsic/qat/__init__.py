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
