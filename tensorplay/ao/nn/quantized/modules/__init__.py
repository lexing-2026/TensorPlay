"""Aliases matching the deeper module layout.

``tensorplay.ao.nn.quantized.modules`` re-exports the classes from
``tensorplay.ao.nn.quantized`` so both import paths expose the same API.
"""

from __future__ import annotations

from tensorplay.ao.nn.quantized import (  # noqa: F401
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    DeQuantize,
    ELU,
    FloatFunctional,
    FXFloatFunctional,
    Hardswish,
    Hardsigmoid,
    LeakyReLU,
    Linear,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    QFunctional,
    Quantize,
    QuantizedLinear,
    ReLU,
    ReLU6,
    Sigmoid,
    Tanh,
)
