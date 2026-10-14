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
    Dropout,
    ELU,
    FloatFunctional,
    FXFloatFunctional,
    GroupNorm,
    Hardswish,
    Hardsigmoid,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
    LeakyReLU,
    Linear,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    PReLU,
    QFunctional,
    Quantize,
    QuantizedLinear,
    ReLU,
    ReLU6,
    Sigmoid,
    Softmax,
    Tanh,
)

from tensorplay.ao.nn.quantized import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Embedding, EmbeddingBag, EmbeddingPackedParams
