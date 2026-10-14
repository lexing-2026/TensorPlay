"""Quantized module library for inference.

Modules here consume and produce native quantized tensors (or, for the
linear family, float tensors produced by the fused Int8 GEMM kernel).  Use
``from_float`` on each class to build them from calibrated float modules.
"""

from __future__ import annotations

from .activation import (
    ELU,
    Hardswish,
    Hardsigmoid,
    LeakyReLU,
    PReLU,
    ReLU,
    ReLU6,
    Sigmoid,
    Softmax,
    Tanh,
)
from .batchnorm import BatchNorm2d, BatchNorm3d
from .conv import Conv1d, Conv2d, Conv3d
from .dropout import Dropout
from .functional_modules import FXFloatFunctional, FloatFunctional, QFunctional
from .linear import QuantizedLinear as Linear
from .linear import QuantizedLinear as QuantizedLinear
from .normalization import GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LayerNorm
from .pooling import MaxPool1d, MaxPool2d, MaxPool3d
from .quantize import DeQuantize, Quantize

__all__ = [
    "Linear",
    "QuantizedLinear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "ReLU",
    "ReLU6",
    "ELU",
    "LeakyReLU",
    "Hardswish",
    "Hardsigmoid",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "PReLU",
    "Dropout",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "Quantize",
    "DeQuantize",
    "FloatFunctional",
    "QFunctional",
    "FXFloatFunctional",
]

from .conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from .embedding_ops import Embedding, EmbeddingBag, EmbeddingPackedParams
__all__ += ['ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'Embedding', 'EmbeddingBag', 'EmbeddingPackedParams']
