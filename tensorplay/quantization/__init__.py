"""Quantization support on the affine Int8 grid.

Exposes real quantize/dequantize kernels (per-tensor and per-channel),
post-training calibration observers, FakeQuantize with a straight-through
estimator, and Quant/DeQuant stubs — the essentials of ``torch.ao.quantization``.
"""

from tensorplay._C import (
    dequantize_per_channel as dequantize_per_channel,
    dequantize_per_tensor as dequantize_per_tensor,
    quantize_per_channel as quantize_per_channel,
    quantize_per_tensor as quantize_per_tensor,
)

from .fake_quant import FakeQuantize as FakeQuantize
from .fake_quant import fake_quantize_per_tensor as fake_quantize_per_tensor
from .observer import MinMaxObserver as MinMaxObserver
from .observer import MovingAverageMinMaxObserver as MovingAverageMinMaxObserver
from .observer import PerChannelMinMaxObserver as PerChannelMinMaxObserver
from .stub import DeQuantStub as DeQuantStub
from .stub import QuantStub as QuantStub

__all__ = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
    "fake_quantize_per_tensor",
    "FakeQuantize",
    "QuantStub",
    "DeQuantStub",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
]
