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
from tensorplay._C import quantized_linear as quantized_linear

from .fake_quant import FakeQuantize as FakeQuantize
from .fake_quant import PerChannelFakeQuantize as PerChannelFakeQuantize
from .fake_quant import fake_quantize_per_channel as fake_quantize_per_channel
from .fake_quant import fake_quantize_per_tensor as fake_quantize_per_tensor
from .observer import FixedQParamsObserver as FixedQParamsObserver
from .observer import HistogramObserver as HistogramObserver
from .observer import MinMaxObserver as MinMaxObserver
from .observer import MovingAverageMinMaxObserver as MovingAverageMinMaxObserver
from .observer import MovingAveragePerChannelMinMaxObserver as MovingAveragePerChannelMinMaxObserver
from .observer import PlaceholderObserver as PlaceholderObserver
from .observer import PerChannelMinMaxObserver as PerChannelMinMaxObserver
from .observer import default_dynamic_quant_observer as default_dynamic_quant_observer
from .observer import default_observer as default_observer
from .observer import default_weight_observer as default_weight_observer
from .observer import get_observer_state_dict as get_observer_state_dict
from .observer import load_observer_state_dict as load_observer_state_dict
from .quantized_modules import QuantizedLinear as QuantizedLinear
from .stub import DeQuantStub as DeQuantStub
from .stub import QuantStub as QuantStub

__all__ = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
    "quantized_linear",
    "fake_quantize_per_tensor",
    "fake_quantize_per_channel",
    "FakeQuantize",
    "PerChannelFakeQuantize",
    "QuantStub",
    "DeQuantStub",
    "QuantizedLinear",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "HistogramObserver",
    "FixedQParamsObserver",
    "PlaceholderObserver",
    "default_observer",
    "default_weight_observer",
    "default_dynamic_quant_observer",
    "get_observer_state_dict",
    "load_observer_state_dict",
]
