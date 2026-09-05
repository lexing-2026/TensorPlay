"""Quantization support on the native affine-quantized grid.

Exposes native quantized tensors (QInt8/QUInt8/QInt32 dtypes carrying an
immutable quantizer with their affine parameters), post-training calibration
observers, FakeQuantize with a straight-through
"""

from tensorplay._C import (
    dequantize as dequantize,
    quantize_per_channel as quantize_per_channel,
    quantize_per_tensor as quantize_per_tensor,
    quantize_per_tensor_dynamic as quantize_per_tensor_dynamic,
)
from tensorplay._C import quantized_linear as quantized_linear
from tensorplay._C import (
    int_repr as int_repr,
    is_quantized as is_quantized,
    q_per_channel_axis as q_per_channel_axis,
    q_per_channel_scales as q_per_channel_scales,
    q_per_channel_zero_points as q_per_channel_zero_points,
    q_scale as q_scale,
    q_zero_point as q_zero_point,
    qscheme as qscheme,
)

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
    "quantize_per_channel",
    "quantize_per_tensor_dynamic",
    "quantized_linear",
    "dequantize",
    "int_repr",
    "is_quantized",
    "q_scale",
    "q_zero_point",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_per_channel_axis",
    "qscheme",
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
