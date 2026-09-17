"""Quantization support on the native affine-quantized grid.

Exposes native quantized tensors (QInt8/QUInt8/QInt32 dtypes carrying an
immutable quantizer with their affine parameters), post-training calibration
observers, FakeQuantize with a straight-through estimator for training, the
eager quantization workflow (configuration, module fusion, prepare/convert)
and the quantized module library.
"""

from tensorplay._C import (
    dequantize as dequantize,
    quantize_per_channel as quantize_per_channel,
    quantize_per_tensor as quantize_per_tensor,
    quantize_per_tensor_dynamic as quantize_per_tensor_dynamic,
)
from tensorplay._C import quantized_linear as quantized_linear
from tensorplay._C import quantized_linear_dynamic as quantized_linear_dynamic
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

from .fake_quantize import FakeQuantize as FakeQuantize
from .fake_quantize import PerChannelFakeQuantize as PerChannelFakeQuantize
from .fake_quantize import fake_quantize_per_channel as fake_quantize_per_channel
from .fake_quantize import fake_quantize_per_tensor as fake_quantize_per_tensor
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
from tensorplay.ao.nn.quantized import QuantizedLinear as QuantizedLinear
from .qconfig import QConfig as QConfig
from .qconfig import QConfigMapping as QConfigMapping
from .qconfig import default_dynamic_qconfig as default_dynamic_qconfig
from .qconfig import default_per_channel_qconfig as default_per_channel_qconfig
from .qconfig import default_qconfig as default_qconfig
from .qconfig import default_weight_only_qconfig as default_weight_only_qconfig
from .quantization_mappings import DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS as DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS
from .quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS as DEFAULT_QAT_MODULE_MAPPINGS
from .quantization_mappings import DEFAULT_STATIC_QUANT_MODULE_MAPPINGS as DEFAULT_STATIC_QUANT_MODULE_MAPPINGS
from .quantization_mappings import get_default_dynamic_quant_module_mappings as get_default_dynamic_quant_module_mappings
from .quantization_mappings import get_default_qat_module_mappings as get_default_qat_module_mappings
from .quantization_mappings import get_default_static_quant_module_mappings as get_default_static_quant_module_mappings
from .quantize import convert as convert
from .quantize import prepare as prepare
from .quantize import quantize as quantize
from .quantize import quantize_dynamic as quantize_dynamic
from .fuse_modules import fuse_modules as fuse_modules
from .stubs import DeQuantStub as DeQuantStub
from .stubs import QuantStub as QuantStub

__all__ = [
    "quantize_per_tensor",
    "quantize_per_channel",
    "quantize_per_tensor_dynamic",
    "quantized_linear",
    "quantized_linear_dynamic",
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
    "FixedQParamsObserver",
    "HistogramObserver",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "PlaceholderObserver",
    "PerChannelMinMaxObserver",
    "default_dynamic_quant_observer",
    "default_observer",
    "default_weight_observer",
    "get_observer_state_dict",
    "load_observer_state_dict",
    "QuantStub",
    "DeQuantStub",
    "QConfig",
    "QConfigMapping",
    "default_qconfig",
    "default_dynamic_qconfig",
    "default_per_channel_qconfig",
    "default_weight_only_qconfig",
    "prepare",
    "convert",
    "quantize",
    "quantize_dynamic",
    "fuse_modules",
]
