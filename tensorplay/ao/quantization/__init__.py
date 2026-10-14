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

from .fake_quantize import *
from .fake_quantize import __all__ as _fake_quant_exports
from .observer import (ObserverBase, UniformQuantizationObserverBase, RecordingObserver, NoopObserver, ReuseInputObserver, default_per_channel_weight_observer, default_histogram_observer, default_float_qparams_observer, default_fixed_qparams_range_neg1to1_observer, default_fixed_qparams_range_0to1_observer)
__all__ += _fake_quant_exports + ['ObserverBase', 'UniformQuantizationObserverBase', 'RecordingObserver', 'NoopObserver', 'ReuseInputObserver', 'default_per_channel_weight_observer', 'default_histogram_observer', 'default_float_qparams_observer', 'default_fixed_qparams_range_neg1to1_observer', 'default_fixed_qparams_range_0to1_observer']
from .qconfig import default_qat_qconfig, default_per_channel_qat_qconfig
from .quantize import prepare_qat, quantize_qat
from .fuse_modules import fuse_modules_qat
__all__ += ['default_qat_qconfig', 'default_per_channel_qat_qconfig', 'prepare_qat', 'quantize_qat', 'fuse_modules_qat']
from .fake_quantize import default_fused_act_fake_quant, default_fused_wt_fake_quant, default_fused_per_channel_wt_fake_quant
from .qconfig import default_qat_qconfig_v2, default_dynamic_qat_qconfig, get_default_qconfig, get_default_qat_qconfig
__all__ += ['default_fused_act_fake_quant', 'default_fused_wt_fake_quant', 'default_fused_per_channel_wt_fake_quant', 'default_qat_qconfig_v2', 'default_dynamic_qat_qconfig', 'get_default_qconfig', 'get_default_qat_qconfig']
from .qconfig import default_embedding_qat_qconfig, float_qparams_weight_only_qconfig, per_channel_dynamic_qconfig
from .quantization_mappings import get_embedding_qat_module_mappings
__all__ += ['default_embedding_qat_qconfig', 'float_qparams_weight_only_qconfig', 'per_channel_dynamic_qconfig', 'get_embedding_qat_module_mappings']
from .observer import default_float_qparams_observer_4bit
from .qconfig import float_qparams_weight_only_qconfig_4bit
__all__ += ['default_float_qparams_observer_4bit', 'float_qparams_weight_only_qconfig_4bit']

from .qconfig import float16_dynamic_qconfig as float16_dynamic_qconfig

from .bitnet import (
    BitNetLinear as BitNetLinear,
    BitNetPacked2bitLinearWeight as BitNetPacked2bitLinearWeight,
    BitNetTrainingConfig as BitNetTrainingConfig,
    BitNetTrainingLinearWeight as BitNetTrainingLinearWeight,
    bitnet_linear as bitnet_linear,
    bitnet_linear_packed as bitnet_linear_packed,
    bitnet_training as bitnet_training,
    precompute_bitnet_scale_for_fsdp as precompute_bitnet_scale_for_fsdp,
    quantize_ as quantize_,
)
from .int8_mixed_precision import (
    Int8MixedPrecisionTrainingConfig as Int8MixedPrecisionTrainingConfig,
    Int8MixedPrecisionTrainingLinear as Int8MixedPrecisionTrainingLinear,
    Int8MixedPrecisionTrainingLinearWeight as Int8MixedPrecisionTrainingLinearWeight,
    int8_mixed_precision_training as int8_mixed_precision_training,
    scaled_int8_mm as scaled_int8_mm,
)
from .int8_weight_only import (
    Int8QuantizedTrainingLinearWeight as Int8QuantizedTrainingLinearWeight,
    Int8WeightOnlyQuantizedTrainingConfig as Int8WeightOnlyQuantizedTrainingConfig,
    int8_weight_only_quantized_training as int8_weight_only_quantized_training,
)
from .float8 import (
    Float8DynamicActivationFloat8WeightConfig as Float8DynamicActivationFloat8WeightConfig,
    Float8MixedPrecisionTrainingLinear as Float8MixedPrecisionTrainingLinear,
    Float8WeightOnlyConfig as Float8WeightOnlyConfig,
    Float8WeightOnlyQuantizedTrainingLinear as Float8WeightOnlyQuantizedTrainingLinear,
    float8_dynamic_activation_float8_weight as float8_dynamic_activation_float8_weight,
    float8_weight_only as float8_weight_only,
)
from .int4_weight_only import (
    Int4WeightOnlyConfig as Int4WeightOnlyConfig,
    Int4WeightOnlyQuantizedTrainingConfig as Int4WeightOnlyQuantizedTrainingConfig,
    Int4WeightOnlyQuantizedTrainingLinear as Int4WeightOnlyQuantizedTrainingLinear,
    WeightOnlyInt4Linear as WeightOnlyInt4Linear,
    int4_weight_only as int4_weight_only,
)
from .int8_dyn_act_int4_weight import (
    Int8DynActInt4WeightLinear as Int8DynActInt4WeightLinear,
    Int8DynamicActivationInt4WeightConfig as Int8DynamicActivationInt4WeightConfig,
    int8_dynamic_activation_int4_weight as int8_dynamic_activation_int4_weight,
)

from .float8 import (
    Float8WeightOnlyInferenceConfig as Float8WeightOnlyInferenceConfig,
    Float8WeightOnlyInferenceLinear as Float8WeightOnlyInferenceLinear,
    float8_weight_only_inference as float8_weight_only_inference,
)
from .int4_weight_only import (
    Int4WeightOnlyQuantizedLinear as Int4WeightOnlyQuantizedLinear,
    quantize_int4_weight_only as quantize_int4_weight_only,
)
from .int2_weight_only import (
    Int2WeightOnlyConfig as Int2WeightOnlyConfig,
    Int2WeightOnlyQuantizedTrainingConfig as Int2WeightOnlyQuantizedTrainingConfig,
    Int2WeightOnlyQuantizedTrainingLinear as Int2WeightOnlyQuantizedTrainingLinear,
    WeightOnlyInt2Linear as WeightOnlyInt2Linear,
    int2_weight_only as int2_weight_only,
    pack_int2 as pack_int2,
    quantize_int2_weight_only as quantize_int2_weight_only,
    unpack_int2 as unpack_int2,
)
from .binary import (
    BinaryWeightOnlyConfig as BinaryWeightOnlyConfig,
    BinaryWeightOnlyQuantizedTrainingConfig as BinaryWeightOnlyQuantizedTrainingConfig,
    BinaryWeightOnlyQuantizedTrainingLinear as BinaryWeightOnlyQuantizedTrainingLinear,
    WeightOnlyBinaryLinear as WeightOnlyBinaryLinear,
    binary_weight_only as binary_weight_only,
    pack_binary as pack_binary,
    quantize_binary_weight_only as quantize_binary_weight_only,
    unpack_binary as unpack_binary,
)
from .awq import (
    AWQConfig as AWQConfig,
    AWQObservedLinear as AWQObservedLinear,
    AWQObserver as AWQObserver,
    awq as awq,
)
from .bitnet import quantize_int8_rowwise as quantize_int8_rowwise
__all__ += [
    "BitNetLinear",
    "BitNetPacked2bitLinearWeight",
    "BitNetTrainingConfig",
    "BitNetTrainingLinearWeight",
    "bitnet_linear",
    "bitnet_linear_packed",
    "bitnet_training",
    "precompute_bitnet_scale_for_fsdp",
    "quantize_",
    "Int8MixedPrecisionTrainingConfig",
    "Int8MixedPrecisionTrainingLinear",
    "Int8MixedPrecisionTrainingLinearWeight",
    "int8_mixed_precision_training",
    "Int8QuantizedTrainingLinearWeight",
    "Int8WeightOnlyQuantizedTrainingConfig",
    "int8_weight_only_quantized_training",
    "Float8WeightOnlyConfig",
    "Float8WeightOnlyInferenceConfig",
    "Float8DynamicActivationFloat8WeightConfig",
    "Float8WeightOnlyQuantizedTrainingLinear",
    "Float8WeightOnlyInferenceLinear",
    "Float8MixedPrecisionTrainingLinear",
    "float8_weight_only",
    "float8_weight_only_inference",
    "float8_dynamic_activation_float8_weight",
    "Int4WeightOnlyConfig",
    "Int4WeightOnlyQuantizedTrainingConfig",
    "WeightOnlyInt4Linear",
    "Int4WeightOnlyQuantizedLinear",
    "Int4WeightOnlyQuantizedTrainingLinear",
    "int4_weight_only",
    "quantize_int4_weight_only",
    "Int2WeightOnlyConfig",
    "Int2WeightOnlyQuantizedTrainingConfig",
    "Int2WeightOnlyQuantizedTrainingLinear",
    "WeightOnlyInt2Linear",
    "int2_weight_only",
    "pack_int2",
    "quantize_int2_weight_only",
    "unpack_int2",
    "BinaryWeightOnlyConfig",
    "BinaryWeightOnlyQuantizedTrainingConfig",
    "BinaryWeightOnlyQuantizedTrainingLinear",
    "WeightOnlyBinaryLinear",
    "binary_weight_only",
    "pack_binary",
    "quantize_binary_weight_only",
    "unpack_binary",
    "Int8DynActInt4WeightLinear",
    "Int8DynamicActivationInt4WeightConfig",
    "int8_dynamic_activation_int4_weight",
    "AWQConfig",
    "AWQObserver",
    "AWQObservedLinear",
    "awq",
    "quantize_int8_rowwise",
    "scaled_int8_mm",
]

# ---- eager workflow completeness ----
from .quant_type import QuantType as QuantType
from .stubs import QuantWrapper as QuantWrapper
from .qconfig import (
    QConfigDynamic as QConfigDynamic,
    QConfigAny as QConfigAny,
    qconfig_equals as qconfig_equals,
    get_default_qconfig_dict as get_default_qconfig_dict,
    get_default_qat_qconfig_dict as get_default_qat_qconfig_dict,
    default_debug_qconfig as default_debug_qconfig,
    float16_static_qconfig as float16_static_qconfig,
    default_activation_only_qconfig as default_activation_only_qconfig,
    default_reuse_input_qconfig as default_reuse_input_qconfig,
    default_quint8_weight_qconfig as default_quint8_weight_qconfig,
    default_embedding_qat_qconfig_4bit as default_embedding_qat_qconfig_4bit,
    default_symmetric_qnnpack_qconfig as default_symmetric_qnnpack_qconfig,
    default_per_channel_symmetric_qnnpack_qconfig as default_per_channel_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qat_qconfig as default_symmetric_qnnpack_qat_qconfig,
    default_per_channel_symmetric_qnnpack_qat_qconfig as default_per_channel_symmetric_qnnpack_qat_qconfig,
)
from .observer import (
    AffineQuantizedObserverBase as AffineQuantizedObserverBase,
    Granularity as Granularity,
    MappingType as MappingType,
    PerAxis as PerAxis,
    PerBlock as PerBlock,
    PerGroup as PerGroup,
    PerRow as PerRow,
    PerTensor as PerTensor,
    PerToken as PerToken,
    TorchAODType as TorchAODType,
    ZeroPointDomain as ZeroPointDomain,
    get_block_size as get_block_size,
    default_debug_observer as default_debug_observer,
    default_reuse_input_observer as default_reuse_input_observer,
    default_affine_fixed_qparams_observer as default_affine_fixed_qparams_observer,
    default_symmetric_fixed_qparams_observer as default_symmetric_fixed_qparams_observer,
    weight_observer_range_neg_127_to_127 as weight_observer_range_neg_127_to_127,
    per_channel_weight_observer_range_neg_127_to_127 as per_channel_weight_observer_range_neg_127_to_127,
)
from .fake_quantize import (
    default_histogram_fake_quant as default_histogram_fake_quant,
    default_embedding_fake_quant_4bit as default_embedding_fake_quant_4bit,
    default_affine_fixed_qparams_fake_quant as default_affine_fixed_qparams_fake_quant,
    default_symmetric_fixed_qparams_fake_quant as default_symmetric_fixed_qparams_fake_quant,
    fused_wt_fake_quant_range_neg_127_to_127 as fused_wt_fake_quant_range_neg_127_to_127,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127 as fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)
from .fuser_method_mappings import (
    fuse_conv_bn as fuse_conv_bn,
    fuse_conv_bn_relu as fuse_conv_bn_relu,
    fuse_linear_bn as fuse_linear_bn,
    fuse_convtranspose_bn as fuse_convtranspose_bn,
    get_fuser_method as get_fuser_method,
    get_fuser_method_new as get_fuser_method_new,
)
from .quantization_mappings import (
    get_default_static_quant_reference_module_mappings as get_default_static_quant_reference_module_mappings,
    get_embedding_static_quant_module_mappings as get_embedding_static_quant_module_mappings,
    get_default_static_sparse_quant_module_mappings as get_default_static_sparse_quant_module_mappings,
    get_static_quant_module_class as get_static_quant_module_class,
    get_dynamic_quant_module_class as get_dynamic_quant_module_class,
    get_embedding_qat_module_mappings as get_embedding_qat_module_mappings,
    get_default_dynamic_sparse_quant_module_mappings as get_default_dynamic_sparse_quant_module_mappings,
    get_default_qconfig_propagation_list as get_default_qconfig_propagation_list,
    get_default_compare_output_module_list as get_default_compare_output_module_list,
    get_default_float_to_quantized_operator_mappings as get_default_float_to_quantized_operator_mappings,
    get_quantized_operator as get_quantized_operator,
    no_observer_set as no_observer_set,
)
from .quantize import (
    get_default_custom_config_dict as get_default_custom_config_dict,
    swap_module as swap_module,
)
from .utils import (
    get_combined_dict as get_combined_dict,
    get_qparam_dict as get_qparam_dict,
    has_no_children_ignoring_parametrizations as has_no_children_ignoring_parametrizations,
    get_quant_type as get_quant_type,
    activation_dtype as activation_dtype,
    weight_dtype as weight_dtype,
    activation_is_statically_quantized as activation_is_statically_quantized,
    activation_is_dynamically_quantized as activation_is_dynamically_quantized,
    weight_is_quantized as weight_is_quantized,
    weight_is_statically_quantized as weight_is_statically_quantized,
    get_qconfig_dtypes as get_qconfig_dtypes,
    determine_qparams as determine_qparams,
    to_underlying_dtype as to_underlying_dtype,
)
__all__ += [
    "QuantType",
    "QuantWrapper",
    "QConfigDynamic",
    "QConfigAny",
    "qconfig_equals",
    "get_default_qconfig_dict",
    "get_default_qat_qconfig_dict",
    "default_debug_qconfig",
    "float16_static_qconfig",
    "default_activation_only_qconfig",
    "default_reuse_input_qconfig",
    "default_quint8_weight_qconfig",
    "default_embedding_qat_qconfig_4bit",
    "default_symmetric_qnnpack_qconfig",
    "default_per_channel_symmetric_qnnpack_qconfig",
    "default_symmetric_qnnpack_qat_qconfig",
    "default_per_channel_symmetric_qnnpack_qat_qconfig",
    "AffineQuantizedObserverBase",
    "Granularity",
    "MappingType",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "TorchAODType",
    "ZeroPointDomain",
    "get_block_size",
    "default_debug_observer",
    "default_reuse_input_observer",
    "default_affine_fixed_qparams_observer",
    "default_symmetric_fixed_qparams_observer",
    "weight_observer_range_neg_127_to_127",
    "per_channel_weight_observer_range_neg_127_to_127",
    "default_histogram_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_affine_fixed_qparams_fake_quant",
    "default_symmetric_fixed_qparams_fake_quant",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
    "fuse_conv_bn",
    "fuse_conv_bn_relu",
    "fuse_linear_bn",
    "fuse_convtranspose_bn",
    "get_fuser_method",
    "get_fuser_method_new",
    "get_default_static_quant_reference_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_static_quant_module_class",
    "get_dynamic_quant_module_class",
    "get_embedding_qat_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_qconfig_propagation_list",
    "get_default_compare_output_module_list",
    "get_default_float_to_quantized_operator_mappings",
    "get_quantized_operator",
    "no_observer_set",
    "get_default_custom_config_dict",
    "swap_module",
    "get_combined_dict",
    "get_qparam_dict",
    "has_no_children_ignoring_parametrizations",
    "get_quant_type",
    "activation_dtype",
    "weight_dtype",
    "activation_is_statically_quantized",
    "activation_is_dynamically_quantized",
    "weight_is_quantized",
    "weight_is_statically_quantized",
    "get_qconfig_dtypes",
    "determine_qparams",
    "to_underlying_dtype",
    "analyze_model",
    "print_model_analysis",
    "model_quantization_summary",
    "quantization_ready",
    "auto_prepare",
    "auto_convert",
    "auto_quantize",
    "prepare_fx",
    "convert_fx",
    "fuse_fx",
    "model_size_bytes",
    "benchmark_latency",
    "benchmark_quantized",
    "compare_outputs",
    "quantize_input_for_model",
]

from . import analyze as analyze
from .analyze import (
    analyze_model as analyze_model,
    model_quantization_summary as model_quantization_summary,
    print_model_analysis as print_model_analysis,
    quantization_ready as quantization_ready,
)
from . import auto as auto
from .auto import (
    auto_convert as auto_convert,
    auto_prepare as auto_prepare,
    auto_quantize as auto_quantize,
)
from .quantize_fx import (
    convert_fx as convert_fx,
    fuse_fx as fuse_fx,
    prepare_fx as prepare_fx,
)
from . import benchmark as benchmark
from .benchmark import (
    benchmark_latency as benchmark_latency,
    benchmark_quantized as benchmark_quantized,
    compare_outputs as compare_outputs,
    model_size_bytes as model_size_bytes,
    quantize_input_for_model as quantize_input_for_model,
)
