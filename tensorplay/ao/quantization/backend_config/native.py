"""Default backend configuration for the native quantized operators."""

import tensorplay as tp

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig

__all__ = [
    "default_op_quint8_dtype_config",
    "default_op_fp16_dtype_config",
    "default_dynamic_int8_dtype_config",
    "default_dynamic_float16_dtype_config",
    "input_output_only_quint8_dtype_config",
    "weight_only_quint8_dtype_config",
    "weight_only_quint4x2_dtype_config",
    "get_native_backend_config",
    "get_native_backend_config_dict",
]

weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=tp.quint8,
    output_dtype=tp.quint8,
    weight_dtype=tp.qint8,
    bias_dtype=tp.float32,
)

default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=tp.quint8,
    output_dtype=tp.quint8,
)

default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=tp.float16,
    output_dtype=tp.float16,
    weight_dtype=tp.float16,
    bias_dtype=tp.float16,
)

default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=tp.quint8,
    output_dtype=tp.float32,
    weight_dtype=tp.qint8,
    bias_dtype=tp.float32,
    is_dynamic=True,
)

default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=tp.float16,
    output_dtype=tp.float32,
    weight_dtype=tp.float16,
    bias_dtype=tp.float32,
    is_dynamic=True,
)

input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=tp.quint8,
    output_dtype=tp.quint8,
    weight_dtype=tp.float32,
    bias_dtype=tp.float32,
)

weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=tp.float32,
    output_dtype=tp.float32,
    weight_dtype=tp.quint8,
    bias_dtype=tp.float32,
)

weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=tp.float32,
    output_dtype=tp.float32,
    weight_dtype=tp.quint4x2,
    bias_dtype=tp.float32,
)


def _get_native_backend_config():
    """Build the default backend config for native quantized operators."""
    weighted_int8_dtype_config = DTypeConfig(
        input_dtype=tp.quint8,
        output_dtype=tp.quint8,
        weight_dtype=tp.qint8,
        bias_dtype=tp.float32,
    )
    dynamic_int8_dtype_config = DTypeConfig(
        input_dtype=tp.quint8,
        output_dtype=tp.float32,
        weight_dtype=tp.qint8,
        bias_dtype=tp.float32,
        is_dynamic=True,
    )
    dynamic_float16_dtype_config = DTypeConfig(
        input_dtype=tp.float16,
        output_dtype=tp.float32,
        weight_dtype=tp.float16,
        bias_dtype=tp.float32,
        is_dynamic=True,
    )
    return BackendConfig("native") \
        .set_backend_pattern_configs(_get_conv_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_linear_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_bn_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_default_op_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs()) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_binary_op_configs(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_cat_config(weighted_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_tensor_info_op_configs()) \
        .set_backend_pattern_configs(_get_rnn_op_configs(dynamic_int8_dtype_config)) \
        .set_backend_pattern_configs(_get_ln_configs(weighted_int8_dtype_config))


def get_native_backend_config() -> BackendConfig:
    """Return the default backend config for native quantized operators."""
    return _get_native_backend_config()


def get_native_backend_config_dict():
    """Return the default backend config as a plain dictionary."""
    return _get_native_backend_config().to_dict()