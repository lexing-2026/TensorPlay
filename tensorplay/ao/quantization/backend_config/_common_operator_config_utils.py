"""Builders for the standard operator-pattern backend configs."""

import copy
import operator
from collections.abc import Callable

import tensorplay as tp
import tensorplay.ao.nn.intrinsic as nni
import tensorplay.ao.nn.intrinsic.qat as nniqat
import tensorplay.ao.nn.qat as nnqat
import tensorplay.ao.nn.quantized.reference as nnqr
from tensorplay import nn
from tensorplay.nn import functional as F

from ..fuser_method_mappings import (
    _sequential_wrapper2,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_convtranspose_bn,
    fuse_linear_bn,
)

from .backend_config import (
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)

_FIXED_QPARAM_OP_0TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=tp.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_exact_match=1.0 / 256.0,
    zero_point_exact_match=0,
)
_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=tp.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_exact_match=2.0 / 256.0,
    zero_point_exact_match=128,
)
_FIXED_QPARAMS_OP_TO_CONSTRAINTS: dict[Callable | str, DTypeWithConstraints] = {
    nn.Hardsigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    F.hardsigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "hardsigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "hardsigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    nn.Sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    tp.sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    nn.Softmax: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    nn.Tanh: _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    tp.tanh: _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    "tanh": _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    "tanh_": _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
}


def _get_binary_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    binary_op_configs: list[BackendPatternConfig] = []
    num_tensor_args_to_observation_type_mapping = {
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op in [operator.add, tp.add, operator.mul, tp.mul]:
        bop_patterns = [
            (op, nn.ReLU),
            (op, F.relu),
            (op, tp.relu),
            op,
        ]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(
                BackendPatternConfig(bop_pattern)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_num_tensor_args_to_observation_type(
                    num_tensor_args_to_observation_type_mapping
                )
            )
    binary_op_configs.append(
        BackendPatternConfig(tp.matmul).set_dtype_configs([copy.deepcopy(dtype_configs)])
    )
    return binary_op_configs


def _get_linear_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    linear_configs: list[BackendPatternConfig] = []

    linear_configs.append(
        BackendPatternConfig(nn.Linear)
        .set_observation_type(observation_type)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_root_module(nn.Linear)
        .set_qat_module(nnqat.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    linear_configs.append(
        BackendPatternConfig(nn.NonDynamicallyQuantizableLinear)
        .set_observation_type(observation_type)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_root_module(nn.Linear)
        .set_qat_module(nnqat.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    linear_configs.append(
        BackendPatternConfig((nn.Linear, nn.ReLU))
        .set_observation_type(observation_type)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_fused_module(nni.LinearReLU)
        .set_fuser_method(_sequential_wrapper2(nni.LinearReLU))
    )
    linear_configs.append(
        BackendPatternConfig(nni.LinearReLU)
        .set_observation_type(observation_type)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_root_module(nn.Linear)
        .set_qat_module(nniqat.LinearReLU)
        .set_reference_quantized_module(nnqr.Linear)
    )
    linear_configs.append(
        BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
        .set_observation_type(observation_type)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_fused_module(nni.LinearBn1d)
        .set_fuser_method(fuse_linear_bn)
    )
    linear_configs.append(
        BackendPatternConfig(nni.LinearBn1d)
        .set_observation_type(observation_type)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_root_module(nn.Linear)
        .set_qat_module(nniqat.LinearBn1d)
        .set_reference_quantized_module(nnqr.Linear)
    )
    # dynamic linear
    dynamic_linear_config = BackendPatternConfig(nn.Linear).set_observation_type(
        ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    )
    dynamic_linear_config.set_dtype_configs(
        [
            DTypeConfig(
                input_dtype=tp.quint8,
                output_dtype=tp.float,
                weight_dtype=tp.qint8,
                bias_dtype=tp.float,
                is_dynamic=True,
            )
        ]
    )
    dynamic_linear_config.set_root_module(nn.Linear)
    dynamic_linear_config.set_reference_quantized_module(nnqr.Linear)
    linear_configs.append(dynamic_linear_config)
    return linear_configs


def _get_conv_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    conv_configs: list[BackendPatternConfig] = []
    for rank in (1, 2, 3):
        conv = getattr(nn, f"Conv{rank}d")
        conv_transpose = getattr(nn, f"ConvTranspose{rank}d")
        bn = getattr(nn, f"BatchNorm{rank}d")
        qref_conv = getattr(nnqr, f"Conv{rank}d")
        qref_conv_transpose = getattr(nnqr, f"ConvTranspose{rank}d")
        fused_conv_relu = getattr(nni, f"ConvReLU{rank}d")
        fused_conv_bn = getattr(nni, f"ConvBn{rank}d")
        fused_conv_bn_relu = getattr(nni, f"ConvBnReLU{rank}d")
        qat_conv = getattr(nnqat, f"Conv{rank}d")
        qat_conv_relu = getattr(nniqat, f"ConvReLU{rank}d")
        qat_conv_bn = getattr(nniqat, f"ConvBn{rank}d")
        qat_conv_bn_relu = getattr(nniqat, f"ConvBnReLU{rank}d")
        conv_fused = f"fuse_conv_bn" if rank != 1 else f"fuse_conv_bn"
        conv_fused_relu = "fuse_conv_bn_relu"
        conv_transpose_fused = "fuse_convtranspose_bn"

        conv_configs.extend(
            [
                BackendPatternConfig(conv)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_root_module(conv)
                .set_qat_module(qat_conv)
                .set_reference_quantized_module(qref_conv),
                BackendPatternConfig((conv, nn.ReLU))
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_fused_module(fused_conv_relu)
                .set_fuser_method(_sequential_wrapper2(fused_conv_relu)),
                BackendPatternConfig(fused_conv_relu)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_root_module(conv)
                .set_qat_module(qat_conv_relu)
                .set_reference_quantized_module(qref_conv),
                BackendPatternConfig((conv, bn))
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_fused_module(fused_conv_bn)
                .set_fuser_method(fuse_conv_bn),
                BackendPatternConfig(fused_conv_bn)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_root_module(conv)
                .set_qat_module(qat_conv_bn)
                .set_reference_quantized_module(qref_conv),
                BackendPatternConfig((conv, bn, nn.ReLU))
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_fused_module(fused_conv_bn_relu)
                .set_fuser_method(fuse_conv_bn_relu),
                BackendPatternConfig(fused_conv_bn_relu)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_root_module(conv)
                .set_qat_module(qat_conv_bn_relu)
                .set_reference_quantized_module(qref_conv),
                BackendPatternConfig(conv_transpose)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_root_module(conv_transpose)
                .set_reference_quantized_module(qref_conv_transpose),
                BackendPatternConfig((conv_transpose, bn))
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_fused_module(fused_conv_bn)
                .set_fuser_method(fuse_convtranspose_bn),
            ]
        )
    return conv_configs


def _get_bn_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    bn_configs: list[BackendPatternConfig] = []
    for rank in (2, 3):
        bn = getattr(nn, f"BatchNorm{rank}d")
        fused_bn_relu = getattr(nni, f"BNReLU{rank}d")
        bn_configs.extend(
            [
                BackendPatternConfig(bn)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)]),
                BackendPatternConfig((bn, nn.ReLU))
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)])
                .set_fused_module(fused_bn_relu)
                .set_fuser_method(_sequential_wrapper2(fused_bn_relu)),
                BackendPatternConfig(fused_bn_relu)
                .set_observation_type(observation_type)
                .set_dtype_configs([copy.deepcopy(dtype_configs)]),
            ]
        )
    return bn_configs


def _get_cat_config(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    return [
        BackendPatternConfig(tp.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig((nn.ReLU, tp.cat))
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
    ]


def _get_default_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    configs = []
    for pattern in [
        nn.ReLU,
        F.relu,
        tp.relu,
        "relu",
        "relu_",
        nn.ReLU6,
        F.hardtanh,
        F.hardtanh_,
        nn.Hardswish,
        F.hardswish,
        nn.ELU,
        F.elu,
        nn.LeakyReLU,
        F.leaky_relu,
        nn.Sigmoid,
        F.sigmoid,
        "sigmoid",
        "sigmoid_",
        nn.Tanh,
        F.tanh,
        "tanh",
        "tanh_",
        nn.Softmax,
        F.dropout,
        nn.Dropout,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.Identity,
        F.instance_norm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        F.layer_norm,
        nn.LayerNorm,
        F.group_norm,
        nn.GroupNorm,
    ]:
        configs.append(
            BackendPatternConfig(pattern)
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs([copy.deepcopy(dtype_configs)])
        )
    return configs


def _get_fixed_qparams_op_configs() -> list[BackendPatternConfig]:
    fixed_qparams_op_configs = []
    for pattern, constraints in _FIXED_QPARAMS_OP_TO_CONSTRAINTS.items():
        dtype_config = DTypeConfig(
            input_dtype=constraints,
            output_dtype=constraints,
        )
        fixed_qparams_op_configs.append(
            BackendPatternConfig(pattern)
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs([dtype_config])
        )
    return fixed_qparams_op_configs


def _get_share_qparams_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    return [
        BackendPatternConfig(tp.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig(tp.max)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig(tp.min)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig(tp.mean)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig(tp.reshape)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig(tp.flatten)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
    ]


def _get_tensor_info_op_configs() -> list[BackendPatternConfig]:
    return [
        BackendPatternConfig(tp.size)
        .set_observation_type(ObservationType.INPUT_OUTPUT_NOT_OBSERVED)
        .set_dtype_configs([]),
    ]


def _get_rnn_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    rnn_op_configs = []
    for module in [nn.RNNCell, nn.GRUCell, nn.LSTMCell, nn.RNN, nn.GRU, nn.LSTM]:
        rnn_op_configs.append(
            BackendPatternConfig(module)
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs([copy.deepcopy(dtype_configs)])
        )
    return rnn_op_configs


def _get_embedding_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    return [
        BackendPatternConfig(nn.Embedding)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_root_module(nn.Embedding)
        .set_reference_quantized_module(nnqr.Embedding),
        BackendPatternConfig(nn.EmbeddingBag)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)])
        .set_root_module(nn.EmbeddingBag)
        .set_reference_quantized_module(nnqr.EmbeddingBag),
    ]


def _get_ln_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    return [
        BackendPatternConfig(nn.LayerNorm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
        BackendPatternConfig(F.layer_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs([copy.deepcopy(dtype_configs)]),
    ]