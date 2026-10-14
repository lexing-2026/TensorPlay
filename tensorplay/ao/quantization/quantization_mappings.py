"""Default float-to-quantized module and operator mappings."""

import copy
from collections.abc import Callable

import tensorplay as tp
import tensorplay.ao.nn as ao_nn
import tensorplay.ao.nn.intrinsic as nni
import tensorplay.ao.nn.intrinsic.qat as nniqat
import tensorplay.ao.nn.intrinsic.quantized as nniq
import tensorplay.ao.nn.intrinsic.quantized.dynamic as nniqd
import tensorplay.ao.nn.qat as nnqat
import tensorplay.ao.nn.qat.dynamic as nnqatd
import tensorplay.ao.nn.quantized as nnq
import tensorplay.ao.nn.quantized.dynamic as nnqd
import tensorplay.ao.nn.quantized.reference as nnqr
import tensorplay.nn.functional as F
from tensorplay import nn
from tensorplay.nn.utils.parametrize import type_before_parametrizations

from .fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant,
    default_fixed_qparams_range_neg1to1_fake_quant,
)
from .stubs import DeQuantStub, QuantStub
from .utils import get_combined_dict

__all__ = [
    "DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_QAT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS",
    "DEFAULT_MODULE_TO_ACT_POST_PROCESS",
    "DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS",
    "no_observer_set",
    "get_default_static_quant_module_mappings",
    "get_default_static_quant_reference_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_static_quant_module_class",
    "get_dynamic_quant_module_class",
    "get_default_qat_module_mappings",
    "get_embedding_qat_module_mappings",
    "get_default_dynamic_quant_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_qconfig_propagation_list",
    "get_default_compare_output_module_list",
    "get_default_float_to_quantized_operator_mappings",
    "get_quantized_operator",
]

DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS: dict[Callable, object] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.Linear: nnqr.Linear,
    nn.Conv1d: nnqr.Conv1d,
    nn.Conv2d: nnqr.Conv2d,
    nn.Conv3d: nnqr.Conv3d,
    nn.ConvTranspose1d: nnqr.ConvTranspose1d,
    nn.ConvTranspose2d: nnqr.ConvTranspose2d,
    nn.ConvTranspose3d: nnqr.ConvTranspose3d,
    nn.Embedding: nnqr.Embedding,
    nn.EmbeddingBag: nnqr.EmbeddingBag,
    nn.GRUCell: nnqr.GRUCell,
    nn.LSTMCell: nnqr.LSTMCell,
    nn.RNNCell: nnqr.RNNCell,
    nn.LSTM: nnqr.LSTM,
}

DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: dict[Callable, object] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.Dropout: nnq.Dropout,
    nn.Conv1d: nnq.Conv1d,
    nn.Conv2d: nnq.Conv2d,
    nn.Conv3d: nnq.Conv3d,
    nn.ConvTranspose1d: nnq.ConvTranspose1d,
    nn.ConvTranspose2d: nnq.ConvTranspose2d,
    nn.ConvTranspose3d: nnq.ConvTranspose3d,
    nn.ELU: nnq.ELU,
    nn.Embedding: nnq.Embedding,
    nn.EmbeddingBag: nnq.EmbeddingBag,
    nn.GroupNorm: nnq.GroupNorm,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.NonDynamicallyQuantizableLinear: nnq.Linear,
    nn.Linear: nnq.Linear,
    nn.ReLU6: nnq.ReLU6,
    nn.PReLU: nnq.PReLU,
    nnq.FloatFunctional: nnq.QFunctional,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.ConvAdd2d: nniq.ConvAdd2d,
    nni.ConvAddReLU2d: nniq.ConvAddReLU2d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.LinearLeakyReLU: nniq.LinearLeakyReLU,
    nni.LinearTanh: nniq.LinearTanh,
    nniqat.ConvBn1d: nnq.Conv1d,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBn3d: nnq.Conv3d,
    nniqat.ConvBnReLU1d: nniq.ConvReLU1d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    nniqat.ConvBnReLU3d: nniq.ConvReLU3d,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.ConvReLU3d: nniq.ConvReLU3d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.LinearBn1d: nnq.Linear,
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
    nnqat.Conv3d: nnq.Conv3d,
}

DEFAULT_QAT_MODULE_MAPPINGS: dict[Callable, object] = {
    nn.Conv2d: nnqat.Conv2d,
    nn.Conv3d: nnqat.Conv3d,
    nn.Linear: nnqat.Linear,
    nn.NonDynamicallyQuantizableLinear: nnqat.Linear,
    nni.ConvBn1d: nniqat.ConvBn1d,
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBn3d: nniqat.ConvBn3d,
    nni.ConvBnReLU1d: nniqat.ConvBnReLU1d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvBnReLU3d: nniqat.ConvBnReLU3d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.ConvReLU3d: nniqat.ConvReLU3d,
    nni.LinearReLU: nniqat.LinearReLU,
    nni.LinearBn1d: nniqat.LinearBn1d,
}

DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: dict[Callable, object] = {
    nn.GRUCell: nnqd.GRUCell,
    nn.Linear: nnqd.Linear,
    nnqatd.Linear: nnqd.Linear,
    nn.NonDynamicallyQuantizableLinear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    nn.GRU: nnqd.GRU,
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
    nni.LinearReLU: nniqd.LinearReLU,
    nn.EmbeddingBag: nnq.EmbeddingBag,
    nn.Embedding: nnq.Embedding,
}

_INCLUDE_QCONFIG_PROPAGATE_LIST: set[Callable] = {
    nn.Sequential,
}

DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS: dict[Callable | str, Callable] = {
    F.elu: tp._C.quantized_elu,
    F.hardswish: tp._C.quantized_hardswish,
    F.instance_norm: tp._C.quantized_instance_norm,
    F.layer_norm: tp._C.quantized_layer_norm,
    F.leaky_relu: tp._C.quantized_leaky_relu,
    F.dropout: tp._C.quantized_dropout,
}

DEFAULT_MODULE_TO_ACT_POST_PROCESS: dict[Callable, Callable] = {
    nn.Hardsigmoid: default_fixed_qparams_range_0to1_fake_quant,
    nn.Sigmoid: default_fixed_qparams_range_0to1_fake_quant,
    nn.Softmax: default_fixed_qparams_range_0to1_fake_quant,
    nn.Tanh: default_fixed_qparams_range_neg1to1_fake_quant,
}

DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS: dict[Callable, object] = {
    nn.Linear: ao_nn.sparse.quantized.Linear
}

DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS: dict[Callable, object] = {
    nn.Linear: ao_nn.sparse.quantized.dynamic.Linear
}


def no_observer_set() -> set:
    """Module types that must not have observers inserted by default."""
    from tensorplay.ao.nn import quantizable as nn_quantizable

    return {nn_quantizable.LSTM, nn_quantizable.MultiheadAttention}


def get_default_static_quant_module_mappings() -> dict:
    return copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)


def get_default_static_quant_reference_module_mappings() -> dict:
    return copy.deepcopy(DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS)


def get_embedding_static_quant_module_mappings() -> dict:
    mapping = copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    mapping[nnqat.EmbeddingBag] = nnq.EmbeddingBag
    mapping[nnqat.Embedding] = nnq.Embedding
    return mapping


def get_default_static_sparse_quant_module_mappings() -> dict:
    return copy.deepcopy(DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS)


def get_static_quant_module_class(
    float_module_class: Callable,
    additional_static_quant_mapping: dict | None = None,
    is_reference: bool = False,
):
    """Return the statically quantized class for a float module class."""
    if additional_static_quant_mapping is None:
        additional_static_quant_mapping = {}
    all_mappings = get_combined_dict(
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS
        if is_reference
        else DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
        additional_static_quant_mapping,
    )
    static_quant_module_class = all_mappings.get(float_module_class, None)
    if static_quant_module_class is None:
        raise AssertionError(
            f"Floating point module class {str(float_module_class)}"
            + " does not have a corresponding quantized module class"
        )
    return copy.deepcopy(static_quant_module_class)


def get_dynamic_quant_module_class(
    float_module_class: Callable,
    additional_dynamic_quant_mapping: dict | None = None,
):
    """Return the dynamically quantized class for a float module class."""
    if additional_dynamic_quant_mapping is None:
        additional_dynamic_quant_mapping = {}
    all_mappings = get_combined_dict(
        DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS, additional_dynamic_quant_mapping
    )
    dynamic_quant_module_class = all_mappings.get(float_module_class, None)
    if dynamic_quant_module_class is None:
        raise AssertionError(
            f"Floating point module class {str(float_module_class)}"
            + " does not have a corresponding quantized module class"
        )
    return copy.deepcopy(dynamic_quant_module_class)


def get_default_qat_module_mappings() -> dict:
    return copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)


def get_embedding_qat_module_mappings() -> dict:
    mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    mapping[nn.EmbeddingBag] = nnqat.EmbeddingBag
    mapping[nn.Embedding] = nnqat.Embedding
    return mapping


def get_default_dynamic_quant_module_mappings() -> dict:
    return copy.deepcopy(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS)


def get_default_dynamic_sparse_quant_module_mappings() -> dict:
    return copy.deepcopy(DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS)


def get_default_qconfig_propagation_list() -> set:
    """Return the default module types that receive a qconfig in prepare."""
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(QCONFIG_PROPAGATE_MODULE_CLASS_LIST)


def get_default_compare_output_module_list() -> set:
    """Return the module types whose outputs are recorded in the numeric suite."""
    NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.values())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST)


def get_default_float_to_quantized_operator_mappings() -> dict:
    return copy.deepcopy(DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS)


def get_quantized_operator(float_op: Callable | str) -> Callable:
    """Return the quantized operator corresponding to a float operator."""
    quantized_op = DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS.get(float_op)
    if quantized_op is None:
        raise AssertionError(
            f"Operator {str(float_op)} does not have corresponding quantized op"
        )
    return quantized_op


def _get_special_act_post_process(module) -> Callable | None:
    """Return the fixed-qparams post process for ``module`` if one exists."""
    return DEFAULT_MODULE_TO_ACT_POST_PROCESS.get(type_before_parametrizations(module))


def _has_special_act_post_process(module) -> bool:
    return module.training and type(module) in DEFAULT_MODULE_TO_ACT_POST_PROCESS