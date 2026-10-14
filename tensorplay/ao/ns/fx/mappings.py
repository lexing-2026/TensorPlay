"""Op mappings for the FX numeric suite."""

import operator

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F
from tensorplay.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_float_to_quantized_operator_mappings,
    get_default_static_quant_module_mappings,
)

__all__ = [
    "get_base_name_to_sets_of_related_ops",
    "get_base_name_for_op",
    "add_op_to_sets_of_related_ops",
    "get_node_type_to_io_type_map",
    "get_unmatchable_types_map",
]


def get_base_name_to_sets_of_related_ops() -> dict[str, set]:
    """Return families of mathematically-related ops (float + quantized).

    Each family is one set of module classes, functionals and method names
    that represent the same operation across quantization flavors, e.g.
    ``nn.Linear`` and the quantized linear module.
    """
    sets_of_related_ops: list[set] = [
        {nn.Conv1d},
        {nn.Conv2d},
        {nn.Conv3d},
        {F.conv1d},
        {F.conv2d},
        {F.conv3d},
        {nn.Linear},
        {F.linear},
        {nn.AvgPool1d},
        {nn.AvgPool2d},
        {nn.AvgPool3d},
        {nn.AdaptiveAvgPool1d},
        {nn.AdaptiveAvgPool2d},
        {nn.AdaptiveAvgPool3d},
        {nn.LSTM},
        {tp.add, operator.add},
        {tp.cat},
        {tp.mul, operator.mul},
        {F.relu, nn.ReLU, "relu", "relu_", tp.relu},
        {nn.MaxPool1d, F.max_pool1d},
        {nn.MaxPool2d, F.max_pool2d},
        {nn.MaxPool3d, F.max_pool3d},
        {nn.Embedding},
        {nn.EmbeddingBag},
        {nn.BatchNorm1d},
        {nn.BatchNorm2d},
        {nn.BatchNorm3d},
        {nn.LayerNorm},
        {nn.GroupNorm},
        {nn.InstanceNorm1d},
        {nn.InstanceNorm2d},
        {nn.InstanceNorm3d},
        {nn.Dropout},
        {nn.Softmax},
        {nn.PReLU},
        {nn.ELU},
        {nn.LeakyReLU},
        {nn.Hardswish},
        {nn.SiLU},
        {nn.ConvTranspose1d},
        {nn.ConvTranspose2d},
        {nn.ConvTranspose3d},
        {nn.RNN},
        {nn.GRU},
    ]

    # connect float ops to their quantized counterparts via the default
    # quantization mappings
    new_connections: list[tuple] = []
    for source_to_target in (
        get_default_static_quant_module_mappings(),
        get_default_dynamic_quant_module_mappings(),
        get_default_float_to_quantized_operator_mappings(),
    ):
        for source, target in source_to_target.items():
            new_connections.append((source, target))

    for item1, item2 in new_connections:
        for set_of_related_ops in sets_of_related_ops:
            if item1 in set_of_related_ops or item2 in set_of_related_ops:
                set_of_related_ops.add(item1)
                set_of_related_ops.add(item2)
                break

    return {str(counter): s for counter, s in enumerate(sets_of_related_ops)}


def get_base_name_for_op(op) -> str | None:
    """Return the base name for a given op, or ``None`` if not found."""
    for base_name, ops in get_base_name_to_sets_of_related_ops().items():
        if op in ops:
            return base_name
    return None


def add_op_to_sets_of_related_ops(base_name, op, sets_of_related_ops):
    sets_of_related_ops[base_name].add(op)
    return sets_of_related_ops


def get_node_type_to_io_type_map() -> dict[str, set]:
    """Return a mapping from node type to input/output types.

    A compact subset of the framework op families is reported; the full set of
    FP32/INT8 flavors follows the related-ops families.
    """
    return {
        "funs_io_type_fp32": {
            F.linear, F.conv1d, F.conv2d, F.conv3d, F.elu, F.hardswish,
            F.instance_norm, F.layer_norm, F.leaky_relu, F.dropout,
            F.prelu, operator.add, operator.mul,
        },
        "funs_io_type_int8": set(),
        "funs_io_type_fp32_or_int8": {
            F.relu, F.tanh, F.sigmoid, F.hardsigmoid, F.dropout, F.hardtanh,
            F.max_pool1d, F.max_pool2d, F.max_pool3d, F.relu6, F.interpolate,
            tp.cat, tp.add, operator.add,
        },
        "mods_io_type_fp32": {
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.BatchNorm2d,
            nn.BatchNorm3d, nn.Dropout, nn.ConvTranspose1d, nn.ConvTranspose2d,
            nn.ConvTranspose3d, nn.ELU, nn.GroupNorm, nn.InstanceNorm1d,
            nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm, nn.Hardswish,
            nn.LeakyReLU, nn.ReLU6, nn.SiLU, nn.Softmax, nn.PReLU,
        },
        "mods_io_type_int8": set(),
        "mods_io_type_fp32_or_int8": {nn.ReLU},
    }


def get_unmatchable_types_map() -> dict[str, set]:
    """Return the sets of ops that cannot be matched for shadowing."""
    FUNS_UNMATCHABLE = {
        tp.quantize_per_tensor,
        operator.getitem,
    }
    MODS_UNMATCHABLE = {
        nn.Identity,
    }
    METHS_UNMATCHABLE = {
        "to",
        "dequantize",
        "reshape",
        "view",
        "unsqueeze_",
        "unsqueeze",
        "transpose",
        "squeeze_",
        "squeeze",
        "size",
        "shape",
        "resize_",
        "repeat_interleave",
        "repeat",
        "permute",
        "numel",
        "mean",
        "detach_",
        "detach",
        "contiguous",
        "clamp",
        "chunk",
    }
    return {
        "funs_unmatchable": FUNS_UNMATCHABLE,
        "mods_unmatchable": MODS_UNMATCHABLE,
        "meths_unmatchable": METHS_UNMATCHABLE,
    }