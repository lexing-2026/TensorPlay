"""Op mappings for the FX numeric suite."""

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

__all__ = [
    "get_base_name_to_sets_of_related_ops",
    "get_base_name_for_op",
    "get_node_type_to_io_type_map",
    "get_unmatchable_types_map",
]


def _get_linear_ops():
    return {nn.Linear, F.linear}


def _get_conv_ops():
    return {nn.Conv1d, nn.Conv2d, nn.Conv3d, F.conv1d, F.conv2d, F.conv3d}


def _get_relu_ops():
    return {nn.ReLU, F.relu, "relu", "relu_"}


def get_base_name_to_sets_of_related_ops() -> dict[str, set]:
    """Return the canonical base name for each family of related ops."""
    return {
        "linear": _get_linear_ops(),
        "conv": _get_conv_ops(),
        "relu": _get_relu_ops(),
    }


def get_base_name_for_op(op) -> str:
    """Return the base name for a given op."""
    for base_name, ops in get_base_name_to_sets_of_related_ops().items():
        if op in ops:
            return base_name
    if isinstance(op, str):
        return op
    return type(op).__name__ if isinstance(op, type) else str(op)


def add_op_to_sets_of_related_ops(base_name, op, sets_of_related_ops):
    sets_of_related_ops[base_name].add(op)
    return sets_of_related_ops


def get_node_type_to_io_type_map() -> dict[str, set]:
    """Return a mapping from node type to input/output types."""
    return {}


def get_unmatchable_types_map() -> dict[str, set]:
    """Return the sets of ops that cannot be matched for shadowing."""
    return {"node_to_io_type_map": {}}