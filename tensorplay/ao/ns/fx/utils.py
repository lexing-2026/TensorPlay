"""Utilities for the FX numeric suite."""

import enum

import tensorplay as tp
from tensorplay import nn

__all__ = [
    "NodeInputOrOutputType",
    "get_node_first_input_and_output_type",
    "get_target_type_str",
    "compute_sqnr",
    "compute_normalized_l2_error",
    "compute_cosine_similarity",
    "op_type_supports_shadowing",
]


class NodeInputOrOutputType(enum.Enum):
    INPUT = 0
    OUTPUT = 1


def get_node_first_input_and_output_type(node, gm, num_non_observer_args=1):
    """Return the dtype types of the first input and output of a node."""
    input_type = None
    output_type = None
    if len(node.args) > 0:
        input_arg = node.args[0]
        if hasattr(input_arg, "meta"):
            input_type = input_arg.meta.get("type", None)
    if input_type is None:
        input_type = tp.float32
    output_type = input_type
    return input_type, output_type


def get_target_type_str(node, gm):
    """Return a human readable target string for a node."""
    if node.op == "call_function":
        return str(node.target)
    if node.op == "call_module":
        return type(gm.get_submodule(node.target)).__name__
    if node.op == "call_method":
        return str(node.target)
    return node.op


def compute_sqnr(x, y):
    """Compute signal-to-quantization-noise ratio between two tensors."""
    x = x.detach()
    y = y.detach()
    if x.dtype != y.dtype:
        x = x.float()
        y = y.float()
    mse = ((x - y) ** 2).mean()
    if mse == 0:
        return tp.tensor(float("inf"))
    signal = (x**2).mean()
    return 10 * tp.log10(signal / mse)


def compute_normalized_l2_error(x, y):
    """Compute normalized L2 error between two tensors."""
    x = x.detach()
    y = y.detach()
    if x.dtype != y.dtype:
        x = x.float()
        y = y.float()
    diff = x - y
    return diff.norm() / x.norm()


def compute_cosine_similarity(x, y):
    x = x.detach().flatten()
    y = y.detach().flatten()
    if x.dtype != y.dtype:
        x = x.float()
        y = y.float()
    return tp.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)


def op_type_supports_shadowing(node):
    """Return whether an op type supports graph shadowing."""
    return node.op in ("call_module", "call_function", "call_method")