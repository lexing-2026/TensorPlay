from __future__ import annotations

import itertools
import math
import operator
from functools import reduce
from typing import Any, Callable, TypeVar

import sympy
import tensorplay as tp
from tensorplay import nn

from ..node import Node
from ..tensor_type import Dyn, TensorType, is_consistent, is_more_precise
from .refinement_types import Equality
from .unification import Var

_T = TypeVar("_T")
_P = TypeVar("_P")

_INFERENCE_RULES: dict[Any, Callable[..., Any]] = {}
_REFINEMENT_RULES: dict[Any, Callable[..., Any]] = {}
_RULES: dict[Any, Callable[..., Any]] = {}

__all__ = [
    "GraphTypeChecker",
    "Refine",
    "adaptiveavgpool2d_check",
    "adaptiveavgpool2d_inference_rule",
    "add_inference_rule",
    "all_eq",
    "broadcast_types",
    "calculate_out_dimension",
    "conv2d_inference_rule",
    "conv_refinement_rule",
    "conv_rule",
    "element_wise_eq",
    "expand_to_tensor_dim",
    "first_two_eq",
    "flatten_check",
    "flatten_inference_rule",
    "flatten_refinement_rule",
    "get_attr_inference_rule",
    "get_greatest_upper_bound",
    "get_parameter",
    "linear_check",
    "linear_inference_rule",
    "maxpool2d_check",
    "maxpool2d_inference_rule",
    "register_algebraic_expressions_inference_rule",
    "register_inference_rule",
    "register_refinement_rule",
    "relu_inference_rule",
    "reshape_inference_rule",
    "transpose_inference_rule",
]


def expand_to_tensor_dim(value: Any, rank: int) -> TensorType:
    if value is Dyn:
        return TensorType((Dyn,) * rank)
    if isinstance(value, TensorType):
        if len(value.dims) != rank:
            raise TypeError(f"tensor rank {len(value.dims)} cannot be used as rank {rank}")
        return value
    raise TypeError(f"expected a tensor type, got {value!r}")


def broadcast_types(left: Any, right: Any) -> tuple[Any, Any]:
    if left is Dyn or right is Dyn or isinstance(left, Var) or isinstance(right, Var):
        return left, right
    if not isinstance(left, TensorType) or not isinstance(right, TensorType):
        raise TypeError(f"cannot broadcast {left!r} and {right!r}")
    dims_left = list(left.dims)
    dims_right = list(right.dims)
    if len(dims_left) < len(dims_right):
        dims_left[:0] = [1] * (len(dims_right) - len(dims_left))
    elif len(dims_right) < len(dims_left):
        dims_right[:0] = [1] * (len(dims_left) - len(dims_right))
    for index, (dim_left, dim_right) in enumerate(zip(dims_left, dims_right)):
        if dim_left == 1:
            dims_left[index] = dim_right
        elif dim_right == 1:
            dims_right[index] = dim_left
        elif not is_consistent(dim_left, dim_right):
            raise TypeError(f"incompatible broadcast dimensions {dim_left!r} and {dim_right!r}")
    return TensorType(tuple(dims_left)), TensorType(tuple(dims_right))


def _register(table: dict[Any, Callable[..., Any]], target: Any) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    def decorate(function: Callable[..., _T]) -> Callable[..., _T]:
        if target in table:
            raise RuntimeError(f"a rule is already registered for {target!r}")
        table[target] = function
        return function

    return decorate


def register_inference_rule(target: Any) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    return _register(_INFERENCE_RULES, target)


def register_refinement_rule(target: Any) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    return _register(_REFINEMENT_RULES, target)


def register_algebraic_expressions_inference_rule(target: Any) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    return _register(_RULES, target)


def _node_arg(node: Node, index: int) -> Node:
    if index >= len(node.args) or not isinstance(node.args[index], Node):
        raise TypeError(f"argument {index} of {node.name} is not a graph value")
    return node.args[index]


@register_inference_rule(tp.add)
@register_inference_rule(operator.add)
def add_inference_rule(node: Node) -> Any:
    left = _node_arg(node, 0).type
    right = _node_arg(node, 1).type
    if left is int and isinstance(right, TensorType):
        node.type = right
        return node.type
    if right is int and isinstance(left, TensorType):
        node.type = left
        return node.type
    new_left, new_right = broadcast_types(left, right)
    node.meta["broadcast"] = new_left != left or new_right != right
    if node.meta["broadcast"]:
        node.meta[str(node.args[0])] = new_left
        node.meta[str(node.args[1])] = new_right
    if not is_consistent(new_left, new_right):
        raise TypeError(f"incompatible add operands {left!r} and {right!r}")
    node.type = new_right if is_more_precise(new_left, new_right) else new_left
    return node.type


@register_inference_rule(getattr)
def get_attr_inference_rule(node: Node, traced: Any) -> Any:
    if len(node.args) < 2 or node.args[1] != "shape":
        raise TypeError("only the shape attribute has a gradual type rule")
    del traced
    node.type = Dyn
    return node.type


@register_inference_rule(tp.transpose)
def transpose_inference_rule(node: Node) -> Any:
    value = _node_arg(node, 0).type
    if value is Dyn:
        node.type = Dyn
        return node.type
    if not isinstance(value, TensorType) or len(node.args) < 3:
        raise TypeError(f"invalid transpose node {node.name}")
    dim0, dim1 = node.args[1], node.args[2]
    if not isinstance(dim0, int) or not isinstance(dim1, int):
        raise TypeError("transpose dimensions must be integers")
    rank = len(value.dims)
    if not (-rank <= dim0 < rank and -rank <= dim1 < rank):
        raise TypeError(f"transpose dimensions {dim0}, {dim1} exceed rank {rank}")
    dims = list(value.dims)
    dim0 %= rank
    dim1 %= rank
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    node.type = get_greatest_upper_bound(node.type, TensorType(tuple(dims)))
    return node.type


@register_inference_rule(tp.reshape)
def reshape_inference_rule(node: Node) -> TensorType:
    value = _node_arg(node, 0).type
    if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
        raise TypeError("reshape requires a sequence shape")
    requested = list(node.args[1])
    result = TensorType(tuple(Dyn if dim == -1 else dim for dim in requested))
    if value is Dyn:
        node.type = result
        return result
    if not isinstance(value, TensorType):
        raise TypeError(f"cannot reshape value of type {value!r}")
    known = [1 if dim is Dyn else dim for dim in value.dims]
    requested_known = [1 if dim == -1 else dim for dim in requested]
    old_count = reduce(operator.mul, known, 1)
    new_count = reduce(operator.mul, requested_known, 1)
    if old_count and new_count and old_count % new_count and new_count % old_count:
        raise TypeError(f"reshape changes the known element count from {old_count} to {new_count}")
    node.type = result
    return result


def _pair(value: Any) -> tuple[Any, Any]:
    if isinstance(value, int):
        return value, value
    return tuple(value)


def _module_attr(module: Any, name: str, default: Any = None) -> Any:
    return getattr(module, name, default)


@register_inference_rule(nn.BatchNorm2d)
def bn2d_inference_rule(node: Node, module: Any) -> Any:
    input_type = expand_to_tensor_dim(_node_arg(node, 0).type, 4)
    output_type = expand_to_tensor_dim(node.type, 4)
    features = _module_attr(module, "num_features")
    if not is_consistent(input_type.dims[1], features) or not is_consistent(output_type.dims[1], features):
        raise TypeError(f"batch normalization channel count is inconsistent for {node.name}")
    if not is_consistent(input_type, output_type):
        raise TypeError(f"batch normalization changes incompatible dimensions for {node.name}")
    node.type = get_greatest_upper_bound(input_type, output_type)
    return node.type


def calculate_out_dimension(d_in: Any, module: Any, index: int) -> Any:
    if d_in is Dyn:
        return Dyn
    if not isinstance(d_in, (int, sympy.Basic)):
        raise TypeError(f"dimension must be numeric or symbolic, got {d_in!r}")
    padding = _pair(module.padding)
    kernel = _pair(module.kernel_size)
    stride = _pair(module.stride)
    dilation = _pair(module.dilation)
    numerator = d_in + 2 * padding[index] - dilation[index] * (kernel[index] - 1) - 1
    return numerator // stride[index] + 1


def get_greatest_upper_bound(left: Any, right: Any) -> Any:
    if left is Dyn:
        return right
    if right is Dyn:
        return left
    if isinstance(left, TensorType) and isinstance(right, TensorType):
        if not is_consistent(left, right):
            raise TypeError(f"inconsistent tensor types {left!r} and {right!r}")
        dims = [a if is_more_precise(a, b) else b for a, b in zip(left.dims, right.dims)]
        return TensorType(tuple(dims))
    if left == right:
        return left
    return Dyn


@register_inference_rule(nn.Conv2d)
def conv2d_inference_rule(node: Node, module: Any) -> Any:
    input_type = expand_to_tensor_dim(_node_arg(node, 0).type, 4)
    current_type = expand_to_tensor_dim(node.type, 4)
    if not is_consistent(input_type.dims[1], module.in_channels):
        raise TypeError(f"conv input channels do not match for {node.name}")
    output = TensorType(
        (
            input_type.dims[0],
            module.out_channels,
            calculate_out_dimension(input_type.dims[2], module, 0),
            calculate_out_dimension(input_type.dims[3], module, 1),
        )
    )
    node.type = get_greatest_upper_bound(output, current_type)
    return node.type


@register_inference_rule(nn.ReLU)
def relu_inference_rule(node: Node, module: Any) -> Any:
    del module
    value = _node_arg(node, 0).type
    if value is Dyn and isinstance(node.type, TensorType):
        value = expand_to_tensor_dim(value, len(node.type.dims))
    if isinstance(value, TensorType):
        node.type = get_greatest_upper_bound(value, node.type)
    return node.type


def maxpool2d_check(value: TensorType, module: Any) -> TensorType:
    if len(value.dims) not in {3, 4}:
        raise TypeError(f"pooling expects rank 3 or 4, got {value!r}")
    dims = list(value.dims)
    dims[-2] = calculate_out_dimension(dims[-2], module, 0)
    dims[-1] = calculate_out_dimension(dims[-1], module, 1)
    return TensorType(tuple(dims))


@register_inference_rule(nn.MaxPool2d)
def maxpool2d_inference_rule(node: Node, module: Any) -> Any:
    value = _node_arg(node, 0).type
    if value is Dyn and isinstance(node.type, TensorType):
        value = expand_to_tensor_dim(value, len(node.type.dims))
    if isinstance(value, TensorType):
        node.type = get_greatest_upper_bound(maxpool2d_check(value, module), node.type)
    return node.type


def linear_check(value: TensorType, module: Any) -> TensorType:
    if len(value.dims) < 2:
        raise TypeError(f"linear input must have rank at least two, got {value!r}")
    if not is_consistent(value.dims[-1], module.in_features):
        raise TypeError(f"linear input feature count does not match for {module!r}")
    dims = list(value.dims)
    dims[-1] = module.out_features
    return TensorType(tuple(dims))


@register_inference_rule(nn.Linear)
def linear_inference_rule(node: Node, module: Any) -> Any:
    value = _node_arg(node, 0).type
    if value is Dyn and isinstance(node.type, TensorType):
        value = expand_to_tensor_dim(value, len(node.type.dims))
    if isinstance(value, TensorType):
        node.type = get_greatest_upper_bound(linear_check(value, module), node.type)
    return node.type


def adaptiveavgpool2d_check(value: TensorType, module: Any) -> TensorType:
    if len(value.dims) not in {3, 4}:
        raise TypeError(f"adaptive pooling expects rank 3 or 4, got {value!r}")
    output_size = module.output_size
    output_size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
    output_h = output_size[0] if output_size[0] is not None else output_size[1]
    output_w = output_size[1] if output_size[1] is not None else output_size[0]
    dims = list(value.dims)
    dims[-2], dims[-1] = output_h, output_w
    return TensorType(tuple(dims))


@register_inference_rule(nn.AdaptiveAvgPool2d)
def adaptiveavgpool2d_inference_rule(node: Node, module: Any) -> Any:
    value = _node_arg(node, 0).type
    if value is Dyn and isinstance(node.type, TensorType):
        value = expand_to_tensor_dim(value, len(node.type.dims))
    if isinstance(value, TensorType):
        node.type = get_greatest_upper_bound(adaptiveavgpool2d_check(value, module), node.type)
    return node.type


def flatten_check(value: TensorType, start_dim: int, end_dim: int) -> TensorType:
    rank = len(value.dims)
    start = start_dim % rank
    end = end_dim + rank if end_dim < 0 else end_dim
    if not 0 <= start <= end < rank:
        raise TypeError(f"invalid flatten range {start_dim}, {end_dim} for rank {rank}")
    middle = value.dims[start : end + 1]
    flattened: Any = Dyn if Dyn in middle else reduce(operator.mul, middle, 1)
    return TensorType(tuple(value.dims[:start]) + (flattened,) + tuple(value.dims[end + 1 :]))


@register_inference_rule(tp.flatten)
def flatten_inference_rule(node: Node) -> Any:
    value = _node_arg(node, 0).type
    start = node.args[1] if len(node.args) > 1 else 0
    end = node.args[2] if len(node.args) > 2 else -1
    if value is Dyn and isinstance(node.type, TensorType):
        value = expand_to_tensor_dim(value, len(node.type.dims))
    if isinstance(value, TensorType):
        node.type = get_greatest_upper_bound(flatten_check(value, start, end), node.type)
    return node.type


def _resolve_module(traced: Any, target: str) -> Any:
    getter = getattr(traced, "get_submodule", None)
    if callable(getter):
        return getter(target)
    value = getattr(traced, "root", traced)
    for part in target.split("."):
        value = getattr(value, part)
    return value


def get_parameter(traced: Any, target: str) -> Any:
    getter = getattr(traced, "_get_attr", None)
    if callable(getter):
        value = getter(target)
    else:
        module_path, _, name = target.rpartition(".")
        value = getattr(_resolve_module(traced, module_path), name)
    if not hasattr(value, "shape"):
        raise AttributeError(f"graph attribute {target!r} is not tensor-like")
    return value


def _map_types(value: Any) -> Any:
    if isinstance(value, Node):
        return value.type
    if isinstance(value, tuple):
        return tuple(_map_types(item) for item in value)
    if isinstance(value, list):
        return [_map_types(item) for item in value]
    if isinstance(value, dict):
        return {key: _map_types(item) for key, item in value.items()}
    return value


class GraphTypeChecker:
    def __init__(self, env: dict[str, Any], traced: Any) -> None:
        self.env = env
        self.traced = traced

    def type_check(self) -> bool:
        for node in self.traced.graph.nodes:
            self.type_check_node(node)
        return True

    def type_check_node(self, node: Node) -> Any:
        if node.type is None:
            node.type = self.env.get(node.name, Dyn)
        if node.op == "placeholder":
            return node.type
        if node.op == "get_attr":
            value = get_parameter(self.traced, node.target)
            shape = value.shape() if callable(value.shape) else value.shape
            node.type = TensorType(tuple(shape))
            return node.type
        if node.op == "call_function":
            if node.target not in _INFERENCE_RULES:
                raise RuntimeError(f"no inference rule registered for {node.target!r}")
            rule = _INFERENCE_RULES[node.target]
            return rule(node, self.traced) if node.target is getattr else rule(node)
        if node.op == "call_method":
            target = {"transpose": tp.transpose, "reshape": tp.reshape, "view": tp.reshape, "flatten": tp.flatten}.get(node.target)
            if target is None or target not in _INFERENCE_RULES:
                raise RuntimeError(f"no inference rule registered for method {node.target!r}")
            return _INFERENCE_RULES[target](node)
        if node.op == "call_module":
            module = _resolve_module(self.traced, node.target)
            rule = _INFERENCE_RULES.get(type(module))
            if rule is None:
                raise RuntimeError(f"no inference rule registered for {type(module)!r}")
            return rule(node, module)
        if node.op == "output":
            node.type = _map_types(node.args[0])
            return node.type
        raise NotImplementedError(f"unsupported graph operation {node.op!r}")


@register_refinement_rule(nn.Conv2d)
def conv_refinement_rule(node: Node) -> list[Any] | None:
    value = _node_arg(node, 0).type
    if isinstance(value, TensorType) and isinstance(node.type, TensorType):
        return [Equality(value.dims[0], node.type.dims[0])]
    return None


@register_refinement_rule(nn.Linear)
def linear_refinement_rule(node: Node) -> list[Any]:
    value = _node_arg(node, 0).type
    if isinstance(value, TensorType) and isinstance(node.type, TensorType):
        return [Equality(value.dims[0], node.type.dims[0])]
    return []


def _same_shape(node: Node, count: int | None = None) -> list[Any]:
    value = _node_arg(node, 0).type
    if not isinstance(value, TensorType) or not isinstance(node.type, TensorType):
        return []
    dims = value.dims if count is None else value.dims[:count]
    target = node.type.dims if count is None else node.type.dims[:count]
    return [Equality(left, right) for left, right in zip(dims, target)]


@register_refinement_rule(nn.BatchNorm2d)
@register_refinement_rule(nn.ReLU)
def all_eq(node: Node) -> list[Any]:
    return _same_shape(node)


@register_refinement_rule(nn.AdaptiveAvgPool2d)
@register_refinement_rule(nn.MaxPool2d)
def first_two_eq(node: Node) -> list[Any]:
    return _same_shape(node, 2)


@register_refinement_rule(tp.add)
@register_refinement_rule(operator.add)
def element_wise_eq(node: Node) -> list[Any]:
    first = _node_arg(node, 0).type
    second = _node_arg(node, 1).type
    if not isinstance(first, TensorType) or not isinstance(second, TensorType) or not isinstance(node.type, TensorType):
        return []
    first, second = broadcast_types(first, second)
    return [Equality(left, output) for left, right, output in zip(first.dims, second.dims, node.type.dims) if left == right]


@register_refinement_rule(tp.flatten)
def flatten_refinement_rule(node: Node) -> list[Any]:
    value = _node_arg(node, 0).type
    if not isinstance(value, TensorType) or not isinstance(node.type, TensorType):
        return []
    start = node.args[1] if len(node.args) > 1 else 0
    end = node.args[2] if len(node.args) > 2 else -1
    rank = len(node.type.dims)
    start %= rank
    end = end + rank if end < 0 else end
    result = [Equality(left, right) for left, right in zip(node.type.dims[:start], value.dims[:start])]
    result.extend(Equality(left, right) for left, right in zip(node.type.dims[start + 1 :], value.dims[end + 1 :]))
    return result


@register_algebraic_expressions_inference_rule(nn.Conv2d)
def conv_rule(node: Node, module: Any) -> TensorType | None:
    value = _node_arg(node, 0).type
    if isinstance(value, TensorType) and isinstance(node.type, TensorType):
        result = TensorType(
            (
                node.type.dims[0],
                node.type.dims[1],
                calculate_out_dimension(value.dims[2], module, 0),
                calculate_out_dimension(value.dims[3], module, 1),
            )
        )
        node.type = result
        return result
    return None


class Refine:
    """Generate equality constraints and symbolic dimension relations."""

    def __init__(self, traced: Any) -> None:
        self.constraints: list[Any] = []
        self.traced = traced
        self.symbol_iter = itertools.count()

    def refine(self) -> bool:
        for node in self.traced.graph.nodes:
            self.refine_node(node)
        return True

    def symbolic_relations(self) -> bool:
        for node in self.traced.graph.nodes:
            self.infer_symbolic_relations(node)
        return True

    def replace_dyn_with_fresh_var(self, value: Any) -> Any:
        if value is Dyn:
            return Var(next(self.symbol_iter))
        if isinstance(value, TensorType):
            return TensorType(tuple(self.replace_dyn_with_fresh_var(item) for item in value.dims))
        if isinstance(value, list):
            return [self.replace_dyn_with_fresh_var(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.replace_dyn_with_fresh_var(item) for item in value)
        return value

    def convert_to_sympy_symbols(self, value: Any) -> Any:
        if isinstance(value, Var):
            return sympy.Symbol(str(value))
        if isinstance(value, TensorType):
            return TensorType(tuple(self.convert_to_sympy_symbols(item) for item in value.dims))
        if isinstance(value, list):
            return [self.convert_to_sympy_symbols(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.convert_to_sympy_symbols(item) for item in value)
        return value

    def refine_node(self, node: Node) -> Any:
        node.type = self.replace_dyn_with_fresh_var(node.type if node.type is not None else Dyn)
        if node.op == "call_function" and node.target in _REFINEMENT_RULES:
            constraints = _REFINEMENT_RULES[node.target](node)
            if constraints:
                self.constraints.extend(constraints)
        elif node.op == "call_module":
            module = _resolve_module(self.traced, node.target)
            rule = _REFINEMENT_RULES.get(type(module))
            if rule is not None:
                constraints = rule(node)
                if constraints:
                    self.constraints.extend(constraints)
        elif node.op == "output":
            node.type = _map_types(node.args[0])
        return node.type

    def infer_symbolic_relations(self, node: Node) -> Any:
        node.type = self.convert_to_sympy_symbols(node.type)
        if node.op == "call_function" and node.target in _RULES:
            return _RULES[node.target](node)
        if node.op == "call_module":
            module = _resolve_module(self.traced, node.target)
            rule = _RULES.get(type(module))
            if rule is not None:
                return rule(node, module)
        if node.op == "output":
            node.type = _map_types(node.args[0])
        return node.type

