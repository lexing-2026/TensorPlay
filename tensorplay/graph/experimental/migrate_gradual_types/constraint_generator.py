from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias, TypeVar

from .... import functional as functional
from .... import nn
from ....nn import functional as nn_functional
from ....nn.modules.batchnorm import BatchNorm2d
from ....nn.modules.conv import Conv2d
from ....nn.modules.linear import Linear
from ....nn.modules.pooling import AdaptiveAvgPool2d, MaxPool2d
from ..._utils import _iter_nodes
from ...node import Node
from ...tensor_type import Dyn, TensorType
from .constraint import (
    ApplyBroadcasting,
    BinConstraintD,
    BinConstraintT,
    BVar,
    CalcConv,
    CalcMaxPool,
    CalcProduct,
    CanReshape,
    Conj,
    Constraint,
    DGreatestUpperBound,
    Disj,
    DVar,
    F,
    GetItem,
    GetItemTensor,
    IndexSelect,
    T,
    TGreatestUpperBound,
    Transpose,
    TVar,
)
from .operation import (
    op_add,
    op_consistency,
    op_div,
    op_eq,
    op_gt,
    op_leq,
    op_lt,
    op_matching,
    op_mul,
    op_neq,
    op_precision,
    op_sub,
)
from .util import gen_bvar, gen_dvar, gen_nat_constraints, gen_tensor_dims, gen_tvar

_T = TypeVar("_T")
_Symbol: TypeAlias = TVar | DVar | BVar
_SymbolDict: TypeAlias = dict[Node, _Symbol]
_Rule: TypeAlias = Callable[..., tuple[list[Constraint], int]]

MAX_TENSOR_RANK = 4
_INFERENCE_RULES: dict[Any, _Rule] = {}

__all__ = [
    "ConstraintGenerator",
    "MAX_TENSOR_RANK",
    "adaptive_inference_rule",
    "add_layer_norm_constraints",
    "add_linear_constraints",
    "arange_inference_rule",
    "assert_inference_rule",
    "batchnorm_inference_rule",
    "bmm_inference_rule",
    "broadcasting_inference_rule",
    "conv2d_inference_rule",
    "cumsum_inference_rule",
    "embedding_inference_rule",
    "embedding_inference_rule_functional",
    "eq_inference_rule",
    "equality_inference_rule",
    "expand_inference_rule",
    "flatten_inference_rule",
    "full_inference_rule",
    "gen_broadcasting_constraints",
    "gen_embedding_rules",
    "gen_layer_norm_constraints",
    "generate_flatten_constraints",
    "get_attr_inference_rule",
    "getitem_inference_rule",
    "gt_inference_rule",
    "index_select_inference_rule",
    "layer_norm_functional",
    "layer_norm_inference_rule",
    "linear_constraints",
    "linear_inference_rule",
    "lt_inference_rule",
    "masked_fill_inference_rule",
    "maxpool_inference_rule",
    "neq_inference_rule",
    "range_check",
    "register_inference_rule",
    "relu_inference_rule",
    "reshape_inference_rule",
    "size_inference_rule",
    "tensor_inference_rule",
    "torch_dim_inference_rule",
    "torch_linear_inference_rule",
    "transpose_inference_rule",
    "type_inference_rule",
    "view_inference_rule",
]


def register_inference_rule(call_target: Any) -> Callable[[_Rule], _Rule]:
    def register(fn: _Rule) -> _Rule:
        previous = _INFERENCE_RULES.get(call_target)
        if previous is not None and previous is not fn:
            raise RuntimeError(f"inference rule already registered for {call_target!r}")
        _INFERENCE_RULES[call_target] = fn
        return fn

    return register


def _register_names(fn: _Rule, *names: str) -> _Rule:
    for name in names:
        _INFERENCE_RULES.setdefault(name, fn)
    return fn


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target.rsplit(".", 1)[-1]
    return getattr(target, "__name__", getattr(target, "name", type(target).__name__))


def _lookup_rule(target: Any) -> _Rule | None:
    direct = _INFERENCE_RULES.get(target)
    if direct is not None:
        return direct
    name = _target_name(target)
    return _INFERENCE_RULES.get(name)


def _require_node(value: Any, position: str) -> Node:
    if not isinstance(value, Node):
        raise AssertionError(f"expected graph node at {position}, got {type(value)}")
    return value


def _symbol(symbols: _SymbolDict, value: Any) -> Any:
    return symbols[value] if isinstance(value, Node) else value


def _dim(value: Any, symbols: _SymbolDict) -> Any:
    resolved = _symbol(symbols, value)
    if isinstance(resolved, (DVar, int)) or resolved == Dyn:
        return resolved
    raise AssertionError(f"expected a dimension, got {type(resolved)}")


def _tensor(value: Any, symbols: _SymbolDict) -> TVar:
    resolved = _symbol(symbols, value)
    if not isinstance(resolved, TVar):
        raise AssertionError(f"expected a tensor variable, got {type(resolved)}")
    return resolved


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(value)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) != 2:
        raise ValueError("a spatial parameter must contain one or two values")
    return int(values[0]), int(values[1])


def generate_flatten_constraints(
    start_dim: int,
    end_dim: int,
    input: TVar,
    flattened: TVar,
    n: int,
    counter: int,
) -> tuple[Conj, int]:
    dims, counter = gen_tensor_dims(n, counter)
    start = n if start_dim == -1 else (start_dim if start_dim >= 0 else n + start_dim)
    end = n - 1 if end_dim == -1 else (end_dim if end_dim >= 0 else n + end_dim)
    if not 0 <= start <= end < n:
        return Conj([F()]), counter
    return (
        Conj(
            [
                BinConstraintT(input, TensorType(dims), op_eq),
                CalcProduct(start, end + 1, flattened, dims),
                *gen_nat_constraints(dims),
            ]
        ),
        counter,
    )


def get_attr_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _require_node(n.args[0], "get_attr receiver")
    output, counter = gen_tvar(counter)
    symbols[n] = output
    if n.args[1] == "device":
        return [BinConstraintT(symbols[source], output, op_eq)], counter
    raise NotImplementedError(f"attribute inference is not implemented for {n.args[1]!r}")


def bmm_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    left = _tensor(n.args[0], symbols)
    right = _tensor(n.args[1], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    left_dims, counter = gen_tensor_dims(3, counter)
    right_dims, counter = gen_tensor_dims(3, counter)
    batch, counter = gen_dvar(counter)
    return [
        Disj(
            [
                Conj([BinConstraintT(left, Dyn, op_eq), BinConstraintT(right, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)]),
                Conj([
                    BinConstraintT(left, Dyn, op_eq),
                    BinConstraintT(right, TensorType(right_dims), op_eq),
                    BinConstraintT(output, TensorType([right_dims[0], Dyn, right_dims[2]]), op_eq),
                    *gen_nat_constraints(right_dims),
                ]),
                Conj([
                    BinConstraintT(right, Dyn, op_eq),
                    BinConstraintT(left, TensorType(left_dims), op_eq),
                    BinConstraintT(output, TensorType([left_dims[0], left_dims[1], Dyn]), op_eq),
                    *gen_nat_constraints(left_dims),
                ]),
                Conj([
                    BinConstraintT(left, TensorType(left_dims), op_eq),
                    BinConstraintT(right, TensorType(right_dims), op_eq),
                    BinConstraintT(output, TensorType([batch, left_dims[1], right_dims[2]]), op_eq),
                    BinConstraintD(left_dims[0], right_dims[0], op_consistency),
                    DGreatestUpperBound(batch, left_dims[0], right_dims[0]),
                    *gen_nat_constraints(left_dims + right_dims),
                ]),
            ]
        )
    ], counter


def index_select_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    index = n.args[1]
    if not isinstance(index, int):
        raise AssertionError("index_select dimension must be an integer")
    index_tensor = _tensor(n.args[2], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    index_dim, counter = gen_dvar(counter)
    branches: list[Constraint] = []
    for rank in range(1, MAX_TENSOR_RANK + 1):
        branches.append(
            Conj([
                BinConstraintT(index_tensor, TensorType([index_dim]), op_eq),
                IndexSelect(rank, source, index_dim, index, output),
                BinConstraintD(0, index_dim, op_leq),
            ])
        )
        branches.append(
            Conj([
                BinConstraintT(index_tensor, Dyn, op_eq),
                IndexSelect(rank, source, Dyn, index, output),
            ])
        )
    return [Disj(branches)], counter


def expand_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    requested = list(n.args[1]) if len(n.args) == 2 and isinstance(n.args[1], (tuple, list)) else list(n.args[1:])
    target_dims = [_dim(item, symbols) for item in requested]
    target, counter = gen_tvar(counter)
    target_type = TensorType(target_dims)
    constraints, counter = gen_broadcasting_constraints(source, target, symbols, counter, output)
    constraints.append(BinConstraintT(target, target_type, op_eq))
    constraints.extend(BinConstraintD(0, dim, op_leq) for dim in target_dims if isinstance(dim, DVar))
    constraints.append(BinConstraintT(output, target_type, op_eq))
    return constraints, counter


def equality_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    output, counter = gen_tvar(counter)
    symbols[n] = output
    if not n.args:
        raise RuntimeError("shape-preserving operation has no input")
    first = n.args[0]
    if isinstance(first, Node):
        value = symbols[first]
        if isinstance(value, TVar):
            return [BinConstraintT(value, output, op_eq)], counter
        dims = [_dim(item, symbols) for item in n.args]
        return [BinConstraintT(output, TensorType(dims), op_eq)], counter
    if isinstance(first, (tuple, list)):
        return [BinConstraintT(output, TensorType([_dim(item, symbols) for item in first]), op_eq)], counter
    raise NotImplementedError(f"cannot infer shape-preserving target {n.target!r}")


def transpose_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    first, second = n.args[1], n.args[2]
    if not isinstance(first, int) or not isinstance(second, int):
        raise AssertionError("transpose dimensions must be integers")
    output, counter = gen_tvar(counter)
    symbols[n] = output
    return [
        Disj([
            Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)]),
            Disj([Transpose(rank, source, first, second, output) for rank in range(1, MAX_TENSOR_RANK + 1)]),
        ])
    ], counter


def type_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    target = _tensor(n.args[1], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    return [BinConstraintT(source, target, op_consistency), BinConstraintT(output, target, op_eq)], counter


def masked_fill_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    mask = _tensor(n.args[1], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    return gen_broadcasting_constraints(source, mask, symbols, counter, output)


def gen_embedding_rules(
    n: Node, symbols: _SymbolDict, embedding_dim: int | DVar, counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    branches: list[Constraint] = [Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])]
    for rank in range(1, MAX_TENSOR_RANK):
        dims, counter = gen_tensor_dims(rank, counter)
        branches.append(
            Conj([
                BinConstraintT(source, TensorType(dims), op_eq),
                BinConstraintT(output, TensorType(dims + [embedding_dim]), op_eq),
                *gen_nat_constraints(dims),
            ])
        )
    return [Disj(branches)], counter


def embedding_inference_rule_functional(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    _require_node(n.args[0], "embedding input")
    weights = _tensor(n.args[1], symbols)
    dims, counter = gen_tensor_dims(2, counter)
    generated, counter = gen_embedding_rules(n, symbols, dims[1], counter)
    return [BinConstraintT(weights, TensorType(dims), op_eq), *generated], counter


def embedding_inference_rule(
    n: Node, module_instance: Any, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    return gen_embedding_rules(n, symbols, int(module_instance.embedding_dim), counter)


def tensor_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    return [], counter


def view_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    shape_args = list(n.args[1]) if len(n.args) == 2 and isinstance(n.args[1], (tuple, list)) else list(n.args[1:])
    target_dims: list[Any] = []
    extra: list[Constraint] = []
    for item in shape_args:
        item = _symbol(symbols, item)
        if item == -1:
            unknown, counter = gen_dvar(counter)
            target_dims.append(unknown)
            extra.append(BinConstraintD(unknown, Dyn, op_neq))
        else:
            target_dims.append(_dim(item, symbols))
            extra.append(BinConstraintD(target_dims[-1], Dyn, op_neq))
    target = TensorType(target_dims)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    return [BinConstraintT(output, target, op_eq), CanReshape(source, target), *extra], counter


def size_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    if len(n.args) == 1:
        output, counter = gen_tvar(counter)
        symbols[n] = output
        return [BinConstraintT(source, output, op_eq)], counter
    if len(n.args) != 2 or not isinstance(n.args[1], int):
        raise NotImplementedError("size expects zero or one dimension selector")
    output, counter = gen_dvar(counter)
    symbols[n] = output
    branches = [
        Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintD(output, Dyn, op_eq)])
    ]
    for rank in range(1, MAX_TENSOR_RANK + 1):
        branches.append(Conj([GetItem(rank, n.args[1], output, source), BinConstraintD(0, output, op_leq)]))
    return [Disj(branches)], counter


def range_check(i: int, n: int) -> T | F:
    return T() if (0 <= i < n or -n <= i < 0) else F()


def cumsum_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    dim = n.args[1] if len(n.args) > 1 else n.kwargs.get("dim", 0)
    if not isinstance(dim, int):
        raise AssertionError("cumsum dimension must be an integer")
    output, counter = gen_tvar(counter)
    symbols[n] = output
    branches: list[Constraint] = [Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])]
    for rank in range(1, MAX_TENSOR_RANK + 1):
        dims, counter = gen_tensor_dims(rank, counter)
        branches.append(Conj([BinConstraintT(source, TensorType(dims), op_eq), BinConstraintT(output, TensorType(dims), op_eq), range_check(dim, rank), *gen_nat_constraints(dims)]))
    return [Disj(branches)], counter


def assert_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if n.users:
        raise AssertionError("assertion helper must be terminal")
    return [], counter


def getitem_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    index = n.args[1]
    if isinstance(index, int):
        output, counter = gen_dvar(counter)
        symbols[n] = output
        return [
            Disj([
                Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintD(output, Dyn, op_eq)]),
                Conj([Disj([GetItem(rank, index, output, source) for rank in range(1, MAX_TENSOR_RANK + 1)]), BinConstraintD(0, output, op_leq)]),
            ])
        ], counter
    if isinstance(index, tuple):
        output, counter = gen_tvar(counter)
        symbols[n] = output
        return [
            Disj([
                Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)]),
                *[GetItemTensor(rank, index, output, source) for rank in range(1, MAX_TENSOR_RANK + 1)],
            ])
        ], counter
    raise RuntimeError(f"unsupported index type {type(index)}")


def _comparison_rule(
    n: Node, symbols: _SymbolDict, counter: int, op: str
) -> tuple[list[Constraint], int]:
    left = _symbol(symbols, n.args[0])
    right = _symbol(symbols, n.args[1])
    if isinstance(left, TVar) and isinstance(right, TVar):
        output, counter = gen_tvar(counter)
        symbols[n] = output
        return gen_broadcasting_constraints(left, right, symbols, counter, output)
    if isinstance(left, (DVar, int)) and isinstance(right, (DVar, int)):
        result, counter = gen_bvar(counter)
        symbols[n] = result
        return [BinConstraintD(result, BinConstraintD(left, right, op), op_eq)], counter
    raise RuntimeError("comparison operands have incompatible inferred sorts")


def gt_inference_rule(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return _comparison_rule(n, symbols, counter, op_gt)


def eq_inference_rule(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return _comparison_rule(n, symbols, counter, op_eq)


def lt_inference_rule(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return _comparison_rule(n, symbols, counter, op_lt)


def neq_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if len(n.args) != 2 or not isinstance(n.args[1], tuple):
        return _comparison_rule(n, symbols, counter, op_neq)
    source = _tensor(n.args[0], symbols)
    target = n.args[1]
    if len(target) not in (3, 4):
        raise NotImplementedError("tensor inequality inference supports ranks three and four")
    dims, counter = gen_tensor_dims(len(target), counter)
    parts: list[Constraint] = [BinConstraintT(source, TensorType(dims), op_eq)]
    differences: list[Constraint] = []
    for value, dimension in zip(target, dims):
        lhs = _symbol(symbols, value)
        differences.append(Conj([BinConstraintD(lhs, Dyn, op_neq), BinConstraintD(dimension, Dyn, op_neq), BinConstraintD(lhs, dimension, op_neq)]))
    result, counter = gen_bvar(counter)
    symbols[n] = result
    return [BinConstraintD(result, Conj(parts + [Disj(differences)]), op_eq)], counter


def full_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    output, counter = gen_tvar(counter)
    symbols[n] = output
    size = n.args[0] if n.args else n.kwargs.get("size")
    if not isinstance(size, Iterable):
        raise AssertionError("full size must be iterable")
    dims = [_dim(item, symbols) for item in size]
    return [BinConstraintT(output, TensorType(dims), op_eq)], counter


def arange_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    args = list(n.args)
    if len(args) == 1:
        start, end, step = 0, _symbol(symbols, args[0]), 1
    elif len(args) == 2:
        start, end, step = _symbol(symbols, args[0]), _symbol(symbols, args[1]), 1
    elif len(args) == 3:
        start, end, step = (_symbol(symbols, value) for value in args)
    else:
        raise NotImplementedError("arange accepts one to three positional arguments")
    size, counter = gen_dvar(counter)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    formula = BinConstraintD(size, BinConstraintD(BinConstraintD(end, start, op_sub), step, op_div), op_eq)
    dynamic = Conj([BinConstraintD(value, Dyn, op_eq) for value in (start, end, step)] + [BinConstraintD(size, Dyn, op_eq)])
    concrete = Conj([BinConstraintD(value, Dyn, op_neq) for value in (start, end, step)] + [BinConstraintD(size, Dyn, op_neq), formula])
    return [BinConstraintT(output, TensorType([size]), op_eq), Disj([dynamic, concrete])], counter


def gen_broadcasting_constraints(
    e1: TVar, e2: TVar, symbols: _SymbolDict, counter: int, output_var: TVar
) -> tuple[list[Constraint], int]:
    first, counter = gen_tvar(counter)
    second, counter = gen_tvar(counter)
    return [
        TGreatestUpperBound(output_var, first, second),
        ApplyBroadcasting(first, second, e1, e2),
        BinConstraintT(first, second, op_consistency),
    ], counter


def broadcasting_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    left = _symbol(symbols, n.args[0])
    right = _symbol(symbols, n.args[1])
    op = op_mul if _target_name(n.target) == "mul" else op_add
    if isinstance(left, TVar) and isinstance(right, TVar):
        output, counter = gen_tvar(counter)
        symbols[n] = output
        return gen_broadcasting_constraints(left, right, symbols, counter, output)
    if isinstance(left, TVar) and isinstance(right, (int, float)):
        output, counter = gen_tvar(counter)
        symbols[n] = output
        return [BinConstraintT(output, left, op_eq)], counter
    if isinstance(right, TVar) and isinstance(left, (int, float)):
        output, counter = gen_tvar(counter)
        symbols[n] = output
        return [BinConstraintT(output, right, op_eq)], counter
    if isinstance(left, DVar) and isinstance(right, (DVar, int, float)):
        output, counter = gen_dvar(counter)
        symbols[n] = output
        return [Conj([BinConstraintD(output, BinConstraintD(left, right, op), op_eq), BinConstraintD(0, output, op_leq)])], counter
    if isinstance(right, DVar) and isinstance(left, (DVar, int, float)):
        output, counter = gen_dvar(counter)
        symbols[n] = output
        return [Conj([BinConstraintD(output, BinConstraintD(right, left, op), op_eq), BinConstraintD(0, output, op_leq)])], counter
    raise NotImplementedError(f"broadcasting inference is unavailable for {type(left)} and {type(right)}")


def flatten_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    start = n.args[1] if len(n.args) > 1 else n.kwargs.get("start_dim", 1)
    end = n.args[2] if len(n.args) > 2 else n.kwargs.get("end_dim", -1)
    if not isinstance(start, int) or not isinstance(end, int):
        raise AssertionError("flatten dimensions must be integers")
    branches: list[Constraint] = [Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])]
    for rank in range(1, MAX_TENSOR_RANK + 1):
        branch, counter = generate_flatten_constraints(start, end, source, output, rank, counter)
        branches.append(branch)
    return [Disj(branches)], counter


def add_layer_norm_constraints(input_dim: list[DVar], normalized_dim: list[int]) -> list[Constraint]:
    if len(normalized_dim) > len(input_dim):
        return [F()]
    return [BinConstraintD(left, right, op_consistency) for left, right in zip(reversed(input_dim), reversed(normalized_dim))]


def gen_layer_norm_constraints(
    n: Node, normalized_shape: Sequence[int], symbols: _SymbolDict, counter: int
) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    branches: list[Constraint] = [Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])]
    for rank in range(1, MAX_TENSOR_RANK + 1):
        dims, counter = gen_tensor_dims(rank, counter)
        branches.append(Conj([BinConstraintT(source, TensorType(dims), op_eq), BinConstraintT(output, TensorType(dims), op_eq), *add_layer_norm_constraints(dims, list(normalized_shape)), *gen_nat_constraints(dims)]))
    return [Disj(branches)], counter


def layer_norm_functional(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return gen_layer_norm_constraints(n, n.args[1], symbols, counter)


def layer_norm_inference_rule(n: Node, module_instance: Any, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return gen_layer_norm_constraints(n, module_instance.normalized_shape, symbols, counter)


def relu_inference_rule(n: Node, module_instance: Any, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    return [BinConstraintT(source, output, op_eq)], counter


def add_linear_constraints(dims1: list[DVar], dims2: list[DVar], in_features: int | DVar, out_features: int | DVar) -> list[Constraint]:
    if len(dims1) != len(dims2):
        raise AssertionError("linear input and output ranks must match")
    return [
        *[BinConstraintD(a, b, op_eq) for a, b in zip(dims1, dims2) if a is not dims1[-1]],
        BinConstraintD(dims1[-1], in_features, op_consistency),
        BinConstraintD(dims2[-1], out_features, op_eq),
    ]


def linear_constraints(n: Node, in_features: int | DVar, out_features: int | DVar, symbols: _SymbolDict, counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    branches: list[Constraint] = [Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])]
    for rank in range(1, MAX_TENSOR_RANK + 1):
        input_dims, counter = gen_tensor_dims(rank, counter)
        output_dims, counter = gen_tensor_dims(rank, counter)
        branches.append(Conj([BinConstraintT(source, TensorType(input_dims), op_eq), BinConstraintT(output, TensorType(output_dims), op_eq), *add_linear_constraints(input_dims, output_dims, in_features, out_features), *gen_nat_constraints(input_dims + output_dims)]))
    return [Disj(branches)], counter


def linear_inference_rule(n: Node, module_instance: Any, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return linear_constraints(n, module_instance.in_features, module_instance.out_features, symbols, counter)


def torch_dim_inference_rule(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_dvar(counter)
    symbols[n] = output
    branches: list[Constraint] = [Conj([BinConstraintT(source, Dyn, op_eq), BinConstraintD(output, Dyn, op_eq)])]
    for rank in range(1, MAX_TENSOR_RANK + 1):
        dims, counter = gen_tensor_dims(rank, counter)
        branches.append(Conj([BinConstraintT(source, TensorType(dims), op_eq), BinConstraintD(output, rank, op_eq)]))
    return [Disj(branches)], counter


def torch_linear_inference_rule(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    weight = _tensor(n.args[1], symbols)
    dims, counter = gen_tensor_dims(2, counter)
    generated, counter = linear_constraints(n, dims[1], dims[0], symbols, counter)
    return [BinConstraintT(weight, TensorType(dims), op_eq), *generated], counter


def reshape_inference_rule(n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    return view_inference_rule(n, symbols, constraints, counter)


def batchnorm_inference_rule(n: Node, module_instance: BatchNorm2d, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    dims, counter = gen_tensor_dims(4, counter)
    return [BinConstraintT(source, TensorType(dims), op_matching), BinConstraintT(source, output, op_eq), *gen_nat_constraints(dims)], counter


def adaptive_inference_rule(n: Node, module_instance: AdaptiveAvgPool2d, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    dims, counter = gen_tensor_dims(4, counter)
    output_size = module_instance.output_size
    return [BinConstraintT(source, TensorType(dims), op_matching), BinConstraintT(output, TensorType([dims[0], dims[1], output_size[0], output_size[1]]), op_eq), *gen_nat_constraints(dims)], counter


def conv2d_inference_rule(n: Node, module_instance: Conv2d, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    dims, counter = gen_tensor_dims(4, counter)
    return [BinConstraintT(source, TensorType(dims), op_matching), BinConstraintD(module_instance.in_channels, dims[1], op_consistency), CalcConv(output, source, module_instance.out_channels, module_instance.kernel_size, module_instance.padding, module_instance.stride, module_instance.dilation, dims), *gen_nat_constraints(dims)], counter


def maxpool_inference_rule(n: Node, module_instance: MaxPool2d, symbols: _SymbolDict, constraints: list[Constraint], counter: int) -> tuple[list[Constraint], int]:
    source = _tensor(n.args[0], symbols)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    dims, counter = gen_tensor_dims(4, counter)
    return [BinConstraintT(source, TensorType(dims), op_matching), CalcMaxPool(output, source, module_instance.kernel_size, module_instance.padding, module_instance.stride, module_instance.dilation, dims), *gen_nat_constraints(dims)], counter


def _register_function_names(fn: _Rule, *names: str) -> None:
    _register_names(fn, *names)
    for module in (functional, nn_functional):
        for name in names:
            target = getattr(module, name, None)
            if target is not None:
                _INFERENCE_RULES.setdefault(target, fn)


_register_function_names(get_attr_inference_rule, "getattr")
_register_function_names(bmm_inference_rule, "bmm")
_register_function_names(index_select_inference_rule, "index_select")
_register_function_names(expand_inference_rule, "expand")
_register_function_names(equality_inference_rule, "gelu", "dropout", "softmax", "detach", "to", "int", "long", "contiguous", "ones", "zeros")
_register_function_names(transpose_inference_rule, "transpose")
_register_function_names(type_inference_rule, "type_as")
_register_function_names(masked_fill_inference_rule, "masked_fill", "masked_fill_")
_register_function_names(embedding_inference_rule_functional, "embedding")
_register_function_names(tensor_inference_rule, "tensor")
_register_function_names(view_inference_rule, "view")
_register_function_names(reshape_inference_rule, "reshape")
_register_function_names(size_inference_rule, "size")
_register_function_names(cumsum_inference_rule, "cumsum")
_register_function_names(getitem_inference_rule, "getitem")
_register_function_names(gt_inference_rule, "gt")
_register_function_names(eq_inference_rule, "eq")
_register_function_names(neq_inference_rule, "ne", "neq")
_register_function_names(lt_inference_rule, "lt")
_register_function_names(full_inference_rule, "full")
_register_function_names(arange_inference_rule, "arange")
_register_function_names(broadcasting_inference_rule, "add", "mul")
_register_function_names(flatten_inference_rule, "flatten")
_register_function_names(layer_norm_functional, "layer_norm")
_register_function_names(torch_dim_inference_rule, "dim")

for _target in (operator.add, operator.mul, operator.eq, operator.ne, operator.lt, operator.gt, operator.getitem):
    _INFERENCE_RULES.setdefault(_target, _lookup_rule(_target.__name__))

for _cls, _fn in (
    (nn.modules.sparse.Embedding, embedding_inference_rule),
    (nn.modules.normalization.LayerNorm, layer_norm_inference_rule),
    (nn.modules.dropout.Dropout, relu_inference_rule),
    (nn.modules.activation.ReLU, relu_inference_rule),
    (Linear, linear_inference_rule),
    (BatchNorm2d, batchnorm_inference_rule),
    (AdaptiveAvgPool2d, adaptive_inference_rule),
    (Conv2d, conv2d_inference_rule),
    (MaxPool2d, maxpool_inference_rule),
):
    _INFERENCE_RULES[_cls] = _fn


class ConstraintGenerator:
    def __init__(self, traced: Any, graph: Any | None = None) -> None:
        self.traced = traced
        self.traced_params = self._collect_attributes(traced)
        self.constraints: list[Constraint] = []
        self.symbol_dict: _SymbolDict = {}
        self.graph = getattr(traced, "graph", graph)

    @staticmethod
    def _collect_attributes(root: Any) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if hasattr(root, "named_parameters"):
            values.update(dict(root.named_parameters()))
        if hasattr(root, "named_buffers"):
            values.update(dict(root.named_buffers()))
        if hasattr(root, "__dict__"):
            values.update({key: value for key, value in vars(root).items() if not key.startswith("_")})
        return values

    def _get_submodule(self, target: str) -> Any:
        if hasattr(self.traced, "get_submodule"):
            return self.traced.get_submodule(target)
        value = self.traced
        for name in target.split("."):
            value = getattr(value, name)
        return value

    def generate_constraints(self, counter: int = 0) -> tuple[Conj, int]:
        if self.graph is None:
            raise ValueError("a graph is required for constraint generation")
        result: list[Constraint] = []
        for node in self.graph.nodes:
            generated, counter = self.generate_constraints_node(node, counter)
            result.extend(generated)
        self.constraints = result
        return Conj(result), counter

    def generate_constraints_node(self, n: Node, counter: int) -> tuple[list[Constraint], int]:
        if n.op == "placeholder":
            symbol, counter = gen_tvar(counter)
            self.symbol_dict[n] = symbol
            annotation = getattr(n, "type", None)
            if annotation is None:
                annotation = Dyn
            if annotation is not Dyn and not isinstance(annotation, TensorType):
                example = n.meta.get("example_value") if hasattr(n, "meta") else None
                shape = getattr(example, "shape", None)
                annotation = TensorType(tuple(shape)) if shape is not None else Dyn
            return [BinConstraintT(annotation, symbol, op_precision)], counter
        if n.op == "call_function" or n.op == "call_method":
            rule = _lookup_rule(n.target)
            if rule is None:
                raise RuntimeError(f"no inference rule registered for target {n.target!r}")
            return rule(n, self.symbol_dict, self.constraints, counter)
        if n.op == "call_module":
            module = self._get_submodule(n.target)
            rule = _INFERENCE_RULES.get(type(module))
            if rule is None:
                for cls, candidate in _INFERENCE_RULES.items():
                    if isinstance(cls, type) and isinstance(module, cls):
                        rule = candidate
                        break
            if rule is None:
                raise RuntimeError(f"no inference rule registered for module {type(module)!r}")
            return rule(n, module, self.symbol_dict, self.constraints, counter)
        if n.op == "get_attr":
            value = self.traced_params.get(n.target)
            shape = getattr(value, "shape", None)
            if shape is None:
                return [], counter
            output, counter = gen_tvar(counter)
            self.symbol_dict[n] = output
            return [BinConstraintT(output, TensorType(tuple(shape)), op_eq)], counter
        if n.op == "output":
            return [], counter
        raise NotImplementedError(f"constraint generation does not support {n.op!r}")
