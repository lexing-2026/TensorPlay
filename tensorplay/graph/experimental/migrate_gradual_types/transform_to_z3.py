from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

from ...graph import Graph
from ...node import Node
from ...tensor_type import Dyn, TensorType, _DynType
from .constraint import (
    BinConstraintD,
    BinConstraintT,
    BVar,
    Conj,
    Constraint,
    Disj,
    DVar,
    F,
    Prod,
    T,
    TVar,
    is_algebraic_expression,
    is_bool_expr,
    is_dim,
)
from .constraint_generator import ConstraintGenerator
from .constraint_transformation import transform_constraint
from .operation import (
    op_add,
    op_consistency,
    op_div,
    op_eq,
    op_gt,
    op_leq,
    op_lt,
    op_mod,
    op_mul,
    op_neq,
    op_precision,
    op_sub,
)
from .z3_types import D, HAS_Z3, tensor_type, z3_dyn

_Z3Expr: TypeAlias = Any
_Z3Result: TypeAlias = Any

__all__ = [
    "evaluate_conditional_with_constraints",
    "iterate_till_fixed_point",
    "transform_algebraic_expression",
    "transform_all_constraints",
    "transform_all_constraints_trace_time",
    "transform_dimension",
    "transform_to_z3",
    "transform_var",
]


def _require_z3() -> Any:
    if not HAS_Z3:
        raise ImportError("the optional constraint solver is required for this operation")
    import z3

    return z3


def _and(z3: Any, values: Iterable[Any]) -> Any:
    values = list(values)
    return z3.And(*values) if values else z3.BoolVal(True)


def _or(z3: Any, values: Iterable[Any]) -> Any:
    values = list(values)
    return z3.Or(*values) if values else z3.BoolVal(False)


def transform_to_z3(
    constraint: Constraint, counter: int = 0, dimension_dict: dict[int, int] | None = None
) -> tuple[_Z3Expr, int]:
    z3 = _require_z3()
    dimensions = {} if dimension_dict is None else dimension_dict
    if isinstance(constraint, Conj):
        values: list[Any] = []
        for item in constraint.conjuncts:
            value, counter = transform_to_z3(item, counter, dimensions)
            values.append(value)
        return _and(z3, values), counter
    if isinstance(constraint, Disj):
        values = []
        for item in constraint.disjuncts:
            value, counter = transform_to_z3(item, counter, dimensions)
            values.append(value)
        return _or(z3, values), counter
    if isinstance(constraint, T):
        return z3.BoolVal(True), counter
    if isinstance(constraint, F):
        return z3.BoolVal(False), counter
    if isinstance(constraint, BinConstraintT):
        lhs, counter = transform_var(constraint.lhs, counter, dimensions)
        rhs, counter = transform_var(constraint.rhs, counter, dimensions)
        if constraint.op in {op_eq, op_consistency, op_precision}:
            return lhs == rhs, counter
        if constraint.op == op_neq:
            return lhs != rhs, counter
        raise NotImplementedError(f"tensor operation {constraint.op!r} is not supported")
    if isinstance(constraint, BinConstraintD):
        if constraint.op in {op_eq, op_neq, op_leq, op_gt, op_lt}:
            if constraint.op in {op_eq, op_neq} and isinstance(constraint.lhs, BVar):
                lhs = z3.Bool(str(constraint.lhs.c))
                rhs, counter = transform_to_z3(constraint.rhs, counter, dimensions)
                return (lhs == rhs, counter) if constraint.op == op_eq else (lhs != rhs, counter)
            if constraint.op in {op_eq, op_neq} and (is_dim(constraint.lhs) and is_dim(constraint.rhs)):
                lhs, counter = transform_dimension(constraint.lhs, counter, dimensions)
                rhs, counter = transform_dimension(constraint.rhs, counter, dimensions)
                if constraint.op == op_eq:
                    return lhs == rhs, counter
                return _dimension_neq(z3, constraint.lhs, constraint.rhs, lhs, rhs), counter
            lhs, counter = transform_algebraic_expression(constraint.lhs, counter, dimensions)
            rhs, counter = transform_algebraic_expression(constraint.rhs, counter, dimensions)
            if constraint.op == op_eq:
                return lhs == rhs, counter
            if constraint.op == op_neq:
                return lhs != rhs, counter
            if constraint.op == op_leq:
                return lhs <= rhs, counter
            if constraint.op == op_gt:
                return lhs > rhs, counter
            if constraint.op == op_lt:
                return lhs < rhs, counter
        raise NotImplementedError(f"dimension operation {constraint.op!r} is not supported")
    raise NotImplementedError(f"constraint kind {type(constraint).__name__!r} is not supported")


def _dimension_neq(z3: Any, left: Any, right: Any, left_expr: Any, right_expr: Any) -> Any:
    if left == Dyn:
        return right_expr.arg(0) == 1
    if right == Dyn:
        return left_expr.arg(0) == 1
    if isinstance(left, int) and not isinstance(right, int):
        return z3.Or(right_expr.arg(0) == 0, z3.And(right_expr.arg(0) == 1, left_expr.arg(1) != right_expr.arg(1)))
    if isinstance(right, int) and not isinstance(left, int):
        return z3.Or(left_expr.arg(0) == 0, z3.And(left_expr.arg(0) == 1, left_expr.arg(1) != right_expr.arg(1)))
    return z3.Or(
        z3.And(left_expr.arg(0) == 0, right_expr.arg(0) != 0),
        z3.And(left_expr.arg(0) != 0, right_expr.arg(0) == 0),
        z3.And(left_expr.arg(0) != 0, right_expr.arg(0) != 0, left_expr.arg(1) != right_expr.arg(1)),
    )


def transform_var(
    tensor: TVar | TensorType | _DynType | int,
    counter: int,
    dimension_dict: dict[int, int],
) -> tuple[_Z3Expr, int]:
    z3 = _require_z3()
    if isinstance(tensor, TensorType):
        values: list[Any] = []
        for dimension in tensor.dims:
            value, counter = transform_dimension(dimension, counter, dimension_dict)
            values.append(value)
        if not 1 <= len(values) <= 4:
            raise AssertionError(f"tensor rank must be between one and four, got {len(values)}")
        constructor = getattr(tensor_type, f"tensor{len(values)}")
        return constructor(*values), counter
    if tensor == Dyn:
        return z3_dyn, counter
    if isinstance(tensor, TVar):
        return z3.Const(str(tensor.tvar), tensor_type), counter
    raise NotImplementedError(f"unsupported tensor value {type(tensor).__name__}")


def transform_dimension(
    dimension: DVar | int | _DynType,
    counter: int,
    dimension_dict: dict[int, int],
) -> tuple[_Z3Expr, int]:
    z3 = _require_z3()
    if dimension == Dyn:
        counter += 1
        return D(0, z3.Int(f"d{counter}")), counter
    if isinstance(dimension, int) and not isinstance(dimension, bool):
        return D(1, dimension), counter
    if isinstance(dimension, DVar):
        mapped = dimension_dict.get(dimension.c)
        if mapped is None:
            counter += 1
            mapped = counter
            dimension_dict[dimension.c] = mapped
        return D(z3.Int(f"k{mapped}"), z3.Int(f"v{dimension.c}")), counter
    raise NotImplementedError(f"unsupported dimension value {type(dimension).__name__}")


def transform_algebraic_expression(
    expr: Any, counter: int, dimension_dict: dict[int, int]
) -> tuple[_Z3Expr, int]:
    z3 = _require_z3()
    if is_dim(expr):
        value, counter = transform_dimension(expr, counter, dimension_dict)
        return value.arg(1), counter
    if isinstance(expr, Prod):
        values: list[Any] = []
        for item in expr.products:
            if not is_dim(item):
                raise AssertionError("product factors must be dimensions")
            value, counter = transform_algebraic_expression(item, counter, dimension_dict)
            values.append(value)
        result: Any = z3.IntVal(1)
        for value in values:
            result = result * value
        return result, counter
    if isinstance(expr, BinConstraintD) and is_algebraic_expression(expr):
        lhs, counter = transform_algebraic_expression(expr.lhs, counter, dimension_dict)
        rhs, counter = transform_algebraic_expression(expr.rhs, counter, dimension_dict)
        if expr.op == op_add:
            return lhs + rhs, counter
        if expr.op == op_sub:
            return lhs - rhs, counter
        if expr.op == op_mul:
            return lhs * rhs, counter
        if expr.op == op_div:
            return lhs / rhs, counter
        if expr.op == op_mod:
            return z3.Mod(lhs, rhs), counter
        raise NotImplementedError(f"algebraic operation {expr.op!r} is not supported")
    if isinstance(expr, (BinConstraintD, Conj, Disj, T, F, BVar)):
        value, counter = transform_to_z3(expr, counter, dimension_dict)
        return value, counter
    if isinstance(expr, bool):
        return z3.BoolVal(expr), counter
    if isinstance(expr, int):
        return z3.IntVal(expr), counter
    raise AssertionError(f"expected an algebraic expression, got {type(expr).__name__}")


def iterate_till_fixed_point(constraints: Constraint, counter: int = 0) -> tuple[Constraint, int]:
    previous: Constraint | None = None
    while previous != constraints:
        previous = constraints
        constraints, counter = transform_constraint(constraints, counter)
    return constraints, counter


def transform_all_constraints(traced: Any, counter: int = 0) -> _Z3Expr:
    generated, counter = ConstraintGenerator(traced).generate_constraints(counter)
    reduced, counter = iterate_till_fixed_point(generated, counter)
    transformed, _ = transform_to_z3(reduced, counter, {})
    return transformed


def transform_all_constraints_trace_time(
    tracer_root: Any, graph: Graph, node: Node, counter: int = 0
) -> tuple[_Z3Expr, _Z3Expr]:
    z3 = _require_z3()
    generated, counter = ConstraintGenerator(tracer_root, graph).generate_constraints(counter)
    if not generated.conjuncts:
        raise ValueError("the graph did not produce a conditional constraint")
    condition = generated.conjuncts[-1]
    generated = Conj(generated.conjuncts[:-1])
    reduced, counter = iterate_till_fixed_point(generated, counter)
    if not isinstance(condition, BinConstraintD) or not isinstance(condition.lhs, BVar) or not is_bool_expr(condition.rhs):
        raise TypeError("the final graph constraint is not a boolean condition")
    condition_rhs, counter = iterate_till_fixed_point(condition.rhs, counter)
    base, counter = transform_to_z3(reduced, counter, {})
    positive, _ = transform_to_z3(condition_rhs, counter, {})
    return z3.And(base, positive), z3.And(base, z3.Not(positive))


def evaluate_conditional_with_constraints(
    tracer_root: Any,
    graph: Graph,
    node: Node,
    counter: int = 0,
    user_constraints: _Z3Expr | None = None,
) -> tuple[_Z3Result, _Z3Result]:
    z3 = _require_z3()
    positive, negative = transform_all_constraints_trace_time(tracer_root, graph, node, counter)
    results = []
    for expression in (positive, negative):
        solver = z3.Solver()
        solver.add(expression)
        if user_constraints is not None:
            solver.add(user_constraints)
        results.append(solver.check())
    return results[0], results[1]
