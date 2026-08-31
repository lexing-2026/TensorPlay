from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

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
    Prod,
    T,
    TGreatestUpperBound,
    TVar,
    Transpose,
    is_algebraic_expression,
    is_bool_expr,
    is_dim,
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
from .util import gen_dvar, gen_nat_constraints, gen_tensor_dims

__all__ = [
    "apply_padding",
    "broadcast_dim",
    "calc_last_two_dims",
    "create_equality_constraints_for_broadcasting",
    "gen_all_reshape_possibilities",
    "gen_broadcasting_constraints",
    "gen_consistency_constraints",
    "gen_greatest_upper_bound",
    "gen_lists_of_dims",
    "generate_all_broadcasting_possibilities_no_padding",
    "generate_all_int_dyn_dim_possibilities",
    "generate_binconstraint_d",
    "generate_binconstraint_t",
    "generate_broadcasting",
    "generate_calc_conv",
    "generate_calc_maxpool",
    "generate_calc_product",
    "generate_conj",
    "generate_d_gub",
    "generate_disj",
    "generate_gub",
    "generate_reshape",
    "is_dim_div_by_target",
    "is_target_div_by_dim",
    "no_broadcast_dim_with_index",
    "register_transformation_rule",
    "transform_constraint",
    "transform_get_item",
    "transform_get_item_tensor",
    "transform_index_select",
    "transform_transpose",
    "valid_index",
    "valid_index_tensor",
]

_RULES: dict[type[Constraint], Callable[[Constraint, int], tuple[Constraint, int]]] = {}


def register_transformation_rule(kind: type[Constraint]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        _RULES[kind] = function
        return function

    return decorate


def valid_index(index: int, dims: list[DVar]) -> Constraint:
    return T() if -len(dims) <= index < len(dims) else F()


def valid_index_tensor(index: tuple[Any, ...], dims: list[DVar]) -> Constraint:
    consumed = sum(item is not None and not isinstance(item, int) or isinstance(item, int) for item in index)
    return T() if consumed <= len(dims) else F()


@register_transformation_rule(Transpose)
def transform_transpose(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, Transpose)
    dimensions, counter = gen_tensor_dims(constraint.tensor_size, counter)
    result = list(dimensions)
    first = constraint.index1 % len(result)
    second = constraint.index2 % len(result)
    result[first], result[second] = result[second], result[first]
    return Conj([
        BinConstraintT(constraint.input_var, TensorType(dimensions), op_eq),
        BinConstraintT(constraint.output, TensorType(result), op_eq),
        *gen_nat_constraints(dimensions),
    ]), counter


@register_transformation_rule(IndexSelect)
def transform_index_select(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, IndexSelect)
    dimensions, counter = gen_tensor_dims(constraint.tensor_size, counter)
    result = list(dimensions)
    if -len(result) <= constraint.index < len(result):
        result[constraint.index % len(result)] = constraint.dim_replace
    else:
        return F(), counter
    return Conj([
        BinConstraintT(constraint.input_var, TensorType(dimensions), op_eq),
        BinConstraintT(constraint.output, TensorType(result), op_eq),
        *gen_nat_constraints(dimensions),
    ]), counter


@register_transformation_rule(GetItem)
def transform_get_item(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, GetItem)
    dimensions, counter = gen_tensor_dims(constraint.tensor_size, counter)
    if not -len(dimensions) <= constraint.index < len(dimensions):
        return F(), counter
    return Conj([
        BinConstraintT(constraint.input_var, TensorType(dimensions), op_eq),
        BinConstraintD(constraint.res, dimensions[constraint.index % len(dimensions)], op_eq),
        *gen_nat_constraints(dimensions),
    ]), counter


@register_transformation_rule(GetItemTensor)
def transform_get_item_tensor(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, GetItemTensor)
    dimensions, counter = gen_tensor_dims(constraint.tensor_size, counter)
    output: list[Any] = []
    consumed = 0
    for item in constraint.index_tuple:
        if item is None:
            output.append(1)
        elif isinstance(item, slice):
            if consumed >= len(dimensions):
                return F(), counter
            output.append(dimensions[consumed])
            consumed += 1
        else:
            if consumed >= len(dimensions):
                return F(), counter
            consumed += 1
    output.extend(dimensions[consumed:])
    if len(output) > 4:
        return F(), counter
    return Conj([
        BinConstraintT(constraint.input_var, TensorType(dimensions), op_eq),
        BinConstraintT(constraint.res, TensorType(output), op_eq),
        *gen_nat_constraints(dimensions),
    ]), counter


@register_transformation_rule(BinConstraintT)
def generate_binconstraint_t(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, BinConstraintT)
    return constraint, counter


@register_transformation_rule(BinConstraintD)
def generate_binconstraint_d(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, BinConstraintD)
    return constraint, counter


@register_transformation_rule(Conj)
def generate_conj(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, Conj)
    values: list[Constraint] = []
    for item in constraint.conjucts:
        transformed, counter = transform_constraint(item, counter)
        if isinstance(transformed, F):
            return F(), counter
        if not isinstance(transformed, T):
            values.append(transformed)
    return Conj(values), counter


@register_transformation_rule(Disj)
def generate_disj(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, Disj)
    values: list[Constraint] = []
    for item in constraint.disjuncts:
        transformed, counter = transform_constraint(item, counter)
        if isinstance(transformed, T):
            return T(), counter
        if not isinstance(transformed, F):
            values.append(transformed)
    return Disj(values), counter


@register_transformation_rule(TGreatestUpperBound)
def generate_gub(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, TGreatestUpperBound)
    return constraint, counter


@register_transformation_rule(DGreatestUpperBound)
def generate_d_gub(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, DGreatestUpperBound)
    return constraint, counter


@register_transformation_rule(CalcConv)
def generate_calc_conv(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    if not isinstance(constraint, CalcConv):
        raise TypeError(type(constraint))
    dimensions, counter = gen_tensor_dims(4, counter)
    result_type = TensorType(dimensions)
    result_constraint = BinConstraintT(constraint.conv_result, result_type, op_eq)
    channel_constraint = Conj([
        BinConstraintD(dimensions[1], constraint.c_out, op_eq),
        BinConstraintD(dimensions[1], Dyn, op_neq),
    ])
    batch_constraint = BinConstraintD(
        constraint.matching_constraint[0], dimensions[0], op_eq
    )
    height_constraint, width_constraint = calc_last_two_dims(constraint, dimensions)
    natural_constraints = Conj([
        BinConstraintD(0, dimension, op_leq) for dimension in dimensions
    ])
    return Conj([
        result_constraint,
        channel_constraint,
        batch_constraint,
        height_constraint,
        width_constraint,
        natural_constraints,
    ]), counter


@register_transformation_rule(CalcMaxPool)
def generate_calc_maxpool(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    if not isinstance(constraint, CalcMaxPool):
        raise TypeError(type(constraint))
    dimensions, counter = gen_tensor_dims(4, counter)
    result_type = TensorType(dimensions)
    result_constraint = BinConstraintT(constraint.maxpool_result, result_type, op_eq)
    channel_constraint = BinConstraintD(
        constraint.matching_constraint[1], dimensions[1], op_eq
    )
    batch_constraint = BinConstraintD(
        constraint.matching_constraint[0], dimensions[0], op_eq
    )
    height_constraint, width_constraint = calc_last_two_dims(constraint, dimensions)
    natural_constraints = Conj([
        BinConstraintD(0, dimension, op_leq) for dimension in dimensions
    ])
    return Conj([
        result_constraint,
        channel_constraint,
        batch_constraint,
        height_constraint,
        width_constraint,
        natural_constraints,
    ]), counter


@register_transformation_rule(CalcProduct)
def generate_calc_product(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, CalcProduct)
    product = Prod(constraint.dims_to_flatten[constraint.start:constraint.end])
    return BinConstraintT(constraint.flattened, product, op_eq), counter


@register_transformation_rule(CanReshape)
def generate_reshape(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, CanReshape)
    return BinConstraintT(constraint.src, constraint.target, op_eq), counter


@register_transformation_rule(ApplyBroadcasting)
def generate_broadcasting(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    assert isinstance(constraint, ApplyBroadcasting)
    return Conj([
        TGreatestUpperBound(constraint.res1, constraint.input1, constraint.input2),
        TGreatestUpperBound(constraint.res2, constraint.input1, constraint.input2),
    ]), counter


def transform_constraint(constraint: Constraint, counter: int = 0) -> tuple[Constraint, int]:
    if isinstance(constraint, (T, F)):
        return constraint, counter
    rule = _RULES.get(type(constraint))
    if rule is None:
        raise NotImplementedError(f"no transformation rule for {type(constraint).__name__}")
    return rule(constraint, counter)


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(value)
    if len(values) != 2:
        raise ValueError("spatial parameters require one or two values")
    return int(values[0]), int(values[1])


def calc_last_two_dims(
    constraint: CalcConv | CalcMaxPool, dimensions: list[DVar]
) -> tuple[Constraint, Constraint]:
    if not isinstance(constraint, (CalcConv, CalcMaxPool)):
        raise AssertionError(
            f"expected convolution or pooling constraint, got {type(constraint)}"
        )
    if len(dimensions) < 4 or len(constraint.matching_constraint) < 4:
        raise ValueError("spatial shape constraints require four dimensions")

    input_height, input_width = constraint.matching_constraint[2:4]
    output_height, output_width = dimensions[2:4]
    padding = _pair(constraint.padding)
    kernel = _pair(constraint.kernel)
    stride = _pair(constraint.stride)
    dilation = _pair(constraint.dilation)

    height_dynamic = Conj([
        BinConstraintD(output_height, Dyn, op_eq),
        BinConstraintD(input_height, Dyn, op_eq),
    ])
    width_dynamic = Conj([
        BinConstraintD(output_width, Dyn, op_eq),
        BinConstraintD(input_width, Dyn, op_eq),
    ])
    height_static = Conj([
        BinConstraintD(output_height, Dyn, op_neq),
        BinConstraintD(input_height, Dyn, op_neq),
    ])
    width_static = Conj([
        BinConstraintD(output_width, Dyn, op_neq),
        BinConstraintD(input_width, Dyn, op_neq),
    ])

    height_numerator = BinConstraintD(
        BinConstraintD(
            BinConstraintD(input_height, BinConstraintD(2, padding[0], op_mul), op_add),
            BinConstraintD(
                dilation[0], BinConstraintD(kernel[0], 1, op_sub), op_mul
            ),
            op_sub,
        ),
        1,
        op_sub,
    )
    height_formula = BinConstraintD(
        BinConstraintD(height_numerator, stride[0], op_div), 1, op_add
    )
    width_numerator = BinConstraintD(
        BinConstraintD(
            BinConstraintD(input_width, BinConstraintD(2, padding[1], op_mul), op_add),
            BinConstraintD(
                dilation[1], BinConstraintD(kernel[1], 1, op_sub), op_mul
            ),
            op_sub,
        ),
        1,
        op_sub,
    )
    width_formula = BinConstraintD(
        BinConstraintD(width_numerator, stride[1], op_div), 1, op_add
    )
    return (
        Disj([
            height_dynamic,
            Conj([height_static, BinConstraintD(output_height, height_formula, op_eq)]),
        ]),
        Disj([
            width_dynamic,
            Conj([width_static, BinConstraintD(output_width, width_formula, op_eq)]),
        ]),
    )


def generate_all_int_dyn_dim_possibilities(dimensions: list[Any]) -> list[list[Any]]:
    result = [[]]
    for dimension in dimensions:
        options = [0, 1] if dimension == Dyn else [dimension]
        result = [prefix + [option] for prefix in result for option in options]
    return result


def is_target_div_by_dim(target: int, dimension: Any) -> bool:
    return isinstance(dimension, int) and dimension != 0 and target % dimension == 0


def is_dim_div_by_target(dimension: int, target: int) -> bool:
    return target != 0 and dimension % target == 0


def gen_all_reshape_possibilities(source: list[Any], target: list[Any]) -> list[list[Any]]:
    return [target] if source and target else [[]]


def broadcast_dim(left: Any, right: Any) -> list[Any]:
    if left == right:
        return [left]
    if left == 1:
        return [right]
    if right == 1:
        return [left]
    if left == Dyn or right == Dyn:
        return [Dyn]
    return []


def apply_padding(dimensions: list[Any], padding: list[Any]) -> list[Any]:
    if len(dimensions) != len(padding):
        raise ValueError("padding and dimensions must have equal lengths")
    return [dimension + 2 * pad if isinstance(dimension, int) and isinstance(pad, int) else Dyn for dimension, pad in zip(dimensions, padding)]


def no_broadcast_dim_with_index(dimensions: list[Any], index: int) -> bool:
    return 0 <= index < len(dimensions) and dimensions[index] != 1


def gen_lists_of_dims(rank: int, current: int) -> tuple[list[list[DVar]], int]:
    dimensions, current = gen_tensor_dims(rank, current)
    return [dimensions], current


def create_equality_constraints_for_broadcasting(left: list[Any], right: list[Any]) -> list[Constraint]:
    result: list[Constraint] = []
    for first, second in zip(reversed(left), reversed(right)):
        result.append(BinConstraintD(first, second, op_consistency))
    return result


def gen_consistency_constraints(left: list[Any], right: list[Any]) -> list[Constraint]:
    return create_equality_constraints_for_broadcasting(left, right)


def gen_greatest_upper_bound(result: list[Any], left: list[Any], right: list[Any]) -> list[Constraint]:
    return [DGreatestUpperBound(out, first, second) for out, first, second in zip(result, left, right)]


def generate_all_broadcasting_possibilities_no_padding(left: list[Any], right: list[Any]) -> list[tuple[list[Any], list[Any]]]:
    length = max(len(left), len(right))
    left = [1] * (length - len(left)) + list(left)
    right = [1] * (length - len(right)) + list(right)
    return [(left, right)] if all(broadcast_dim(a, b) for a, b in zip(left, right)) else []


def gen_broadcasting_constraints(left: TVar, right: TVar, result: TVar, counter: int = 0) -> tuple[Constraint, int]:
    dimensions, counter = gen_tensor_dims(4, counter)
    return Conj([
        BinConstraintT(left, TensorType(dimensions), op_eq),
        BinConstraintT(right, TensorType(dimensions), op_eq),
        BinConstraintT(result, TensorType(dimensions), op_eq),
    ]), counter
