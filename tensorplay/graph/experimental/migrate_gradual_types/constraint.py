from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

from ...tensor_type import Dyn, TensorType, _DynType
from .operation import (
    op_add,
    op_div,
    op_eq,
    op_gt,
    op_lt,
    op_mod,
    op_mul,
    op_neq,
    op_sub,
)

__all__ = [
    "ApplyBroadcasting",
    "BinConstraintD",
    "BinConstraintT",
    "BinaryConstraint",
    "BVar",
    "CalcConv",
    "CalcMaxPool",
    "CalcProduct",
    "CanReshape",
    "Conj",
    "Constraint",
    "DGreatestUpperBound",
    "Disj",
    "DVar",
    "F",
    "GetItem",
    "GetItemTensor",
    "IndexSelect",
    "Prod",
    "T",
    "TGreatestUpperBound",
    "TVar",
    "Transpose",
    "is_algebraic_expression",
    "is_bool_expr",
    "is_dim",
]


class Constraint:
    pass


class Conj(Constraint):
    def __init__(self, conjuncts: Sequence[Constraint]) -> None:
        self.conjucts = list(conjuncts)

    @property
    def conjuncts(self) -> list[Constraint]:
        return self.conjucts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Conj) and self.conjucts == other.conjucts

    def __repr__(self) -> str:
        return f"And({self.conjucts})"


class Disj(Constraint):
    def __init__(self, disjuncts: Sequence[Constraint]) -> None:
        self.disjuncts = list(disjuncts)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Disj) and self.disjuncts == other.disjuncts

    def __repr__(self) -> str:
        return f"Or({self.disjuncts})"


class Prod(Constraint):
    def __init__(self, products: Sequence[Any]) -> None:
        self.products = list(products)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Prod) and self.products == other.products

    def __repr__(self) -> str:
        return f"Product({self.products})"


class T(Constraint):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, T)

    def __repr__(self) -> str:
        return "True"


class F(Constraint):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, F)

    def __repr__(self) -> str:
        return "False"


class BinaryConstraint(Constraint):
    def __init__(self, lhs: Any, rhs: Any, op: str | None) -> None:
        self.lhs, self.rhs, self.op = lhs, rhs, op

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BinaryConstraint) and (
            self.lhs, self.rhs, self.op
        ) == (other.lhs, other.rhs, other.op)

    def __repr__(self) -> str:
        return f"({self.lhs} {self.op} {self.rhs})"


class BinConstraintT(BinaryConstraint):
    def __init__(self, lhs: Any, rhs: Any, op: str | None) -> None:
        valid = (TVar, TensorType, int, _DynType)
        if not isinstance(lhs, valid) or not isinstance(rhs, valid):
            raise AssertionError(f"invalid tensor operands: lhs={type(lhs)}, rhs={type(rhs)}")
        super().__init__(lhs, rhs, op)


class BinConstraintD(BinaryConstraint):
    def __init__(self, lhs: Any, rhs: Any, op: str | None) -> None:
        if not (is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)):
            raise AssertionError(f"invalid dimension lhs: {type(lhs)}")
        if not (is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)):
            raise AssertionError(f"invalid dimension rhs: {type(rhs)}")
        super().__init__(lhs, rhs, op)


class TGreatestUpperBound(Constraint):
    def __init__(self, res: TVar, rhs1: TVar, rhs2: TVar) -> None:
        self.res, self.rhs1, self.rhs2 = res, rhs1, rhs2

    def __repr__(self) -> str:
        return f"{self.res} = {self.rhs1} ⋃* {self.rhs2}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TGreatestUpperBound) and (
            self.res, self.rhs1, self.rhs2
        ) == (other.res, other.rhs1, other.rhs2)


class DGreatestUpperBound(Constraint):
    def __init__(self, res: Any, rhs1: Any, rhs2: Any) -> None:
        if not all(is_dim(value) for value in (res, rhs1, rhs2)):
            raise AssertionError("greatest-upper-bound operands must be dimensions")
        self.res, self.rhs1, self.rhs2 = res, rhs1, rhs2

    def __repr__(self) -> str:
        return f"{self.res} = {self.rhs1} ⋃ {self.rhs2}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DGreatestUpperBound) and (
            self.res, self.rhs1, self.rhs2
        ) == (other.res, other.rhs1, other.rhs2)


class CanReshape(Constraint):
    def __init__(self, src: TVar, target: TensorType) -> None:
        self.src, self.target = src, target

    def __repr__(self) -> str:
        return f"can-reshape({self.src}, {self.target})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CanReshape) and (self.src, self.target) == (other.src, other.target)


class IndexSelect(Constraint):
    def __init__(self, tensor_size: int, input_var: TVar, dim_replace: Any, index: int, output: TVar) -> None:
        self.tensor_size, self.input_var, self.dim_replace, self.index, self.output = tensor_size, input_var, dim_replace, index, output

    def __repr__(self) -> str:
        return f"{self.output} = IndexSelect({self.input_var}, {self.tensor_size}, {self.dim_replace}, {self.index})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IndexSelect) and self.__dict__ == other.__dict__


class Transpose(Constraint):
    def __init__(self, tensor_size: int, input_var: TVar, index1: int, index2: int, output: TVar) -> None:
        self.tensor_size, self.input_var, self.index1, self.index2, self.output = tensor_size, input_var, index1, index2, output

    def __repr__(self) -> str:
        return f"{self.output} = Transpose({self.input_var}, {self.tensor_size}, {self.index1}, {self.index2})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Transpose) and self.__dict__ == other.__dict__


class GetItem(Constraint):
    def __init__(self, tensor_size: int, index: int, res: DVar, input_var: TVar) -> None:
        self.tensor_size, self.index, self.res, self.input_var = tensor_size, index, res, input_var

    def __repr__(self) -> str:
        return f"{self.res} = GetItem({self.input_var}, {self.tensor_size}, {self.index})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GetItem) and self.__dict__ == other.__dict__


class GetItemTensor(Constraint):
    def __init__(self, tensor_size: int, index_tuple: tuple[Any, ...], res: TVar, input_var: TVar) -> None:
        self.tensor_size, self.index_tuple, self.res, self.input_var = tensor_size, index_tuple, res, input_var

    def __repr__(self) -> str:
        return f"{self.res} = GetItemT({self.input_var}, {self.tensor_size}, {self.index_tuple})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GetItemTensor) and self.__dict__ == other.__dict__


class CalcConv(Constraint):
    def __init__(self, conv_result: TVar, input_var: TVar, c_out: int, kernel: Any, padding: Any, stride: Any, dilation: Any, matching_constraint_vars: list[DVar]) -> None:
        self.conv_result, self.input_var, self.c_out = conv_result, input_var, c_out
        self.kernel, self.padding, self.stride, self.dilation = kernel, padding, stride, dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        return f"{self.conv_result} = calc-conv({self.input_var}, {self.c_out}, {self.kernel}, {self.padding}, {self.stride}, {self.dilation})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CalcConv) and self.__dict__ == other.__dict__


class CalcMaxPool(Constraint):
    def __init__(self, maxpool_result: TVar, input_var: TVar, kernel: Any, padding: Any, stride: Any, dilation: Any, matching_constraint_vars: list[DVar]) -> None:
        self.maxpool_result, self.input_var = maxpool_result, input_var
        self.kernel, self.padding, self.stride, self.dilation = kernel, padding, stride, dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        return f"{self.maxpool_result} = calc-maxpool({self.input_var}, {self.kernel}, {self.padding}, {self.stride}, {self.dilation})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CalcMaxPool) and self.__dict__ == other.__dict__


class ApplyBroadcasting(Constraint):
    def __init__(self, res1: TVar, res2: TVar, input1: TVar, input2: TVar) -> None:
        self.res1, self.res2, self.input1, self.input2 = res1, res2, input1, input2

    def __repr__(self) -> str:
        return f"{self.res1}, {self.res2} = apply-broadcasting({self.input1}, {self.input2})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ApplyBroadcasting) and self.__dict__ == other.__dict__


class CalcProduct(Constraint):
    def __init__(self, start: int, end: int, flattened: TVar, dims_to_flatten: list[DVar]) -> None:
        self.start, self.end, self.flattened, self.dims_to_flatten = start, end, flattened, dims_to_flatten

    def __repr__(self) -> str:
        return f"{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CalcProduct) and self.__dict__ == other.__dict__


class _Var:
    prefix = "V"

    def __init__(self, value: int) -> None:
        self.c = value

    def __repr__(self) -> str:
        return f"{self.prefix}({self.c})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.c == other.c

    def __hash__(self) -> int:
        return hash((type(self), self.c))


class TVar(_Var):
    prefix = "TV"

    @property
    def tvar(self) -> int:
        return self.c


class DVar(_Var):
    prefix = "DV"

    @property
    def c(self) -> int:
        return self._c

    @c.setter
    def c(self, value: int) -> None:
        self._c = value


class BVar(_Var):
    prefix = "BV"


_Operand: TypeAlias = Any


def is_algebraic_expression(value: object) -> bool:
    return isinstance(value, (Prod,)) or isinstance(value, BinConstraintD) and value.op in {op_add, op_sub, op_div, op_mul, op_mod}


def is_bool_expr(value: object) -> bool:
    return isinstance(value, (BVar, Conj, Disj)) or isinstance(value, BinConstraintD) and value.op in {op_gt, op_lt, op_neq, op_eq}


def is_dim(value: object) -> bool:
    return isinstance(value, (DVar, int)) and not isinstance(value, bool) or value == Dyn
