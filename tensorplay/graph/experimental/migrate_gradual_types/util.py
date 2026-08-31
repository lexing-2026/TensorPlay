from __future__ import annotations

from .constraint import BinConstraintD, BVar, DVar, TVar
from .operation import op_leq

__all__ = ["gen_bvar", "gen_dvar", "gen_nat_constraints", "gen_tensor_dims", "gen_tvar"]


def gen_tvar(current: int) -> tuple[TVar, int]:
    current += 1
    return TVar(current), current


def gen_dvar(current: int) -> tuple[DVar, int]:
    current += 1
    return DVar(current), current


def gen_bvar(current: int) -> tuple[BVar, int]:
    current += 1
    return BVar(current), current


def gen_tensor_dims(count: int, current: int) -> tuple[list[DVar], int]:
    dimensions: list[DVar] = []
    for _ in range(count):
        dimension, current = gen_dvar(current)
        dimensions.append(dimension)
    return dimensions, current


def gen_nat_constraints(dimensions: list[DVar]) -> list[BinConstraintD]:
    return [BinConstraintD(0, dimension, op_leq) for dimension in dimensions]
