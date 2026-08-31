"""Reduction helpers for layouts whose local values are partial."""

from __future__ import annotations

from typing import Any

from .. import distributed_core as dist
from .placement_types import Partial

__all__ = ["is_linear_reduction", "reduce_partial"]


def is_linear_reduction(reduce_op: str) -> bool:
    return reduce_op in Partial.LINEAR_REDUCE_OPS


def reduce_partial(value: Any, placement: Partial, group: Any) -> Any:
    operation = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "product": dist.ReduceOp.PRODUCT,
    }[placement.reduce_op]
    dist.all_reduce(value, op=operation, group=group)
    return value
