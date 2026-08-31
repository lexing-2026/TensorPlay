"""Single-mesh-dimension strategy records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..placement_types import Placement, Replicate, Shard, _StridedShard
from .utils import register_op_strategy

__all__ = ["PreparedSingleDimStrategy", "register_single_dim_strategy"]


@dataclass(frozen=True)
class PreparedSingleDimStrategy:
    input_placements: tuple[Placement, ...]
    output_placements: tuple[Placement, ...]
    redistribute_cost: int = 0


def register_single_dim_strategy(operation: Any, strategy: Callable[..., Any]) -> Callable[..., Any]:
    return register_op_strategy(operation)(strategy)
