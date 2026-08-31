"""Validation helpers for placement combinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..placement_types import Partial, Replicate, Shard

__all__ = ["ComparisonStats", "Discrepancy", "is_fully_replicated", "is_trivial_shard", "normalize_placement", "parse_placement"]


@dataclass
class Discrepancy:
    message: str
    expected: Any = None
    actual: Any = None


@dataclass
class ComparisonStats:
    checked: int = 0
    passed: int = 0
    failed: int = 0


def parse_placement(value: str) -> Any:
    value = value.strip()
    if value in {"R", "Replicate()"}:
        return Replicate()
    if value.startswith("S(") and value.endswith(")"):
        return Shard(int(value[2:-1]))
    if value.startswith("P(") and value.endswith(")"):
        return Partial(value[2:-1])
    raise ValueError(f"invalid placement string {value!r}")


def is_fully_replicated(placements: tuple[Any, ...]) -> bool:
    return all(isinstance(placement, Replicate) for placement in placements)


def is_trivial_shard(placement: Any, tensor_shape: tuple[int, ...]) -> bool:
    return isinstance(placement, Shard) and (placement.dim < 0 or placement.dim >= len(tensor_shape) or tensor_shape[placement.dim] <= 1)


def normalize_placement(placement: Any, tensor_shape: tuple[int, ...]) -> Any:
    if isinstance(placement, Shard) and placement.dim < 0:
        return Shard(placement.dim + len(tensor_shape))
    return placement
