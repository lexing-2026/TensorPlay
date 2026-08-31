"""Declarative module sharding plans."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..sharding_spec.api import ShardingSpec

__all__ = ["ShardingPlan", "ShardingPlanner"]


@dataclass
class ShardingPlan:
    plan: dict[str, ShardingSpec | Any]
    output_plan: dict[str, ShardingSpec] | None = None
    return_local_tensor: list[str] | None = None


class ShardingPlanner(ABC):
    @abstractmethod
    def build_plan(self, module: Any) -> ShardingPlan:
        raise NotImplementedError
