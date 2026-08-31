"""Sequence index balancers for causal attention workloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["_HeadTailLoadBalancer", "_LoadBalancer"]


class _LoadBalancer(ABC):
    @abstractmethod
    def _generate_indices(self, restore: bool = False) -> Any:
        raise NotImplementedError


class _HeadTailLoadBalancer(_LoadBalancer):
    def __init__(self, seq_length: int, world_size: int, device: Any = None) -> None:
        self.seq_length = int(seq_length)
        self.world_size = int(world_size)
        self.device = device

    def _generate_indices(self, restore: bool = False) -> Any:
        if self.seq_length % (self.world_size * 2):
            raise ValueError("sequence length must be divisible by twice the world size")
        chunk = self.seq_length // (self.world_size * 2)
        indices = []
        for rank in range(self.world_size):
            head = range(rank * chunk, (rank + 1) * chunk)
            tail = range(self.seq_length - (rank + 1) * chunk, self.seq_length - rank * chunk)
            indices.extend((*head, *tail))
        if restore:
            inverse = [0] * len(indices)
            for current, original in enumerate(indices):
                inverse[original] = current
            indices = inverse
        return indices
