"""Remove graph operations that cannot contribute to the result."""

from __future__ import annotations

from ..graph import dead_code_elimination
from .base import PassBase, PassResult

__all__ = ["DeadCodeElimination"]


class DeadCodeElimination(PassBase):
    """Drop nodes that cannot reach the output while retaining inputs."""

    def __call__(self, graph_module) -> PassResult:
        removed = dead_code_elimination(graph_module.graph)
        return PassResult(graph_module, removed > 0)
