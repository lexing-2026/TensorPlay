"""Common protocol and result types for graph transformations."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from ..graph import Graph
from ..graph_module import GraphModule

__all__ = ["PassBase", "PassResult", "_as_graph_module"]


class PassResult(NamedTuple):
    """Result returned by a graph transformation."""

    graph_module: GraphModule
    modified: bool


class PassBase:
    """Base protocol for transformations that operate on a graph module."""

    def __call__(self, graph_module: GraphModule) -> PassResult:
        raise NotImplementedError

    def constraint(self) -> Optional[str]:
        """Return a human-readable precondition, or ``None``."""

        return None


def _as_graph_module(target: Any) -> GraphModule:
    """Normalize a graph or graph module to the pass input type."""

    if isinstance(target, GraphModule):
        return target
    if isinstance(target, Graph):
        return GraphModule(None, target, None)
    raise TypeError(
        f"passes expect a GraphModule or Graph, got {type(target)!r}"
    )
