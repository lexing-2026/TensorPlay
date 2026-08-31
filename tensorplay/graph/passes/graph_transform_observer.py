"""Observe graph changes around a transformation callback."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from ..graph import Graph
from ..graph_module import GraphModule

__all__ = ["GraphTransformObserver"]

_T = TypeVar("_T")


class GraphTransformObserver:
    _pass_count = 0

    def __init__(self, gm: GraphModule, passname: str, subsystem: str | None = None, log_url: str | None = None) -> None:
        self.gm = gm
        self.passname = passname
        self.subsystem = subsystem
        self.log_url = log_url
        self.active = True
        self.input_dot_graph: str | None = None
        self.output_dot_graph: str | None = None
        self.created_nodes: set[str] = set()
        self.erased_nodes: set[str] = set()

    @classmethod
    def get_current_pass_count(cls) -> int:
        return cls._pass_count

    def __enter__(self) -> "GraphTransformObserver":
        GraphTransformObserver._pass_count += 1
        self.input_dot_graph = self.gm.graph.to_dot()
        self._before = {node.name for node in self.gm.graph.nodes}
        return self

    def __exit__(self, exc_type, value, traceback) -> None:
        if exc_type is None:
            after = {node.name for node in self.gm.graph.nodes}
            self.created_nodes = after - self._before
            self.erased_nodes = self._before - after
            self.output_dot_graph = self.gm.graph.to_dot()

    def apply_gm_pass(self, pass_fn: Callable[[GraphModule], _T]) -> _T:
        with self:
            return pass_fn(self.gm)

    def apply_graph_pass(self, pass_fn: Callable[[Graph], _T]) -> _T:
        with self:
            return pass_fn(self.gm.graph)
