from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..graph_module import GraphModule

__all__ = ["set_trace"]


def insert_pdb(body: Sequence[str]) -> list[str]:
    return ["import pdb; pdb.set_trace()", *body]


def set_trace(graph_module: GraphModule) -> GraphModule:
    """Insert a debugger breakpoint into the generated graph function."""

    if not isinstance(graph_module, GraphModule):
        raise TypeError(f"expected GraphModule, got {type(graph_module).__name__}")

    with graph_module.graph.on_generate_code(
        make_transformer=lambda current: (
            lambda body: insert_pdb(current(body) if current else body)
        )
    ):
        graph_module.recompile()
    return graph_module
