"""Insert explicit runtime checks collected during graph analysis."""

from __future__ import annotations

from typing import Any

from ..graph import Graph
from ..graph_module import GraphModule

__all__ = ["insert_deferred_runtime_asserts"]


def _check(condition: Any, message: str) -> Any:
    if not bool(condition):
        raise RuntimeError(message)
    return condition


def insert_deferred_runtime_asserts(
    graph_module: GraphModule,
    shape_env: Any,
    name: str,
    export: bool = False,
) -> None:
    """Materialize deferred boolean checks stored by a shape environment."""

    del export
    assertions = getattr(shape_env, "deferred_runtime_asserts", None)
    if assertions is None:
        assertions = getattr(shape_env, "runtime_asserts", ())
    if not assertions:
        return
    graph = graph_module.graph
    output = graph.output_node
    with graph.inserting_before(output):
        for index, assertion in enumerate(assertions):
            condition = getattr(assertion, "expr", assertion)
            graph.call_function(_check, (condition, f"runtime assertion {name}:{index}"))
    graph.lint()
