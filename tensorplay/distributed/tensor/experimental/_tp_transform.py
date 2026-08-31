"""Annotate an exported graph with tensor-parallel placement metadata."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["tensor_parallel_transformation"]


def tensor_parallel_transformation(exported_program: Any, device_mesh: Any, placements: Mapping[str, Any] | None = None) -> Any:
    graph_module = getattr(exported_program, "graph_module", exported_program)
    for node in getattr(getattr(graph_module, "graph", None), "nodes", ()):
        if placements and node.name in placements:
            node.meta["placements"] = placements[node.name]
        node.meta.setdefault("device_mesh", device_mesh)
    return exported_program
