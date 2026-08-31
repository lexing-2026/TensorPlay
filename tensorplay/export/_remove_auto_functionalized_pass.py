"""Remove graph nodes that only describe an automatic functional wrapper."""

from __future__ import annotations

from typing import Any

from ..graph.graph import dead_code_elimination

__all__ = ["remove_self_clone", "unsafe_remove_auto_functionalized_pass"]


def _target_name(target: Any) -> str:
    return getattr(target, "__name__", getattr(target, "name", str(target)))


def remove_self_clone(graph: Any) -> None:
    for node in list(graph.nodes):
        if node.op != "call_function" or not node.args:
            continue
        if _target_name(node.target) not in {"copy_", "clone"}:
            continue
        if len(node.args) >= 2 and node.args[0] is node.args[1]:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)


def unsafe_remove_auto_functionalized_pass(exported_program: Any) -> Any:
    """Inline automatic mutation wrappers and remove redundant self-copies."""

    graph = exported_program.graph
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if _target_name(node.target) not in {"auto_functionalized", "auto_functionalized_v2"}:
            continue
        if not node.args or not callable(node.args[0]):
            raise TypeError("automatic functional wrapper does not contain a callable")
        node.target = node.args[0]
        node.args = tuple(node.args[1:])
    remove_self_clone(graph)
    dead_code_elimination(graph)
    exported_program.graph_module.recompile()
    exported_program.validate()
    return exported_program
