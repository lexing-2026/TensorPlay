"""Passes that operate on exported graph programs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from ...graph.passes.infra.pass_base import PassResult

__all__ = [
    "PassResult",
    "RuntimeDependentDecompositionPass",
    "move_to_device_pass",
    "register_runtime_dependent_op",
]


_RUNTIME_DEPENDENT_OPS: dict[str, Callable[[Any, Any], Any]] = {}


def register_runtime_dependent_op(
    name: str, decomposition: Callable[[Any, Any], Any]
) -> Callable[[Any, Any], Any]:
    """Declare an op whose result is only knowable at runtime.

    ``decomposition(graph, node)`` rewrites the call into runtime-checked
    nodes; the pass applies it where the op name appears.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("op name must be a non-empty string")
    if not callable(decomposition):
        raise TypeError("decomposition must be callable")
    _RUNTIME_DEPENDENT_OPS[name] = decomposition
    return decomposition


def _target_name(target: Any) -> str:
    return getattr(target, "__name__", getattr(target, "name", str(target)))


class RuntimeDependentDecompositionPass:
    """Replace runtime-dependent ops with their checked rewrites.

    Ops registered via :func:`register_runtime_dependent_op` cannot keep
    their symbolic result through later passes; this pass rewrites each such
    call site into the decomposition's runtime-validated node sequence.
    """

    def __init__(self, additional_ops: Mapping[str, Callable[[Any, Any], Any]] | None = None) -> None:
        self._ops: dict[str, Callable[[Any, Any], Any]] = dict(_RUNTIME_DEPENDENT_OPS)
        if additional_ops:
            self._ops.update(additional_ops)

    def __call__(self, graph_module: Any) -> PassResult | None:
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op not in {"call_function", "call_method"}:
                continue
            decomposition = self._ops.get(_target_name(node.target))
            if decomposition is None:
                continue
            if not callable(decomposition):
                raise TypeError(
                    f"runtime-dependent decomposition for {node.target!r} is not callable"
                )
            with graph.inserting_before(node):
                replacement = decomposition(graph, node)
            if replacement is None or replacement is node:
                continue
            node.replace_all_uses_with(replacement)
            if not node.users:
                graph.erase_node(node)
            modified = True
        if not modified:
            return None
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)


def _device_for(value: Any, location: Any) -> Any:
    current = getattr(value, "device", None)
    if isinstance(location, Mapping):
        return location.get(str(current), current)
    return location


def _move_value(value: Any, location: Any) -> Any:
    if hasattr(value, "to") and callable(value.to):
        target = _device_for(value, location)
        return value.to(target) if target is not None else value
    if isinstance(value, tuple):
        return tuple(_move_value(item, location) for item in value)
    if isinstance(value, list):
        return [_move_value(item, location) for item in value]
    if isinstance(value, dict):
        return {key: _move_value(item, location) for key, item in value.items()}
    return value


def move_to_device_pass(exported_program: Any, location: Any) -> Any:
    """Move captured state, examples, and device metadata to a destination."""

    result = copy.deepcopy(exported_program)
    root = result.graph_module.root
    for name, value in list(result.named_parameters()) + list(result.named_buffers()):
        parent_name, _, attribute = name.rpartition(".")
        parent = root if not parent_name else result.graph_module.get_submodule(parent_name)
        setattr(parent, attribute, _move_value(value, location))
    result.example_inputs = _move_value(result.example_inputs, location)
    for node in result.graph.nodes:
        node.args = _move_value(node.args, location)
        node.kwargs = _move_value(node.kwargs, location)
        node.meta = _move_value(node.meta, location)
    result.graph_module.recompile()
    result.validate()
    return result
