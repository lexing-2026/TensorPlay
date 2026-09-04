"""Experimental export transforms and lazy capture helpers."""

from __future__ import annotations

import copy
import dataclasses
import types
from collections.abc import Callable
from typing import Any

from .._trace import export
from ..exported_program import ExportedProgram
from ..graph_signature import ExportGraphSignature
from ._utils import _get_main_cpp_file, _get_make_file

__all__ = [
    "_ExportMethod",
    "_ExportPackage",
    "_export_forward_backward",
    "_get_main_cpp_file",
    "_get_make_file",
    "_remove_detach_pass",
    "_sticky_export",
    "update_tensor_list_mutable",
]


def update_tensor_list_mutable(
    ep: ExportedProgram, mutable_from_list: Any = None
) -> ExportedProgram:
    """Normalize tensor-list mutation bookkeeping in a captured program.

    Tensor lists flatten into per-element user inputs, so in-place element
    updates already surface as user-input mutation outputs.  This helper
    verifies that contract for every element of ``mutable_from_list`` and
    returns the program unchanged when it holds; a mismatch raises, since a
    silent repair would change what the graph writes back.
    """

    if mutable_from_list is None:
        return ep
    if not isinstance(mutable_from_list, (list, tuple)):
        raise TypeError("mutable_from_list must be a list or tuple of tensors")
    declared = {id(value) for value in mutable_from_list}
    example_inputs = ep.example_inputs
    for spec in ep.graph_signature.output_specs:
        if spec.kind.name != "USER_INPUT_MUTATION":
            continue
        container_name = spec.target
        example = example_inputs.get(container_name) if container_name else None
        if example is None:
            continue
        container = example if isinstance(example, (list, tuple)) else (example,)
        if all(id(item) not in declared for item in container):
            raise ValueError(
                f"mutation output {spec.arg.name!r} does not alias an element of "
                f"mutable_from_list; the declared mutable list is incomplete"
            )
    return ep


def _copy_graph_module_and_signature(ep: ExportedProgram) -> tuple[Any, ExportGraphSignature]:
    return copy.deepcopy(ep.graph_module), copy.deepcopy(ep.graph_signature)


def _remove_detach_pass(graph_module: Any, signature: ExportGraphSignature | None = None) -> None:
    del signature
    for node in list(graph_module.graph.nodes):
        target_name = getattr(node.target, "__name__", str(node.target))
        if node.op == "call_function" and target_name == "detach" and node.args:
            node.replace_all_uses_with(node.args[0])
            if not node.users:
                node.graph.erase_node(node)
    if hasattr(graph_module.graph, "eliminate_dead_code"):
        graph_module.graph.eliminate_dead_code()
    graph_module.recompile()


def _export_forward_backward(ep: ExportedProgram, joint_loss_index: int = 0) -> ExportedProgram:
    del joint_loss_index
    result = ep.run_decompositions()
    _remove_detach_pass(result.graph_module, result.graph_signature)
    result.validate()
    return result


def _sticky_export(
    forward_func: Callable[..., Any],
    dynamic_shapes_callback: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    if getattr(forward_func, "__self__", None) is None:
        raise TypeError("sticky export requires a bound forward method")
    model = forward_func.__self__
    original = forward_func.__func__

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        artifact = getattr(wrapper, "_exported_artifact", None)
        if artifact is None:
            model.forward = types.MethodType(original, model)
            shapes = dynamic_shapes_callback(*args, **kwargs) if dynamic_shapes_callback else None
            try:
                artifact = export(model, *args, dynamic_shapes=shapes, **kwargs)
                wrapper._exported_artifact = artifact
            finally:
                model.forward = wrapper
        return artifact(*args, **kwargs)

    wrapper.__name__ = getattr(forward_func, "__name__", "forward")
    wrapper.__doc__ = getattr(forward_func, "__doc__", None)
    return wrapper


@dataclasses.dataclass
class _ExportMethod:
    overloads: dict[str, ExportedProgram]
    fallbacks: list[ExportedProgram]


class _ExportPackage:
    def __init__(self) -> None:
        self.methods: dict[str, _ExportMethod] = {}

    def add(self, name: str, program: ExportedProgram, *, overload: str = "default") -> None:
        method = self.methods.setdefault(name, _ExportMethod({}, []))
        method.overloads[overload] = program

    def resolve(self, name: str, overload: str = "default") -> ExportedProgram:
        try:
            return self.methods[name].overloads[overload]
        except KeyError as exc:
            raise KeyError(f"exported method {name!r} with overload {overload!r} is missing") from exc
