"""Graph export frontend.

``tensorplay.export`` captures a model once into a static :class:`ExportedProgram`
whose graph is guaranteed to be free of Python data-dependent control flow and
whose parameters/buffers are separated from user inputs via
sympy constraint solving, guards machinery, or pt2 archive serialization.

The capture itself reuses the single compiler frontend
(:mod:`tensorplay.compiler.graph`), so an exported program is directly
consumable by every registered compiler backend.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .compiler.graph import GraphCaptureError, GraphModule, Node, Tracer

__all__ = [
    "Dim",
    "ExportedProgram",
    "GraphSignature",
    "export",
]


class Dim:
    """Symbolic dimension marker for ``dynamic_shapes``.

    Reuse the same instance across arguments to bind dimensions together::

        batch = export.Dim("batch")
        tp.export(model, x, y, dynamic_shapes={"x": {0: batch}, "y": {0: batch}})
    """

    _auto_counter = 0

    def __init__(self, name: str | None = None) -> None:
        if name is None:
            Dim._auto_counter += 1
            name = f"d{Dim._auto_counter}"
        self.name = str(name)

    def __repr__(self) -> str:
        return f"Dim({self.name!r})"


@dataclass(frozen=True)
class GraphSignature:
    """Separation of graph inputs into parameters, buffers, and user inputs.

    targets are qualified attribute paths into the root module (e.g.
    ``"linear.weight"``), while ``user_inputs`` are placeholder names bound to
    the exported callable's arguments.
    """

    parameters: tuple[str, ...]
    buffers: tuple[str, ...]
    non_persistent_buffers: tuple[str, ...]
    user_inputs: tuple[str, ...]


@dataclass
class ExportedProgram:
    """A captured, validated static program."""

    graph_module: GraphModule
    graph_signature: GraphSignature
    example_inputs: dict[str, Any] = field(default_factory=dict)
    dynamic_shapes: dict[str, dict[int, Any]] = field(default_factory=dict)

    @property
    def graph(self):
        return self.graph_module.graph

    def module(self) -> Callable[..., Any]:
        """Return a standalone executable built from the captured graph."""

        self.graph_module.recompile()
        return self.graph_module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Arguments omitted at call time fall back to the export-time binding
        # (``example_inputs``), so ``export(fn, x, offset=5.0)`` keeps the
        # keyword override when invoked as ``program(x)``.
        placeholders = list(self.graph.placeholders)
        if len(args) > len(placeholders):
            raise TypeError(
                f"exported program takes at most {len(placeholders)} "
                f"argument(s), got {len(args)}"
            )
        call_kwargs = dict(kwargs)
        for node, value in zip(placeholders, args):
            if node.name in call_kwargs:
                raise TypeError(f"duplicate value for argument {node.name!r}")
            call_kwargs[node.name] = value
        for node in placeholders:
            if node.name in call_kwargs:
                continue
            try:
                call_kwargs[node.name] = self.example_inputs[node.name]
            except KeyError as exc:
                raise TypeError(
                    f"missing required export input: {node.name}"
                ) from exc
        return self.graph_module(**call_kwargs)

    def print_readable(self) -> str:
        signature = self.graph_signature
        lines = [self.graph.python_code(), ""]
        lines.append(f"user_inputs            = {list(signature.user_inputs)}")
        lines.append(f"parameters             = {list(signature.parameters)}")
        lines.append(f"buffers                = {list(signature.buffers)}")
        if signature.non_persistent_buffers:
            lines.append(
                f"non_persistent_buffers = {list(signature.non_persistent_buffers)}"
            )
        if self.dynamic_shapes:
            lines.append(f"dynamic_shapes         = {self.dynamic_shapes}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.print_readable()


def _collect_attributes(root: Any) -> dict[str, tuple[str, bool]]:
    """Map qualified attribute path -> ("parameter"|"buffer", persistent)."""

    attributes: dict[str, tuple[str, bool]] = {}
    for module_name, module in root.named_modules(remove_duplicate=True):
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for attr_name, value in getattr(module, "_parameters", {}).items():
            if value is None:
                continue
            attributes[_qualified(module_name, attr_name)] = ("parameter", True)
        for attr_name, value in getattr(module, "_buffers", {}).items():
            if value is None:
                continue
            attributes[_qualified(module_name, attr_name)] = (
                "buffer",
                attr_name not in non_persistent,
            )
    return attributes


def _qualified(module_name: str, attribute: str) -> str:
    return f"{module_name}.{attribute}" if module_name else attribute


def _validate_graph(graph_module: GraphModule, attributes: dict[str, Any]) -> None:
    """Fail at export time instead of silently producing a wrong program."""

    for node in graph_module.graph.nodes:
        kind = node.op
        if kind == "get_attr":
            if node.target not in attributes:
                raise GraphCaptureError(
                    f"get_attr target {node.target!r} does not exist on the "
                    f"exported model; known attributes: {sorted(attributes)}"
                )
        elif kind == "call_function":
            if not callable(node.target):
                raise GraphCaptureError(
                    f"call_function target {node.target!r} is not callable"
                )
        elif kind == "call_method":
            if not node.args or not isinstance(node.target, str):
                raise GraphCaptureError(f"malformed call_method node: {node}")
        elif kind not in {"placeholder", "output", "call_module"}:
            raise GraphCaptureError(f"unsupported graph node kind: {kind!r}")

    if not graph_module.graph.outputs:
        raise GraphCaptureError("captured graph has no output")


def _normalize_dynamic_shapes(
    spec: dict[str, dict[int, Any]] | None,
    parameter_names: list[str],
) -> dict[str, dict[int, Any]]:
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise TypeError("dynamic_shapes must be a dict keyed by argument name")

    normalized: dict[str, dict[int, Any]] = {}
    for arg_name, dims in spec.items():
        if arg_name not in parameter_names:
            raise ValueError(
                f"dynamic_shapes key {arg_name!r} does not match any argument; "
                f"expected one of {parameter_names}"
            )
        if not isinstance(dims, dict):
            raise TypeError(
                f"dynamic_shapes[{arg_name!r}] must be a dict of dim index -> int|Dim"
            )
        entry: dict[int, Any] = {}
        for dim_index, value in dims.items():
            if not isinstance(dim_index, int) or dim_index < 0:
                raise TypeError(
                    f"dimension index must be a non-negative int, got {dim_index!r}"
                )
            if isinstance(value, Dim):
                entry[dim_index] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                entry[dim_index] = value
            else:
                raise TypeError(
                    f"dimension spec must be int or Dim, got {type(value)!r}"
                )
        normalized[arg_name] = entry
    return normalized


def export(
    model: Callable[..., Any],
    *args: Any,
    dynamic_shapes: dict[str, dict[int, Any]] | None = None,
    **kwargs: Any,
) -> ExportedProgram:
    """Capture ``model`` into a validated static :class:`ExportedProgram`.

    Args:
        model: an ``nn.Module`` or plain callable.  Child modules are inlined;
            parameters and buffers become ``get_attr`` nodes.
        args/kwargs: example inputs used to bind defaults and recorded as
            ``example_inputs``.  They are not executed against the model.
        dynamic_shapes: optional mapping of argument name to
            ``{dim_index: int | Dim}`` describing which dimensions are dynamic.
            Static entries record the expected size; :class:`Dim` entries are
            symbolic names shared by identity across arguments.

    Raises:
        GraphCaptureError: when the model contains constructs that cannot be
            captured statically (data-dependent control flow, unsupported node
            kinds, dangling attribute references).
    """

    graph_module = Tracer().trace(model)
    attributes = _collect_attributes(model) if hasattr(model, "named_modules") else {}
    _validate_graph(graph_module, attributes)

    placeholder_nodes: list[Node] = graph_module.graph.placeholders
    user_inputs = tuple(node.name for node in placeholder_nodes)

    parameters: list[str] = []
    buffers: list[str] = []
    non_persistent_buffers: list[str] = []
    referenced = {
        node.target for node in graph_module.graph.nodes if node.op == "get_attr"
    }
    for qualified_name, (attr_kind, persistent) in sorted(attributes.items()):
        if qualified_name not in referenced:
            continue
        if attr_kind == "parameter":
            parameters.append(qualified_name)
        else:
            buffers.append(qualified_name)
            if not persistent:
                non_persistent_buffers.append(qualified_name)

    bound = graph_module.signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    example_inputs = {
        node.name: bound.arguments[node.name] for node in placeholder_nodes
    }

    return ExportedProgram(
        graph_module=graph_module,
        graph_signature=GraphSignature(
            parameters=tuple(parameters),
            buffers=tuple(buffers),
            non_persistent_buffers=tuple(non_persistent_buffers),
            user_inputs=user_inputs,
        ),
        example_inputs=example_inputs,
        dynamic_shapes=_normalize_dynamic_shapes(dynamic_shapes, user_inputs),
    )
