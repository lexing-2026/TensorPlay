"""Capture routines for building structured exported programs."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable

from ..graph import GraphCaptureError, GraphModule, Node, Proxy, Tracer
from .dynamic_shapes import AdditionalInputs, ConstraintsExceededError, Dim, ShapesCollection, _DimHint
from .exported_program import EqualityConstraint, ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)

__all__ = ["ExportTracer", "draft_export", "export", "export_for_training"]

_STATE_PREFIXES = {"parameter": "p", "buffer": "b", "constant": "c"}


def _qualified(module_name: str, attribute: str) -> str:
    return f"{module_name}.{attribute}" if module_name else attribute


def _collect_attributes(root: Any) -> dict[str, tuple[str, bool]]:
    """Map qualified attribute paths to ``(kind, persistent)``.

    Kinds cover parameters, persistent and non-persistent buffers, and plain
    tensor attributes (recorded as constants).  Constant entries carry the
    value so the tracer can lift them alongside parameters and buffers.
    """

    attributes: dict[str, tuple[str, bool]] = {}
    named_modules = getattr(root, "named_modules", None)
    if not callable(named_modules):
        return attributes
    import tensorplay as tp

    for module_name, module in named_modules(remove_duplicate=True):
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for name, value in getattr(module, "_parameters", {}).items():
            if value is not None:
                attributes[_qualified(module_name, name)] = ("parameter", True)
        for name, value in getattr(module, "_buffers", {}).items():
            if value is not None:
                attributes[_qualified(module_name, name)] = (
                    "buffer",
                    name not in non_persistent,
                )
        parameters = getattr(module, "_parameters", {})
        buffers = getattr(module, "_buffers", {})
        children = getattr(module, "_modules", {})
        for name, value in vars(module).items():
            if name.startswith("_") or name in parameters or name in buffers:
                continue
            if name in children or callable(value):
                continue
            if isinstance(value, tp.nn.Parameter) or not isinstance(value, tp.Tensor):
                continue
            attributes[_qualified(module_name, name)] = ("constant", True)
    return attributes


def _resolve_attribute(root: Any, target: str) -> Any:
    value = root
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


class ExportTracer(Tracer):
    """Tracer that lifts module state into graph inputs.

    Parameters, buffers, and constant tensors become placeholders ahead of the
    user inputs, so the captured graph is functional: it reads no attributes
    and its only inputs are the flat value list described by the graph
    signature.

    Child module forwards are additionally wrapped so every recorded node
    carries the qualified path of the module that produced it
    (``nn_module_stack``), and module call boundaries (argument and result
    nodes) are recorded for later hierarchy reconstruction.
    """

    def __init__(self, concrete_args: dict[str, Any] | None = None) -> None:
        super().__init__(concrete_args)
        # qualified attribute path -> (placeholder node, kind, persistent)
        self.state_targets: dict[str, tuple[Node, str, bool]] = {}
        self._constant_patches: list[tuple[Any, str, Any]] = []
        self._missing_sentinel = object()
        self._module_stack: tuple[str, ...] = ()
        self._forward_patches: list[tuple[Any, Any]] = []
        self.module_calls: list[dict[str, Any]] = []
        self._call_keys: set[tuple[Any, ...]] = set()

    def _register_state(self, root: Any) -> None:
        for target, (kind, persistent) in _collect_attributes(root).items():
            value = _resolve_attribute(root, target)
            mangled = f"{_STATE_PREFIXES[kind]}_{target.replace('.', '_')}"
            node = self.graph.create_node("placeholder", mangled, (), {}, name=mangled)
            node.meta["state_target"] = target
            node.meta["state_kind"] = kind
            node.meta["state_persistent"] = persistent
            self.state_targets[target] = (node, kind, persistent)

    def _patch_constants(self, root: Any) -> None:
        for target, (node, kind, _persistent) in self.state_targets.items():
            if kind != "constant":
                continue
            parent_name, _, leaf = target.rpartition(".")
            parent = _resolve_attribute(root, parent_name) if parent_name else root
            previous = getattr(parent, leaf, self._missing_sentinel)
            self._constant_patches.append((parent, leaf, previous))
            setattr(parent, leaf, Proxy(node, self))

    def _restore_constants(self) -> None:
        for parent, leaf, previous in reversed(self._constant_patches):
            if previous is self._missing_sentinel:
                delattr(parent, leaf)
            else:
                setattr(parent, leaf, previous)
        self._constant_patches.clear()

    def _wrap_child_forwards(self, root: Any) -> None:
        """Route child module calls through stack-tracking wrappers."""

        from ..graph._pytree import tree_flatten

        named_modules = getattr(root, "named_modules", None)
        if not callable(named_modules):
            return
        for module_name, module in named_modules(remove_duplicate=True):
            if not module_name:
                continue  # the root call is described by the program itself
            original = getattr(module, "forward", None)
            if not callable(original):
                continue

            def wrapper(
                *args: Any,
                _original: Any = original,
                _qualname: str = module_name,
                **kwargs: Any,
            ) -> Any:
                arg_nodes: list[Any] = []
                for value in args:
                    arg_nodes.append(
                        value.node.name if isinstance(value, Proxy) else value
                    )
                kwargs_nodes = {
                    key: value.node.name if isinstance(value, Proxy) else value
                    for key, value in kwargs.items()
                }
                self._module_stack = (*self._module_stack, _qualname)
                try:
                    result = _original(*args, **kwargs)
                finally:
                    self._module_stack = self._module_stack[:-1]
                result_nodes: list[str] = []

                def visit(item: Any) -> None:
                    if isinstance(item, Proxy):
                        result_nodes.append(item.node.name)
                    elif isinstance(item, (tuple, list)):
                        for entry in item:
                            visit(entry)
                    elif isinstance(item, dict):
                        for entry in item.values():
                            visit(entry)

                visit(result)
                _in_spec = tree_flatten(args)[1]
                _out_spec = tree_flatten(result)[1]
                key = (_qualname, tuple(map(repr, arg_nodes)), tuple(result_nodes))
                if key not in self._call_keys:
                    self._call_keys.add(key)
                    self.module_calls.append(
                        {
                            "fqn": _qualname,
                            "args": arg_nodes,
                            "kwargs": kwargs_nodes,
                            "result": result_nodes,
                            "in_spec": _in_spec,
                            "out_spec": _out_spec,
                        }
                    )
                return result

            try:
                module.forward = wrapper  # type: ignore[method-assign]
            except Exception:
                continue
            self._forward_patches.append((module, original))

    def _restore_child_forwards(self) -> None:
        for module, _original in self._forward_patches:
            try:
                del module.forward
            except AttributeError:
                pass
        self._forward_patches.clear()

    def trace(self, root: Any, sample_inputs: dict[str, Any] | None = None) -> GraphModule:
        self.root = root
        if callable(getattr(root, "named_modules", None)):
            self._register_state(root)
            self._patch_constants(root)
            self._wrap_child_forwards(root)
            try:
                return super().trace(root, sample_inputs)
            finally:
                self._restore_child_forwards()
                self._restore_constants()
        return super().trace(root, sample_inputs)

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Proxy:
        if kind == "get_attr":
            entry = self.state_targets.get(target)
            if entry is not None:
                return Proxy(entry[0], self)
        proxy = super().create_proxy(kind, target, args, kwargs)
        if self._module_stack:
            proxy.node.meta["nn_module_stack"] = self._module_stack
        return proxy


def _validate_graph(graph_module: GraphModule, attributes: Mapping[str, Any]) -> None:
    for node in graph_module.graph.nodes:
        op = node.op
        if op == "get_attr":
            if node.target not in attributes:
                raise GraphCaptureError(
                    f"get_attr target {node.target!r} is not present on the captured model"
                )
        elif op == "call_function":
            if not callable(node.target):
                raise GraphCaptureError(f"call_function target {node.target!r} is not callable")
        elif op == "call_method":
            if not node.args or not isinstance(node.target, str):
                raise GraphCaptureError(f"malformed call_method node: {node}")
        elif op not in {"placeholder", "output", "call_module"}:
            raise GraphCaptureError(f"unsupported graph node kind: {op!r}")
    if not graph_module.graph.outputs:
        raise GraphCaptureError("captured graph has no output")


def _normalize_dynamic_shapes(
    spec: Any,
    parameter_names: list[str],
    model: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    if spec is None:
        return {}
    if isinstance(spec, AdditionalInputs):
        return spec.dynamic_shapes(model, args, kwargs)
    if isinstance(spec, ShapesCollection):
        return spec.dynamic_shapes(model, args, kwargs)
    if isinstance(spec, (list, tuple)):
        if len(spec) > len(parameter_names):
            raise ValueError("dynamic_shapes has more entries than graph inputs")
        return {
            name: _validate_dimension_spec(value)
            for name, value in zip(parameter_names, spec)
            if value is not None
        }
    if not isinstance(spec, dict):
        raise TypeError("dynamic_shapes must be a mapping or a sequence")

    normalized: dict[str, Any] = {}
    for name, dims_spec in spec.items():
        if name not in parameter_names:
            raise ValueError(
                f"dynamic_shapes key {name!r} does not match any argument; "
                f"expected one of {parameter_names}"
            )
        if isinstance(dims_spec, dict):
            entry: dict[int, Any] = {}
            for index, value in dims_spec.items():
                if type(index) is not int or index < 0:
                    raise TypeError(f"dimension index must be a non-negative int, got {index!r}")
                entry[index] = _validate_dimension_value(value)
            normalized[name] = entry
        elif isinstance(dims_spec, (tuple, list)):
            normalized[name] = _validate_dimension_spec(dims_spec)
        elif dims_spec is None:
            normalized[name] = None
        else:
            raise TypeError(f"dynamic_shapes[{name!r}] must describe dimensions")
    return normalized


def _validate_dimension_value(value: Any) -> Any:
    if value is None or isinstance(value, (Dim, _DimHint)):
        return value
    if type(value) is int:
        return value
    raise TypeError(
        "dimension spec must be int or Dim (a dim hint or None is also accepted), "
        f"got {type(value)!r}"
    )


def _validate_dimension_spec(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[int, Any] = {}
        for index, item in value.items():
            if type(index) is not int or index < 0:
                raise TypeError(f"dimension index must be a non-negative int, got {index!r}")
            result[index] = _validate_dimension_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return tuple(_validate_dimension_value(item) for item in value)
    return _validate_dimension_value(value)


def _argument_for_node(node: Any) -> Any:
    if isinstance(node, Node):
        return TensorArgument(node.name)
    if isinstance(node, (str, int, float, bool)) or node is None:
        return ConstantArgument(f"constant_{abs(hash(repr(node))) % 100000}", node)
    return ConstantArgument(f"constant_{abs(hash(type(node).__name__)) % 100000}", node)


def _flatten_leaves(value: Any) -> list[Any]:
    leaves: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, Node):
            leaves.append(item)
        elif isinstance(item, (tuple, list)):
            for entry in item:
                visit(entry)
        elif isinstance(item, dict):
            for entry in item.values():
                visit(entry)
        else:
            leaves.append(item)

    visit(value)
    return leaves


def _mutation_chain_root(node: Node) -> Node:
    """Walk in-place op and element-read chains back to the state holder.

    Two hop rules apply: in-place methods (``add_``) consume the previous
    value of the object they update, and ``getitem`` reads reach into a
    container that itself entered the graph as one placeholder.  Hopping
    through both attributes an element update to the container input.
    """

    import operator

    seen: set[int] = set()
    current = node
    while id(current) not in seen:
        seen.add(id(current))
        if (
            current.op == "call_method"
            and isinstance(current.target, str)
            and current.target.endswith("_")
            and current.args
            and isinstance(current.args[0], Node)
        ):
            current = current.args[0]
            continue
        if (
            current.op == "call_function"
            and current.target is operator.getitem
            and current.args
            and isinstance(current.args[0], Node)
        ):
            current = current.args[0]
            continue
        break
    return current


def _detect_mutations(
    graph_module: GraphModule,
    state_targets: Mapping[str, tuple[Node, str, bool]],
) -> list[tuple[Node, OutputKind, str | None]]:
    """Find in-place updates of lifted state or user inputs.

    Returns ``(holder, kind, target)`` tuples where ``holder`` is the graph
    node carrying the final value of the mutated object.
    """

    state_by_placeholder: dict[str, tuple[str, str, bool]] = {
        node.name: (target, kind, persistent)
        for target, (node, kind, persistent) in state_targets.items()
    }
    holders: dict[str, tuple[OutputKind, str | None]] = {}
    for node in graph_module.graph.placeholders:
        entry = state_by_placeholder.get(node.name)
        if entry is not None:
            _target, kind, _persistent = entry
            holders[node.name] = (
                (
                    OutputKind.PARAMETER_MUTATION
                    if kind == "parameter"
                    else OutputKind.BUFFER_MUTATION
                ),
                _target,
            )
        else:
            holders[node.name] = (OutputKind.USER_INPUT_MUTATION, None)
    mutations: dict[str, tuple[Node, OutputKind, str | None]] = {}
    for node in graph_module.graph.nodes:
        if node.op != "call_method" or not isinstance(node.target, str):
            continue
        if not node.target.endswith("_") or not node.args or not isinstance(node.args[0], Node):
            continue
        root = _mutation_chain_root(node)
        info = holders.get(root.name)
        if info is None:
            continue
        # only a container element (getitem chain) may update a user input;
        # a direct in-place call on a whole input tensor is graph-invisible
        # by construction because capture runs on value copies
        if info[0] is OutputKind.USER_INPUT_MUTATION and root is node.args[0]:
            continue
        target = info[1] if info[1] is not None else root.name
        mutations[node.name] = (node, info[0], target)
    return list(mutations.values())


def _state_entry_for_placeholder(
    state_targets: Mapping[str, tuple[Node, str, bool]],
    placeholder_name: str,
) -> tuple[str, str, bool] | None:
    for target, (node, kind, persistent) in state_targets.items():
        if node.name == placeholder_name:
            return target, kind, persistent
    return None


def _state_map_for_placeholders(
    state_targets: Mapping[str, tuple[Node, str, bool]],
) -> dict[str, tuple[str, str, bool]]:
    """Placeholder-name-keyed view of the lifted-state table.

    Signature construction visits placeholders in graph order; a name-keyed
    map keeps that walk linear instead of scanning the state table per node.
    """

    return {
        node.name: (target, kind, persistent)
        for target, (node, kind, persistent) in state_targets.items()
    }


def _rewrite_container_reads(
    graph_module: GraphModule,
    mutations: list[tuple[Node, OutputKind, str | None]],
) -> None:
    """Point element reads after a mutation at the mutated value.

    A second ``items[0]`` read records its own getitem node; without this
    rewrite it would observe the pre-mutation element and diverge from eager
    execution, where both reads return the same object.
    """

    import operator

    if not mutations:
        return
    graph = graph_module.graph
    order = {node.name: position for position, node in enumerate(graph.nodes)}
    finals: dict[tuple[str, int], Node] = {}
    for node, _kind, _target in mutations:
        current = node
        container: Node | None = None
        index: int | None = None
        while True:
            if (
                current.op == "call_function"
                and current.target is operator.getitem
                and current.args
                and isinstance(current.args[0], Node)
            ):
                if index is None and len(current.args) > 1 and isinstance(current.args[1], int):
                    container = current.args[0]
                    index = current.args[1]
                current = current.args[0]
                continue
            break
        if container is not None and index is not None:
            finals[(container.name, index)] = node
    if not finals:
        return
    for read in list(graph.nodes):
        if read.op != "call_function" or read.target is not operator.getitem:
            continue
        if len(read.args) != 2 or not isinstance(read.args[0], Node):
            continue
        key = (read.args[0].name, read.args[1]) if isinstance(read.args[1], int) else None
        final = finals.get(key) if key is not None else None
        if final is None or final is read:
            continue
        if order.get(final.name, -1) >= order.get(read.name, 1 << 30):
            # the read happens before the mutation; it keeps the old value
            continue
        read.replace_all_uses_with(final)
        if not read.users:
            graph.erase_node(read)


def _output_specs(
    graph_module: GraphModule,
    mutations: list[tuple[Node, OutputKind, str | None]],
) -> list[OutputSpec]:
    output = graph_module.graph.output_node
    leaves = _flatten_leaves(output.args[0])
    specs = [
        OutputSpec(kind, TensorArgument(node.name), target)
        for node, kind, target in mutations
    ]
    specs.extend(
        OutputSpec(OutputKind.USER_OUTPUT, _argument_for_node(value))
        for value in leaves[len(mutations):]
    )
    return specs


def _restructure_output(
    graph_module: GraphModule,
    mutations: list[tuple[Node, OutputKind, str | None]],
) -> None:
    """Rewrite the graph result to ``(mutations..., flattened_user_outputs...)``.

    A mutated value may also be a user output; the flat contract repeats the
    node in both roles, exactly as the signature describes.
    """

    if not mutations:
        return
    graph = graph_module.graph
    output = graph.output_node
    leaves = _flatten_leaves(output.args[0])
    mutation_nodes = [node for node, _kind, _target in mutations]
    graph.output(tuple([*mutation_nodes, *leaves]))


# -- runtime assertions for dynamic-shape contracts ------------------------


def _assert_dim_range(tensor: Any, index: int, min: Any, max: Any, name: str) -> Any:
    size = tuple(tensor.shape)[index]
    if (min is not None and size < min) or (max is not None and size > max):
        raise ConstraintsExceededError(
            f"runtime assertion failed for {name!r}: expected dimension {index} "
            f"in [{min if min is not None else '-inf'}, {max if max is not None else 'inf'}], "
            f"got {size}"
        )
    return size


def _assert_dims_equal(tensor_a: Any, index_a: int, tensor_b: Any, index_b: int, name: str) -> Any:
    size_a = tuple(tensor_a.shape)[index_a]
    size_b = tuple(tensor_b.shape)[index_b]
    if size_a != size_b:
        raise ConstraintsExceededError(
            f"runtime assertion failed for {name!r}: dimensions "
            f"{index_a} and {index_b} must agree, got {size_a} and {size_b}"
        )
    return size_a


def _assert_dim_relation(
    tensor_root: Any,
    index_root: int,
    tensor_derived: Any,
    index_derived: int,
    scale: int,
    offset: int,
    name: str,
) -> Any:
    size_root = tuple(tensor_root.shape)[index_root]
    size_derived = tuple(tensor_derived.shape)[index_derived]
    expected = scale * size_root + offset
    if size_derived != expected:
        raise ConstraintsExceededError(
            f"runtime assertion failed for {name!r}: expected dimension "
            f"{index_derived} == {scale} * dim {index_root} + {offset} "
            f"({expected}), got {size_derived}"
        )
    return size_derived


def _apply_dynamic_shape_constraints(
    graph_module: GraphModule,
    combined_args: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> tuple[dict[str, dict[str, int | None]], list[EqualityConstraint]]:
    """Validate the spec, insert runtime assertions, and describe shared dims.

    Returns the per-name range bounds and one equality record per named
    dimension that appears at more than one input site (shared dims must hold
    equal sizes across those sites at runtime).
    """

    from .dynamic_shapes import _constraint_program, _process_dynamic_shapes
    from .dim_constraints import DimConstraints, attach_observed_sizes

    constraints = _process_dynamic_shapes(combined_args, normalized)
    if not constraints:
        return {}, []

    # strict checks first: contradictory range declarations for one name are
    # specification errors regardless of the example inputs
    asserts, ranges = _constraint_program(constraints)

    solver = DimConstraints()
    for constraint, observed in attach_observed_sizes(constraints, combined_args, normalized):
        solver.add(constraint, observed)
    if not solver.solve():
        raise ConstraintsExceededError(
            "export-time dimension constraints are inconsistent:\n"
            + solver.pretty_print()
        )

    placeholders = {node.name: node for node in graph_module.graph.placeholders}
    identity_to_name = {
        id(value): name
        for name, value in combined_args.items()
        if not isinstance(value, (dict, list, tuple))
    }

    # one equality record per name spanning several distinct input sites
    site_pairs: dict[str, set[tuple[str, int]]] = {}
    for constraint in constraints:
        if constraint.name is None:
            continue
        input_name = identity_to_name.get(id(constraint.source))
        if input_name is None:
            continue
        site_pairs.setdefault(constraint.name, set()).add((input_name, constraint.dim))
    equality_constraints = [
        EqualityConstraint(tuple(sorted(pairs, key=repr)), name=name)
        for name, pairs in site_pairs.items()
        if len(pairs) > 1
    ]

    def node_of(source: Any) -> Node | None:
        name = identity_to_name.get(id(source))
        return placeholders.get(name) if name is not None else None

    anchors: dict[str, tuple[Node, int]] = {}
    graph = graph_module.graph
    output = graph.output_node
    with graph.inserting_before(output):
        for constraint in asserts:
            if constraint.root is None and constraint.name is not None:
                node = node_of(constraint.source)
                if node is not None:
                    anchors[constraint.name] = (node, constraint.dim)
        for constraint in asserts:
            node = node_of(constraint.source)
            if node is None:
                continue
            if constraint.root is not None:
                anchor = anchors.get(constraint.root)
                if anchor is None:
                    continue
                graph.call_function(
                    _assert_dim_relation,
                    (
                        anchor[0],
                        anchor[1],
                        node,
                        constraint.dim,
                        constraint.scale,
                        constraint.offset,
                        constraint.name or constraint.root,
                    ),
                )
                continue
            if constraint.name is not None:
                anchor = anchors.get(constraint.name)
                if anchor is not None and anchor[0] is not node:
                    graph.call_function(
                        _assert_dims_equal,
                        (
                            anchor[0],
                            anchor[1],
                            node,
                            constraint.dim,
                            constraint.name,
                        ),
                    )
                    continue
            if constraint.min is None and constraint.max is None:
                continue
            graph.call_function(
                _assert_dim_range,
                (
                    node,
                    constraint.dim,
                    constraint.min,
                    constraint.max,
                    constraint.name or f"dim {constraint.dim}",
                ),
            )
    return ranges, equality_constraints


def _graph_signature(
    graph_module: GraphModule,
    state_targets: Mapping[str, tuple[Node, str, bool]],
    mutations: list[tuple[Node, OutputKind, str | None]],
) -> ExportGraphSignature:
    """Signature over the flat input contract: state first, then user inputs."""

    state_by_placeholder = _state_map_for_placeholders(state_targets)
    inputs: list[InputSpec] = []
    for node in graph_module.graph.placeholders:
        entry = state_by_placeholder.get(node.name)
        if entry is None:
            inputs.append(InputSpec(InputKind.USER_INPUT, TensorArgument(node.name), None))
            continue
        target, kind, persistent = entry
        if kind == "parameter":
            inputs.append(
                InputSpec(InputKind.PARAMETER, TensorArgument(node.name), target)
            )
        elif kind == "buffer":
            inputs.append(
                InputSpec(
                    InputKind.BUFFER,
                    TensorArgument(node.name),
                    target,
                    persistent=persistent,
                )
            )
        else:
            inputs.append(
                InputSpec(
                    InputKind.CONSTANT_TENSOR,
                    TensorArgument(node.name),
                    target,
                    persistent=None,
                )
            )
    return ExportGraphSignature(inputs, _output_specs(graph_module, mutations))


def _bind_examples(graph_module: GraphModule, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = graph_module.signature
    if signature is None:
        signature = inspect.signature(getattr(graph_module.root, "forward", graph_module.root))
    bound = signature.bind_partial(*args, **dict(kwargs))
    bound.apply_defaults()
    return {
        node.name: bound.arguments[
            node.target if isinstance(node.target, str) else node.name
        ]
        for node in graph_module.graph.placeholders
        if (
            node.target if isinstance(node.target, str) else node.name
        ) in bound.arguments
    }


def _flat_signature(
    graph_module: GraphModule,
    state_targets: Mapping[str, tuple[Node, str, bool]],
    example_inputs: Mapping[str, Any] | None = None,
) -> inspect.Signature:
    """Signature covering every flat input: state placeholders, then user args.

    User parameters keep their kinds; defaults prefer the export-time binding
    over the source declaration so omitted call arguments replay the capture.
    """

    example_inputs = example_inputs or {}
    state_names = {node.name for node, _kind, _persistent in state_targets.values()}
    user_parameters = (
        dict(graph_module.signature.parameters)
        if graph_module.signature is not None
        else {}
    )
    parameters: list[inspect.Parameter] = []
    for node in graph_module.graph.placeholders:
        if node.name in state_names:
            parameters.append(
                inspect.Parameter(node.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
            continue
        original = user_parameters.get(node.name)
        default = original.default if original is not None else inspect.Parameter.empty
        if node.name in example_inputs:
            default = example_inputs[node.name]
        if original is not None and default is not original.default:
            original = inspect.Parameter(
                original.name, original.kind, default=default
            )
        if original is not None:
            parameters.append(original)
        elif node.args:
            parameters.append(
                inspect.Parameter(
                    node.name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=node.args[0],
                )
            )
        else:
            parameters.append(
                inspect.Parameter(node.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
    return inspect.Signature(parameters)


def _capture(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    dynamic_shapes: Any,
) -> ExportedProgram:
    from ..graph._pytree import tree_flatten
    from .dynamic_shapes import _combine_args

    tracer = ExportTracer()
    target = model.forward if callable(getattr(model, "forward", None)) else model
    try:
        bound = inspect.signature(target).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        sample_inputs = dict(bound.arguments)
    except (TypeError, ValueError):
        sample_inputs = None
    from tensorplay.compiler import _exporting_context

    with _exporting_context():
        graph_module = tracer.trace(model, sample_inputs=sample_inputs)
    attributes = _collect_attributes(model)
    _validate_graph(graph_module, attributes)
    placeholders = graph_module.graph.placeholders
    names = [node.name for node in placeholders]
    examples = _bind_examples(graph_module, args, kwargs)
    normalized = _normalize_dynamic_shapes(dynamic_shapes, names, model, args, kwargs)

    meta = graph_module.meta
    meta["user_signature"] = graph_module.signature
    meta["state_targets"] = dict(tracer.state_targets)
    constants: dict[str, Any] = {}
    for target, (_node, kind, persistent) in tracer.state_targets.items():
        if kind == "constant" or (kind == "buffer" and not persistent):
            constants[target] = _resolve_attribute(model, target)
    meta["constants"] = constants

    mutations = _detect_mutations(graph_module, tracer.state_targets)
    user_out_spec = tree_flatten(graph_module.graph.output_node.args[0])[1]
    ranges, equality_constraints = _apply_dynamic_shape_constraints(
        graph_module, _combine_args(model, args, kwargs), normalized
    )
    _rewrite_container_reads(graph_module, mutations)
    _restructure_output(graph_module, mutations)
    if mutations or ranges:
        graph_module.recompile()
    meta["num_mutations"] = len(mutations)
    meta["out_spec"] = user_out_spec
    meta["in_spec"] = tree_flatten((tuple(args), dict(kwargs)))[1]
    graph_module.signature = _flat_signature(
        graph_module, tracer.state_targets, examples
    )

    meta["module_calls"] = list(tracer.module_calls)
    signature = _graph_signature(graph_module, tracer.state_targets, mutations)
    user_output_args = [
        spec.arg
        for spec in signature.output_specs
        if spec.kind is OutputKind.USER_OUTPUT
    ]
    root_entry = ModuleCallEntry(
        "",
        ModuleCallSignature(
            inputs=[TensorArgument(node.name) for node in placeholders],
            outputs=user_output_args,
            in_spec=meta.get("in_spec"),
            out_spec=meta.get("out_spec"),
            forward_arg_names=[node.name for node in placeholders],
        ),
    )
    call_entries: list[ModuleCallEntry] = []
    for record in tracer.module_calls:
        arg_specs = [
            TensorArgument(value) if isinstance(value, str) else ConstantArgument("", value)
            for value in record["args"]
        ]
        call_entries.append(
            ModuleCallEntry(
                record["fqn"],
                ModuleCallSignature(
                    inputs=arg_specs,
                    outputs=[TensorArgument(name) for name in record["result"]],
                    in_spec=record.get("in_spec"),
                    out_spec=record.get("out_spec"),
                    forward_arg_names=[
                        value if isinstance(value, str) else f"arg_{index}"
                        for index, value in enumerate(record["args"])
                    ],
                ),
            )
        )
    calls = [root_entry, *call_entries]
    program = ExportedProgram(
        graph_module=graph_module,
        graph_signature=signature,
        example_inputs=examples,
        dynamic_shapes=normalized,
        module_call_graph=calls,
        range_constraints=ranges,
        equality_constraints=equality_constraints,
    )
    program.validate()
    return program


def export(
    model: Callable[..., Any],
    *args: Any,
    dynamic_shapes: Any = None,
    strict: bool = False,
    preserve_module_call_signature: Any = (),
    **kwargs: Any,
) -> ExportedProgram:
    """Capture a callable and return an executable graph program.

    Args:
        model: an ``nn.Module`` or plain callable; child modules are inlined.
        args/kwargs: example inputs binding argument defaults.
        dynamic_shapes: dimension specification per argument (dict, sequence,
            :class:`ShapesCollection`, or :class:`AdditionalInputs`).
        strict: reserved for callers of the strict capture contract; capture
            validation is identical in both modes.
        preserve_module_call_signature: submodule paths whose call metadata is
            recorded in ``module_call_graph`` for module-level tooling.
    """

    if not callable(model):
        raise TypeError(f"model must be callable, got {type(model).__name__}")
    if isinstance(model, object) and hasattr(model, "named_modules"):
        known = {name for name, _ in model.named_modules(remove_duplicate=True)}
        unknown = [p for p in preserve_module_call_signature if p not in known]
        if unknown:
            raise GraphCaptureError(
                f"preserve_module_call_signature paths {sorted(unknown)} do not "
                f"exist on the model; known: {sorted(known)}"
            )
    program = _capture(model, args, kwargs, dynamic_shapes)
    if preserve_module_call_signature:
        existing = {entry.fqn for entry in program.module_call_graph}
        program.module_call_graph.extend(
            ModuleCallEntry(path)
            for path in preserve_module_call_signature
            if path not in existing
        )
    del strict
    return program


def export_for_training(
    model: Callable[..., Any],
    *args: Any,
    dynamic_shapes: Any = None,
    **kwargs: Any,
) -> ExportedProgram:
    """Capture a callable while retaining its mutable training state."""

    return export(model, *args, dynamic_shapes=dynamic_shapes, **kwargs)


def draft_export(
    model: Callable[..., Any],
    *args: Any,
    dynamic_shapes: Any = None,
    **kwargs: Any,
) -> Any:
    """Capture with failure reporting; see :mod:`tensorplay.export._draft_export`."""

    from ._draft_export import draft_export as _draft_export

    return _draft_export(model, *args, dynamic_shapes=dynamic_shapes, **kwargs)
