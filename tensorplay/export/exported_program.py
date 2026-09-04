"""Executable graph containers and export-time call metadata."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

from ..graph import GraphCaptureError, Graph, GraphModule
from ..graph._pytree import TreeSpec, tree_flatten, tree_unflatten
from .graph_signature import (
    ArgumentSpec,
    ExportGraphSignature,
    GraphSignature,
    InputKind,
    TensorArgument,
)

__all__ = [
    "EqualityConstraint",
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "Verifier",
    "default_decompositions",
]


@dataclasses.dataclass
class ModuleCallSignature:
    inputs: list[ArgumentSpec]
    outputs: list[ArgumentSpec]
    in_spec: TreeSpec | None = None
    out_spec: TreeSpec | None = None
    forward_arg_names: list[str] | None = None

    def replace_all_uses_with(self, original_node: Any, new_node: Any) -> None:
        old_name = getattr(original_node, "name", original_node)
        new_name = getattr(new_node, "name", new_node)
        for argument in (*self.inputs, *self.outputs):
            if argument.name == old_name:
                argument.name = new_name


@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: ModuleCallSignature | None = None


@dataclasses.dataclass(frozen=True)
class EqualityConstraint:
    """Ties several input sites to one shared dimension size.

    ``sites`` lists every ``(input placeholder name, dim index)`` pair whose
    runtime sizes must stay equal; ``name`` is the symbolic dimension they
    implement when the tie comes from a shared :class:`Dim`, else ``None``.
    """

    sites: tuple[tuple[str, int], ...]
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sites", tuple((str(item), int(dim)) for item, dim in self.sites)
        )
        if len(self.sites) < 2:
            raise ValueError("an equality constraint needs at least two sites")

    @property
    def dim_pairs(self) -> tuple[tuple[str, int], ...]:
        return self.sites

    def __repr__(self) -> str:
        return f"EqualityConstraint(sites={self.sites!r}, name={self.name!r})"


def _strip_mutation_outputs(graph_module: GraphModule, value: Any) -> Any:
    """Consume mutation outputs so callers observe the user's return value."""

    meta = getattr(graph_module, "meta", {})
    count = int(meta.get("num_mutations", 0) or 0)
    if count <= 0:
        return value
    flat = list(value) if isinstance(value, (tuple, list)) else [value]
    user_leaves = flat[count:]
    out_spec = meta.get("out_spec")
    if out_spec is None:
        return user_leaves[0] if len(user_leaves) == 1 else tuple(user_leaves)
    return tree_unflatten(user_leaves, out_spec)


def _user_output_count(graph_module: GraphModule) -> int:
    return int(getattr(graph_module, "meta", {}).get("num_mutations", 0) or 0)


def _ensure_parent_module(root: Any, path: str) -> Any:
    import tensorplay as tp

    parent: Any = root
    for atom in path.split("."):
        child = getattr(parent, atom, None)
        if child is None:
            child = tp.nn.Module()
            setattr(parent, atom, child)
        parent = child
    return parent


def _unlift_exported_program_lifted_states(program: "ExportedProgram") -> GraphModule:
    """Fold lifted state back into module attributes on a fresh root."""

    import inspect

    import tensorplay as tp
    from ..graph import Graph, GraphModule

    old_graph = program.graph
    state_specs = program._state_specs()
    state_by_name = {spec.arg.name: spec for spec in state_specs}

    root = tp.nn.Module()
    for spec in state_specs:
        value = program._resolve_state_value(spec.target)
        parent_name, _, leaf = str(spec.target).rpartition(".")
        parent = _ensure_parent_module(root, parent_name) if parent_name else root
        if spec.kind is InputKind.PARAMETER:
            parent.register_parameter(leaf, value)
        elif spec.kind is InputKind.BUFFER:
            parent.register_buffer(leaf, value, persistent=spec.persistent is not False)
        else:
            setattr(parent, leaf, value)

    new_graph = Graph()
    val_map: dict[Any, Any] = {}
    examples = program.example_inputs
    for node in old_graph.placeholders:
        spec = state_by_name.get(node.name)
        if spec is not None:
            val_map[node] = new_graph.create_node(
                "get_attr", str(spec.target), name=node.name
            )
        else:
            default = node.args[0] if node.args else inspect.Parameter.empty
            if node.name in examples:
                default = examples[node.name]
            val_map[node] = new_graph.placeholder(node.name, default)
    output_value = new_graph.graph_copy(old_graph, val_map)
    new_graph.output(output_value)

    user_signature = program.graph_module.meta.get("user_signature")
    unlifted = GraphModule(root, new_graph, user_signature)
    unlifted.meta = dict(program.graph_module.meta)
    unlifted.meta.pop("state_targets", None)
    if _user_output_count(unlifted) > 0:
        return _UserFacingModule(unlifted)
    return unlifted


class _UserFacingModule(GraphModule):
    """Executable view whose forward returns user outputs only.

    Mutation outputs stay in the captured graph (they are part of the flat
    contract); this wrapper consumes them so callers observe the same return
    structure as the original callable.
    """

    def __init__(self, graph_module: GraphModule) -> None:
        super().__init__(graph_module.root, graph_module.graph, graph_module.signature)
        self.meta = dict(getattr(graph_module, "meta", {}))
        self._mutation_count = int(self.meta.get("num_mutations", 0) or 0)
        self._out_spec = self.meta.get("out_spec")
        self._compiled_forward = getattr(graph_module, "_compiled_forward", None)
        self.__dict__.pop("forward", None)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return _strip_mutation_outputs(self, super().forward(*args, **kwargs))


class _CallSpec(NamedTuple):
    in_spec: TreeSpec | None
    out_spec: TreeSpec | None


class _ProgramBindings(NamedTuple):
    """Precomputed input-layout facts used on every program invocation."""

    key: int
    state_specs: tuple[Any, ...]
    state_names: frozenset[str]
    user_placeholders: tuple[str, ...]


class Verifier:
    """Structural checks every captured program must satisfy."""

    dialect = "STABLE"

    def check(self, program: "ExportedProgram") -> None:
        graph = program.graph
        signature = program.graph_signature
        placeholders = {node.name for node in graph.placeholders}
        user_specs = [
            spec for spec in signature.input_specs
            if spec.kind is InputKind.USER_INPUT
        ]
        for spec in user_specs:
            if spec.arg.name not in placeholders:
                raise GraphCaptureError(
                    f"input spec {spec.arg.name!r} has no matching placeholder"
                )
        non_user = [
            spec for spec in signature.input_specs
            if spec.kind is not InputKind.USER_INPUT
        ]
        attr_targets = {
            node.target for node in graph.nodes if node.op == "get_attr"
        }
        for spec in non_user:
            if spec.arg.name in placeholders:
                # flat-lifted capture: state enters as placeholders
                continue
            if isinstance(spec.target, str) and spec.target in attr_targets:
                continue
            raise GraphCaptureError(
                f"{spec.kind.name} spec {spec.target!r} is neither a placeholder "
                f"nor a graph attribute"
            )
        flat_outputs = self._flat_outputs(graph)
        output_specs = signature.output_specs
        if len(flat_outputs) != len(output_specs):
            raise GraphCaptureError(
                f"graph produces {len(flat_outputs)} outputs but the signature "
                f"describes {len(output_specs)}"
            )
        for value, spec in zip(flat_outputs, output_specs):
            name = getattr(value, "name", None)
            if name is not None and name != spec.arg.name:
                raise GraphCaptureError(
                    f"output value {name!r} does not match output spec "
                    f"{spec.arg.name!r}"
                )
        mutation_targets = (
            *signature.buffers_to_mutate.values(),
            *signature.parameters_to_mutate.values(),
        )
        state_targets = set(signature.parameters) | set(signature.buffers)
        for target in mutation_targets:
            if target not in state_targets:
                raise GraphCaptureError(
                    f"mutation target {target!r} is not lifted graph state"
                )

    @staticmethod
    def _flat_outputs(graph: Graph) -> list[Any]:
        leaves: list[Any] = []
        stack = [graph.output_node.args[0]]
        while stack:
            item = stack.pop(0)
            if isinstance(item, (tuple, list)):
                stack.extend(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
            else:
                leaves.append(item)
        return leaves


@dataclasses.dataclass
class ExportedProgram:
    """A validated graph together with its state and example bindings."""

    graph_module: GraphModule
    graph_signature: ExportGraphSignature | GraphSignature
    example_inputs: dict[str, Any] = dataclasses.field(default_factory=dict)
    dynamic_shapes: Any = None
    module_call_graph: list[ModuleCallEntry] = dataclasses.field(default_factory=list)
    range_constraints: dict[Any, Any] = dataclasses.field(default_factory=dict)
    equality_constraints: list[EqualityConstraint] = dataclasses.field(default_factory=list)
    verifier: Any = None
    _bindings: Any = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    _unlifted: Any = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.graph_signature, GraphSignature):
            self.graph_signature = self.graph_signature.to_export_signature()
        if not isinstance(self.graph_signature, ExportGraphSignature):
            raise TypeError("graph_signature must describe an exported graph")
        self.example_inputs = dict(self.example_inputs)
        if self.dynamic_shapes is None:
            self.dynamic_shapes = {}
        self.module_call_graph = list(self.module_call_graph)
        self.range_constraints = dict(self.range_constraints)
        self.equality_constraints = list(self.equality_constraints)
        if self.verifier is None:
            self.verifier = Verifier()

    @property
    def graph(self):
        return self.graph_module.graph

    @property
    def code(self) -> str:
        """Python source of the captured graph's generated forward."""
        return self.graph_module.graph.python_code()

    @property
    def call_spec(self) -> _CallSpec:
        meta = getattr(self.graph_module, "meta", {})
        return _CallSpec(meta.get("in_spec"), meta.get("out_spec"))

    @property
    def constants(self) -> dict[str, Any]:
        constants = getattr(self.graph_module, "meta", {}).get("constants", {})
        return dict(constants)

    @property
    def tensor_constants(self) -> dict[str, Any]:
        """Lifted non-parameter, non-buffer tensor values."""
        return self.constants

    def _mutation_count(self) -> int:
        return int(getattr(self.graph_module, "meta", {}).get("num_mutations", 0) or 0)

    def _user_output(self, value: Any) -> Any:
        """Strip mutation outputs from a flattened graph result."""

        return _strip_mutation_outputs(self.graph_module, value)

    def module(self) -> GraphModule:
        """Return a self-contained module with lifted state folded back in.

        The returned module takes only the user arguments: state placeholders
        are rewritten into attribute reads on a fresh module that owns the
        parameter, buffer, and constant values.  The result is cached; pass
        ``rebind`` to drop state changes made through this program view.
        """
        if self._unlifted is not None:
            return self._unlifted
        state_specs = self._state_specs()
        if not state_specs:
            self.graph_module.recompile()
            result = self.graph_module
        else:
            result = _unlift_exported_program_lifted_states(self)
        self._unlifted = result
        return result

    def invalidate_unlifted(self) -> None:
        """Drop the cached unlifted module so the next call rebuilds it."""
        self._unlifted = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_kwargs = self._user_call_kwargs(args, kwargs)
        return self._user_output(self.graph_module(**call_kwargs))

    def validate(self) -> None:
        self.graph.lint()
        if len(self.graph.outputs) != 1:
            raise GraphCaptureError("exported graph must have exactly one output")
        verifier = self.verifier
        if verifier is None:
            verifier = Verifier()
        if callable(verifier) and not hasattr(verifier, "check"):
            verifier(self)
        else:
            verifier.check(self)

    def run_decompositions(self, decomp_table: Any = None) -> "ExportedProgram":
        """Return a copied program after applying registered graph rewrites.

        Entries map a graph target (a callable, a method name, or a target
        string) to a builder invoked as ``builder(graph, node)``; the builder
        creates the replacement nodes and returns the value users should
        consume.  Nodes whose target has no entry are left untouched.
        """

        if decomp_table is None:
            decomp_table = default_decompositions()
        if not hasattr(decomp_table, "get"):
            raise TypeError("decomp_table must provide mapping access")
        result = copy.deepcopy(self)
        graph = result.graph
        for _round in range(8):
            changed = False
            for node in list(graph.nodes):
                if node.op not in {"call_function", "call_method"}:
                    continue
                replacement = _lookup_decomp(decomp_table, node)
                if replacement is None:
                    continue
                if not callable(replacement):
                    raise TypeError(f"decomposition for {node.target!r} is not callable")
                with graph.inserting_before(node):
                    new_value = replacement(graph, node)
                if new_value is None:
                    continue
                if new_value is not node:
                    node.replace_all_uses_with(new_value)
                graph.erase_node(node)
                changed = True
            if not changed:
                break
        graph.eliminate_dead_code()
        result.graph_module.recompile()
        result.validate()
        return result

    @property
    def state_dict(self) -> dict[str, Any]:
        """Tensor values of the lifted parameters and persistent buffers."""

        state: dict[str, Any] = {}
        for spec in self._state_specs():
            if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER):
                continue
            if spec.persistent is False:
                continue
            if not isinstance(spec.target, str):
                continue
            state[spec.target] = self._resolve_state_value(spec.target)
        return state

    def parameters(self) -> Iterator[Any]:
        """Iterate over the captured module's parameters."""

        for _, param in self.named_parameters():
            yield param

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        method = getattr(self.graph_module.root, "named_parameters", None)
        if callable(method):
            yield from method()

    def buffers(self) -> Iterator[Any]:
        """Iterate over the captured module's buffers."""

        for _, buf in self.named_buffers():
            yield buf

    def named_buffers(self) -> Iterator[tuple[str, Any]]:
        method = getattr(self.graph_module.root, "named_buffers", None)
        if callable(method):
            yield from method()

    def _state_specs(self) -> list[Any]:
        return list(self._layout().state_specs)

    def _layout(self) -> "_ProgramBindings":
        """Cached input-layout facts, keyed by the signature spec list object.

        The cache is refreshed whenever the signature's spec list is replaced
        (deep copies, passes, token removal); in-place arg renames do not
        change the layout, so they keep the cache valid.
        """

        key = id(self.graph_signature.input_specs)
        cached = self._bindings
        if cached is not None and cached.key == key:
            return cached
        state_specs = tuple(
            spec
            for spec in self.graph_signature.input_specs
            if spec.kind is not InputKind.USER_INPUT
        )
        state_names = frozenset(spec.arg.name for spec in state_specs)
        user_placeholders = tuple(
            node.name
            for node in self.graph.placeholders
            if node.name not in state_names
        )
        bindings = _ProgramBindings(key, state_specs, state_names, user_placeholders)
        self._bindings = bindings
        return bindings

    def _resolve_state_value(self, target: str) -> Any:
        value: Any = self.graph_module.root
        for atom in str(target).split("."):
            value = getattr(value, atom)
        return value

    def _user_call_kwargs(self, args: Any, kwargs: Any) -> dict[str, Any]:
        """Bind user arguments by name and fill lifted state values."""

        bindings = self._layout()
        call_kwargs = dict(kwargs)
        placeholders = self.graph.placeholders
        user_names = bindings.user_placeholders
        nodes_by_name = {node.name: node for node in placeholders}
        for index, name in enumerate(user_names):
            if index < len(args):
                if name in call_kwargs:
                    raise TypeError(f"duplicate value for argument {name!r}")
                call_kwargs[name] = args[index]
        for name in user_names:
            if name in call_kwargs:
                continue
            if name not in self.example_inputs:
                raise TypeError(f"missing required export input: {name}")
            call_kwargs[name] = self.example_inputs[name]
        if len(args) > len(user_names):
            raise TypeError(
                f"expected at most {len(user_names)} positional arguments, got {len(args)}"
            )
        for spec in bindings.state_specs:
            call_kwargs[spec.arg.name] = self._resolve_state_value(spec.target)
        return call_kwargs

    def _get_flat_args_with_check(self, args: Any, kwargs: Any) -> tuple[tuple[Any, ...], TreeSpec]:
        flat, spec = tree_flatten((args, kwargs))
        return tuple(flat), spec

    def _graph_module_flat_inputs(self, args: Any, kwargs: Any) -> tuple[Any, ...]:
        """Map user arguments onto the flat input contract.

        The flat graph expects the lifted state values first (in input spec
        order) followed by the flattened user inputs.
        """

        call_kwargs = self._user_call_kwargs(args, kwargs)
        ordered = [
            call_kwargs[node.name]
            for node in self.graph.placeholders
        ]
        return tuple(ordered)

    def _check_input_constraints(self, flat_args_with_path: Any) -> None:
        """Fail fast on structurally invalid inputs.

        Checks the leaf count against the user-input specs and, when the
        capture recorded a call contract, the tree structure of the caller's
        argument container.
        """

        user_specs = [
            spec for spec in self.graph_signature.input_specs
            if spec.kind is InputKind.USER_INPUT
        ]
        flat = [value for _path, value in flat_args_with_path]
        if len(flat) != len(user_specs):
            raise TypeError(
                f"expected {len(user_specs)} flattened user inputs, got {len(flat)}"
            )
        in_spec = self.call_spec.in_spec
        if in_spec is None:
            return
        from ..graph._pytree import tree_flatten as _flatten

        _leaves, actual = _flatten(tuple(flat))
        if actual != in_spec:
            raise TypeError(
                "input tree structure does not match the captured call contract"
            )

    @staticmethod
    def call_exported(program: "ExportedProgram") -> Callable[..., Any]:
        """Return a callable executing the flat contract on user arguments."""

        def runner(*args: Any, **kwargs: Any) -> Any:
            flat_inputs = program._graph_module_flat_inputs(args, kwargs)
            raw = program.graph_module(*flat_inputs)
            return program._user_output(raw)

        return runner

    def _transform_do_not_use(self, *passes: Callable[..., Any]) -> "ExportedProgram":
        """Run graph passes and rebuild the signature for the new node names."""

        transformed = copy.deepcopy(self)
        modified = False
        for pass_fn in passes:
            res = pass_fn(transformed.graph_module)
            if res is None:
                continue
            graph_module, did_modify = res
            modified = modified or did_modify
            transformed.graph_module = graph_module
        if not modified:
            return self
        old = self.graph_signature
        new_inputs = []
        for index, node in enumerate(transformed.graph.placeholders):
            spec = old.input_specs[index]
            arg = spec.arg
            if isinstance(arg, TensorArgument):
                arg = TensorArgument(node.name)
            new_inputs.append(
                type(spec)(spec.kind, arg, spec.target, spec.persistent)
            )
        flat_outputs = Verifier._flat_outputs(transformed.graph)
        new_outputs = []
        for index, value in enumerate(flat_outputs):
            spec = old.output_specs[index]
            arg = spec.arg
            if isinstance(arg, TensorArgument) and getattr(value, "name", None):
                arg = TensorArgument(value.name)
            new_outputs.append(type(spec)(spec.kind, arg, spec.target))
        transformed.graph_signature = ExportGraphSignature(new_inputs, new_outputs)
        transformed.validate()
        return transformed

    def serialize(self, opset_version: Any = None, pickle_protocol: int = 4) -> Any:
        """Return serialized program artifacts (JSON program + example inputs)."""

        from .serde import serialize

        return serialize(self, opset_version, pickle_protocol)

    @classmethod
    def deserialize(
        cls,
        artifact: Any,
        state_dict: Any = None,
        constants: Any = None,
        example_inputs: Any = None,
    ) -> "ExportedProgram":
        """Rebuild a program from :meth:`serialize` artifacts."""

        from .serde import deserialize

        return deserialize(artifact, state_dict, constants, example_inputs)

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

    def __repr__(self) -> str:
        return f"ExportedProgram({self.graph_module!r})"


def _lookup_decomp(decomp_table: Any, node: Any) -> Callable[..., Any] | None:
    """Resolve the decomposition entry for a node, if one is registered."""

    target = node.target
    for key in (target, getattr(target, "__name__", None), getattr(target, "name", None)):
        if key is None:
            continue
        try:
            entry = decomp_table.get(key)
        except Exception:
            entry = None
        if entry is not None:
            return entry
    return None


def default_decompositions() -> Any:
    """Return the mutable table of built-in graph rewrites."""

    from .decomp_utils import CustomDecompTable

    return CustomDecompTable()
