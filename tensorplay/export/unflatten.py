"""Module views and argument adapters for exported graphs."""

from __future__ import annotations

import abc
import copy
from typing import Any

from ..graph._pytree import TreeSpec, tree_flatten, tree_unflatten
from ..nn import Module as _Module
from .exported_program import ExportedProgram

__all__ = [
    "FlatArgsAdapter",
    "InterpreterModule",
    "UnflattenedModule",
    "unflatten",
]


class InterpreterModule:
    """A module that executes its graph through the stepwise interpreter.

    Interpreted execution gives precise per-node error reporting, which makes
    unflattened hierarchies easier to debug than generated executors.
    """

    def __init__(self, graph_module: Any, ty: str | None = None) -> None:
        from ..graph import GraphModule

        if not isinstance(graph_module, GraphModule):
            raise TypeError(
                f"expected a GraphModule, got {type(graph_module).__name__}"
            )
        self.graph_module = graph_module
        self.graph = graph_module.graph
        self._ty = ty

    @property
    def ty(self) -> str | None:
        return self._ty

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        from ..graph import Interpreter

        raw = Interpreter(self.graph_module).run(*args)
        from .exported_program import _strip_mutation_outputs

        return _strip_mutation_outputs(self.graph_module, raw)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def print_readable(self, print_output: bool = False) -> str:
        text = f"class {type(self).__name__}({self._ty or 'Module'}):" if self._ty else ""
        body = self.graph_module.print_readable()
        text = f"{text}\n{body}" if text else body
        if print_output:
            print(text)
        return text

    def __repr__(self) -> str:
        return f"InterpreterModule(ty={self._ty!r})"


class FlatArgsAdapter(abc.ABC):
    """Adapt one flattened argument layout into another layout."""

    @abc.abstractmethod
    def adapt(
        self,
        target_spec: TreeSpec,
        input_spec: TreeSpec,
        input_args: list[Any],
        metadata: dict[str, Any] | None = None,
        obj: Any | None = None,
    ) -> list[Any]:
        raise NotImplementedError

    def get_flat_arg_paths(self) -> list[str]:
        return []


class _TreeAdapter(FlatArgsAdapter):
    def __init__(self, target_spec: TreeSpec) -> None:
        self.target_spec = target_spec

    def adapt(
        self,
        target_spec: TreeSpec,
        input_spec: TreeSpec,
        input_args: list[Any],
        metadata: dict[str, Any] | None = None,
        obj: Any | None = None,
    ) -> list[Any]:
        del metadata, obj
        value = tree_unflatten(input_args, input_spec)
        flat, actual_spec = tree_flatten(value)
        if actual_spec != input_spec:
            raise ValueError("input values do not match the supplied tree specification")
        if target_spec != self.target_spec:
            raise ValueError("target specification does not match this adapter")
        return list(tree_flatten(tree_unflatten(flat, target_spec))[0])


class _FrameModule(_Module):
    """A reconstructed module whose body is one frame of the flat graph."""

    def __init__(self, graph_module: Any, ty: str | None = None) -> None:
        super().__init__()
        self.graph_module = graph_module
        self._ty = ty

    @property
    def root(self) -> Any:
        return self.graph_module.root

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            graph_module = self.__dict__.get("graph_module")
            if graph_module is not None:
                return getattr(graph_module.root, name)
            raise

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph_module(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def print_readable(self, print_output: bool = False) -> str:
        text = self.graph_module.print_readable()
        if print_output:
            print(text)
        return text

    def __repr__(self) -> str:
        return f"_FrameModule(ty={self._ty!r})"


def _node_owner(node: Any) -> str:
    """Qualified path of the module whose forward produced ``node``."""

    if node.op == "get_attr":
        return str(node.target).rpartition(".")[0]
    stack = node.meta.get("nn_module_stack")
    return stack[-1] if stack else ""


def _unflatten_nodes(leaves: list[Any], spec: Any) -> Any:
    """Rebuild a node structure shaped like ``spec`` from flat leaves."""

    import tensorplay as tp

    def build(current: Any, iterator: Any) -> Any:
        if current.type is None:
            return next(iterator)
        children = [build(child, iterator) for child in current.children_specs]
        if current.type is dict:
            return dict(zip(current.context, children))
        if current.type is list:
            return list(children)
        if current.type is tuple:
            return tuple(children)
        if isinstance(current.type, type) and issubclass(current.type, tuple):
            if hasattr(current.type, "_make"):
                return current.type._make(children)
            return current.type(*children)
        raise TypeError(f"cannot rebuild tree node of type {current.type!r}")

    return build(spec, iter(leaves))


def _map_value_tree(value: Any, map_node: Any) -> Any:
    """Apply a node mapping across a structured value tree."""

    if isinstance(value, tuple):
        return tuple(_map_value_tree(item, map_node) for item in value)
    if isinstance(value, list):
        return [_map_value_tree(item, map_node) for item in value]
    if isinstance(value, dict):
        return {key: _map_value_tree(item, map_node) for key, item in value.items()}
    return map_node(value)


def _frame_module_for(
    fqn: str,
    frame_nodes: list[Any],
    call_records: list[dict[str, Any]],
    calls_by_fqn: dict[str, list[dict[str, Any]]],
    graph: Any,
    name_to_node: dict[str, Any],
    state_root: Any,
    user_signature: Any,
    output_value: Any,
) -> Any:
    """Assemble one module frame from its slice of the flat graph.

    ``output_value`` is the structured return value for this frame: the
    output subtree for the root frame, or the recorded result-node names for
    a child frame.
    """

    import operator as _operator

    import tensorplay as tp
    from ..graph import Graph, GraphModule
    from ..graph.node import Node as _NodeType

    _NODE_TYPES = (_NodeType,)

    record = call_records[0]
    sub = Graph()
    val_map: dict[Any, Any] = {}
    external: dict[Any, Any] = {}
    child_results: dict[str, Any] = {}

    def placeholder_for(node: Any, name: str, default: Any = None) -> Any:
        if node not in external:
            args = (default,) if default is not None else ()
            external[node] = sub.create_node("placeholder", name, args, name=name)
        return external[node]

    if fqn:
        for index, value in enumerate(record["args"]):
            if isinstance(value, str) and value in name_to_node:
                placeholder_for(name_to_node[value], f"arg_{index}")
        for key, value in record["kwargs"].items():
            if isinstance(value, str) and value in name_to_node:
                placeholder_for(name_to_node[value], key)
    else:
        for node in graph.placeholders:
            default = node.args[0] if node.args else None
            placeholder_for(node, node.name, default)

    def child_call_for(owner: str, requested: Any) -> Any:
        """Emit the submodule call site that produced ``requested``.

        A module invoked several times yields one call site per recorded
        invocation; sites are keyed by the invocation record that produced
        the requested value.
        """

        requested_name = requested.name if hasattr(requested, "name") else requested
        cache_key = (owner, requested_name)
        if cache_key in child_results:
            return child_results[cache_key]
        records = calls_by_fqn[owner]
        record_index = 0
        for index, child_record in enumerate(records):
            if requested_name in child_record["result"]:
                record_index = index
                break
        child_record = records[record_index]
        args = tuple(
            _map_node(name_to_node[value]) if isinstance(value, str) else value
            for value in child_record["args"]
        )
        target = owner[len(fqn) + 1:] if fqn else owner
        call_node = sub.create_node("call_module", target, args, {})
        if len(child_record["result"]) > 1:
            results = tuple(
                sub.create_node("call_function", _operator.getitem, (call_node, index))
                for index in range(len(child_record["result"]))
            )
        elif len(child_record["result"]) == 1:
            results = (call_node,)
        else:
            results = ()
        for name, mapped in zip(child_record["result"], results):
            node = name_to_node.get(name)
            if node is not None:
                val_map[node] = mapped
        child_results[cache_key] = results
        return results

    def _map_node(node: Any) -> Any:
        if node in val_map:
            return val_map[node]
        if node in external:
            return external[node]
        owner = _node_owner(node)
        descendant = (not fqn) or owner.startswith(f"{fqn}.")
        if owner != fqn and descendant and owner in calls_by_fqn:
            child_call_for(owner, node)
            if node in val_map:
                return val_map[node]
        raise NotImplementedError(
            f"node {node.name!r} (owned by {owner!r}) crosses the boundary of "
            f"module {fqn!r} without passing through its call signature"
        )

    for node in graph.nodes:
        if node.op in {"output", "placeholder"}:
            continue
        if _node_owner(node) != fqn:
            continue
        target = node.target
        if node.op == "get_attr" and fqn:
            target = str(node.target)[len(fqn) + 1:]
        copied = sub.node_copy(node, _map_node)
        if node.op == "get_attr":
            copied.target = target
        val_map[node] = copied

    def finish(value: Any) -> Any:
        if isinstance(value, _NODE_TYPES):
            return _map_node(value)
        if isinstance(value, str) and value in name_to_node:
            return _map_node(name_to_node[value])
        if isinstance(value, (tuple, list, dict)):
            return _map_value_tree(value, _map_node)
        return value

    sub.output(finish(output_value))

    frame_root = tp.nn.Module()
    prefix = f"{fqn}." if fqn else ""
    for node in frame_nodes:
        if node.op != "get_attr":
            continue
        target = str(node.target)
        if not target.startswith(prefix):
            continue
        value = state_root
        for atom in target.split("."):
            value = getattr(value, atom)
        relative = target[len(prefix):]
        parent_name, _, leaf = relative.rpartition(".")
        parent: Any = frame_root
        if parent_name:
            for atom in parent_name.split("."):
                child = getattr(parent, atom, None)
                if child is None:
                    child = tp.nn.Module()
                    setattr(parent, atom, child)
                parent = child
        setattr(parent, leaf, value)

    # Graph construction needs every direct call target to exist on the
    # frame root.  The hierarchy is attached after all frame graphs compile,
    # so temporary module objects reserve those names during construction.
    for child_fqn in calls_by_fqn:
        if not child_fqn:
            continue
        parent_name, _, leaf = child_fqn.rpartition(".")
        if parent_name != fqn:
            continue
        if getattr(frame_root, leaf, None) is None:
            setattr(frame_root, leaf, tp.nn.Module())

    signature = user_signature if not fqn else None
    graph_module = GraphModule(frame_root, sub, signature)
    return graph_module if not fqn else _FrameModule(graph_module)


def _flat_output_names(graph: Any) -> list[str]:
    leaves: list[str] = []
    stack = [graph.output_node.args[0]]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif hasattr(item, "name"):
            leaves.append(item.name)
        else:
            leaves.append(str(item))
    return leaves


def _rebuild_hierarchy(program: ExportedProgram) -> Any:
    """Reconstruct the module hierarchy from recorded call boundaries."""

    import tensorplay as tp
    from ..graph import GraphModule

    records = program.graph_module.meta.get("module_calls") or []
    if not records:
        return None
    base = program.module()
    graph = base.graph
    mutation_count = int(program.graph_module.meta.get("num_mutations", 0) or 0)
    if mutation_count > 0:
        graph = copy.deepcopy(graph)
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
        user_spec = program.graph_module.meta.get("out_spec")
        if user_spec is not None:
            user_value = _unflatten_nodes(leaves[mutation_count:], user_spec)
        else:
            rest = leaves[mutation_count:]
            user_value = rest[0] if len(rest) == 1 else tuple(rest)
        graph.output(user_value)

    calls_by_fqn: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        calls_by_fqn.setdefault(record["fqn"], []).append(record)
    for fqn, group in calls_by_fqn.items():
        first = group[0]
        first_args = [repr(value) for value in first["args"]]
        for other in group[1:]:
            if [repr(value) for value in other["args"]] != first_args:
                raise NotImplementedError(
                    f"module {fqn!r} is invoked with differing argument sets; "
                    f"hierarchy reconstruction supports a single argument "
                    f"wiring per module"
                )

    name_to_node = {node.name: node for node in graph.nodes}
    frames: dict[str, list[Any]] = {}
    for node in graph.nodes:
        if node.op in {"output", "placeholder"}:
            continue
        frames.setdefault(_node_owner(node), []).append(node)

    user_signature = program.graph_module.meta.get("user_signature")
    built: dict[str, Any] = {}
    for fqn in sorted(calls_by_fqn, key=lambda f: (f.count("."), f), reverse=True):
        record = calls_by_fqn[fqn][0]
        results: Any = record["result"]
        if len(results) == 1:
            results = results[0]
        built[fqn] = _frame_module_for(
            fqn,
            frames.get(fqn, []),
            calls_by_fqn[fqn],
            calls_by_fqn,
            graph,
            name_to_node,
            base.root,
            user_signature,
            results,
        )

    root_record = {
        "args": [node.name for node in graph.placeholders],
        "kwargs": {},
        "result": [],
    }
    root_graph_module = _frame_module_for(
        "",
        frames.get("", []),
        [root_record],
        calls_by_fqn,
        graph,
        name_to_node,
        base.root,
        user_signature,
        graph.output_node.args[0],
    )

    for fqn in sorted(built, key=lambda f: (f.count("."), f)):
        parent_name, _, leaf = fqn.rpartition(".")
        child = built[fqn]
        if parent_name:
            owner = built.get(parent_name)
            if owner is None:
                continue
            parent = owner.root
            setattr(parent, leaf, child)
            setattr(owner.graph_module, leaf, child)
        else:
            parent = root_graph_module.root
            setattr(parent, leaf, child)
            setattr(root_graph_module, leaf, child)
    return root_graph_module


class UnflattenedModule:
    """Executable module view retaining the captured root module hierarchy."""

    def __init__(
        self,
        export_module: ExportedProgram,
        flat_args_adapter: FlatArgsAdapter | None = None,
    ) -> None:
        if not isinstance(export_module, ExportedProgram):
            raise TypeError("unflatten expects an ExportedProgram")
        self.exported_program = export_module
        self.graph_signature = copy.deepcopy(export_module.graph_signature)
        self.module_call_graph = copy.deepcopy(export_module.module_call_graph)
        self.range_constraints = copy.deepcopy(export_module.range_constraints)
        self.flat_args_adapter = flat_args_adapter
        rebuilt = None
        if flat_args_adapter is None:
            rebuilt = _rebuild_hierarchy(export_module)
        if rebuilt is not None:
            # hierarchy view: submodule calls are preserved as call_module
            # nodes and attribute access reaches the reconstructed modules
            self.graph_module = rebuilt
        else:
            # flat view: lifted state folded back into module attributes
            self.graph_module = export_module.module()
        self.graph = self.graph_module.graph
        self.root = self.graph_module.root
        self._hierarchical = rebuilt is not None

    def __getattr__(self, name: str) -> Any:
        if name in {"root", "graph_module", "exported_program"}:
            raise AttributeError(name)
        root = self.__dict__.get("root")
        if root is not None:
            return getattr(root, name)
        raise AttributeError(name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._hierarchical:
            return self.graph_module(*args, **kwargs)
        if self.flat_args_adapter is None:
            return self.exported_program(*args, **kwargs)
        flat, input_spec = tree_flatten((args, kwargs))
        target_spec = getattr(self.flat_args_adapter, "target_spec", input_spec)
        adapted = self.flat_args_adapter.adapt(target_spec, input_spec, list(flat), obj=self)
        values = tree_unflatten(adapted, target_spec)
        if not isinstance(values, tuple) or len(values) != 2:
            raise ValueError("argument adapter must produce an (args, kwargs) tree")
        call_args, call_kwargs = values
        return self.exported_program(*call_args, **call_kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def recompile(self) -> Any:
        return self.graph_module.recompile()

    def print_readable(self, print_output: bool = False) -> str:
        if self._hierarchical:
            sections = []
            for name, module in sorted(self.root.__dict__.items()):
                if hasattr(module, "print_readable"):
                    sections.append(module.print_readable())
            text = self.graph_module.print_readable()
            if sections:
                text = "\n".join([text, *sections])
        else:
            text = self.exported_program.print_readable()
        if print_output:
            print(text)
        return text

    def __repr__(self) -> str:
        return f"UnflattenedModule({self.graph_module!r})"


def unflatten(
    module: ExportedProgram,
    flat_args_adapter: FlatArgsAdapter | None = None,
    preserve_ops: Any = (),
) -> UnflattenedModule:
    """Build an executable module view from an exported program.

    When the capture recorded module call boundaries, the view reconstructs
    the original module hierarchy (attribute access and submodule calls work
    as in the source model).  Otherwise it falls back to the flat view.
    """

    del preserve_ops
    return UnflattenedModule(module, flat_args_adapter)
