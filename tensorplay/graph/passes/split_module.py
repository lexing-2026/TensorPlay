"""Partition a graph module into independently callable child modules."""

from __future__ import annotations

import operator
import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .._utils import _iter_nodes, _map_arg
from ..graph import Graph
from ..graph_module import GraphModule
from ..node import Node
from .split_utils import HolderModule, getattr_recursive

__all__ = ["Partition", "split_module", "split_module_simple"]


@dataclass
class Partition:
    """Construction state for one partition."""

    name: str
    submod_name: str = ""
    node_names: list[str] = field(default_factory=list)
    inputs: OrderedDict[str, None] = field(default_factory=OrderedDict)
    outputs: OrderedDict[str, None] = field(default_factory=OrderedDict)
    dependencies: OrderedDict[str, None] = field(default_factory=OrderedDict)
    dependents: OrderedDict[str, None] = field(default_factory=OrderedDict)
    graph: Graph = field(default_factory=Graph)
    environment: dict[Node, Node] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    call_inputs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.submod_name:
            self.submod_name = f"submod_{self.name}"

    def __repr__(self) -> str:
        return (
            f"name: {self.name},\n"
            f" nodes: {self.node_names},\n"
            f" inputs: {dict(self.inputs)},\n"
            f" outputs: {dict(self.outputs)},\n"
            f" partitions depended on: {dict(self.dependencies)},\n"
            f" partition dependents: {dict(self.dependents)}"
        )


def _get_attr_from_qualname(mod: Any, qualname: str) -> Any:
    value = getattr_recursive(mod, qualname)
    if value is None:
        raise AttributeError(f"graph attribute {qualname!r} was not found")
    return value


def _partition_name(value: Any, affix: str | None) -> str:
    name = str(value)
    return f"{affix}_{name}" if affix else name


def _safe_attr_name(target: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]", "_", target)
    return value if value and not value[0].isdigit() else "attribute_" + value


def _stable_partition_order(partitions: dict[str, Partition]) -> list[str]:
    indegree = {name: len(partition.dependencies) for name, partition in partitions.items()}
    ready = deque(name for name, count in indegree.items() if count == 0)
    result: list[str] = []
    while ready:
        name = ready.popleft()
        result.append(name)
        for dependent in partitions[name].dependents:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if len(result) != len(partitions):
        raise RuntimeError("cycle exists between graph partitions")
    return result


def _copy_value(value: Any, mapping: dict[Node, Node]) -> Any:
    if isinstance(value, Node):
        return mapping[value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item, mapping) for item in value)
    if isinstance(value, list):
        return [_copy_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _copy_value(item, mapping) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(
            _copy_value(value.start, mapping),
            _copy_value(value.stop, mapping),
            _copy_value(value.step, mapping),
        )
    return value


def _copy_metadata(source: Node, target: Node) -> None:
    target.meta = dict(source.meta)
    target.type = source.type
    target.tag = source.tag
    target.size_bytes = source.size_bytes


def split_module(
    m: GraphModule,
    root_m: Any,
    split_callback: Callable[[Node], int | str],
    qualname_map: dict[str, str] | None = None,
    keep_original_order: bool | None = False,
    keep_original_node_name: bool | None = False,
    keep_original_input_name: bool = True,
    *,
    partition_affix: str | None = None,
    tuple_return: bool = False,
) -> GraphModule:
    """Create child graph modules according to ``split_callback``.

    The callback is evaluated for every callable node.  Dependencies between
    partitions are inferred from use-def edges, so callbacks may assign IDs in
    any order as long as the resulting dependency graph is acyclic.
    """

    if not isinstance(m, GraphModule):
        raise TypeError("m must be a GraphModule")
    partitions: OrderedDict[str, Partition] = OrderedDict()
    original_nodes = {node.name: node for node in m.graph.nodes}
    node_partition: dict[Node, str] = {}
    for node in m.graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        name = _partition_name(split_callback(node), partition_affix)
        partition = partitions.get(name)
        if partition is None:
            partition = Partition(name)
            if partition_affix:
                partition.submod_name = f"submod_{partition_affix}_{name.rsplit('_', 1)[-1]}"
            partitions[name] = partition
        partition.node_names.append(node.name)
        node_partition[node] = name

    def record_use(def_node: Node, use_node: Node | None) -> None:
        defined = node_partition.get(def_node)
        used = node_partition.get(use_node) if use_node is not None else None
        if defined == used:
            return
        if defined is not None:
            partitions[defined].outputs.setdefault(def_node.name)
            if used is not None:
                partitions[defined].dependents.setdefault(used)
        if used is not None:
            partitions[used].inputs.setdefault(def_node.name)
            if defined is not None:
                partitions[used].dependencies.setdefault(defined)

    for node in m.graph.nodes:
        if node.op == "output":
            for value in _iter_nodes(node.args):
                record_use(value, None)
            continue
        if node.op in {"placeholder", "get_attr"}:
            continue
        for value in _iter_nodes(node.args):
            record_use(value, node)
        for value in _iter_nodes(node.kwargs):
            record_use(value, node)

    partition_order = _stable_partition_order(partitions)
    base_graph = Graph()
    base_env: dict[Node, Node] = {}
    base_root = HolderModule()
    for node in m.graph.nodes:
        if node.op != "placeholder":
            continue
        copied = base_graph.placeholder(node.name, node.meta.get("default")) if "default" in node.meta else base_graph.placeholder(node.name)
        _copy_metadata(node, copied)
        base_env[node] = copied

    construction_order = (
        list(partitions) if keep_original_order else partition_order
    )
    for partition_name in construction_order:
        partition = partitions[partition_name]
        input_nodes = [original_nodes[name] for name in partition.inputs]
        counter = 0
        for input_node in input_nodes:
            if input_node.op == "get_attr":
                attr = _get_attr_from_qualname(root_m, input_node.target)
                target = _safe_attr_name(input_node.target)
                copied = partition.graph.get_attr(target)
                partition.targets[target] = attr
            else:
                name = input_node.name if keep_original_input_name else f"arg_{counter}"
                copied = partition.graph.placeholder(name)
                counter += 1
            _copy_metadata(input_node, copied)
            partition.environment[input_node] = copied
            if copied.op == "placeholder":
                partition.inputs[input_node.name] = None
                partition.call_inputs.append(input_node.name)

        for node_name in partition.node_names:
            node = original_nodes[node_name]
            args = _copy_value(node.args, partition.environment)
            kwargs = _copy_value(node.kwargs, partition.environment)
            target = node.target
            if node.op in {"call_module", "get_attr"} and isinstance(target, str):
                target_value = _get_attr_from_qualname(root_m, target)
                target = _safe_attr_name(target)
                partition.targets[target] = target_value
                if qualname_map is not None:
                    qualname_map[f"{partition.submod_name}.{target}"] = node.target
            copied = partition.graph.create_node(
                node.op,
                target,
                args,
                kwargs,
                name=node.name if keep_original_node_name else None,
            )
            _copy_metadata(node, copied)
            partition.environment[node] = copied

        output_nodes = [original_nodes[name] for name in partition.outputs]
        output_values = tuple(partition.environment[node] for node in output_nodes)
        partition.graph.output(
            output_values if tuple_return or len(output_values) != 1 else output_values[0]
        )
        partition.graph.lint()
        child_root = HolderModule(partition.targets)
        child = GraphModule(child_root, partition.graph, None)
        child.meta["partition_name"] = partition.name
        child.meta["partition_targets"] = dict(partition.targets)
        setattr(base_root, partition.submod_name, child)
        for input_node in input_nodes:
            if input_node.op != "get_attr" or input_node in base_env:
                continue
            target = _safe_attr_name(input_node.target)
            value = _get_attr_from_qualname(root_m, input_node.target)
            setattr(base_root, target, value)
            base_env[input_node] = base_graph.get_attr(target)
        missing = [
            name for name in partition.call_inputs if original_nodes[name] not in base_env
        ]
        if missing:
            raise RuntimeError(
                "partition order does not define inputs before use: "
                + ", ".join(missing)
            )
        call_args = tuple(
            base_env[original_nodes[name]] for name in partition.call_inputs
        )
        call = base_graph.call_module(partition.submod_name, call_args)
        if len(output_nodes) == 1 and not tuple_return:
            base_env[output_nodes[0]] = call
        else:
            for index, output_node in enumerate(output_nodes):
                base_env[output_node] = base_graph.call_function(
                    operator.getitem, (call, index)
                )

    def map_base(value: Any) -> Any:
        if isinstance(value, Node):
            if value.op == "get_attr" and value not in base_env:
                target = _safe_attr_name(value.target)
                copied = base_graph.get_attr(target)
                _copy_metadata(value, copied)
                base_env[value] = copied
            return base_env[value]
        return _map_arg(value, map_base)

    output = m.graph.output_node
    base_graph.output(map_base(output.args[0]))
    base_graph.lint()
    result = GraphModule(base_root, base_graph, m.signature)
    result.meta["partition_names"] = {partition.submod_name for partition in partitions.values()}
    if not keep_original_order:
        result.meta["partition_order"] = tuple(partition_order)
    result.recompile()
    return result


def split_module_simple(
    m: GraphModule,
    node_to_partition: dict[Node, int],
    *,
    partition_affix: str | None = None,
) -> GraphModule:
    """Split an inference graph from a precomputed node-to-partition map."""

    return split_module(
        m,
        m.root,
        lambda node: node_to_partition[node],
        partition_affix=partition_affix,
    )
