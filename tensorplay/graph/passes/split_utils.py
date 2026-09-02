"""Utilities for splitting a captured graph into tagged subgraphs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .._utils import _iter_nodes, _map_arg
from ..graph import Graph
from ..graph_module import GraphModule
from ..node import Node
from ...nn.modules.module import Module
from .tools_common import CALLABLE_NODE_OPS, is_node_output_tensor

if TYPE_CHECKING:
    from .splitter_base import Subgraph

__all__ = [
    "Component",
    "getattr_recursive",
    "setattr_recursive",
    "split_by_tags",
    "move_non_tensor_nodes_on_boundary",
]


class HolderModule(Module):
    """Attribute container used as the root of a stitched graph."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        super().__init__()
        for name, value in (values or {}).items():
            setattr(self, name, value)

    def add_module(self, name: str, module: Any) -> None:
        super().add_module(name, module)

    def named_children(self):
        yield from super().named_children()

    def children(self):
        yield from super().children()

    def named_modules(
        self,
        memo: set[int] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        yield from super().named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )


def getattr_recursive(obj: object, name: str) -> Any:
    value: Any = obj
    for layer in name.split("."):
        modules = getattr(value, "_modules", None)
        if isinstance(modules, dict) and layer in modules:
            value = modules[layer]
        elif hasattr(value, layer):
            value = getattr(value, layer)
        else:
            return None
    return value


def setattr_recursive(obj: object, attr: str, value: object) -> None:
    parts = attr.split(".")
    if len(parts) == 1:
        setattr(obj, attr, value)
        return
    holder = getattr_recursive(obj, parts[0])
    if holder is None:
        raise AttributeError(f"missing parent attribute {parts[0]!r}")
    setattr_recursive(holder, ".".join(parts[1:]), value)


@dataclass
class Component:
    """Container for one extracted graph and its boundary bookkeeping."""

    graph: Graph
    order: int
    name: str
    input_placeholders: list[Node] = field(default_factory=list)
    orig_inputs: list[Node] = field(default_factory=list)
    orig_outputs: list[Node] = field(default_factory=list)
    getattr_maps: dict[Node, Node] = field(default_factory=dict)
    constructor_args: list[str] = field(default_factory=list)
    gm: GraphModule | None = None
    nodes: list[Node] = field(default_factory=list)


def _flatten(value: Any) -> list[Node]:
    return list(_iter_nodes(value))


def _copy_meta(source: Node, target: Node) -> None:
    target.meta = copy.copy(source.meta)
    target.type = source.type
    target.tag = source.tag
    target.size_bytes = source.size_bytes


def _make_graph_module(
    cls: type[GraphModule], root: Any, graph: Graph, signature: Any
) -> GraphModule:
    try:
        return cls(root, graph, signature)
    except TypeError as exc:
        if signature is not None:
            raise
        try:
            return cls(root, graph)  # type: ignore[call-arg]
        except TypeError:
            raise exc


def split_by_tags(
    gm: GraphModule,
    tags: list[str],
    return_fqn_mapping: bool = False,
    return_tuple: bool = False,
    GraphModuleCls: type[GraphModule] = GraphModule,
):
    """Extract tagged callable nodes and stitch their modules in tag order.

    Structural nodes are created by the splitter: placeholders stay in the
    parent graph, attribute reads are copied into each child that needs them,
    and cross-component values become child placeholders.  Every component is
    emitted, including a component with no callable nodes, so the requested
    order remains observable and stable.
    """

    if not isinstance(gm, GraphModule):
        raise TypeError("split_by_tags expects a GraphModule")
    if not tags:
        raise ValueError("tags must contain at least one component")

    tag_to_component: dict[str, Component] = {}
    for order, tag in enumerate(tags):
        if tag in tag_to_component:
            raise ValueError(f"duplicate graph component tag: {tag!r}")
        tag_to_component[tag] = Component(Graph(), order, str(tag))
    all_components = list(tag_to_component.values())

    node_remapping: dict[Node, Node] = {}
    node_to_component: dict[Node, Component] = {}
    used_in_main: dict[Node, None] = {}
    main_graph = Graph()
    main_remapping: dict[Node, Node] = {}
    output_node: Node | None = None

    def flatten(value: Any) -> list[Node]:
        return _flatten(value)

    def copy_meta(source: Node, target: Node) -> None:
        _copy_meta(source, target)

    for node in gm.graph.nodes:
        if node.op == "output":
            if output_node is not None:
                raise RuntimeError("graph contains multiple output nodes")
            output_node = node
            continue
        if node.op == "placeholder":
            default = node.meta.get("default")
            if default is None and "default" not in node.meta:
                copied = main_graph.placeholder(node.name)
            else:
                copied = main_graph.placeholder(node.name, default)
            copy_meta(node, copied)
            main_remapping[node] = copied
            continue
        if node.op == "get_attr":
            continue

        tag = node.tag if node.tag is not None else node.meta.get("tag")
        if tag not in tag_to_component:
            raise AssertionError(
                f"callable node {node.name!r} has no requested component tag"
            )
        component = tag_to_component[tag]
        component.nodes.append(node)
        upstream = [
            node_to_component[input_node]
            for input_node in flatten(node.args) + flatten(node.kwargs)
            if input_node.op not in {"placeholder", "get_attr"}
        ]
        max_upstream_order = max(
            (item.order for item in upstream), default=component.order
        )
        if component.order < max_upstream_order:
            raise AssertionError(
                f"component {component.name!r} must follow its dependencies"
            )
        node_to_component[node] = component

        def remap_input(value: Node) -> Node:
            if value.op == "get_attr":
                if not isinstance(value.target, str):
                    raise TypeError("graph attribute targets must be strings")
                mapped = component.getattr_maps.get(value)
                if mapped is None:
                    mapped = component.graph.get_attr(value.target)
                    copy_meta(value, mapped)
                    component.getattr_maps[value] = mapped
                return mapped
            if value.op != "placeholder" and node_to_component[value] is component:
                return node_remapping[value]
            if value not in component.orig_inputs:
                component.orig_inputs.append(value)
                placeholder = component.graph.placeholder(value.name)
                copy_meta(value, placeholder)
                component.input_placeholders.append(placeholder)
                used_in_main[value] = None
            return component.input_placeholders[component.orig_inputs.index(value)]

        copied = component.graph.node_copy(node, remap_input)
        copy_meta(node, copied)
        node_remapping[node] = copied

    if output_node is None:
        raise RuntimeError("graph has no output node")

    for value in flatten(output_node.args[0]):
        if value.op == "get_attr":
            main_remapping[value] = main_graph.get_attr(value.name)
        elif value.op != "placeholder":
            used_in_main[value] = None

    for original in used_in_main:
        if original.op != "placeholder":
            component = node_to_component.get(original)
            if component is not None and original not in component.orig_outputs:
                component.orig_outputs.append(original)

    fqn_mapping: dict[str, str] = {}
    for component in all_components:
        outputs = tuple(node_remapping[node] for node in component.orig_outputs)
        component.graph.output(
            outputs if return_tuple or len(outputs) != 1 else outputs[0]
        )
        component.graph.lint()
        component.gm = _make_graph_module(
            GraphModuleCls,
            gm.root,
            component.graph,
            None,
        )
        component.gm.meta["component_name"] = component.name
        component.gm.meta["component_order"] = component.order
        for original in component.nodes if hasattr(component, "nodes") else ():
            if original.op in {"call_module", "get_attr"}:
                fqn_mapping[original.target] = f"{component.name}.{original.target}"
        args = tuple(main_remapping[node] for node in component.orig_inputs)
        main_node = main_graph.call_module(component.name, args)
        if len(outputs) == 1 and not return_tuple:
            main_remapping[component.orig_outputs[0]] = main_node
        else:
            for index, original in enumerate(component.orig_outputs):
                item = main_graph.call_function(operator.getitem, (main_node, index))
                main_remapping[original] = item

    direct_attr_nodes: dict[Node, Node] = {}
    for value in flatten(output_node.args[0]):
        if value.op != "get_attr":
            continue
        attr_node = main_remapping[value]
        direct_attr_nodes[value] = attr_node

    def remap_main(value: Any) -> Any:
        if isinstance(value, Node):
            if value not in main_remapping:
                raise RuntimeError(f"node {value.name!r} has no split mapping")
            return main_remapping[value]
        return _map_arg(value, remap_main)

    main_graph.output(remap_main(output_node.args[0]))
    main_root = HolderModule({
        component.name: component.gm for component in all_components
    })
    for original, attr_node in direct_attr_nodes.items():
        value = getattr_recursive(gm.root, original.target)
        if value is None:
            raise AttributeError(f"missing graph attribute {original.target!r}")
        setattr(main_root, attr_node.target, value)
        fqn_mapping[original.target] = attr_node.target
    main_graph.lint()
    result = _make_graph_module(GraphModuleCls, main_root, main_graph, gm.signature)
    if return_fqn_mapping:
        return result, fqn_mapping
    return result


def move_non_tensor_nodes_on_boundary(subgraphs: list["Subgraph"]) -> None:
    """Move movable scalar regions across accelerator subgraph boundaries."""

    node_to_subgraph: dict[Node, int] = {}
    for index, subgraph in enumerate(subgraphs):
        for node in subgraph.nodes:
            node_to_subgraph[node] = index

    def children(node: Node) -> list[Node]:
        return [
            user
            for user in node.users
            if user.op in CALLABLE_NODE_OPS and user in node_to_subgraph
        ]

    def parents(node: Node) -> list[Node]:
        return [
            value
            for value in _flatten(node.args) + _flatten(node.kwargs)
            if value.op in CALLABLE_NODE_OPS and value in node_to_subgraph
        ]

    def crosses(node: Node, current: int) -> bool:
        return any(node_to_subgraph[user] != current for user in children(node))

    def movable(
        node: Node, source: int, destination: int
    ) -> tuple[bool, set[Node]]:
        moving: set[Node] = set()
        visited: set[Node] = set()
        valid = True

        def visit(current: Node) -> None:
            nonlocal valid
            if current in visited or not valid:
                return
            visited.add(current)
            owner = node_to_subgraph.get(current)
            if owner is None or owner == destination:
                return
            if owner != source:
                valid = False
                return
            moving.add(current)
            for child in children(current):
                visit(child)

        visit(node)
        return valid, moving

    for source_index, source_subgraph in enumerate(subgraphs):
        if not getattr(source_subgraph, "is_acc", True):
            continue
        queue = [
            node
            for node in source_subgraph.nodes
            if not is_node_output_tensor(node) and crosses(node, source_index)
        ]
        processed: set[Node] = set()
        while queue:
            current = queue.pop(0)
            if current in processed:
                continue
            processed.add(current)
            if node_to_subgraph.get(current) != source_index:
                continue
            current_children = children(current)
            destinations = {
                node_to_subgraph[child]
                for child in current_children
                if node_to_subgraph[child] != source_index
            }
            if len(destinations) != 1:
                continue
            destination = destinations.pop()
            valid, moving = movable(current, source_index, destination)
            if not valid:
                continue
            for node in moving:
                source_subgraph.nodes.remove(node)
                subgraphs[destination].nodes.append(node)
                node_to_subgraph[node] = destination
                for parent in parents(node):
                    if parent in source_subgraph.nodes and not is_node_output_tensor(parent):
                        if crosses(parent, source_index):
                            queue.append(parent)
