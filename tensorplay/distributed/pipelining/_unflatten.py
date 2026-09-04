"""Graph outlining helpers for pipeline stages."""

from __future__ import annotations

import copy
import operator
import re
from collections import OrderedDict
from typing import Any

from tensorplay.graph.graph import Graph
from tensorplay.graph.graph_module import GraphModule
from tensorplay.graph.node import Node, map_arg
from tensorplay.nn.modules.module import Module

__all__ = ["_outline_submodules"]


def _module_stack_path(node: Node) -> str:
    stack = node.meta.get("nn_module_stack")
    if isinstance(stack, dict):
        values = list(stack.values())
        if not values:
            return ""
        value = values[-1]
        if isinstance(value, (tuple, list)):
            return str(value[0]) if value else ""
        return str(value)
    if isinstance(stack, (tuple, list)):
        return str(stack[-1]) if stack else ""
    return str(stack) if isinstance(stack, str) else ""


def _top_level_module_path(node: Node) -> str:
    path = _module_stack_path(node)
    return path.split(".", 1)[0] if path else ""


def _copy_node_metadata(source: Node, target: Node) -> None:
    target.meta = copy.copy(source.meta)
    target.tag = source.tag
    target.size_bytes = source.size_bytes


def _resolve_attr(root: Any, target: str) -> Any:
    value = root
    for part in target.split("."):
        value = getattr(value, part)
    return value


def _attach_reference(root: Module, source_root: Any, target: str) -> None:
    if source_root is None or not target:
        return
    first = target.split(".", 1)[0]
    try:
        value = getattr(source_root, first)
    except AttributeError:
        return
    if not hasattr(root, first):
        setattr(root, first, value)


def _safe_name(path: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z_]", "_", path) or "outlined"
    if base[0].isdigit():
        base = f"outlined_{base}"
    name = base
    index = 0
    while name in used:
        name = f"{base}_{index}"
        index += 1
    used.add(name)
    return name


def _outline_submodules(orig_graph: Graph) -> GraphModule:
    if not isinstance(orig_graph, Graph):
        raise TypeError(f"expected Graph, got {type(orig_graph).__name__}")

    owner_module = getattr(orig_graph, "owning_module", None)
    source_root = getattr(owner_module, "root", None)
    source_signature = getattr(owner_module, "signature", None)
    owners: OrderedDict[str, list[Node]] = OrderedDict()
    for node in orig_graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        owner = _top_level_module_path(node)
        if owner:
            owners.setdefault(owner, []).append(node)

    if not owners:
        graph = Graph()
        root = source_root if isinstance(source_root, Module) else Module()
        node_map: dict[Node, Node] = {}
        for node in orig_graph.nodes:
            if node.op == "output":
                continue
            copied = graph.node_copy(node, lambda value: node_map[value])
            _copy_node_metadata(node, copied)
            node_map[node] = copied
        output = orig_graph.output_node
        graph.output(map_arg(output.args[0], lambda value: node_map[value]))
        result = GraphModule(root, graph, source_signature)
        result.meta = copy.copy(getattr(owner_module, "meta", {}))
        return result

    root_graph = Graph()
    root = Module()
    root_map: dict[Node, Node] = {}
    for node in orig_graph.nodes:
        if node.op != "placeholder":
            continue
        copied = root_graph.placeholder(
            node.name,
            node.type,
            node.args[0] if node.args else None,
        )
        _copy_node_metadata(node, copied)
        root_map[node] = copied

    owner_to_call_outputs: dict[str, tuple[Node, ...]] = {}
    used_names: set[str] = set()
    owner_by_node = {
        node: _top_level_module_path(node)
        for nodes in owners.values()
        for node in nodes
    }

    def map_root_value(value: Any) -> Any:
        if isinstance(value, Node):
            if value.op == "get_attr" and value not in root_map:
                _attach_reference(root, source_root, str(value.target))
                copied_attr = root_graph.get_attr(str(value.target), value.type)
                _copy_node_metadata(value, copied_attr)
                root_map[value] = copied_attr
            if value not in root_map:
                raise RuntimeError(f"graph value {value.name!r} is not available")
            return root_map[value]
            return map_arg(value, map_root_value)

    def ensure_root_node(value: Any) -> Any:
        if not isinstance(value, Node):
            return map_arg(value, ensure_root_node)
        if value in root_map:
            return root_map[value]
        if value.op == "get_attr":
            return map_root_value(value)
        if value.op == "placeholder":
            raise RuntimeError(f"graph input {value.name!r} is not available")
        if value in owner_by_node:
            raise RuntimeError(f"graph stage input {value.name!r} is not available")
        copied = root_graph.create_node(
            value.op,
            value.target,
            map_arg(value.args, ensure_root_node),
            map_arg(value.kwargs, ensure_root_node),
            value.name,
            value.type,
        )
        _copy_node_metadata(value, copied)
        root_map[value] = copied
        return copied

    for owner, owner_nodes in owners.items():
        child_graph = Graph()
        child_root = Module()
        child_map: dict[Node, Node] = {}
        child_inputs: OrderedDict[Node, Node] = OrderedDict()

        def map_child_value(value: Any) -> Any:
            if not isinstance(value, Node):
                return map_arg(value, map_child_value)
            if value in child_map:
                return child_map[value]
            if value.op == "get_attr":
                _attach_reference(child_root, source_root, str(value.target))
                copied_attr = child_graph.get_attr(str(value.target), value.type)
                _copy_node_metadata(value, copied_attr)
                child_map[value] = copied_attr
                return copied_attr
            copied_input = child_inputs.get(value)
            if copied_input is None:
                copied_input = child_graph.placeholder(
                    f"arg_{len(child_inputs)}", value.type
                )
                _copy_node_metadata(value, copied_input)
                child_inputs[value] = copied_input
            return copied_input

        for node in owner_nodes:
            copied = child_graph.create_node(
                node.op,
                node.target,
                map_arg(node.args, map_child_value),
                map_arg(node.kwargs, map_child_value),
                node.name,
                node.type,
            )
            _copy_node_metadata(node, copied)
            child_map[node] = copied
            if node.op == "call_module":
                _attach_reference(child_root, source_root, str(node.target))

        output_nodes: list[Node] = []
        for node in owner_nodes:
            if any(
                user.op == "output" or owner_by_node.get(user, owner) != owner
                for user in node.users
            ):
                output_nodes.append(node)
        if not output_nodes:
            output_nodes.append(owner_nodes[-1])
        child_output_values = tuple(child_map[node] for node in output_nodes)
        child_graph.output(
            child_output_values[0]
            if len(child_output_values) == 1
            else child_output_values
        )
        child_graph.lint()
        child = GraphModule(child_root, child_graph)
        child.meta["outlined_path"] = owner
        child_name = _safe_name(owner, used_names)
        root.__dict__.setdefault("_modules", {})[child_name] = child

        call_args: list[Node] = []
        for original_input in child_inputs:
            try:
                call_args.append(ensure_root_node(original_input))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"graph input {original_input.name!r} is not available before {owner!r}"
                ) from exc
        call = root_graph.call_module(child_name, tuple(call_args))
        if len(output_nodes) == 1:
            _copy_node_metadata(output_nodes[0], call)
        call.meta["outlined_path"] = owner
        if len(output_nodes) == 1:
            root_map[output_nodes[0]] = call
        else:
            outputs: list[Node] = []
            for index, original_output in enumerate(output_nodes):
                item = root_graph.call_function(operator.getitem, (call, index))
                _copy_node_metadata(original_output, item)
                outputs.append(item)
                root_map[original_output] = item
        owner_to_call_outputs[owner] = tuple(root_map[node] for node in output_nodes)

    for node in orig_graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        if _top_level_module_path(node):
            continue
        ensure_root_node(node)

    output_node = orig_graph.output_node
    root_graph.output(map_root_value(output_node.args[0]))
    root_graph.lint()
    result = GraphModule(root, root_graph, source_signature)
    result.meta = copy.copy(getattr(owner_module, "meta", {}))
    result.meta["outlined_modules"] = tuple(owner_to_call_outputs)
    return result
