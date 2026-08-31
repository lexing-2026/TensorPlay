from __future__ import annotations

import operator
import re
from collections.abc import Callable
from typing import Any

from ..graph import Graph
from ..graph_module import GraphModule
from ..node import Node
from ..symbolic_trace import symbolic_trace

__all__ = [
    "FoldedGraphModule",
    "_inline_module",
    "get_unique_attr_name_in_module",
    "split_const_subgraphs",
]


class FoldedGraphModule(GraphModule):
    """Graph module that evaluates a constant region on first invocation."""

    def __init__(
        self,
        root: Any,
        graph: Graph,
        const_subgraph: Graph | None = None,
        const_folded_attrs_name: str | None = None,
        device_for_folded_attrs: Any = "cpu",
    ) -> None:
        super().__init__(root, graph)
        self.const_subgraph_module = (
            None if const_subgraph is None else GraphModule(root, const_subgraph)
        )
        self.has_folding_been_run = False
        self.const_folded_attrs_name = const_folded_attrs_name
        self.device_for_folded_attrs = device_for_folded_attrs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.has_folding_been_run:
            self.run_folding()
        return super().__call__(*args, **kwargs)

    def run_folding(self) -> None:
        if self.const_subgraph_module is None or self.const_folded_attrs_name is None:
            self.has_folding_been_run = True
            return
        if self.has_folding_been_run:
            raise RuntimeError("constant folding has already run")
        values = self.const_subgraph_module()
        setattr(self.root, self.const_folded_attrs_name, values)
        self.has_folding_been_run = True


def _get_attr(root: Any, target: str) -> Any:
    value = root
    for part in target.split("."):
        value = getattr(value, part)
    return value


def _inline_module(
    module: GraphModule,
    inline_mod_name: str,
    run_dce: bool = True,
    is_impure_node: Callable[[Node], bool] | None = None,
) -> dict[Node, Node]:
    """Inline one child graph module at its call site."""

    inline_mod = module.get_submodule(inline_mod_name)
    if not isinstance(inline_mod, GraphModule):
        raise TypeError(f"{inline_mod_name!r} is not a GraphModule")
    call_node = next(
        (
            node
            for node in module.graph.nodes
            if node.op == "call_module" and node.target == inline_mod_name
        ),
        None,
    )
    if call_node is None:
        raise ValueError(f"no call site for {inline_mod_name!r}")
    mapping: dict[Node, Node] = {}
    placeholders = [node for node in inline_mod.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(call_node.args):
        raise ValueError("child graph inputs do not match its call site")
    mapping.update(zip(placeholders, call_node.args))
    for source in inline_mod.graph.nodes:
        if source.op in {"placeholder", "output"}:
            continue
        with module.graph.inserting_before(call_node):
            copied = module.graph.node_copy(source, lambda value: mapping[value])
        mapping[source] = copied
    output = next(node for node in inline_mod.graph.nodes if node.op == "output")
    replacement = _map_value(output.args[0], mapping)
    call_node.replace_all_uses_with(replacement)
    module.graph.erase_node(call_node)
    if run_dce:
        module.graph.eliminate_dead_code()
    module.delete_submodule(inline_mod_name)
    return mapping


def _map_value(value: Any, mapping: dict[Node, Node]) -> Any:
    if isinstance(value, Node):
        return mapping[value]
    if isinstance(value, tuple):
        return tuple(_map_value(item, mapping) for item in value)
    if isinstance(value, list):
        return [_map_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _map_value(item, mapping) for key, item in value.items()}
    return value


def get_unique_attr_name_in_module(module: GraphModule, name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not name:
        name = "constant"
    if name[0].isdigit():
        name = "_" + name
    candidate = name
    index = 1
    while hasattr(module.root, candidate) or hasattr(module, candidate):
        candidate = f"{name}_{index}"
        index += 1
    return candidate


def split_const_subgraphs(
    module: Any,
    skip_folding_node_fn: Callable[[Node], bool] | None = None,
    device_for_folded_attrs: Any = "cpu",
    is_impure_node: Callable[[Node], bool] | None = None,
) -> FoldedGraphModule:
    """Extract graph nodes whose inputs are all module constants."""

    del device_for_folded_attrs
    graph_module = module if isinstance(module, GraphModule) else symbolic_trace(module)
    constants: set[Node] = set()
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if skip_folding_node_fn is not None and skip_folding_node_fn(node):
            continue
        if is_impure_node is not None and is_impure_node(node):
            continue
        if node.is_impure():
            continue
        if node.op == "get_attr":
            try:
                value = _get_attr(graph_module.root, str(node.target))
            except AttributeError:
                continue
            if callable(value) and not hasattr(value, "shape"):
                continue
            constants.add(node)
        elif set(node.all_input_nodes).issubset(constants):
            constants.add(node)
    folded_nodes = [node for node in graph_module.graph.nodes if node in constants and node.op != "get_attr"]
    if not folded_nodes:
        return FoldedGraphModule(graph_module.root, graph_module.graph)

    const_graph = Graph()
    copied: dict[Node, Node] = {}
    for source in graph_module.graph.nodes:
        if source not in constants:
            continue
        copied[source] = const_graph.node_copy(source, lambda value: copied[value])
    const_value = tuple(copied[node] for node in folded_nodes)
    const_graph.output(const_value[0] if len(const_value) == 1 else const_value)
    const_graph.lint()

    attr_name = get_unique_attr_name_in_module(graph_module, "_constant_values")
    anchor = folded_nodes[0]
    replacements: dict[Node, Node] = {}
    with graph_module.graph.inserting_before(anchor):
        attr_node = graph_module.graph.get_attr(attr_name)
        if len(folded_nodes) == 1:
            replacements[folded_nodes[0]] = attr_node
        else:
            for index, node in enumerate(folded_nodes):
                replacements[node] = graph_module.graph.call_function(
                    operator.getitem, (attr_node, index)
                )
    for source in folded_nodes:
        source.replace_all_uses_with(replacements[source])
        graph_module.graph.erase_node(source)
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    return FoldedGraphModule(
        graph_module.root,
        graph_module.graph,
        const_graph,
        attr_name,
        "cpu",
    )

