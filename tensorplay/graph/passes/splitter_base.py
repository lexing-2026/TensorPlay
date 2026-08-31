"""Common accelerator/host graph splitting infrastructure."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, IO, Literal, NamedTuple

from .._utils import _iter_nodes
from ..graph_module import GraphModule
from ..node import Node
from .graph_manipulation import get_size_of_node
from .operator_support import OperatorSupportBase
from .shape_prop import ShapeProp
from .split_utils import move_non_tensor_nodes_on_boundary, split_by_tags
from .tools_common import (
    CALLABLE_NODE_OPS,
    GraphAccFusionsFinder,
    NodeList,
    NodeSet,
    Tensors,
    get_node_target,
    is_node_output_tensor,
)

__all__ = [
    "GraphNetAccNodesFinder",
    "GraphNetSplitterInternalError",
    "Subgraph",
    "SplitResult",
    "generate_inputs_for_submodules",
    "NodeEvent",
    "NodeEventTracker",
]

DEFAULT_MIN_ACC_MODULE_SIZE = 1
DEFAULT_SKIP_FUSION = False
DEFAULT_ALLOW_NON_TENSOR = False
TRACKER_DUMP_PATH = "_graph_net_tracker"
NODES_SUFFIX = "_nodes.txt"
ALL_SUFFIX = "_all.txt"
ENV_TRACKER_MODE = "GRAPH_NET_SPLITTER_TRACKER_MODE"
ENV_TRACKER_DUMP_PATH = "GRAPH_NET_SPLITTER_TRACKER_DUMP_PATH"
ENV_TRACKER_NODES = "GRAPH_NET_SPLITTER_TRACKER_TRACKED_NODES"
DUMP_PREFIX = os.environ.get(ENV_TRACKER_DUMP_PATH, TRACKER_DUMP_PATH)
TRACKER_MODE: Literal["0", "1", "2", "3"] = os.environ.get(
    ENV_TRACKER_MODE, "0"
)  # type: ignore[assignment]


class _SplitterSettingBase:
    def __init__(
        self,
        min_acc_module_size: int = DEFAULT_MIN_ACC_MODULE_SIZE,
        skip_fusion: bool = DEFAULT_SKIP_FUSION,
        allow_non_tensor: bool = DEFAULT_ALLOW_NON_TENSOR,
        max_acc_splits: int = -1,
        move_non_tensor_nodes_on_boundary: bool = False,
    ) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--min-acc-module-size", type=int)
        parser.add_argument("--max-acc-splits", type=int)
        parser.add_argument("--skip-fusion", action="store_true")
        parser.add_argument("--allow-non-tensor", action="store_true")
        parser.add_argument("--move-non-tensor-nodes-on-boundary", action="store_true")
        args, _ = parser.parse_known_args()
        self.min_acc_module_size = args.min_acc_module_size or min_acc_module_size
        self.skip_fusion = bool(args.skip_fusion or skip_fusion)
        self.allow_non_tensor = bool(args.allow_non_tensor or allow_non_tensor)
        self.max_acc_splits = max_acc_splits if args.max_acc_splits is None else args.max_acc_splits
        self.move_non_tensor_nodes_on_boundary = bool(
            args.move_non_tensor_nodes_on_boundary or move_non_tensor_nodes_on_boundary
        )


class NodeEvent:
    """One decision recorded while classifying a graph node."""

    def __init__(self, source: Node, desc: str, dep: Node | None = None) -> None:
        self.source = source
        self.desc = desc
        self.dep = dep

    def to_str(self) -> str:
        return f"{self.source.name}: {self.desc} {self.dep.name if self.dep else '#'}"


class NodeEventTracker:
    """Collect and optionally persist node classification events."""

    def __init__(self, tracker_mode: int, dump_prefix: str) -> None:
        self.tracker_mode = tracker_mode
        self.dump_prefix = dump_prefix
        self.events: list[NodeEvent] = []
        self.node_events: dict[str, list[int]] = {}
        self.writer: Callable[[str], object] = print

    def add(self, node: Node, desc: str, dep: Node | None = None) -> None:
        event = NodeEvent(node, desc, dep)
        self.events.append(event)
        self.node_events.setdefault(node.name, []).append(len(self.events) - 1)

    def print_node(
        self,
        node_name: str,
        recursive: bool = False,
        tab: str = "",
        writer: Callable[[str], object] | None = None,
    ) -> None:
        writer = writer or self.writer
        for index in self.node_events.get(node_name, ()):
            event = self.events[index]
            writer(tab + event.to_str())
            if recursive and event.dep is not None:
                self.print_node(event.dep.name, True, "| " + tab, writer)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            name: [self.events[index].to_str() for index in indexes]
            for name, indexes in self.node_events.items()
        }

    def print_all(self, writer: Callable[[str], object] | None = None) -> None:
        writer = writer or self.writer
        for name in self.node_events:
            writer(f"Node: {name}:")
            self.print_node(name, writer=writer)

    def dump(self) -> None:
        if self.tracker_mode >= 1:
            with open(self.dump_prefix + ALL_SUFFIX, "w") as stream:
                self.print_all(lambda text: stream.write(text + "\n"))
        if self.tracker_mode not in {2, 3}:
            return
        if self.tracker_mode == 2:
            names = os.environ.get(ENV_TRACKER_NODES, "").split(",")
        else:
            names = [name for name, events in self.node_events.items() if len(events) > 1]
        with open(self.dump_prefix + NODES_SUFFIX, "w") as stream:
            writer = lambda text: stream.write(text + "\n")
            for name in names:
                writer(f"===== Tracking node {name} =====")
                self.print_node(name, recursive=True, tab="|-", writer=writer)
                writer(f"===== End of tracking node {name} =====")


class GraphNetAccNodesFinder:
    """Find supported nodes while preventing non-tensor backend crossings."""

    def __init__(
        self,
        module: GraphModule,
        operator_support: OperatorSupportBase,
        allow_non_tensor: bool,
    ) -> None:
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor
        self.acc_nodes: NodeSet = set()
        self.tracker = NodeEventTracker(int(TRACKER_MODE), DUMP_PREFIX)

    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList) -> None:
        while cpu_worklist:
            node = cpu_worklist.pop(0)
            for user in node.users:
                if user not in self.acc_nodes:
                    continue
                self.acc_nodes.remove(user)
                self.tracker.add(user, "acc_del|user_of_new_cpu_node", node)
                if not is_node_output_tensor(user):
                    self.tracker.add(user, "new_cpu_node|non_tensor_output")
                    cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self) -> None:
        non_tensor_cpu_nodes: NodeList = []
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS or node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            self.tracker.add(node, "new_cpu_node|callable_non_tensor_input")
            non_tensor_cpu_nodes.append(node)
        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self) -> None:
        while True:
            new_cpu_nodes: NodeList = []
            for node in list(self.acc_nodes):
                if is_node_output_tensor(node):
                    continue
                if any(user not in self.acc_nodes for user in node.users):
                    new_cpu_nodes.append(node)
                    self.tracker.add(node, "acc_del|non_tensor_output_with_cpu_user")
            if not new_cpu_nodes:
                return
            for node in new_cpu_nodes:
                self.acc_nodes.remove(node)
            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)

    def __call__(self) -> NodeSet:
        named_modules = getattr(self.module.root, "named_modules", None)
        submodules = dict(named_modules()) if callable(named_modules) else {"": self.module.root}
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                self.tracker.add(node, "init_cpu|not_callable")
                continue
            if not self.operator_support.is_node_supported(submodules, node):
                self.tracker.add(node, "init_cpu|operator_support")
                continue
            self.tracker.add(node, "init_acc|callable_and_operator_supported")
            self.acc_nodes.add(node)
        if not self.allow_non_tensor:
            self.reduce_acc_nodes_non_tensor_input()
            self.reduce_acc_nodes_non_tensor_output()
        self.tracker.dump()
        return self.acc_nodes


class GraphNetSplitterInternalError(Exception):
    pass


@dataclass
class Subgraph:
    is_acc: bool
    nodes: NodeList
    device_ordinal: int | None = None


class SplitResult(NamedTuple):
    split_module: GraphModule
    submodule_inputs: dict[str, Any]
    non_acc_submodule_prefix: str


def generate_inputs_for_submodules(
    model: Any,
    inputs: Sequence[Any],
    target_submodules: Iterable[str],
    deepcopy: bool = False,
) -> dict[str, Any]:
    """Run a model and collect the positional inputs seen by named children."""

    target = set(target_submodules)
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("model must expose named_modules()")
    submodule_to_names = {module: name for name, module in named_modules()}
    handles: list[Any] = []
    results: dict[str, Any] = {}

    def pre_forward(module: Any, module_inputs: tuple[Any, ...]) -> None:
        value = copy.deepcopy(module_inputs) if deepcopy else module_inputs
        results[submodule_to_names[module]] = value

    for name, module in named_modules():
        if name in target and hasattr(module, "register_forward_pre_hook"):
            handles.append(module.register_forward_pre_hook(pre_forward))
    try:
        no_grad = getattr(__import__("tensorplay.autograd.grad_mode", fromlist=["no_grad"]), "no_grad", None)
        if no_grad is None:
            model(*inputs)
        else:
            with no_grad():
                model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return results


class _SplitterBase:
    """Build alternating accelerator and host regions from a graph."""

    PCIe_BW = 100 * 2**30

    def __init__(
        self,
        module: GraphModule,
        sample_input: Sequence[Any],
        operator_support: OperatorSupportBase,
        settings: _SplitterSettingBase,
        non_acc_submodule_name: str = "_run_on_cpu_",
        return_tuple: bool = False,
        nodes_finder: GraphNetAccNodesFinder | None = None,
    ) -> None:
        if not isinstance(module, GraphModule):
            raise AssertionError(f"expected GraphModule, got {type(module)}")
        self.module = module
        self.sample_input = sample_input
        ShapeProp(self.module).propagate(*sample_input)
        self.settings = settings
        self.operator_support = operator_support
        finder = nodes_finder or GraphNetAccNodesFinder(
            self.module, operator_support, settings.allow_non_tensor
        )
        self.acc_nodes = finder()
        if settings.skip_fusion:
            self.fusions: dict[Node, NodeSet] = {}
        else:
            self.fusions = GraphAccFusionsFinder(module, self.acc_nodes)()
            self._merge_overlapping_fusions()
        self.deps = self.find_deps()
        self.update_deps_for_fusions()
        self.non_acc_submodule_name = non_acc_submodule_name
        self._node_submodule_map: dict[str, str] = {}
        self._return_tuple = return_tuple
        self.tags: list[str] = []

    def get_node_submodule_map(self) -> dict[str, str]:
        return self._node_submodule_map

    def find_deps(self) -> dict[Node, NodeSet]:
        deps: dict[Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            for user in node.users:
                if user.op != "output":
                    deps[user].add(node)
        return deps

    def update_deps_for_fusions(self) -> None:
        processed: set[Node] = set()
        for node, fusion in self.fusions.items():
            if node in processed:
                continue
            shared = set().union(*(self.deps[item] for item in fusion)) - fusion
            for fused_node in fusion:
                self.deps[fused_node].update(shared)
                for user in fused_node.users:
                    if user not in fusion:
                        self.deps[user].add(node)
                processed.add(fused_node)

    def _merge_overlapping_fusions(self) -> None:
        if os.environ.get("_SPLITTER_MERGE_OVERLAPPING_FUSIONS", "0") != "1":
            return
        groups: dict[int, NodeSet] = {id(group): group for group in self.fusions.values()}
        parent = {key: key for key in groups}

        def find(key: int) -> int:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        owners: dict[Node, list[int]] = defaultdict(list)
        for group_id, group in groups.items():
            for node in group:
                owners[node].append(group_id)
        for group_ids in owners.values():
            root = find(group_ids[0])
            for other in group_ids[1:]:
                parent[find(other)] = root
        merged: dict[int, NodeSet] = defaultdict(set)
        for group_id, group in groups.items():
            merged[find(group_id)].update(group)
        self.fusions = {
            node: group for group in merged.values() for node in group
        }

    def _lower_model_to_backend(self, mod: GraphModule, inputs: Tensors) -> Any:
        return mod

    def _find_culprit(self, mod: GraphModule, inputs: Tensors) -> str:
        del mod, inputs
        return "unable to identify a failing node"

    def node_support_preview(self, dump_graph: bool = False) -> str:
        del dump_graph
        named_modules = getattr(self.module.root, "named_modules", None)
        submodules = dict(named_modules()) if callable(named_modules) else {"": self.module.root}
        supported: defaultdict[str, set[tuple[Any, ...]]] = defaultdict(set)
        unsupported: defaultdict[str, set[tuple[Any, ...]]] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            target = get_node_target(submodules, node)
            arg_dtypes = tuple(
                getattr(node.meta.get("tensor_meta"), "dtype", None)
                if isinstance(value, Node)
                else None
                for value in node.args
            )
            kwarg_dtypes = tuple(
                (key, getattr(value.meta.get("tensor_meta"), "dtype", None))
                for key, value in node.kwargs.items()
                if isinstance(value, Node)
            )
            (supported if self.operator_support.is_node_supported(submodules, node) else unsupported)[
                target
            ].add((arg_dtypes, kwarg_dtypes))
        lines = ["Supported node types in the model:"]
        for target, values in supported.items():
            lines.extend(f"{target}: {value}" for value in sorted(map(repr, values)))
        lines.append("Unsupported node types in the model:")
        for target, values in unsupported.items():
            lines.extend(f"{target}: {value}" for value in sorted(map(repr, values)))
        return "\n".join(lines)

    def find_reverse_deps(self, tag_id: int | None = None) -> dict[Node, NodeSet]:
        result: dict[Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue
                if tag_id is None or (
                    user.tag is not None
                    and int(str(user.tag).rsplit("_", 1)[-1]) < tag_id
                ):
                    result[node].add(user)
        return result

    def update_reverse_deps_for_fusions(self, deps: dict[Node, NodeSet]) -> None:
        processed: set[Node] = set()
        for node, fusion in self.fusions.items():
            if node in processed:
                continue
            new_dep = set().union(*(deps[item] for item in fusion)) - fusion
            for fused_node in fusion:
                deps[fused_node] = new_dep
                for arg in (*_iter_nodes(fused_node.args), *_iter_nodes(fused_node.kwargs)):
                    if arg not in fusion:
                        deps[arg].update(fusion)
                processed.add(fused_node)

    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        result: set[Node] = set()
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS or node.tag != tag:
                continue
            result.update(
                value
                for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs))
                if value.op in CALLABLE_NODE_OPS and value.tag != tag
            )
        return result

    def extend_acc_subgraph(self, tag: str) -> None:
        tag_id = int(tag.rsplit("_", 1)[-1])
        deps = self.find_reverse_deps(tag_id)
        self.update_reverse_deps_for_fusions(deps)
        parents = self.find_parent_nodes_of_subgraph(tag)
        visited: set[Node] = set()
        while parents:
            node = next(
                (candidate for candidate in parents if deps[candidate] <= visited and candidate in self.acc_nodes),
                None,
            )
            if node is None:
                break
            node.tag = tag
            parents.remove(node)
            visited.add(node)
            if node in self.fusions:
                parents.update(self.fusions[node] - visited)
            parents.update(
                value
                for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs))
                if value.op in CALLABLE_NODE_OPS and value not in visited
            )

    def starter_nodes(self) -> tuple[NodeSet, NodeSet]:
        cpu: set[Node] = set()
        acc: set[Node] = set()
        for node in self.module.graph.nodes:
            if node.op == "call_function" and not list(_iter_nodes(node.args)) and not list(_iter_nodes(node.kwargs)):
                (acc if node in self.acc_nodes else cpu).add(node)
            if node.op not in {"placeholder", "get_attr"}:
                continue
            for user in node.users:
                (acc if user in self.acc_nodes else cpu).add(user)
        return cpu, acc

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        current_cpu, current_acc = self.starter_nodes()
        visited: set[Node] = set()
        acc_subgraph = not any(len(self.deps[node]) == 0 for node in current_cpu)
        current_nodes: NodeList = []
        result: list[Subgraph] = []
        while current_cpu or current_acc:
            candidates = current_acc if acc_subgraph else current_cpu
            node = next((item for item in candidates if self.deps[item] <= visited), None)
            if node is None:
                if not current_nodes:
                    raise GraphNetSplitterInternalError("subgraph cannot be empty")
                result.append(Subgraph(acc_subgraph, current_nodes))
                acc_subgraph = not acc_subgraph
                current_nodes = []
                continue
            candidates.remove(node)
            visited.add(node)
            current_nodes.append(node)
            if node in self.fusions:
                target = current_acc if node in self.acc_nodes else current_cpu
                target.update(self.fusions[node] - visited)
            for user in node.users:
                if user.op in CALLABLE_NODE_OPS:
                    (current_acc if user in self.acc_nodes else current_cpu).add(user)
        if current_nodes:
            result.append(Subgraph(acc_subgraph, current_nodes))
        if not result:
            raise GraphNetSplitterInternalError("no subgraphs were created")
        return result

    def remove_small_acc_subgraphs(self, subgraphs: list[Subgraph]) -> list[Subgraph]:
        result: list[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc and len(subgraph.nodes) < self.settings.min_acc_module_size:
                if result:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    subgraph.is_acc = False
                    result.append(subgraph)
            elif not subgraph.is_acc and result and not result[-1].is_acc:
                result[-1].nodes.extend(subgraph.nodes)
            else:
                result.append(subgraph)
        return result

    def tag(self, subgraphs: list[Subgraph]) -> None:
        self.tags = []
        for index, subgraph in enumerate(subgraphs):
            tag = (
                f"_run_on_acc_{index}"
                if subgraph.is_acc
                else f"{self.non_acc_submodule_name}{index}"
            )
            self.tags.append(tag)
            for node in subgraph.nodes:
                if node.tag is not None:
                    raise GraphNetSplitterInternalError(f"node {node.name} was already tagged")
                node.tag = tag
                self._node_submodule_map[node.name] = tag

    def split(self, remove_tag: bool = False) -> GraphModule:
        result = split_by_tags(self.module, self.tags, return_tuple=self._return_tuple)
        if remove_tag:
            for node in self.module.graph.nodes:
                node.tag = None
        return result

    def __call__(self) -> GraphModule:
        subgraphs = self.put_nodes_into_subgraphs()
        if self.settings.move_non_tensor_nodes_on_boundary:
            move_non_tensor_nodes_on_boundary(subgraphs)
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        self.tag(subgraphs)
        return self.split()

    def generate_split_results(self) -> SplitResult:
        split_module = self()
        root = split_module.root
        children = (
            dict(root.named_children())
            if callable(getattr(root, "named_children", None))
            else {
                name: value
                for name, value in vars(root).items()
                if isinstance(value, GraphModule)
            }
        )
        names = list(children)
        if self.settings.max_acc_splits > 0 and len(names) > self.settings.max_acc_splits:
            raise ValueError("maximum accelerator split count was exceeded")
        inputs = generate_inputs_for_submodules(split_module, self.sample_input, names)
        return SplitResult(split_module, inputs, self.non_acc_submodule_name)
