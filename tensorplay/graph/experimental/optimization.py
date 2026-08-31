from __future__ import annotations

import copy
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, cast

import tensorplay as tp
from tensorplay import nn

from ..graph import Graph
from ..graph_module import GraphModule
from ..interpreter import Transformer
from ..node import Node
from ..symbolic_trace import symbolic_trace
from ..tracer import NodePathTracer, Tracer

__all__ = [
    "MklSubgraph",
    "UnionFind",
    "extract_subgraph",
    "fuse",
    "gen_mkl_autotuner",
    "matches_module_pattern",
    "modules_to_mkldnn",
    "optimize_for_inference",
    "remove_dropout",
    "replace_node_module",
    "reset_modules",
    "use_mkl_length",
]


def _parent_name(target: str) -> tuple[str, str]:
    parent, separator, name = target.rpartition(".")
    return (parent if separator else ""), name if separator else target


def matches_module_pattern(
    pattern: Iterable[type], node: Node, modules: dict[str, Any]
) -> bool:
    """Check whether a node and its direct producer match a module pattern."""

    pattern = tuple(pattern)
    if not node.args or len(pattern) != 2:
        return False
    producer = node.args[0]
    if not isinstance(producer, Node):
        return False
    for expected, candidate in zip(pattern, (producer, node)):
        if candidate.op != "call_module" or not isinstance(candidate.target, str):
            return False
        module = modules.get(candidate.target)
        if module is None or type(module) is not expected:
            return False
    return True


def replace_node_module(
    node: Node, modules: dict[str, Any], new_module: Any
) -> None:
    if not isinstance(node.target, str):
        raise TypeError(f"module target must be a string, got {type(node.target).__name__}")
    parent_name, name = _parent_name(node.target)
    parent = modules.get(parent_name)
    if parent is None:
        raise AttributeError(f"module path {parent_name!r} does not exist")
    modules[node.target] = new_module
    add_module = getattr(parent, "add_module", None)
    if callable(add_module):
        add_module(name, new_module)
    else:
        setattr(parent, name, new_module)


def _shape_scale(norm: Any) -> Any:
    shape = getattr(norm, "shape", None)
    if callable(shape):
        shape = shape()
    return tuple(int(item) for item in shape)


def _fuse_affine_normalization(layer: Any, norm: Any) -> Any:
    if getattr(norm, "training", True):
        raise ValueError("normalization fusion requires evaluation mode")
    if not getattr(norm, "track_running_stats", False):
        raise ValueError("normalization fusion requires running statistics")
    running_mean = getattr(norm, "running_mean", None)
    running_var = getattr(norm, "running_var", None)
    if running_mean is None or running_var is None:
        raise ValueError("normalization fusion requires initialized statistics")
    channels = int(getattr(layer, "out_channels", getattr(layer, "out_features", 0)))
    if channels <= 0:
        raise ValueError("layer has no output channel metadata")
    weight = getattr(norm, "weight", None)
    bias = getattr(norm, "bias", None)
    if weight is None:
        weight = tp.ones((channels,), dtype=running_var.dtype, device=running_var.device)
    if bias is None:
        bias = tp.zeros((channels,), dtype=running_var.dtype, device=running_var.device)
    scale = weight * tp.rsqrt(running_var + norm.eps)
    old_weight = layer.weight
    old_shape = _shape_scale(old_weight)
    reshape = (channels,) + (1,) * (len(old_shape) - 1)
    new_weight = old_weight * scale.reshape(reshape)
    old_bias = getattr(layer, "bias", None)
    if old_bias is None:
        old_bias = tp.zeros((channels,), dtype=new_weight.dtype, device=new_weight.device)
    new_bias = (old_bias - running_mean) * scale + bias
    from tensorplay.nn.parameter import Parameter

    layer.weight = Parameter(new_weight, requires_grad=getattr(old_weight, "requires_grad", True))
    layer.bias = Parameter(new_bias, requires_grad=getattr(old_bias, "requires_grad", True))
    return layer


def fuse(model: Any, inplace: bool = False, no_trace: bool = False) -> Any:
    """Fold evaluation-time normalization modules into preceding affine layers."""

    if not inplace:
        model = copy.deepcopy(model)
    if no_trace and isinstance(model, GraphModule):
        graph_module = model
    else:
        graph_module = NodePathTracer().trace(model)
    modules = dict(graph_module.named_modules())
    patterns = (
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (nn.Linear, nn.BatchNorm1d),
    )
    for pattern in patterns:
        for node in list(graph_module.graph.nodes):
            if not matches_module_pattern(pattern, node, modules):
                continue
            producer = cast(Node, node.args[0])
            if len(producer.users) > 1:
                continue
            layer = modules[producer.target]
            norm = modules[node.target]
            if getattr(norm, "training", True) or not getattr(norm, "track_running_stats", False):
                continue
            with tp.no_grad():
                _fuse_affine_normalization(layer, norm)
            node.replace_all_uses_with(producer)
            graph_module.graph.erase_node(node)
            graph_module.delete_submodule(node.target)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


class _DropoutRemover(Transformer):
    def call_module(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        module = self.fetch_attr(target)
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            if len(args) != 1 or kwargs:
                raise ValueError("dropout removal requires a single positional input")
            return args[0]
        return super().call_module(target, args, kwargs)


def remove_dropout(model: Any) -> Any:
    """Remove dropout modules from an inference graph."""

    graph_module = model if isinstance(model, GraphModule) else NodePathTracer().trace(model)
    return _DropoutRemover(graph_module).transform()


def extract_subgraph(
    orig_module: Any,
    nodes: list[Node],
    inputs: list[Node],
    outputs: list[Node],
) -> GraphModule:
    """Create a graph module from selected nodes and explicit boundaries."""

    graph = Graph()
    env: dict[Node, Node] = {}
    for input_node in inputs:
        env[input_node] = graph.placeholder(input_node.name, type_expr=input_node.type)
    for node in nodes:
        env[node] = graph.node_copy(node, lambda value: env[value])
    result = graph.output([env[node] for node in outputs])
    result.meta.update(getattr(outputs[0], "meta", {})) if outputs else None
    graph.lint()
    return GraphModule(orig_module, graph, None)


def modules_to_mkldnn(nodes: list[Node], modules: dict[str, Any]) -> dict[Any, Any]:
    """Convert supported modules when a native layout implementation is installed."""

    old_modules: dict[Any, Any] = {}
    for node in nodes:
        if node.op != "call_module" or not isinstance(node.target, str):
            continue
        module = modules[node.target]
        converter = getattr(module, "to_mkldnn", None)
        if callable(converter):
            converted = converter()
            old_modules[converted] = copy.deepcopy(module)
            replace_node_module(node, modules, converted)
        elif type(module) in {nn.Conv2d, nn.Linear, nn.BatchNorm2d}:
            raise NotImplementedError(
                f"native layout conversion is unavailable for {type(module).__name__}"
            )
    return old_modules


def reset_modules(
    nodes: Iterable[Node], modules: dict[str, Any], old_modules: dict[Any, Any]
) -> None:
    for node in nodes:
        if node.op == "call_module" and isinstance(node.target, str):
            current = modules[node.target]
            if current in old_modules:
                replace_node_module(node, modules, old_modules[current])


class MklSubgraph:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.nodes: list[Node] = []
        self.start_nodes: list[Node] = []
        self.end_nodes: list[Node] = []


def gen_mkl_autotuner(
    example_inputs: list[Any], iters: int = 10, warmup: int = 1
) -> Callable[[MklSubgraph], bool]:
    if iters <= 0 or warmup < 0:
        raise ValueError("iters must be positive and warmup must be non-negative")

    def heuristic(graph: MklSubgraph) -> bool:
        if not graph.nodes:
            return False
        module = extract_subgraph(
            graph.graph.owning_module.root if graph.graph.owning_module else None,
            graph.nodes,
            graph.start_nodes,
            [node.args[0] for node in graph.end_nodes if node.args],
        )
        if not all(callable(getattr(value, "to_mkldnn", None)) for value in example_inputs):
            raise NotImplementedError("autotuning requires native layout conversion")
        samples = [value.to_mkldnn() for value in example_inputs]
        for _ in range(warmup):
            module(*samples)
        start = time.perf_counter()
        for _ in range(iters):
            module(*samples)
        return time.perf_counter() - start >= 0.0

    return heuristic


def use_mkl_length(graph: MklSubgraph) -> bool:
    return len(graph.nodes) > 2


class UnionFind:
    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("size must be non-negative")
        self.parent: list[int | None] = [None] * n
        self.size: list[int] = [0] * n

    def make_set(self, value: int) -> None:
        self.parent[value] = value
        self.size[value] = 1

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent is None:
            raise ValueError(f"element {value} has not been initialized")
        if parent != value:
            self.parent[value] = self.find(parent)
        return cast(int, self.parent[value])

    def join(self, left: int, right: int) -> int | None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return left
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return left


def optimize_for_inference(
    model: Any,
    pass_config: dict[str, Any] | None = None,
    tracer: type[Tracer] = NodePathTracer,
) -> Any:
    """Apply graph transformations intended for evaluation workloads."""

    config: dict[str, Any] = {
        "conv_bn_fuse": True,
        "remove_dropout": True,
        "mkldnn_layout_optimize": False,
    }
    config.update(pass_config or {})
    result = fuse(model) if config["conv_bn_fuse"] else model
    if config["remove_dropout"]:
        result = remove_dropout(result)
    layout_config = config["mkldnn_layout_optimize"]
    if layout_config is not False:
        if not isinstance(layout_config, dict) or "heuristic" not in layout_config:
            raise ValueError("mkldnn_layout_optimize requires a heuristic mapping")
        raise NotImplementedError("layout optimization is not available without native layout modules")
    if not isinstance(result, GraphModule):
        result = tracer().trace(copy.deepcopy(result))
    return result
