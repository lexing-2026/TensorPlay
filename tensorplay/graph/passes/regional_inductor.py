"""Extract and compile explicitly marked graph regions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from ..graph_module import GraphModule
from ..node import Node
from .infra.partitioner import CapabilityBasedPartitioner
from .operator_support import create_op_support

__all__ = ["regional_inductor"]

_REGION_PREFIX = "__marked_region_"


def _compile_marker(node: Node) -> Any:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    return custom.get("compile_with_inductor")


def _needs_compile(node: Node) -> bool:
    return node.op not in {"placeholder", "output"} and _compile_marker(node) is not None


def _extract_regions(gm: GraphModule) -> GraphModule:
    regions: dict[Any, set[Node]] = defaultdict(set)
    for node in gm.graph.nodes:
        if not _needs_compile(node):
            continue
        marker = _compile_marker(node)
        region_id = marker.get("inductor_region") if isinstance(marker, dict) else None
        regions[region_id].add(node)
    for index, region_nodes in enumerate(regions.values()):
        support = create_op_support(lambda _mods, node, nodes=region_nodes: node in nodes)
        partitioner = CapabilityBasedPartitioner(
            gm, support, allows_single_node_partition=True
        )
        partitions = partitioner.propose_partitions()
        if partitions:
            partitioner.fuse_partitions(partitions, prefix=f"{_REGION_PREFIX}{index}_")
    return gm


def _compile_regions(gm: GraphModule) -> GraphModule:
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not str(node.target).startswith(_REGION_PREFIX):
            continue
        region = gm._get_attr(node.target)
        marker = None
        for subnode in region.graph.nodes:
            marker = _compile_marker(subnode)
            if marker is not None:
                break
        compiler: Callable[..., Any] | None = None
        if isinstance(marker, dict):
            candidate = marker.get("compiler") or marker.get("compile")
            if callable(candidate):
                compiler = candidate
        if compiler is None:
            continue
        examples = [
            input_node.meta.get("val")
            for input_node in region.graph.placeholders
            if input_node.meta.get("val") is not None
        ]
        compiled = compiler(region, examples)
        if not callable(compiled):
            raise TypeError("regional compiler must return a callable artifact")
        node.op = "call_function"
        node.target = compiled
        node.args = node.args
    gm.graph.lint()
    gm.recompile()
    return gm


def regional_inductor(gm: GraphModule, *example_args: object) -> GraphModule:
    """Extract marked regions and invoke their explicitly supplied compiler."""

    del example_args
    gm = _extract_regions(gm)
    return _compile_regions(gm)
