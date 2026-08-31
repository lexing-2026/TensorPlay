"""Partition graphs into regions eligible for device graph capture."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from typing import Any

from ..._utils import _iter_nodes
from ...graph_module import GraphModule
from ...node import Node
from ..fake_tensor_prop import FakeTensorProp
from ..infra.partitioner import CapabilityBasedPartitioner
from ..operator_support import OperatorSupport
from ..tools_common import CALLABLE_NODE_OPS

__all__ = ["CudaGraphsSupport", "partition_cudagraphs"]


def _device_type(value: Any) -> str | None:
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type)
    if getattr(value, "is_cuda", False):
        return "cuda"
    return None


def _metadata_values(value: Any):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _metadata_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _metadata_values(item)
    else:
        yield value


class CudaGraphsSupport(OperatorSupport):
    """Accept nodes whose known tensor values all reside on the device."""

    def is_node_supported(
        self, submodules: Mapping[str, Any], node: Node
    ) -> bool:
        del submodules
        if node.op not in CALLABLE_NODE_OPS:
            return False
        if node.target is operator.getitem:
            return True
        values = []
        for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            values.extend(_metadata_values(input_node.meta.get("val")))
        values.extend(_metadata_values(node.meta.get("val")))
        known = [_device_type(value) for value in values]
        return all(device is None or device == "cuda" for device in known)


def partition_cudagraphs(
    gm: GraphModule, inputs: Sequence[object]
) -> GraphModule:
    """Partition a graph so each fused region is device-graph eligible."""

    FakeTensorProp(gm).run(*inputs)
    partitioner = CapabilityBasedPartitioner(
        gm,
        CudaGraphsSupport(),
        allows_single_node_partition=True,
    )
    return partitioner.fuse_partitions(partitioner.propose_partitions())
