from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any, NamedTuple

from .._utils import _iter_nodes
from ..node import Node

__all__ = [
    "Device",
    "NodeLatency",
    "Partition",
    "PartitionLatency",
    "PartitionMode",
    "PartitionerConfig",
    "get_comm_latency_between",
    "get_device_partition_stats",
    "get_device_to_partitions_mapping",
    "get_extra_size_of",
    "get_latency_of_one_partition",
    "get_latency_of_partitioned_graph",
    "get_logical_id_to_device",
    "get_node_to_partition_mapping",
    "get_partition_to_latency_mapping",
]


class Partition:
    """State and dependency information for one graph partition."""

    def __init__(self, partition_id: int) -> None:
        self.nodes: set[Node] = set()
        self.partition_id = partition_id
        self.parents: set[Partition] = set()
        self.children: set[Partition] = set()
        self.bfs_level = -1
        self.used_mem_bytes = 0
        self.logical_device_ids: list[int] = []

    def __str__(self) -> str:
        return str(self.partition_id)

    def recalculate_mem_size(self) -> None:
        self.used_mem_bytes = sum(get_extra_size_of(node, self.nodes) for node in self.nodes)

    def add_node(self, node: Node) -> None:
        for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            if value.op in {"placeholder", "get_attr"}:
                self.nodes.add(value)
        self.nodes.add(node)
        self.recalculate_mem_size()

    def remove_node(self, node: Node) -> None:
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            if value.op in {"placeholder", "get_attr"} and not any(
                user in self.nodes for user in value.users
            ):
                self.nodes.discard(value)
        self.recalculate_mem_size()


class Device(NamedTuple):
    name: str
    available_mem_bytes: int
    logical_id: int


class NodeLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float


class PartitionLatency(NamedTuple):
    mem_latency_sec: float
    computer_latency_sec: float
    overall_latency_sec: float


class PartitionMode(Enum):
    size_based = 0
    sparse_nn = 1
    cost_aware = 2
    kl_based = 3
    aot_based = 4


class PartitionerConfig(NamedTuple):
    devices: list[Device]
    mode: PartitionMode = PartitionMode.size_based
    transfer_rate_bytes_per_sec: float = 0.0
    node_to_latency_mapping: dict[Node, NodeLatency] = {}
    node_to_partition_mapping: dict[Node, int] = {}
    partition_to_logical_device_mapping: dict[int, list[int]] = {}
    saturate_host: bool = False


def _size_field(node: Node, name: str) -> int:
    size = getattr(node, "size_bytes", None)
    if size is None:
        raise RuntimeError(f"node {node.name!r} has no size metadata")
    value = getattr(size, name, None)
    if value is None and isinstance(size, int):
        return size
    if value is None:
        raise RuntimeError(f"node {node.name!r} has incomplete size metadata")
    return int(value)


def get_extra_size_of(node: Node, nodes: set[Node]) -> int:
    total = _size_field(node, "total_size")
    for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
        if value not in nodes:
            total += _size_field(value, "output_size")
    return total


def _top_nodes(partition: Partition) -> list[Node]:
    result = []
    for node in partition.nodes:
        if node.op in {"placeholder", "get_attr"}:
            continue
        if not any(
            value in partition.nodes and value.op not in {"placeholder", "get_attr"}
            for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs))
        ):
            result.append(node)
    return result


def get_latency_of_one_partition(
    partition: Partition, node_to_latency_mapping: dict[Node, NodeLatency]
) -> PartitionLatency:
    def visit(node: Node, prefix: PartitionLatency, active: set[Node]) -> PartitionLatency:
        if node in active:
            raise RuntimeError("partition dependency cycle detected")
        active.add(node)
        latency = node_to_latency_mapping[node]
        current = PartitionLatency(
            prefix.mem_latency_sec + latency.mem_latency_sec,
            prefix.computer_latency_sec + latency.computer_latency_sec,
            prefix.overall_latency_sec + max(latency.mem_latency_sec, latency.computer_latency_sec),
        )
        users = [user for user in node.users if user in partition.nodes]
        if not users:
            active.remove(node)
            return current
        result = max(
            (visit(user, current, active) for user in users),
            key=lambda value: value.overall_latency_sec,
        )
        active.remove(node)
        return result

    empty = PartitionLatency(0.0, 0.0, 0.0)
    return max((visit(node, empty, set()) for node in _top_nodes(partition)), key=lambda value: value.overall_latency_sec, default=empty)


def get_partition_to_latency_mapping(
    partitions: list[Partition], node_to_latency_mapping: dict[Node, NodeLatency]
) -> dict[Partition, PartitionLatency]:
    return {partition: get_latency_of_one_partition(partition, node_to_latency_mapping) for partition in partitions}


def get_comm_latency_between(
    parent_partition: Partition,
    child_partition: Partition,
    transfer_rate_bytes_per_sec: float,
) -> float:
    if parent_partition.logical_device_ids and parent_partition.logical_device_ids == child_partition.logical_device_ids:
        return 0.0
    if transfer_rate_bytes_per_sec <= 0:
        raise ValueError("transfer rate must be positive when partitions cross devices")
    seen: set[Node] = set()
    bytes_to_transfer = 0
    for node in child_partition.nodes:
        for value in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            if value in parent_partition.nodes and value not in seen:
                bytes_to_transfer += _size_field(value, "output_size")
                seen.add(value)
    return bytes_to_transfer / transfer_rate_bytes_per_sec


def get_latency_of_partitioned_graph(
    partitions: list[Partition],
    partition_to_latency_mapping: dict[Partition, PartitionLatency],
    transfer_rate_bytes_per_sec: float,
) -> float:
    def visit(partition: Partition, prefix: float, active: set[Partition]) -> float:
        if partition in active:
            raise RuntimeError("partition dependency cycle detected")
        active.add(partition)
        current = prefix + partition_to_latency_mapping[partition].overall_latency_sec
        result = current
        for child in partition.children:
            result = max(
                result,
                visit(
                    child,
                    current + get_comm_latency_between(partition, child, transfer_rate_bytes_per_sec),
                    active,
                ),
            )
        active.remove(partition)
        return result

    starts = [partition for partition in partitions if not partition.parents]
    return max((visit(partition, 0.0, set()) for partition in starts), default=0.0)


def get_node_to_partition_mapping(partitions: list[Partition]) -> dict[Node, int]:
    return {node: partition.partition_id for partition in partitions for node in partition.nodes}


def get_logical_id_to_device(devices: list[Device]) -> dict[int, Device]:
    return {device.logical_id: device for device in devices}


def get_device_partition_stats(
    partitions: list[Partition], devices: list[Device]
) -> tuple[dict[Device, list[Partition]], dict[Device, int], list[Partition]]:
    by_device = {device: [] for device in devices}
    remaining = {device: device.available_mem_bytes for device in devices}
    lookup = get_logical_id_to_device(devices)
    unassigned = []
    for partition in partitions:
        if not partition.logical_device_ids:
            unassigned.append(partition)
            continue
        for logical_id in partition.logical_device_ids:
            device = lookup[logical_id]
            by_device[device].append(partition)
            remaining[device] -= partition.used_mem_bytes
    return by_device, remaining, unassigned


def get_device_to_partitions_mapping(partitions: list[Partition], devices: list[Device]) -> bool:
    by_device, remaining, unassigned = get_device_partition_stats(partitions, devices)
    for partition in unassigned:
        def extra_size(device: Device) -> int:
            existing = {node for old in by_device[device] for node in old.nodes}
            return sum(get_extra_size_of(node, existing | partition.nodes) for node in partition.nodes)

        ordered = sorted(remaining, key=lambda device: remaining[device])
        placed = False
        for device in ordered:
            needed = extra_size(device)
            if needed <= remaining[device]:
                by_device[device].append(partition)
                remaining[device] -= needed
                partition.logical_device_ids.append(device.logical_id)
                placed = True
                break
        if not placed:
            return False
    return True
