from __future__ import annotations

import operator
from collections import deque
from typing import Any, NamedTuple

from .._utils import _iter_nodes, _map_arg
from ..graph_module import GraphModule
from ..node import Node
from ..passes.graph_manipulation import get_size_of_all_nodes
from ..passes.split_module import split_module
from .partitioner_utils import (
    Device,
    NodeLatency,
    Partition,
    PartitionLatency,
    PartitionMode,
    PartitionerConfig,
    get_comm_latency_between,
    get_device_partition_stats,
    get_device_to_partitions_mapping,
    get_extra_size_of,
    get_latency_of_one_partition,
    get_latency_of_partitioned_graph,
    get_partition_to_latency_mapping,
)

__all__ = [
    "DAG",
    "DAGNode",
    "PartitionResult",
    "Partitioner",
    "check_dependency",
    "combine_two_partitions",
    "get_bfs_level_partition",
    "get_comm_latency_between",
    "get_device_partition_stats",
    "get_device_to_partitions_mapping",
    "get_latency_of_one_partition",
    "get_latency_of_partitioned_graph",
    "get_logical_id_to_device",
    "get_node_to_partition_mapping",
    "get_partition_to_latency_mapping",
    "reorganize_partitions",
    "reset_partition_device",
    "set_parents_and_children",
]


class DAGNode:
    """Describe one executable partition and its graph boundaries."""

    def __init__(
        self,
        submodule_node: Node,
        input_nodes: list[Node],
        output_nodes: list[Node],
        logical_device_ids: list[int],
        size_bytes: int,
    ) -> None:
        self.submodule_node = submodule_node
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.logical_device_ids = list(logical_device_ids)
        self.size_bytes = int(size_bytes)

    def __str__(self) -> str:
        return str(self.submodule_node)


class DAG:
    """A dependency list for the partition submodules."""

    def __init__(self) -> None:
        self.nodes: list[DAGNode] = []

    def create_node(
        self,
        submodule_node: Node,
        input_nodes: list[Node],
        output_nodes: list[Node],
        logical_devices: list[int],
        size_bytes: int,
    ) -> None:
        self.nodes.append(
            DAGNode(
                submodule_node,
                input_nodes,
                output_nodes,
                logical_devices,
                size_bytes,
            )
        )


class PartitionResult(NamedTuple):
    dag: DAG
    module_with_submodules: GraphModule


def reset_partition_device(partitions: list[Partition]) -> None:
    for partition in partitions:
        partition.logical_device_ids = []


def _recalculate_partition(partition: Partition) -> None:
    partition.recalculate_mem_size()
    partition.parents = set()
    partition.children = set()


def combine_two_partitions(
    partition_0: Partition, partition_1: Partition, partitions: list[Partition]
) -> Partition:
    """Replace two partitions by their union and rebuild graph relations."""

    if partition_0 is partition_1:
        return partition_0
    if partition_0 not in partitions or partition_1 not in partitions:
        raise ValueError("both partitions must belong to the supplied list")
    merged = Partition(len(partitions))
    merged.nodes = set(partition_0.nodes) | set(partition_1.nodes)
    merged.logical_device_ids = list(
        dict.fromkeys(
            [*partition_0.logical_device_ids, *partition_1.logical_device_ids]
        )
    )
    _recalculate_partition(merged)
    partitions.remove(partition_0)
    partitions.remove(partition_1)
    partitions.append(merged)
    reorganize_partitions(partitions)
    return merged


def set_parents_and_children(partitions: list[Partition]) -> None:
    """Compute partition dependencies from node use-def edges."""

    owner: dict[Node, Partition] = {}
    for partition in partitions:
        partition.parents.clear()
        partition.children.clear()
        for node in partition.nodes:
            old = owner.get(node)
            if old is not None and old is not partition:
                raise RuntimeError(f"node {node.name!r} belongs to multiple partitions")
            owner[node] = partition

    for partition in partitions:
        for node in partition.nodes:
            for user in node.users:
                child = owner.get(user)
                if child is not None and child is not partition:
                    partition.children.add(child)
                    child.parents.add(partition)


def reorganize_partitions(partitions: list[Partition]) -> None:
    for index, partition in enumerate(partitions):
        partition.partition_id = index
    set_parents_and_children(partitions)


def get_bfs_level_partition(partitions: list[Partition]) -> None:
    """Assign the shortest source distance to every partition."""

    for partition in partitions:
        partition.bfs_level = -1
    queue = deque(
        (partition, 0) for partition in partitions if not partition.parents
    )
    while queue:
        partition, level = queue.popleft()
        if partition.bfs_level >= 0 and partition.bfs_level <= level:
            continue
        partition.bfs_level = level
        queue.extend((child, level + 1) for child in partition.children)
    if any(partition.bfs_level < 0 for partition in partitions):
        raise RuntimeError("partition dependency graph contains a cycle")


def get_node_to_partition_mapping(partitions: list[Partition]) -> dict[Node, int]:
    return {
        node: partition.partition_id
        for partition in partitions
        for node in partition.nodes
    }


def get_logical_id_to_device(devices: list[Device]) -> dict[int, Device]:
    result: dict[int, Device] = {}
    for device in devices:
        if device.logical_id in result:
            raise ValueError(f"duplicate logical device id {device.logical_id}")
        result[device.logical_id] = device
    return result


def check_dependency(partition: Partition) -> bool:
    """Return whether ``partition`` can be reached from itself downstream."""

    visited = {partition}
    queue: deque[Partition] = deque([partition])
    while queue:
        current = queue.popleft()
        for child in current.children:
            if child is partition:
                return True
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return False


class Partitioner:
    """Partition a captured graph across devices with bounded memory."""

    def __init__(self) -> None:
        self.partitions: list[Partition] = []
        self.node_to_partition: dict[Node, int] = {}
        self.devices: list[Device] = []
        self.graph_module: GraphModule | None = None
        self.root_module: Any = None

    def _reset(self) -> None:
        self.partitions = []
        self.node_to_partition = {}

    @staticmethod
    def _operation_nodes(graph_module: GraphModule) -> list[Node]:
        return [
            node
            for node in graph_module.graph.nodes
            if node.op in {"call_module", "call_method", "call_function"}
        ]

    def partition_graph(
        self,
        graph_module: GraphModule,
        root_module: Any,
        partitioner_config: PartitionerConfig,
    ) -> PartitionResult:
        """Build partitions, rewrite the graph, and expose its boundary DAG."""

        if not isinstance(graph_module, GraphModule):
            raise TypeError("graph_module must be a GraphModule")
        if not partitioner_config.devices:
            raise RuntimeError("at least one device is required")
        self._reset()
        self.graph_module = graph_module
        self.root_module = root_module if root_module is not None else graph_module.root
        self.devices = list(partitioner_config.devices)
        get_logical_id_to_device(self.devices)
        get_size_of_all_nodes(graph_module)
        operations = self._operation_nodes(graph_module)
        if not operations:
            raise RuntimeError("graph has no executable operations")

        total_size = sum(
            int(getattr(node.size_bytes, "total_size", node.size_bytes or 0))
            for node in graph_module.graph.nodes
            if node.op != "output"
        )
        largest = max(self.devices, key=lambda device: device.available_mem_bytes)
        mode = partitioner_config.mode
        if mode == PartitionMode.aot_based:
            self.aot_based_partition(
                partitioner_config.node_to_partition_mapping,
                partitioner_config.partition_to_logical_device_mapping,
            )
        elif total_size <= largest.available_mem_bytes:
            self.find_single_partition(total_size, largest.logical_id)
        elif total_size > sum(device.available_mem_bytes for device in self.devices):
            raise RuntimeError("devices do not have enough memory for the graph")
        elif mode == PartitionMode.sparse_nn:
            capacity = self.devices[0].available_mem_bytes
            if any(device.available_mem_bytes != capacity for device in self.devices):
                raise RuntimeError("sparse partitioning requires equal device capacity")
            self.sparse_nn_partition(capacity)
        elif mode == PartitionMode.cost_aware:
            self.cost_aware_partition(
                partitioner_config.transfer_rate_bytes_per_sec,
                partitioner_config.node_to_latency_mapping,
            )
        elif mode == PartitionMode.kl_based:
            self.kl_based_partition(
                partitioner_config.transfer_rate_bytes_per_sec,
                partitioner_config.node_to_latency_mapping,
            )
        else:
            self.size_based_partition()

        if partitioner_config.saturate_host:
            self.saturate_host()
        module_with_submodules = self.do_partition()
        return PartitionResult(self.dump_dag(module_with_submodules), module_with_submodules)

    def find_single_partition(self, total_size_of_graph: int, logical_device_id: int = 0) -> None:
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        partition = self.create_partition()
        partition.nodes = {
            node for node in self.graph_module.graph.nodes if node.op != "output"
        }
        partition.used_mem_bytes = int(total_size_of_graph)
        partition.logical_device_ids = [logical_device_id]
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)

    def _find_unused_device(self, needed: int, occupied: set[Device]) -> Device | None:
        candidates = [
            device
            for device in self.devices
            if device not in occupied and device.available_mem_bytes >= needed
        ]
        return min(candidates, key=lambda device: device.available_mem_bytes) if candidates else None

    def size_based_partition(self) -> None:
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        occupied: set[Device] = set()
        current: Partition | None = None
        remaining = 0
        for node in self._operation_nodes(self.graph_module):
            needed = get_extra_size_of(node, current.nodes if current else set())
            if current is None:
                device = self._find_unused_device(needed, occupied)
                if device is not None:
                    current = self.create_partition()
                    current.logical_device_ids = [device.logical_id]
                    occupied.add(device)
                    remaining = device.available_mem_bytes
            elif needed > remaining:
                device = self._find_unused_device(
                    get_extra_size_of(node, set()), occupied
                )
                if device is not None:
                    current = self.create_partition()
                    current.logical_device_ids = [device.logical_id]
                    occupied.add(device)
                    remaining = device.available_mem_bytes
                    needed = get_extra_size_of(node, set())
                else:
                    current = None

            if current is None:
                singleton = self.create_partition()
                singleton.add_node(node)
                continue
            current.add_node(node)
            remaining -= needed

        reorganize_partitions(self.partitions)
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        if not get_device_to_partitions_mapping(self.partitions, self.devices):
            raise RuntimeError("cannot assign partitions to the supplied devices")

    def saturate_host(self) -> None:
        if not self.partitions:
            return
        device_to_partitions, left, unassigned = get_device_partition_stats(
            self.partitions, self.devices
        )
        if unassigned:
            raise RuntimeError("all partitions must be assigned before replication")
        used = [device for device in self.devices if device_to_partitions[device]]
        idle = [device for device in self.devices if not device_to_partitions[device]]
        replicas: dict[Device, Device] = {}
        while used and idle:
            round_assignments: dict[Device, Device] = {}
            for source in used:
                occupied_bytes = source.available_mem_bytes - left[source]
                choices = [
                    device
                    for device in idle
                    if device.available_mem_bytes >= occupied_bytes
                ]
                if not choices:
                    round_assignments = {}
                    break
                target = min(choices, key=lambda device: device.available_mem_bytes)
                idle.remove(target)
                round_assignments[target] = source
            if not round_assignments:
                break
            replicas.update(round_assignments)
        for target, source in replicas.items():
            for partition in device_to_partitions[source]:
                if target.logical_id not in partition.logical_device_ids:
                    partition.logical_device_ids.append(target.logical_id)

    def do_partition(self) -> GraphModule:
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        if self.root_module is None:
            self.root_module = self.graph_module.root
        return split_module(
            self.graph_module,
            self.root_module,
            lambda node: self.node_to_partition[node],
        )

    def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
        dag = DAG()
        by_id = {partition.partition_id: partition for partition in self.partitions}
        for node in module_with_submodules.graph.nodes:
            if node.op == "output":
                break
            if node.op in {"placeholder", "get_attr"}:
                continue
            if node.op == "call_function" and node.target is operator.getitem:
                continue
            input_nodes: dict[Node, None] = {}
            _map_arg(node.args, input_nodes.setdefault)
            _map_arg(node.kwargs, input_nodes.setdefault)
            try:
                partition_id = int(str(node.target).rsplit("_", 1)[-1])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"cannot identify partition for {node.name}") from exc
            partition = by_id.get(partition_id)
            if partition is None:
                raise RuntimeError(f"partition {partition_id} is not present")
            outputs = list(node.users) if len(node.users) > 1 else [node]
            dag.create_node(
                node,
                list(input_nodes),
                outputs,
                partition.logical_device_ids,
                partition.used_mem_bytes,
            )
        return dag

    def create_partition(self, partition_id: int | None = None) -> Partition:
        identifier = len(self.partitions) if partition_id is None else partition_id
        partition = Partition(identifier)
        self.partitions.append(partition)
        return partition

    def create_single_node_partition(self, node: Node) -> None:
        partition = self.create_partition()
        partition.add_node(node)

    def _module_for_target(self, target: str) -> Any:
        if self.graph_module is None:
            return None
        try:
            return self.graph_module._get_attr(target)
        except (AttributeError, KeyError):
            pass
        current = self.root_module
        for atom in target.split("."):
            if current is None or not hasattr(current, atom):
                return None
            current = getattr(current, atom)
        return current

    def _is_embedding_node(self, node: Node) -> bool:
        if node.op != "call_module" or not isinstance(node.target, str):
            return False
        module = self._module_for_target(node.target)
        return module is not None and "embedding" in type(module).__name__.lower()

    def sparse_nn_partition(self, available_mem_bytes: int) -> None:
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        embedding: list[Partition] = []
        dense: list[Partition] = []
        current: Partition | None = None
        in_embedding = False
        for node in self._operation_nodes(self.graph_module):
            node_is_embedding = self._is_embedding_node(node)
            if node_is_embedding != in_embedding:
                if current is not None and current.nodes:
                    (embedding if in_embedding else dense).append(current)
                    current = None
                in_embedding = node_is_embedding
            needed = get_extra_size_of(node, current.nodes if current else set())
            if current is None:
                current = self.create_partition()
            elif current.used_mem_bytes + needed > available_mem_bytes:
                (embedding if in_embedding else dense).append(current)
                current = self.create_partition()
                needed = get_extra_size_of(node, set())
            if needed > available_mem_bytes:
                raise RuntimeError(f"operation {node.name!r} exceeds device capacity")
            current.add_node(node)
        if current is not None and current.nodes:
            (embedding if in_embedding else dense).append(current)

        def combine(group: list[Partition]) -> None:
            changed = True
            while changed:
                changed = False
                set_parents_and_children(self.partitions)
                get_bfs_level_partition(self.partitions)
                ordered = sorted(group, key=lambda item: item.used_mem_bytes)
                for small in ordered:
                    candidates = sorted(
                        (item for item in ordered if item is not small),
                        key=lambda item: item.used_mem_bytes,
                        reverse=True,
                    )
                    candidate = next(
                        (
                            item
                            for item in candidates
                            if abs(item.bfs_level - small.bfs_level) <= 1
                            and sum(
                                get_extra_size_of(node, small.nodes | item.nodes)
                                for node in small.nodes | item.nodes
                            )
                            <= available_mem_bytes
                        ),
                        None,
                    )
                    if candidate is not None:
                        merged = combine_two_partitions(candidate, small, self.partitions)
                        group[:] = [item for item in group if item not in {candidate, small}]
                        group.append(merged)
                        changed = True
                        break

        reorganize_partitions(self.partitions)
        combine(dense)
        combine(embedding)
        if not embedding:
            if not get_device_to_partitions_mapping(self.partitions, self.devices):
                raise RuntimeError("cannot assign sparse partitions to devices")
        else:
            dense_size = sum(partition.used_mem_bytes for partition in dense)
            if len(embedding) > len(self.devices):
                raise RuntimeError("not enough devices for embedding partitions")
            used_ids: list[int] = []
            for index, partition in enumerate(embedding):
                if dense_size + partition.used_mem_bytes > available_mem_bytes:
                    raise RuntimeError("embedding and dense partitions exceed capacity")
                partition.logical_device_ids = [self.devices[index].logical_id]
                used_ids.append(self.devices[index].logical_id)
            for partition in dense:
                partition.logical_device_ids = list(used_ids)
        reorganize_partitions(self.partitions)
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)

    def _validate_latency(self, mapping: dict[Node, NodeLatency]) -> None:
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        missing = [node.name for node in self._operation_nodes(self.graph_module) if node not in mapping]
        if missing:
            raise ValueError("latency metadata is missing for: " + ", ".join(missing))

    def _partition_cost(
        self,
        partitions: list[Partition],
        transfer_rate_bytes_per_sec: float,
        latency: dict[Node, NodeLatency],
    ) -> float:
        return get_latency_of_partitioned_graph(
            partitions,
            get_partition_to_latency_mapping(partitions, latency),
            transfer_rate_bytes_per_sec,
        )

    @staticmethod
    def _copy_partition_state(partitions: list[Partition]) -> list[Partition]:
        copies: list[Partition] = []
        for source in partitions:
            target = Partition(source.partition_id)
            target.nodes = set(source.nodes)
            target.logical_device_ids = list(source.logical_device_ids)
            target.used_mem_bytes = source.used_mem_bytes
            copies.append(target)
        reorganize_partitions(copies)
        return copies

    def _try_merge_cost(
        self,
        first: int,
        second: int,
        transfer_rate_bytes_per_sec: float,
        latency: dict[Node, NodeLatency],
    ) -> tuple[float, list[Partition]]:
        candidate = self._copy_partition_state(self.partitions)
        left, right = candidate[first], candidate[second]
        get_bfs_level_partition(candidate)
        if not (
            abs(left.bfs_level - right.bfs_level) <= 1
            or right in left.children
            or left in right.children
        ):
            return float("inf"), candidate
        combine_two_partitions(left, right, candidate)
        if any(check_dependency(partition) for partition in candidate):
            return float("inf"), candidate
        reset_partition_device(candidate)
        if not get_device_to_partitions_mapping(candidate, self.devices):
            return float("inf"), candidate
        return self._partition_cost(candidate, transfer_rate_bytes_per_sec, latency), candidate

    def cost_aware_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: dict[Node, NodeLatency],
    ) -> None:
        self._validate_latency(node_to_latency_mapping)
        if transfer_rate_bytes_per_sec <= 0:
            raise ValueError("transfer rate must be positive")
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        for node in self._operation_nodes(self.graph_module):
            self.create_single_node_partition(node)
        reorganize_partitions(self.partitions)
        get_bfs_level_partition(self.partitions)
        reset_partition_device(self.partitions)
        if not get_device_to_partitions_mapping(self.partitions, self.devices):
            raise RuntimeError("cannot assign initial partitions to devices")
        while len(self.partitions) > 1:
            current_cost = self._partition_cost(
                self.partitions, transfer_rate_bytes_per_sec, node_to_latency_mapping
            )
            best_cost = current_cost
            best_pair: tuple[int, int] | None = None
            for first in range(len(self.partitions) - 1):
                for second in range(first + 1, len(self.partitions)):
                    candidate_cost, _ = self._try_merge_cost(
                        first,
                        second,
                        transfer_rate_bytes_per_sec,
                        node_to_latency_mapping,
                    )
                    if candidate_cost <= best_cost:
                        best_cost = candidate_cost
                        best_pair = (first, second)
            if best_pair is None:
                break
            combine_two_partitions(
                self.partitions[best_pair[0]],
                self.partitions[best_pair[1]],
                self.partitions,
            )
            reset_partition_device(self.partitions)
            if not get_device_to_partitions_mapping(self.partitions, self.devices):
                raise RuntimeError("partition merge lost a valid device assignment")
        reorganize_partitions(self.partitions)
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)

    def _evaluate_state(
        self,
        partitions: list[Partition],
        transfer_rate_bytes_per_sec: float,
        latency: dict[Node, NodeLatency],
    ) -> float:
        reorganize_partitions(partitions)
        if any(check_dependency(partition) for partition in partitions):
            return float("inf")
        reset_partition_device(partitions)
        if not get_device_to_partitions_mapping(partitions, self.devices):
            return float("inf")
        return self._partition_cost(partitions, transfer_rate_bytes_per_sec, latency)

    def kl_based_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: dict[Node, NodeLatency],
    ) -> None:
        self._validate_latency(node_to_latency_mapping)
        if transfer_rate_bytes_per_sec <= 0:
            raise ValueError("transfer rate must be positive")
        self.size_based_partition()
        best_cost = self._partition_cost(
            self.partitions, transfer_rate_bytes_per_sec, node_to_latency_mapping
        )
        operations = self._operation_nodes(self.graph_module) if self.graph_module else []
        for node in operations:
            owner_index = next(
                (index for index, partition in enumerate(self.partitions) if node in partition.nodes),
                None,
            )
            if owner_index is None:
                continue
            for other_index in range(len(self.partitions)):
                if other_index == owner_index:
                    continue
                other = self.partitions[other_index]
                candidates = [None] + [value for value in other.nodes if value.op not in {"placeholder", "get_attr"}]
                for exchanged in candidates:
                    candidate = self._copy_partition_state(self.partitions)
                    source = candidate[owner_index]
                    target = candidate[other_index]
                    source.nodes.discard(node)
                    target.nodes.add(node)
                    if exchanged is not None:
                        target.nodes.discard(exchanged)
                        source.nodes.add(exchanged)
                    for partition in candidate:
                        _recalculate_partition(partition)
                    score = self._evaluate_state(
                        candidate,
                        transfer_rate_bytes_per_sec,
                        node_to_latency_mapping,
                    )
                    if score < best_cost:
                        self.partitions = candidate
                        best_cost = score
                        reorganize_partitions(self.partitions)
                        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
                        owner_index = next(
                            index
                            for index, partition in enumerate(self.partitions)
                            if node in partition.nodes
                        )
                        break
                else:
                    continue
                break
        reorganize_partitions(self.partitions)
        reset_partition_device(self.partitions)
        if not get_device_to_partitions_mapping(self.partitions, self.devices):
            raise RuntimeError("cannot assign optimized partitions to devices")
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)

    def aot_based_partition(
        self,
        node_to_partition_mapping: dict[Node, int],
        partition_to_logical_device_mapping: dict[int, list[int]],
    ) -> None:
        if self.graph_module is None:
            raise RuntimeError("partitioner has not been initialized")
        operations = self._operation_nodes(self.graph_module)
        missing = [node.name for node in operations if node not in node_to_partition_mapping]
        if missing:
            raise ValueError("partition assignment is missing: " + ", ".join(missing))
        by_id: dict[int, Partition] = {}
        for node in operations:
            partition_id = int(node_to_partition_mapping[node])
            partition = by_id.get(partition_id)
            if partition is None:
                if partition_id not in partition_to_logical_device_mapping:
                    raise ValueError(f"device assignment is missing for partition {partition_id}")
                partition = self.create_partition(partition_id)
                partition.logical_device_ids = list(
                    partition_to_logical_device_mapping[partition_id]
                )
                by_id[partition_id] = partition
            partition.add_node(node)
        self.partitions.sort(key=lambda partition: partition.partition_id)
        set_parents_and_children(self.partitions)
        self.node_to_partition = dict(node_to_partition_mapping)


