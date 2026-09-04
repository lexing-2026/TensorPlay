"""Layout planning and collective execution for distributed tensor values."""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import logging
import math
import weakref
from collections import defaultdict
from functools import cache
from typing import Any, Sequence, TypedDict, cast

from ...autograd.function import Function
from ...autograd.grad_mode import is_grad_enabled
from .._functional_collectives import AsyncCollectiveTensor
from ..device_mesh import DeviceMesh
from ._collective_utils import one_step_redistribute_cost
from ._dtensor_spec import (
    _StridedShardNotDecodableError,
    DTensorSpec,
    ShardOrder,
    ShardOrderEntry,
    TensorMeta,
)
from ._utils import assert_no_mixed_partial_types, _strided_shard_indices
from .placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = [
    "DTensorRedistributePlanner",
    "NestedRedistribute",
    "Redistribute",
    "clear_redistribute_planner_cache",
    "disable_redistribute_transform_optimization",
    "get_redistribute_planner",
    "redistribute_local_tensor",
    "use_min_cost_redistribution_plan",
]


logger = logging.getLogger(__name__)

_FORCE_MIN_COST_REDISTRIBUTION_PLAN: bool | None = None
_DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION = False


def _redistribute_cost_sort_key(cost: Any) -> float:
    try:
        return float(cost)
    except (TypeError, ValueError):
        hint = getattr(cost, "__float__", None)
        if callable(hint):
            return float(hint())
        return 0.0


@contextlib.contextmanager
def use_min_cost_redistribution_plan(enabled: bool = True):
    global _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    previous = _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    _FORCE_MIN_COST_REDISTRIBUTION_PLAN = bool(enabled)
    try:
        yield
    finally:
        _FORCE_MIN_COST_REDISTRIBUTION_PLAN = previous


@contextlib.contextmanager
def disable_redistribute_transform_optimization(disabled: bool = True):
    global _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION
    previous = _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION
    _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION = bool(disabled)
    try:
        yield
    finally:
        _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION = previous


@dataclasses.dataclass(frozen=True, slots=True)
class _TransformInfo:
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    logical_shape: Sequence[int]

    def __post_init__(self) -> None:
        if self.mesh_dim < 0:
            raise AssertionError
        if self.src_dst_placements[0] == self.src_dst_placements[1]:
            raise AssertionError("a transform must change its placement")

    def _comm_type_key(self) -> str | None:
        source, target = self.src_dst_placements
        if source.is_partial() and target.is_replicate():
            return "all_reduce"
        if source.is_partial() and _is_shard_like(target):
            return "reduce_scatter"
        if _is_shard_like(source) and target.is_replicate():
            return "all_gather"
        if _is_shard_like(source) and _is_shard_like(target):
            return "all_to_all"
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class _FlattenedTransformInfo(_TransformInfo):
    mesh: DeviceMesh
    original_mesh_dims: tuple[int, ...]
    avg_scale: int | None = None

    def __post_init__(self) -> None:
        _TransformInfo.__post_init__(self)
        if self.avg_scale is not None and self.avg_scale <= 1:
            raise AssertionError("avg_scale must be greater than one")


def _update_shard_order_and_placements(
    transform_info: _TransformInfo,
    current_placements: list[Placement],
    shard_order_dict: dict[int, list[int]],
) -> None:
    source, target = transform_info.src_dst_placements
    mesh_dims = (
        transform_info.original_mesh_dims
        if isinstance(transform_info, _FlattenedTransformInfo)
        else (transform_info.mesh_dim,)
    )
    if _is_shard_like(source):
        source_dim = source.dim
        removed = set()
        for _ in mesh_dims:
            if not shard_order_dict.get(source_dim):
                raise ValueError("shard order cannot remove a missing entry")
            removed.add(shard_order_dict[source_dim].pop())
        if removed != set(mesh_dims):
            raise ValueError("shard order update removed unexpected mesh dimensions")
    if _is_shard_like(target):
        target_dim = target.dim
        shard_order_dict.setdefault(target_dim, [])
        shard_order_dict[target_dim].extend(mesh_dims)
    for mesh_dim in mesh_dims:
        current_placements[mesh_dim] = target


def _get_flattened_mesh_by_layout_impl(
    mesh: DeviceMesh, mesh_dims: tuple[int, ...]
) -> DeviceMesh | None:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is None or any(dim < 0 or dim >= len(names) for dim in mesh_dims):
        return None
    root = mesh._get_root_mesh()
    axes = mesh._get_axis_root_dims()
    expected_axis = tuple(root_dim for dim in mesh_dims for root_dim in axes[dim])
    expected_size = math.prod(int(mesh.size(dim)) for dim in mesh_dims)
    expected_name = "_".join(str(names[dim]) for dim in mesh_dims)
    for flattened in root._flatten_mapping.values():
        flattened_names = getattr(flattened, "mesh_dim_names", None)
        if flattened_names != (expected_name,):
            continue
        if tuple(flattened.size(dim) for dim in range(int(flattened.ndim))) != (
            expected_size,
        ):
            continue
        if flattened._get_axis_root_dims() == (expected_axis,):
            return flattened
    return None


def _get_flattened_mesh_by_layout(
    mesh: DeviceMesh, mesh_dims: tuple[int, ...]
) -> DeviceMesh | None:
    return _get_flattened_mesh_by_layout_impl(mesh, mesh_dims)


_warned_flatten_issues: set[tuple[int, tuple[int, ...], str]] = set()


def _warn_flatten_optimization_not_possible(
    device_mesh: DeviceMesh,
    mesh_dims: tuple[int, ...],
    src_placements: tuple[Placement, ...],
    dst_placements: tuple[Placement, ...],
    num_ops: int,
    comm_type: str,
    reason: str,
) -> None:
    key = (hash(device_mesh), mesh_dims, reason)
    if key in _warned_flatten_issues:
        return
    _warned_flatten_issues.add(key)
    names = getattr(device_mesh, "mesh_dim_names", None)
    dims = str(mesh_dims) if names is None else str(tuple(names[dim] for dim in mesh_dims))
    if reason == "no_flattened_mesh":
        detail = f"create a flattened mesh for dimensions {dims}"
    elif reason == "uneven_tensor_shape":
        detail = "the affected dimension cannot be evenly divided"
    elif reason == "non_ascending_mesh_dims":
        detail = f"the {comm_type} dimensions are not in executable order"
    else:
        raise AssertionError(f"unexpected optimization reason: {reason}")
    logger.warning(
        "layout conversion %s -> %s uses %d %s operations; %s",
        src_placements,
        dst_placements,
        num_ops,
        comm_type,
        detail,
    )


def _optimize_transform_infos(
    transform_infos: list[_TransformInfo],
    device_mesh: DeviceMesh,
    src_placements: tuple[Placement, ...],
    dst_placements: tuple[Placement, ...],
) -> list[_TransformInfo | _FlattenedTransformInfo]:
    if len(transform_infos) < 2 or _DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION:
        return transform_infos

    mergeable = frozenset({"all_gather", "all_reduce", "reduce_scatter"})

    def mergeable_placements(
        first: tuple[Placement, Placement], second: tuple[Placement, Placement]
    ) -> bool:
        if first == second:
            return True
        source_a, target_a = first
        source_b, target_b = second
        if target_a != target_b or not source_a.is_partial() or not source_b.is_partial():
            return False
        return {cast(Partial, source_a).reduce_op, cast(Partial, source_b).reduce_op} <= {
            "sum",
            "avg",
        }

    def make_flattened(
        infos: list[_TransformInfo],
    ) -> tuple[_FlattenedTransformInfo | None, str | None]:
        if len(infos) < 2:
            return None, "too_few_transforms"
        first = infos[0].src_dst_placements
        if not all(mergeable_placements(info.src_dst_placements, first) for info in infos):
            raise AssertionError("incompatible transforms were grouped")
        comm_type = infos[0]._comm_type_key()
        mesh_dims = tuple(info.mesh_dim for info in infos)
        sorted_mesh_dims = tuple(sorted(mesh_dims))
        if comm_type == "reduce_scatter" and mesh_dims != sorted_mesh_dims:
            return None, "non_ascending_mesh_dims"
        if comm_type == "all_gather" and mesh_dims != sorted_mesh_dims[::-1]:
            return None, "non_ascending_mesh_dims"
        flattened = _get_flattened_mesh_by_layout(device_mesh, sorted_mesh_dims)
        if flattened is None:
            return None, "no_flattened_mesh"
        source, target = first
        if comm_type == "all_gather":
            affected_dim = cast(Shard, source).dim
            outermost = max(infos, key=lambda info: info.logical_shape[affected_dim])
        elif comm_type == "reduce_scatter":
            affected_dim = cast(Shard, target).dim
            outermost = max(infos, key=lambda info: info.logical_shape[affected_dim])
            if int(outermost.logical_shape[affected_dim]) % math.prod(
                int(device_mesh.size(info.mesh_dim)) for info in infos
            ):
                return None, "uneven_tensor_shape"
        elif comm_type == "all_reduce":
            outermost = infos[0]
        else:
            raise NotImplementedError(f"unsupported collective type: {comm_type}")
        average_scale = None
        merged_source = source
        if source.is_partial():
            average_scale = math.prod(
                int(device_mesh.size(info.mesh_dim))
                for info in infos
                if cast(Partial, info.src_dst_placements[0]).reduce_op == "avg"
            )
            if average_scale <= 1:
                average_scale = None
            else:
                merged_source = Partial("sum")
        return (
            _FlattenedTransformInfo(
                mesh_dim=0,
                src_dst_placements=(merged_source, target),
                logical_shape=outermost.logical_shape,
                mesh=flattened,
                original_mesh_dims=sorted_mesh_dims,
                avg_scale=average_scale,
            ),
            None,
        )

    result: list[_TransformInfo | _FlattenedTransformInfo] = []
    index = 0
    while index < len(transform_infos):
        info = transform_infos[index]
        if info._comm_type_key() not in mergeable:
            result.append(info)
            index += 1
            continue
        group = [info]
        next_index = index + 1
        while (
            next_index < len(transform_infos)
            and transform_infos[next_index]._comm_type_key() in mergeable
            and mergeable_placements(
                transform_infos[next_index].src_dst_placements,
                info.src_dst_placements,
            )
        ):
            group.append(transform_infos[next_index])
            next_index += 1
        flattened, reason = make_flattened(group)
        if flattened is None:
            result.extend(group)
            if reason in {
                "no_flattened_mesh",
                "uneven_tensor_shape",
                "non_ascending_mesh_dims",
            }:
                _warn_flatten_optimization_not_possible(
                    device_mesh,
                    tuple(sorted(item.mesh_dim for item in group)),
                    src_placements,
                    dst_placements,
                    len(group),
                    cast(str, info._comm_type_key()),
                    reason,
                )
        else:
            result.append(flattened)
        index = next_index
    logger.debug("redistribution transforms: %s -> %s", transform_infos, result)
    return result


_planner_cache: dict[
    tuple[weakref.ReferenceType[DeviceMesh], TensorMeta], DTensorRedistributePlanner
] = {}


def get_redistribute_planner(
    device_mesh: DeviceMesh, dtensor_meta: TensorMeta
) -> "DTensorRedistributePlanner":
    key = (weakref.ref(device_mesh), dtensor_meta)
    planner = _planner_cache.get(key)
    if planner is None:
        planner = DTensorRedistributePlanner(device_mesh, dtensor_meta)
        _planner_cache[key] = planner
    return planner


def clear_redistribute_planner_cache() -> None:
    _planner_cache.clear()


class DTensorRedistributePlanner:
    @dataclasses.dataclass(frozen=True, slots=True)
    class DistState:
        placements: tuple[Placement, ...]
        tensor_dim_to_mesh_dim: ShardOrder
        _hash: int | None = dataclasses.field(
            default=None, init=False, repr=False, compare=False
        )

        def __post_init__(self) -> None:
            object.__setattr__(self, "_hash", hash((self.placements, self.tensor_dim_to_mesh_dim)))

        def __hash__(self) -> int:
            return cast(int, self._hash)

        def __str__(self) -> str:
            return DTensorSpec.format_shard_order_str(
                self.placements, self.tensor_dim_to_mesh_dim
            )

        def __repr__(self) -> str:
            return str(self)

    def _to_tuple(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(self._to_tuple(item) for item in value)
        return value

    @staticmethod
    def _dict_to_ShardOrder(value: dict[int, list[int]]) -> ShardOrder:
        return tuple(
            ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(mesh_dims))
            for key, mesh_dims in sorted(value.items())
            if mesh_dims
        )

    @staticmethod
    def _ShardOrder_to_dict(value: ShardOrder) -> dict[int, list[int]]:
        result: defaultdict[int, list[int]] = defaultdict(list)
        for entry in value:
            result[entry.tensor_dim] = list(entry.mesh_dims)
        return result

    @staticmethod
    def stringify_transform_infos(
        mesh: DeviceMesh,
        transform_infos: Sequence[_TransformInfo],
        src_placement: tuple[Placement, ...],
        src_shard_order: ShardOrder | None = None,
        use_strided_shard_as_shard_order: bool = False,
    ) -> str:
        if len(src_placement) != int(mesh.ndim):
            raise AssertionError("placement and mesh dimensions differ")
        if use_strided_shard_as_shard_order:
            src_placement, src_shard_order = DTensorSpec._normalize_placements_into_shard_order(
                src_placement, mesh, use_strided_shard_as_shard_order=True
            )
        if src_shard_order is None:
            src_shard_order = DTensorSpec.compute_default_shard_order(src_placement)
        current = list(src_placement)
        order = DTensorRedistributePlanner._ShardOrder_to_dict(src_shard_order)
        states = [DTensorRedistributePlanner.DistState(tuple(current), src_shard_order)]
        flattened: list[bool] = []
        for info in transform_infos:
            flattened.append(isinstance(info, _FlattenedTransformInfo))
            _update_shard_order_and_placements(info, current, order)
            states.append(
                DTensorRedistributePlanner.DistState(
                    tuple(current), DTensorRedistributePlanner._dict_to_ShardOrder(order)
                )
            )
        output = [str(states[0])]
        for index, is_flattened in enumerate(flattened):
            output.extend(("-->" if is_flattened else "->", str(states[index + 1])))
        return "".join(output)

    def __init__(self, device_mesh: DeviceMesh, dtensor_meta: TensorMeta) -> None:
        if not _participates(device_mesh):
            raise AssertionError
        if dtensor_meta is None:
            raise AssertionError
        self.device_mesh = device_mesh
        self.dtensor_meta = dtensor_meta
        self.tensor_dimension = len(dtensor_meta.shape)
        self.strided_shard_placements_in_target: set[_StridedShard] = set()
        self.partial_reduce_ops_in_target: set[str] = set()
        self.setup_cost_callbacks()

    def setup_cost_callbacks(self) -> None:
        def state_to_spec(state: DTensorRedistributePlanner.DistState) -> DTensorSpec:
            return DTensorSpec(
                mesh=self.device_mesh,
                placements=state.placements,
                tensor_meta=self.dtensor_meta,
                shard_order=state.tensor_dim_to_mesh_dim,
                use_strided_shard_as_shard_order=False,
            )

        self.cost_function = lambda source, target: one_step_redistribute_cost(
            state_to_spec(source), state_to_spec(target)
        )

    def get_next_state(
        self,
        placements: tuple[Placement, ...],
        tensor_mesh_dim_tuple: ShardOrder,
    ) -> dict["DTensorRedistributePlanner.DistState", float]:
        result: dict[DTensorRedistributePlanner.DistState, float] = {}
        order = self._ShardOrder_to_dict(tensor_mesh_dim_tuple)
        current = self.DistState(self._to_tuple(placements), tensor_mesh_dim_tuple)

        for entry in tensor_mesh_dim_tuple:
            source_dim = entry.tensor_dim
            source_mesh_dim = order[source_dim][-1]
            if type(placements[source_mesh_dim]) is not Shard:
                continue
            move_mesh_dim = order[source_dim].pop()
            for target_dim in range(self.tensor_dimension):
                if source_dim == target_dim:
                    continue
                order[target_dim].append(move_mesh_dim)
                next_placements = list(placements)
                next_placements[move_mesh_dim] = Shard(target_dim)
                next_state = self.DistState(
                    tuple(next_placements), self._dict_to_ShardOrder(order)
                )
                result[next_state] = self.cost_function(current, next_state)
                order[target_dim].pop()
            order[source_dim].append(move_mesh_dim)

        for entry in tensor_mesh_dim_tuple:
            source_dim = entry.tensor_dim
            source_mesh_dim = order[source_dim][-1]
            if type(placements[source_mesh_dim]) is not Shard:
                continue
            move_mesh_dim = order[source_dim].pop()
            next_placements = list(placements)
            next_placements[move_mesh_dim] = Replicate()
            next_state = self.DistState(
                tuple(next_placements), self._dict_to_ShardOrder(order)
            )
            order[source_dim].append(move_mesh_dim)
            result[next_state] = self.cost_function(current, next_state)

        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Partial):
                continue
            next_placements = list(placements)
            next_placements[mesh_dim] = Replicate()
            next_state = self.DistState(tuple(next_placements), tensor_mesh_dim_tuple)
            result[next_state] = self.cost_function(current, next_state)

        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for target_dim in range(self.tensor_dimension):
                next_placements = list(placements)
                next_placements[mesh_dim] = Shard(target_dim)
                order[target_dim].append(mesh_dim)
                next_state = self.DistState(
                    tuple(next_placements), self._dict_to_ShardOrder(order)
                )
                result[next_state] = self.cost_function(current, next_state)
                order[target_dim].pop()

        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Partial):
                continue
            for target_dim in range(self.tensor_dimension):
                next_placements = list(placements)
                next_placements[mesh_dim] = Shard(target_dim)
                order[target_dim].append(mesh_dim)
                next_state = self.DistState(
                    tuple(next_placements), self._dict_to_ShardOrder(order)
                )
                result[next_state] = self.cost_function(current, next_state)
                order[target_dim].pop()

        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for reduce_op in self.partial_reduce_ops_in_target:
                next_placements = list(placements)
                next_placements[mesh_dim] = Partial(reduce_op)
                reduce_ops = {
                    item.reduce_op
                    for item in next_placements
                    if isinstance(item, Partial)
                }
                if len(reduce_ops) > 1 and reduce_ops != {"sum", "avg"}:
                    continue
                next_state = self.DistState(tuple(next_placements), tensor_mesh_dim_tuple)
                result[next_state] = self.cost_function(current, next_state)

        for entry in tensor_mesh_dim_tuple:
            source_dim = entry.tensor_dim
            source_mesh_dim = order[source_dim][-1]
            if not isinstance(placements[source_mesh_dim], _StridedShard):
                continue
            move_mesh_dim = order[source_dim].pop()
            next_placements = list(placements)
            next_placements[move_mesh_dim] = Replicate()
            next_state = self.DistState(
                tuple(next_placements), self._dict_to_ShardOrder(order)
            )
            order[source_dim].append(move_mesh_dim)
            result[next_state] = self.cost_function(current, next_state)

        if not self.strided_shard_placements_in_target:
            return result
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for target in self.strided_shard_placements_in_target:
                order[target.dim].append(mesh_dim)
                next_placements = list(placements)
                next_placements[mesh_dim] = target
                next_state = self.DistState(
                    tuple(next_placements), self._dict_to_ShardOrder(order)
                )
                result[next_state] = self.cost_function(current, next_state)
                order[target.dim].pop()
        return result

    def _get_shard_to_partial_target_states(
        self, current_state: DistState, target_state: DistState
    ) -> dict[DistState, float]:
        result: dict[DTensorRedistributePlanner.DistState, float] = {}
        order = self._ShardOrder_to_dict(current_state.tensor_dim_to_mesh_dim)
        for entry in current_state.tensor_dim_to_mesh_dim:
            tensor_dim = entry.tensor_dim
            mesh_dim = order[tensor_dim][-1]
            source = current_state.placements[mesh_dim]
            target = target_state.placements[mesh_dim]
            if not (
                type(source) is Shard
                and type(target) is Partial
                and target.reduce_op == "sum"
            ):
                continue
            order[tensor_dim].pop()
            placements = list(current_state.placements)
            placements[mesh_dim] = target
            next_state = self.DistState(tuple(placements), self._dict_to_ShardOrder(order))
            order[tensor_dim].append(mesh_dim)
            result[next_state] = 0.0
        return result

    def find_min_cost_path(
        self, source: DistState, target: DistState
    ) -> list["DTensorRedistributePlanner.DistState"]:
        import heapq

        counter = 0
        queue: list[
            tuple[
                float,
                int,
                float,
                DTensorRedistributePlanner.DistState,
                list[DTensorRedistributePlanner.DistState],
            ]
        ] = [(0.0, counter, 0.0, source, [source])]
        visited: set[DTensorRedistributePlanner.DistState] = set()
        while queue:
            _, _, cost, current, path = heapq.heappop(queue)
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)
            next_states = self.get_next_state(
                current.placements, current.tensor_dim_to_mesh_dim
            )
            next_states.update(self._get_shard_to_partial_target_states(current, target))
            for next_state, transition_cost in next_states.items():
                if next_state in visited:
                    continue
                new_cost = cost + transition_cost
                counter += 1
                heapq.heappush(
                    queue,
                    (
                        _redistribute_cost_sort_key(new_cost),
                        counter,
                        new_cost,
                        next_state,
                        path + [next_state],
                    ),
                )
        raise AssertionError(f"no layout path from {source} to {target}")

    def get_logical_shape(
        self,
        source: "DTensorRedistributePlanner.DistState",
        mesh_dim: int,
        full_tensor_shape: tuple[int, ...],
    ) -> list[int]:
        shape = list(full_tensor_shape)
        for entry in source.tensor_dim_to_mesh_dim:
            tensor_dim = entry.tensor_dim
            if not entry.mesh_dims:
                raise AssertionError
            for other_mesh_dim in entry.mesh_dims:
                if other_mesh_dim == mesh_dim:
                    continue
                placement = source.placements[other_mesh_dim]
                rank = int(self.device_mesh.get_local_rank(other_mesh_dim))
                if isinstance(placement, _StridedShard):
                    size = len(
                        _strided_shard_indices(
                            int(shape[tensor_dim]),
                            int(self.device_mesh.size(other_mesh_dim)),
                            rank,
                            placement.split_factor,
                        )
                    )
                elif type(placement) is Shard:
                    size, _ = placement.local_shard_size_and_offset(
                        int(shape[tensor_dim]),
                        int(self.device_mesh.size(other_mesh_dim)),
                        rank,
                    )
                else:
                    raise ValueError(f"unsupported placement type: {placement}")
                shape[tensor_dim] = int(size)
        return shape

    def generate_graph_based_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        full_tensor_shape: tuple[int, ...],
    ) -> list[_TransformInfo]:
        def normalize(spec: DTensorSpec) -> tuple[tuple[Placement, ...], ShardOrder]:
            if spec.use_strided_shard_as_shard_order:
                return DTensorSpec._normalize_placements_into_shard_order(
                    spec.placements,
                    spec.mesh,
                    use_strided_shard_as_shard_order=True,
                )
            if spec.shard_order is None:
                raise ValueError(f"missing shard order in {spec}")
            return spec.placements, spec.shard_order

        source_placements, source_order = normalize(src_spec)
        target_placements, target_order = normalize(dst_spec)
        for placement in target_placements:
            if isinstance(placement, _StridedShard):
                self.strided_shard_placements_in_target.add(placement)
        for placement in itertools.chain(source_placements, target_placements):
            if isinstance(placement, Partial):
                self.partial_reduce_ops_in_target.add(placement.reduce_op)
        source_state = self.DistState(source_placements, source_order)
        target_state = self.DistState(target_placements, target_order)
        transforms: list[_TransformInfo] = []
        path = self.find_min_cost_path(source_state, target_state)
        for current, next_state in itertools.pairwise(path):
            if current.placements == next_state.placements:
                continue
            changed = [
                index
                for index, (source, target) in enumerate(
                    zip(current.placements, next_state.placements)
                )
                if source != target
            ]
            if len(changed) != 1:
                raise AssertionError("a path step must change one mesh dimension")
            mesh_dim = changed[0]
            transforms.append(
                _TransformInfo(
                    mesh_dim=mesh_dim,
                    src_dst_placements=(
                        current.placements[mesh_dim],
                        next_state.placements[mesh_dim],
                    ),
                    logical_shape=self.get_logical_shape(
                        current, mesh_dim, full_tensor_shape
                    ),
                )
            )
        return transforms

    def generate_greedy_transform_infos(
        self, src_spec: DTensorSpec, dst_spec: DTensorSpec
    ) -> list[_TransformInfo]:
        initial_shape = list(src_spec.shape)
        logical_shapes = [initial_shape]
        transforms: list[_TransformInfo] = []
        mesh_ndim = int(self.device_mesh.ndim)
        if mesh_ndim == 1:
            if src_spec.placements[0] != dst_spec.placements[0]:
                transforms.append(
                    _TransformInfo(
                        mesh_dim=0,
                        src_dst_placements=(
                            src_spec.placements[0],
                            dst_spec.placements[0],
                        ),
                        logical_shape=initial_shape,
                    )
                )
            return transforms

        for mesh_dim, source in enumerate(src_spec.placements):
            current_shape = logical_shapes[mesh_dim]
            if _is_shard_like(source) and mesh_dim < mesh_ndim - 1:
                rank = int(self.device_mesh.get_local_rank(mesh_dim))
                if isinstance(source, _StridedShard):
                    local_size = len(
                        _strided_shard_indices(
                            int(current_shape[source.dim]),
                            int(self.device_mesh.size(mesh_dim)),
                            rank,
                            source.split_factor,
                        )
                    )
                else:
                    local_size, _ = source._local_shard_size_and_offset(
                        int(current_shape[source.dim]),
                        int(self.device_mesh.size(mesh_dim)),
                        rank,
                    )
                next_shape = list(current_shape)
                next_shape[source.dim] = int(local_size)
                logical_shapes.append(next_shape)
            else:
                logical_shapes.append(current_shape)

        current_placements = list(src_spec.placements)
        target_placements = list(dst_spec.placements)
        if src_spec.num_shards > 1:
            for mesh_dim in reversed(range(mesh_ndim)):
                current = current_placements[mesh_dim]
                target = target_placements[mesh_dim]
                if type(target) is Shard:
                    shard_dim = target.dim
                    current_order = [
                        index
                        for index, placement in enumerate(current_placements[:mesh_dim])
                        if _is_shard_like(placement) and placement.dim == shard_dim
                    ]
                    target_order = [
                        index
                        for index, placement in enumerate(target_placements[:mesh_dim])
                        if _is_shard_like(placement) and placement.dim == shard_dim
                    ]
                    if current_order != target_order:
                        target = Replicate()
                if current != target:
                    transforms.append(
                        _TransformInfo(
                            mesh_dim=mesh_dim,
                            src_dst_placements=(current, target),
                            logical_shape=logical_shapes[mesh_dim],
                        )
                    )
                    current_placements[mesh_dim] = target
        for mesh_dim, (current, target) in enumerate(
            zip(current_placements, target_placements)
        ):
            if current != target:
                transforms.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(current, target),
                        logical_shape=logical_shapes[mesh_dim],
                    )
                )
                current_placements[mesh_dim] = target
        return transforms


def _gen_transform_infos_non_cached(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    use_graph_based_transform: bool | None = None,
) -> list[_TransformInfo]:
    source_order = src_spec.shard_order
    target_order = dst_spec.shard_order
    non_default_order = not all(
        DTensorSpec.is_default_device_order(order)
        for order in (source_order, target_order)
    )
    has_strided = any(
        isinstance(placement, _StridedShard)
        for placement in (*src_spec.placements, *dst_spec.placements)
    )
    if non_default_order or has_strided:
        use_graph_based_transform = True
    elif _FORCE_MIN_COST_REDISTRIBUTION_PLAN is not None:
        use_graph_based_transform = _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    elif use_graph_based_transform is None:
        use_graph_based_transform = False
    if src_spec.tensor_meta is None:
        raise AssertionError
    planner = get_redistribute_planner(src_spec.device_mesh, src_spec.tensor_meta)
    if use_graph_based_transform:
        try:
            return planner.generate_graph_based_transform_infos(
                src_spec, dst_spec, src_spec.shape
            )
        except _StridedShardNotDecodableError:
            return planner.generate_greedy_transform_infos(src_spec, dst_spec)
    return planner.generate_greedy_transform_infos(src_spec, dst_spec)


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    use_graph_based_transform: bool | None = None,
) -> list[_TransformInfo]:
    return _gen_transform_infos_non_cached(
        src_spec, dst_spec, use_graph_based_transform
    )


def _participates(mesh: DeviceMesh) -> bool:
    coordinate = getattr(mesh, "get_coordinate", None)
    return coordinate is None or coordinate() is not None


def _convert_placement(
    value: Any,
    source: Placement,
    target: Placement,
    mesh: DeviceMesh,
    mesh_dim: int,
    global_shape: Sequence[int],
    async_op: bool = False,
) -> Any:
    if source == target:
        return value
    if isinstance(value, AsyncCollectiveTensor):
        value = value.wait()
    if isinstance(target, _StridedShard):
        if isinstance(source, Partial):
            replicated = source._reduce_value(value, mesh, mesh_dim)
            return target._replicate_to_strided_shard(
                replicated,
                mesh,
                mesh_dim,
                mesh.get_local_rank(mesh_dim),
            )
        if isinstance(source, Replicate):
            return target._replicate_to_strided_shard(
                value,
                mesh,
                mesh_dim,
                mesh.get_local_rank(mesh_dim),
            )
        if isinstance(source, _StridedShard):
            replicated = source._to_replicate_tensor(
                value, mesh, mesh_dim, global_shape
            )
            return target._replicate_to_strided_shard(
                replicated,
                mesh,
                mesh_dim,
                mesh.get_local_rank(mesh_dim),
            )
        if type(source) is Shard:
            replicated = source._to_replicate_tensor(
                value, mesh, mesh_dim, global_shape
            )
            return target._replicate_to_strided_shard(
                replicated,
                mesh,
                mesh_dim,
                mesh.get_local_rank(mesh_dim),
            )
        raise ValueError(f"unsupported placement conversion: {source!r} to {target!r}")
    if isinstance(target, Replicate):
        if isinstance(source, Partial):
            return source._reduce_value(value, mesh, mesh_dim)
        if isinstance(source, _StridedShard):
            return source._to_replicate_tensor(value, mesh, mesh_dim, global_shape)
        if type(source) is Shard:
            return source._to_replicate_tensor(value, mesh, mesh_dim, global_shape)
        raise RuntimeError(f"redistribution from {source} to {target} is unsupported")
    if type(target) is Shard:
        if isinstance(source, Partial):
            return source._reduce_shard_value(value, mesh, mesh_dim, target)
        if isinstance(source, Replicate):
            return target._replicate_to_shard(
                value,
                mesh,
                mesh_dim,
                mesh.get_local_rank(mesh_dim),
            )
        if isinstance(source, _StridedShard):
            replicated = source._to_replicate_tensor(
                value, mesh, mesh_dim, global_shape
            )
            return target._replicate_to_shard(
                replicated,
                mesh,
                mesh_dim,
                mesh.get_local_rank(mesh_dim),
            )
        if type(source) is Shard:
            return source._to_new_shard_dim(
                value,
                mesh,
                mesh_dim,
                global_shape,
                target.dim,
            )
        raise ValueError(f"unsupported placement conversion: {source!r} to {target!r}")
    if isinstance(target, Partial):
        if isinstance(source, Replicate):
            return target._partition_value(value, mesh, mesh_dim)
        if type(source) is Shard:
            return source._to_partial_tensor(
                value,
                mesh,
                mesh_dim,
                global_shape,
            )
        raise RuntimeError(f"redistribution from {source} to {target} is unsupported")
    raise ValueError(f"unsupported placement conversion: {source!r} to {target!r}")


def redistribute_local_tensor(
    local_tensor: Any,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    use_graph_based_transform: bool | None = None,
    is_explicit: bool = False,
) -> Any:
    del is_explicit
    if current_spec.mesh != target_spec.mesh:
        raise NotImplementedError("cross-mesh redistribution is not supported")
    assert_no_mixed_partial_types(current_spec.placements)
    assert_no_mixed_partial_types(target_spec.placements)
    if not _participates(current_spec.mesh):
        return local_tensor
    transforms = _gen_transform_infos(
        current_spec, target_spec, use_graph_based_transform
    )
    transforms = _optimize_transform_infos(
        transforms,
        current_spec.mesh,
        current_spec.placements,
        target_spec.placements,
    )
    result = local_tensor
    for info in transforms:
        mesh = (
            info.mesh if isinstance(info, _FlattenedTransformInfo) else current_spec.mesh
        )
        mesh_dim = info.mesh_dim
        source, target = info.src_dst_placements
        if source == target or int(mesh.size(mesh_dim)) <= 1:
            continue
        result = _convert_placement(
            result,
            source,
            target,
            mesh,
            mesh_dim,
            info.logical_shape,
            async_op,
        )
        if isinstance(info, _FlattenedTransformInfo):
            if info.avg_scale is not None and (
                isinstance(target, (Replicate, Partial)) or _is_shard_like(target)
            ):
                if isinstance(result, AsyncCollectiveTensor):
                    result = result.wait()
                result = result / info.avg_scale
        if not async_op and isinstance(result, AsyncCollectiveTensor):
            result = result.wait()
    return result


def _spec_for_dtensor(value: Any, dtype: Any = None) -> DTensorSpec:
    return DTensorSpec(
        mesh=value.device_mesh,
        placements=value.placements,
        tensor_meta=TensorMeta(
            shape=value.shape,
            stride=value.stride(),
            dtype=value.dtype if dtype is None else dtype,
        ),
    )


def _redistribute_backward(
    grad_output: Any,
    current_spec: DTensorSpec,
    previous_spec: DTensorSpec,
    *,
    out_dtype: Any,
    op_dtype: Any,
    async_op: bool = False,
) -> tuple[Any, DTensorSpec]:
    local_tensor = grad_output
    if local_tensor.dtype != op_dtype:
        local_tensor = local_tensor.to(dtype=op_dtype)
        current_spec = DTensorSpec(
            mesh=current_spec.device_mesh,
            placements=current_spec.placements,
            tensor_meta=TensorMeta(
                shape=current_spec.shape,
                stride=current_spec.stride,
                dtype=op_dtype,
            ),
            use_strided_shard_as_shard_order=current_spec.use_strided_shard_as_shard_order,
        )
        previous_spec = DTensorSpec(
            mesh=previous_spec.device_mesh,
            placements=previous_spec.placements,
            tensor_meta=TensorMeta(
                shape=current_spec.shape,
                stride=current_spec.stride,
                dtype=op_dtype,
            ),
            use_strided_shard_as_shard_order=previous_spec.use_strided_shard_as_shard_order,
        )
    normalized = tuple(
        Replicate()
        if (_is_shard_like(source) or source.is_replicate()) and target.is_partial()
        else target
        for source, target in zip(current_spec.placements, previous_spec.placements)
    )
    previous_spec = DTensorSpec(
        mesh=previous_spec.device_mesh,
        placements=normalized,
        tensor_meta=previous_spec.tensor_meta,
        use_strided_shard_as_shard_order=previous_spec.use_strided_shard_as_shard_order,
    )
    result = redistribute_local_tensor(
        local_tensor, current_spec, previous_spec, async_op=async_op
    )
    if result.dtype != out_dtype:
        result = result.to(dtype=out_dtype)
    return result, DTensorSpec(
        mesh=previous_spec.device_mesh,
        placements=normalized,
        tensor_meta=TensorMeta(
            shape=current_spec.shape,
            stride=current_spec.stride,
            dtype=result.dtype,
        ),
        use_strided_shard_as_shard_order=previous_spec.use_strided_shard_as_shard_order,
    )


class _BackwardDtypeConfig(TypedDict):
    op_dtype: Any
    out_dtype: Any


class _DtypeConfig(TypedDict):
    op_dtype: Any
    out_dtype: Any
    backward_options: _BackwardDtypeConfig


class Redistribute(Function):
    @classmethod
    def apply(
        cls,
        input: Any,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool = False,
        dtype_config: _DtypeConfig | None = None,
    ) -> Any:
        input_dtype = input.dtype
        config = dtype_config or {
            "op_dtype": input_dtype,
            "out_dtype": input_dtype,
            "backward_options": {
                "op_dtype": input_dtype,
                "out_dtype": input_dtype,
            },
        }
        op_dtype = config["op_dtype"]
        out_dtype = config["out_dtype"]
        current_spec = _spec_for_dtensor(input)
        target_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=current_spec.tensor_meta,
        )
        local_input = input.to_local()
        if not is_grad_enabled() and bool(getattr(local_input, "requires_grad", False)):
            local_input = local_input.detach()
        local_tensor = super().apply(
            local_input,
            current_spec,
            target_spec,
            async_op,
            op_dtype,
            out_dtype,
            config["backward_options"]["op_dtype"],
            config["backward_options"]["out_dtype"],
        )
        return type(input)(
            local_tensor,
            device_mesh,
            target_spec.placements,
            shape=input.shape,
            stride=input.stride(),
            backward_dtype=config["backward_options"]["op_dtype"],
        )

    @staticmethod
    def forward(
        ctx: Any,
        local_tensor: Any,
        current_spec: DTensorSpec,
        target_spec: DTensorSpec,
        async_op: bool,
        op_dtype: Any,
        out_dtype: Any,
        bwd_op_dtype: Any,
        bwd_out_dtype: Any,
    ) -> Any:
        ctx.async_op = async_op
        ctx.bwd_op_dtype = bwd_op_dtype
        ctx.bwd_out_dtype = bwd_out_dtype
        if local_tensor.dtype != op_dtype:
            local_tensor = local_tensor.to(dtype=op_dtype)
            current_spec = DTensorSpec(
                mesh=current_spec.device_mesh,
                placements=current_spec.placements,
                tensor_meta=TensorMeta(
                    shape=current_spec.shape,
                    stride=current_spec.stride,
                    dtype=op_dtype,
                ),
                use_strided_shard_as_shard_order=current_spec.use_strided_shard_as_shard_order,
            )
        ctx.current_spec = current_spec
        if current_spec.placements != target_spec.placements:
            target_spec = DTensorSpec(
                mesh=target_spec.device_mesh,
                placements=target_spec.placements,
                tensor_meta=current_spec.tensor_meta,
                use_strided_shard_as_shard_order=target_spec.use_strided_shard_as_shard_order,
            )
            output = redistribute_local_tensor(
                local_tensor,
                current_spec,
                target_spec,
                async_op=async_op,
                is_explicit=True,
            )
        else:
            output = local_tensor
            target_spec = current_spec
        if output.dtype != out_dtype:
            output = output.to(dtype=out_dtype)
            target_spec = DTensorSpec(
                mesh=target_spec.device_mesh,
                placements=target_spec.placements,
                tensor_meta=TensorMeta(
                    shape=target_spec.shape,
                    stride=target_spec.stride,
                    dtype=out_dtype,
                ),
                use_strided_shard_as_shard_order=target_spec.use_strided_shard_as_shard_order,
            )
        ctx.target_spec = target_spec
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None, None, None, None, None, None]:
        output = NestedRedistribute.apply(
            grad_output,
            ctx.target_spec,
            ctx.current_spec,
            ctx.async_op,
            ctx.bwd_op_dtype,
            ctx.bwd_out_dtype,
        )
        return output, None, None, None, None, None, None, None


class NestedRedistribute(Function):
    @classmethod
    def apply(
        cls,
        grad_output: Any,
        current_spec: DTensorSpec,
        previous_spec: DTensorSpec,
        async_op: bool,
        op_dtype: Any,
        out_dtype: Any,
    ) -> Any:
        return super().apply(
            grad_output,
            current_spec,
            previous_spec,
            async_op,
            op_dtype,
            out_dtype,
        )

    @staticmethod
    def forward(
        ctx: Any,
        grad_output: Any,
        current_spec: DTensorSpec,
        previous_spec: DTensorSpec,
        async_op: bool,
        op_dtype: Any,
        out_dtype: Any,
    ) -> Any:
        ctx.async_op = async_op
        ctx.original_dtype = grad_output.dtype
        ctx.op_dtype = op_dtype
        ctx.current_spec = current_spec
        output, spec = _redistribute_backward(
            grad_output,
            current_spec,
            previous_spec,
            out_dtype=out_dtype,
            op_dtype=op_dtype,
            async_op=async_op,
        )
        ctx.previous_spec = spec
        return output

    @staticmethod
    def backward(ctx: Any, grad2_output: Any) -> tuple[Any, None, None, None, None, None]:
        output = NestedRedistribute.apply(
            grad2_output,
            ctx.previous_spec,
            ctx.current_spec,
            ctx.async_op,
            ctx.op_dtype,
            ctx.original_dtype,
        )
        return output, None, None, None, None, None
