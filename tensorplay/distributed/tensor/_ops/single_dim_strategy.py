"""Single-mesh-dimension strategy expansion and registration."""

from __future__ import annotations

import functools
import heapq
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any, TypeAlias, TypeVar, cast

from .._collective_utils import MeshTopoInfo, _compute_placement_transition_cost
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from ..placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
)

logger = logging.getLogger(__name__)

_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_ShardingPlaceholderT = TypeVar("_ShardingPlaceholderT", bound="_ShardingPlaceholder")
_SingleDimStrategyFunc: TypeAlias = Callable[
    [Any, tuple[Any, ...], dict[str, Any]],
    list[list[Placement | _ShardingPlaceholderT | None]],
]
_ExpandedSingleDimStrategyFunc: TypeAlias = Callable[
    [Any, tuple[Any, ...], dict[str, Any]], _StrategyTypeT
]
_FullMeshStrategyFilter: TypeAlias = Callable[
    [Any, OpSchema, list[DTensorSpec], DTensorSpec | tuple[DTensorSpec | None, ...]],
    bool,
]

__all__ = [
    "PreparedSingleDimStrategy",
    "_PreparedSingleDimStrategy",
    "_ShardingPlaceholder",
    "_expand_single_dim_strategy_to_mesh",
    "_dijkstra_expand_single_dim_strategy_to_mesh",
    "register_single_dim_strategy",
]


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def _is_sharding(value: Placement | None) -> bool:
    return isinstance(value, (Shard, _StridedShard))


def _operation_name(operation: Any) -> str:
    value = getattr(operation, "name", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(operation, "__name__", operation)
    return str(value).rsplit(".", 1)[-1]


def _base_operation_name(operation: Any) -> str:
    return _operation_name(operation).split("::")[-1].split(".", 1)[0]


def _strategy_output_specs(value: OpStrategy) -> list[DTensorSpec]:
    result: list[DTensorSpec] = []
    for strategy in value.strategies:
        outputs = strategy.output_specs
        if isinstance(outputs, DTensorSpec):
            result.append(outputs)
        elif isinstance(outputs, (tuple, list)):
            result.extend(item for item in outputs if isinstance(item, DTensorSpec))
    return result


class _ShardingPlaceholder:
    """A placement marker whose tensor dimension is filled in later."""

    def __init__(self, dim: int):
        if type(dim) is not int:
            raise TypeError("sharding dimension must be an integer")
        self.dim = dim

    def __repr__(self) -> str:
        return f"_ShardingPlaceholder(dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ShardingPlaceholder) and self.dim == other.dim

    def __hash__(self) -> int:
        return hash((type(self), self.dim))


@dataclass
class _SingleDimStrategyInfo:
    func: _SingleDimStrategyFunc
    allow_unbacked_sharding: bool | None = None
    allow_uneven_sharding: bool = False
    full_mesh_strategy_filter: _FullMeshStrategyFilter | None = None
    different_mesh_args: list[int] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def _insert_single_dim_replication_strategy(
    strategies: list[list[Placement | _ShardingPlaceholder | None]],
    num_outputs: int,
    num_inputs: int,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    for strategy in strategies:
        if all(isinstance(value, Replicate) or value is None for value in strategy):
            return strategies

    total = num_outputs + num_inputs
    replicate_rule: list[Placement | None] = [Replicate()] * total
    if isinstance(output_tensor_meta, Sequence) and not isinstance(
        output_tensor_meta, TensorMeta
    ):
        for index, meta in enumerate(output_tensor_meta):
            if meta is None and index < num_outputs:
                replicate_rule[index] = None
    for index in range(num_outputs):
        if strategies and all(strategy[index] is None for strategy in strategies):
            replicate_rule[index] = None
    strategies.insert(0, replicate_rule)
    return strategies


def _fill_single_dim_strategy_placeholders(
    unique_input_placements: set[Placement],
    strategies: list[list[Placement | _ShardingPlaceholder | None]],
) -> list[list[Placement | None]]:
    shard_builders: dict[str, Callable[[int], Placement]] = {}
    for placement in unique_input_placements:
        if isinstance(placement, _StridedShard):
            key = f"StridedShard({placement.split_factor})"
            shard_builders.setdefault(
                key,
                functools.partial(_StridedShard, split_factor=placement.split_factor),
            )
        elif isinstance(placement, Shard):
            shard_builders.setdefault("Shard", Shard)

    expanded: list[list[Placement | None]] = []
    for strategy in strategies:
        if any(isinstance(value, _ShardingPlaceholder) for value in strategy):
            for builder in shard_builders.values():
                expanded.append(
                    [
                        builder(value.dim)
                        if isinstance(value, _ShardingPlaceholder)
                        else cast(Placement | None, value)
                        for value in strategy
                    ]
                )
        else:
            if not all(isinstance(value, Placement) or value is None for value in strategy):
                raise TypeError(f"invalid single-dimension strategy {strategy!r}")
            expanded.append(cast(list[Placement | None], strategy))
    return expanded


def _get_unique_placements(op_schema: OpSchema) -> set[Placement]:
    result: set[Placement] = set()

    def visit(value: Any) -> None:
        if isinstance(value, DTensorSpec):
            result.update(value.placements)
        elif isinstance(value, OpStrategy):
            outputs = _strategy_output_specs(value)
            if outputs:
                result.update(outputs[0].placements)
        elif isinstance(value, TupleStrategy):
            for child in value.children:
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    for value in op_schema.args_schema:
        visit(value)
    for value in op_schema.kwargs_schema.values():
        visit(value)
    return result


def _get_num_tensor_inputs(op_schema: OpSchema) -> int:
    def count_inputs(value: Any) -> int:
        if isinstance(value, OpStrategy):
            return 1
        if isinstance(value, TupleStrategy):
            return sum(count_inputs(child) for child in value.children)
        if isinstance(value, (tuple, list)):
            return sum(count_inputs(child) for child in value)
        if isinstance(value, dict):
            return sum(count_inputs(child) for child in value.values())
        return 0

    return sum(count_inputs(value) for value in op_schema.args_schema) + sum(
        count_inputs(value) for value in op_schema.kwargs_schema.values()
    )


def _output_count_from_schema(op_schema: OpSchema) -> int | None:
    returns = getattr(getattr(op_schema.op, "_schema", None), "returns", ())
    if not returns:
        return None
    count_value = 0
    for value in returns:
        type_value = getattr(value, "type", value)
        if "tensor" in str(type_value).lower():
            count_value += 1
    return count_value


def _build_output_specs(
    mesh: Any,
    per_mesh_dim_placements: list[tuple[Placement | None, ...]],
    num_outputs: int,
    output_metas: tuple[TensorMeta | None, ...],
) -> DTensorSpec | tuple[DTensorSpec | None, ...]:
    if num_outputs <= 0:
        raise ValueError("an output spec requires at least one output")
    if len(output_metas) != num_outputs:
        raise ValueError(
            f"expected {num_outputs} output metadata records, got {len(output_metas)}"
        )

    def build(index: int) -> DTensorSpec | None:
        placements = tuple(values[index] for values in per_mesh_dim_placements)
        if all(value is None for value in placements):
            return None
        if any(value is None for value in placements):
            raise ValueError("a strategy cannot mix missing and concrete mesh layouts")
        if output_metas[index] is None:
            return None
        return DTensorSpec(
            mesh,
            tuple(cast(Placement, value) for value in placements),
            tensor_meta=output_metas[index],
        )

    if num_outputs == 1:
        result = build(0)
        if result is None:
            raise ValueError("a single-output strategy must produce a tensor spec")
        return result
    return tuple(build(index) for index in range(num_outputs))


class _PreparedSingleDimStrategy:
    """Materialize one-dimensional rules for one operation schema."""

    def __init__(
        self,
        strategy_fn: _SingleDimStrategyInfo | _SingleDimStrategyFunc,
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
        num_inputs: int | None = None,
    ) -> None:
        if isinstance(strategy_fn, _SingleDimStrategyInfo):
            self.allow_unbacked_sharding = strategy_fn.allow_unbacked_sharding
            self.allow_uneven_sharding = strategy_fn.allow_uneven_sharding
            self.full_mesh_strategy_filter = strategy_fn.full_mesh_strategy_filter
            different_mesh_args = strategy_fn.different_mesh_args
            function = strategy_fn.func
        else:
            self.allow_unbacked_sharding = None
            self.allow_uneven_sharding = False
            self.full_mesh_strategy_filter = None
            different_mesh_args = None
            function = strategy_fn

        self.element_mesh = None
        for value in op_schema.args_schema:
            if isinstance(value, OpStrategy) and value.strategies:
                self.element_mesh = value.mesh
                break
            if isinstance(value, TupleStrategy):
                for child in value.children:
                    if isinstance(child, OpStrategy) and child.strategies:
                        self.element_mesh = child.mesh
                        break
                if self.element_mesh is not None:
                    break

        if self.element_mesh is not None:
            allowed = set(different_mesh_args or ())
            for index, value in enumerate(op_schema.args_schema):
                if not isinstance(value, OpStrategy) or index in allowed:
                    continue
                if value.mesh != self.element_mesh:
                    raise ValueError(
                        f"operation inputs use different meshes: {self.element_mesh!r} and {value.mesh!r}"
                    )

        self.remapped_different_mesh_args: list[int] | None = None
        if different_mesh_args is not None:
            schema_to_strategy: dict[int, int] = {}
            strategy_index = 0
            for schema_index, value in enumerate(op_schema.args_schema):
                if isinstance(value, OpStrategy):
                    schema_to_strategy[schema_index] = strategy_index
                    strategy_index += 1
            self.remapped_different_mesh_args = [
                schema_to_strategy[index]
                for index in different_mesh_args
                if index in schema_to_strategy
            ]

        if num_inputs is None:
            num_inputs = _get_num_tensor_inputs(op_schema)
        self.num_inputs = num_inputs
        raw_strategies = function(
            op_schema.op, op_schema.args_meta, op_schema.kwargs_meta
        )
        strategies = cast(
            list[list[Placement | _ShardingPlaceholder | None]], raw_strategies
        )

        if strategies:
            if isinstance(output_tensor_meta, Sequence) and not isinstance(
                output_tensor_meta, TensorMeta
            ):
                schema_outputs = len(output_tensor_meta)
            else:
                schema_outputs = _output_count_from_schema(op_schema)
                if schema_outputs is None:
                    schema_outputs = len(strategies[0]) - num_inputs
            expected = schema_outputs + num_inputs
            if len(strategies[0]) != expected:
                raise ValueError(
                    f"strategy length {len(strategies[0])} does not match expected {expected}"
                )

        if strategies:
            num_outputs = len(strategies[0]) - num_inputs
        elif output_tensor_meta is None:
            num_outputs = 0
        elif isinstance(output_tensor_meta, TensorMeta):
            num_outputs = 1
        else:
            num_outputs = len(output_tensor_meta)
        self.num_outputs = num_outputs

        strategies = _insert_single_dim_replication_strategy(
            strategies, num_outputs, num_inputs, output_tensor_meta
        )
        self.expanded_strategies = _fill_single_dim_strategy_placeholders(
            _get_unique_placements(op_schema), strategies
        )
        self.strategy_lookup: dict[
            tuple[Placement | None, ...], tuple[Placement | None, ...]
        ] = {}
        for strategy in self.expanded_strategies:
            key = tuple(strategy[num_outputs:])
            self.strategy_lookup.setdefault(key, tuple(strategy[:num_outputs]))

        self.allowed_sharding_per_input: dict[int, set[Shard | _StridedShard]] = defaultdict(set)
        self.allowed_partial_per_input: dict[int, set[Placement]] = defaultdict(set)
        for strategy in self.expanded_strategies:
            for index in range(num_inputs):
                placement = strategy[num_outputs + index]
                if _is_sharding(placement):
                    self.allowed_sharding_per_input[index].add(cast(Shard, placement))
                elif isinstance(placement, Partial):
                    self.allowed_partial_per_input[index].add(placement)

        if output_tensor_meta is None:
            self.output_metas = (None,) * num_outputs
        elif isinstance(output_tensor_meta, TensorMeta):
            self.output_metas = (output_tensor_meta,)
        else:
            self.output_metas = tuple(output_tensor_meta)

    def try_propagate(
        self,
        mesh: Any,
        input_placements: tuple[tuple[Placement, ...], ...],
        input_specs: list[DTensorSpec],
    ) -> OpStrategy | None:
        selected: list[tuple[Placement | None, ...]] = []
        if len(input_placements) != self.num_inputs:
            return None
        for mesh_dim in range(_mesh_ndim(mesh)):
            key = tuple(placements[mesh_dim] for placements in input_placements)
            output = self.strategy_lookup.get(key)
            if output is None:
                return None
            selected.append(output)

        from .utils import is_tensor_shardable

        candidate_inputs = [
            DTensorSpec(mesh, placements, tensor_meta=spec.tensor_meta)
            for placements, spec in zip(input_placements, input_specs)
        ]
        for candidate, original in zip(candidate_inputs, input_specs):
            if candidate.tensor_meta is None:
                continue
            if is_tensor_shardable(
                candidate.shape,
                candidate,
                allow_unbacked_sharding=self.allow_unbacked_sharding,
            ):
                continue
            if self.allow_uneven_sharding and candidate.placements == original.placements:
                continue
            return None

        output_spec = (
            _build_output_specs(mesh, selected, self.num_outputs, self.output_metas)
            if self.num_outputs
            else None
        )
        return OpStrategy(
            [
                OpSpec(
                    output_specs=output_spec,
                    input_specs=candidate_inputs,
                    redistribute_cost=[[0.0] for _ in input_specs],
                )
            ]
        )


PreparedSingleDimStrategy = _PreparedSingleDimStrategy


def _replace_tuple_strategy(value: Any, index: int) -> Any:
    if isinstance(value, TupleStrategy):
        return value.children[index]
    if isinstance(value, tuple):
        return tuple(_replace_tuple_strategy(item, index) for item in value)
    if isinstance(value, list):
        return [_replace_tuple_strategy(item, index) for item in value]
    if isinstance(value, dict):
        return {key: _replace_tuple_strategy(item, index) for key, item in value.items()}
    return value


def _tuple_strategy_length(op_schema: OpSchema) -> int | None:
    length: int | None = None

    def visit(value: Any) -> None:
        nonlocal length
        if isinstance(value, TupleStrategy):
            if length is None:
                length = len(value.children)
            elif length != len(value.children):
                raise ValueError("tuple strategy inputs must have equal lengths")
            for child in value.children:
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    visit(op_schema.args_schema)
    visit(op_schema.kwargs_schema)
    return length


def _expand_single_dim_strategy_to_mesh(
    mesh: Any,
    op_schema: OpSchema,
    strategy_info: _SingleDimStrategyInfo,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
) -> _ExpandedSingleDimStrategyFunc:
    from .utils import expand_to_full_mesh_op_strategy

    def create(
        schema: OpSchema,
        meta: TensorMeta | Sequence[TensorMeta | None] | None,
    ) -> Callable[[Any, tuple[Any, ...], dict[str, Any]], StrategyType]:
        def expanded(
            operation: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
        ) -> StrategyType:
            del args_schema, kwargs_schema
            prepared = _PreparedSingleDimStrategy(strategy_info, schema, meta)
            full_filter = None
            if prepared.full_mesh_strategy_filter is not None:

                def full_filter(
                    input_specs: list[DTensorSpec],
                    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
                ) -> bool:
                    callback = prepared.full_mesh_strategy_filter
                    if callback is None:
                        return True
                    return callback(
                        prepared.element_mesh or mesh, schema, input_specs, output_specs
                    )

            return expand_to_full_mesh_op_strategy(
                prepared.element_mesh or mesh,
                schema,
                prepared.expanded_strategies,
                output_tensor_meta=meta,
                input_index=prepared.num_outputs,
                inplace_op=_base_operation_name(operation).endswith("_"),
                allow_unbacked_sharding=prepared.allow_unbacked_sharding,
                allow_uneven_sharding=prepared.allow_uneven_sharding,
                full_mesh_strategy_filter=full_filter,
                different_mesh_args=prepared.remapped_different_mesh_args,
            )

        return expanded

    cache = functools.lru_cache(maxsize=None)(create)

    def create_cached(
        schema: OpSchema,
        meta: TensorMeta | Sequence[TensorMeta | None] | None,
    ) -> Callable[[Any, tuple[Any, ...], dict[str, Any]], StrategyType]:
        try:
            return cache(schema, meta)
        except TypeError:
            return create(schema, meta)

    tuple_length = _tuple_strategy_length(op_schema)
    operation_name = _base_operation_name(op_schema.op)
    if tuple_length is None or not operation_name.startswith(
        ("_foreach_", "_amp_foreach_", "_fused_")
    ):
        return create_cached(op_schema, output_tensor_meta)

    def expanded_foreach(
        operation: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
    ) -> StrategyType:
        del args_schema, kwargs_schema
        children: list[StrategyType] = []
        for index in range(tuple_length):
            child_schema = OpSchema(
                op_schema.op,
                args_schema=_replace_tuple_strategy(op_schema.args_schema, index),
                kwargs_schema=_replace_tuple_strategy(op_schema.kwargs_schema, index),
                schema_info=op_schema.schema_info,
            )
            child_meta: TensorMeta | None = None
            if isinstance(output_tensor_meta, Sequence) and not isinstance(
                output_tensor_meta, TensorMeta
            ):
                if index < len(output_tensor_meta):
                    child_meta = output_tensor_meta[index]
            child = create_cached(child_schema, child_meta)(
                operation, child_schema.args_meta, child_schema.kwargs_meta
            )
            children.append(child)
        return TupleStrategy(children)

    return expanded_foreach


@dataclass(order=True)
class _PQEntry:
    cost: float
    counter: int
    placements: tuple[tuple[Placement, ...], ...] = field(compare=False)
    transitions: list[tuple[int, int, Placement, Placement]] = field(compare=False)
    per_input_costs: tuple[float, ...] = field(compare=False)
    per_input_comm_bytes_gb: tuple[float, ...] = field(compare=False)


def _get_neighbor_placements(
    allowed_sharding: set[Shard | _StridedShard],
    allowed_partial: set[Placement],
    current: Placement,
    input_placements: tuple[Placement, ...],
    mesh_dim: int,
) -> list[Placement]:
    right_shards = {
        placement.dim
        for placement in input_placements[mesh_dim + 1 :]
        if _is_sharding(placement)
    }
    neighbors: list[Placement] = []
    if isinstance(current, Replicate):
        neighbors.extend(value for value in allowed_sharding if value.dim not in right_shards)
        neighbors.extend(allowed_partial)
    elif _is_sharding(current):
        if current.dim not in right_shards:
            neighbors.append(Replicate())
            neighbors.extend(
                value
                for value in allowed_sharding
                if value != current and value.dim not in right_shards
            )
    elif isinstance(current, Partial):
        neighbors.append(Replicate())
        neighbors.extend(value for value in allowed_sharding if value.dim not in right_shards)
    return neighbors


def _dtype_itemsize(dtype: Any) -> int:
    value = getattr(dtype, "itemsize", None)
    if callable(value):
        value = value()
    if value is not None:
        return int(value)
    names = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "uint64": 8,
        "int64": 8,
        "float64": 8,
        "complex64": 8,
        "complex128": 16,
    }
    name = str(dtype).rsplit(".", 1)[-1].lower()
    if name not in names:
        raise TypeError(f"cannot estimate communication size for {dtype!r}")
    return names[name]


def _dijkstra_expand_single_dim_strategy_to_mesh(
    mesh: Any,
    op_schema: OpSchema,
    single_dim_strategy: _SingleDimStrategyInfo | _SingleDimStrategyFunc,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    _collect_all_matches: set[tuple[tuple[Placement, ...], ...]] | None = None,
) -> OpStrategy | None:
    input_specs: list[DTensorSpec] = []
    for value in op_schema.args_schema:
        if isinstance(value, OpStrategy):
            if len(value.strategies) != 1:
                return None
            input_specs.append(value.strategies[0].output_spec)
        elif isinstance(value, TupleStrategy):
            return None
    for value in op_schema.kwargs_schema.values():
        if isinstance(value, (OpStrategy, TupleStrategy)):
            return None
    if not input_specs:
        return None
    for spec in input_specs:
        if any(isinstance(value, _StridedShard) for value in spec.placements):
            return None
        if spec.tensor_meta is not None and any(
            type(value).__name__ in {"SymInt", "SymFloat", "SymBool"}
            for value in spec.tensor_meta.shape
        ):
            return None

    prepared = _PreparedSingleDimStrategy(
        single_dim_strategy, op_schema, output_tensor_meta, num_inputs=len(input_specs)
    )
    initial = tuple(spec.placements for spec in input_specs)
    fast = prepared.try_propagate(mesh, initial, input_specs)
    if fast is not None and _collect_all_matches is None:
        return fast
    first_result = fast
    if _collect_all_matches is not None and fast is not None:
        _collect_all_matches.add(initial)

    topo = MeshTopoInfo.build_from_mesh(mesh)
    initial_bytes: list[float] = []
    for spec in input_specs:
        if spec.tensor_meta is None:
            return None
        total = _dtype_itemsize(spec.tensor_meta.dtype) * math.prod(
            int(value) for value in spec.tensor_meta.shape
        )
        shard_count = 1
        for mesh_dim, placement in enumerate(spec.placements):
            if _is_sharding(placement):
                shard_count *= int(mesh.size(mesh_dim))
        initial_bytes.append(total / shard_count / (1024**3))

    queue: list[_PQEntry] = []
    visited: set[tuple[tuple[Placement, ...], ...]] = set()
    sequence = count()
    heapq.heappush(
        queue,
        _PQEntry(
            0.0,
            next(sequence),
            initial,
            [],
            (0.0,) * len(input_specs),
            tuple(initial_bytes),
        ),
    )

    def push(
        input_index: int,
        mesh_dim: int,
        target: Placement,
        source: _PQEntry,
    ) -> None:
        values = [list(item) for item in source.placements]
        current = values[input_index][mesh_dim]
        values[input_index][mesh_dim] = target
        candidate = tuple(tuple(item) for item in values)
        if candidate in visited:
            return
        original = initial[input_index][mesh_dim]
        net_cost, _ = _compute_placement_transition_cost(
            original, target, topo, mesh_dim, initial_bytes[input_index]
        )
        if math.isinf(net_cost):
            return
        step_cost, new_bytes = _compute_placement_transition_cost(
            current, target, topo, mesh_dim, source.per_input_comm_bytes_gb[input_index]
        )
        if math.isinf(step_cost):
            return
        per_input = list(source.per_input_costs)
        per_input[input_index] += step_cost
        comm = list(source.per_input_comm_bytes_gb)
        comm[input_index] = new_bytes
        heapq.heappush(
            queue,
            _PQEntry(
                sum(per_input),
                next(sequence),
                candidate,
                source.transitions + [(input_index, mesh_dim, current, target)],
                tuple(per_input),
                tuple(comm),
            ),
        )

    while queue:
        current = heapq.heappop(queue)
        if current.placements in visited:
            continue
        visited.add(current.placements)
        match = prepared.try_propagate(mesh, current.placements, input_specs)
        if match is not None:
            match_spec = match.strategies[0]
            result = OpStrategy(
                [
                    OpSpec(
                        output_specs=match_spec.output_specs,
                        input_specs=list(match_spec.input_specs or ()),
                        redistribute_cost=[[value] for value in current.per_input_costs],
                    )
                ]
            )
            if first_result is None:
                first_result = result
            if _collect_all_matches is None:
                return result
            _collect_all_matches.add(current.placements)
        for mesh_dim in range(_mesh_ndim(mesh)):
            for input_index, placements in enumerate(current.placements):
                for target in _get_neighbor_placements(
                    prepared.allowed_sharding_per_input[input_index],
                    prepared.allowed_partial_per_input[input_index],
                    placements[mesh_dim],
                    placements,
                    mesh_dim,
                ):
                    push(input_index, mesh_dim, target, current)

    if first_result is not None:
        return first_result
    logger.warning("no valid placement strategy found for %s", op_schema.op)
    return None


def register_single_dim_strategy(
    operation: Any,
    schema_info: RuntimeSchemaInfo | None = None,
    allow_unbacked_sharding: bool | None = None,
    allow_uneven_sharding: bool = False,
    full_mesh_strategy_filter: _FullMeshStrategyFilter | None = None,
    different_mesh_args: list[int] | None = None,
) -> Callable[[_SingleDimStrategyFunc], _SingleDimStrategyFunc]:
    operations = tuple(operation) if isinstance(operation, (list, tuple, set)) else (operation,)

    def register(function: _SingleDimStrategyFunc) -> _SingleDimStrategyFunc:
        from .._api import DTensor

        info = _SingleDimStrategyInfo(
            function,
            allow_unbacked_sharding=allow_unbacked_sharding,
            allow_uneven_sharding=allow_uneven_sharding,
            full_mesh_strategy_filter=full_mesh_strategy_filter,
            different_mesh_args=different_mesh_args,
        )
        dispatcher = getattr(DTensor, "_op_dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("distributed tensor dispatcher is not initialized")
        propagator = dispatcher.sharding_propagator
        for item in operations:
            propagator.register_single_dim_op_strategy(item, info, schema_info)
        return function

    return register
