"""Shape maps and placement propagation for view-like operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OpSpec, OpStrategy, RuntimeSchemaInfo
from ..placement_types import Partial, Replicate, Shard, _StridedShard, _is_shard_like
from .single_dim_strategy import _ShardingPlaceholder, register_single_dim_strategy
from .utils import normalize_dim, normalize_dims, prod
from .utils import generate_redistribute_costs

__all__ = [
    "Broadcast",
    "ClaimedDim",
    "DimSpec",
    "Flatten",
    "InputDim",
    "NewDim",
    "Repeat",
    "Singleton",
    "Split",
    "dim_atleast_3d",
    "dim_flatten",
    "dim_movedim",
    "dim_pad_left",
    "dim_reduction",
    "dim_repeat",
    "dim_squeeze",
    "dim_transpose",
    "dim_unsqueeze",
    "dim_view_as_real",
    "expand",
    "infer_size",
    "propagate_shape_and_sharding",
    "register_view_ops",
    "view_groups",
]


class ClaimedDim(NamedTuple):
    input_dim: int
    output_dim: int


@dataclass
class DimSpec:
    def inputs(self) -> Iterable["DimSpec"]:
        return ()


@dataclass
class Singleton(DimSpec):
    pass


@dataclass(eq=False)
class InputDim(DimSpec):
    input_dim: int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InputDim):
            return self.input_dim == other.input_dim
        if not isinstance(other, DimSpec):
            raise TypeError(
                f"cannot compare InputDim with {type(other).__name__}"
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((InputDim, self.input_dim))


@dataclass
class Broadcast(DimSpec):
    dim: DimSpec
    dim_size: int

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return cls(dim, dim_size)

    @property
    def input_dim(self) -> int:
        if not isinstance(self.dim, InputDim):
            raise AttributeError("broadcast dimension is not direct")
        return self.dim.input_dim

    def inputs(self) -> Iterable[DimSpec]:
        return (self.dim,)


@dataclass
class NewDim(DimSpec):
    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        return Singleton() if size == 1 else cls(size)


@dataclass
class Repeat(DimSpec):
    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        if times == 1:
            return dim
        return Broadcast(dim, times) if isinstance(dim, Singleton) else cls(dim, times)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


@dataclass
class Flatten(DimSpec):
    input_dims: Sequence[DimSpec]

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        if len(dims) == 0:
            return Singleton()
        return dims[0] if len(dims) == 1 else cls(dims)

    def inputs(self) -> Iterable[DimSpec]:
        return self.input_dims


@dataclass
class Split(DimSpec):
    input_dim: DimSpec
    group_shape: Sequence[int]
    split_id: int

    @classmethod
    def new(cls, dim: DimSpec, group_shape: Sequence[int], index: int) -> DimSpec:
        if len(group_shape) == 1:
            if index != 0:
                raise IndexError("split index must be zero")
            return dim
        if group_shape[index] == 1:
            return Singleton()
        group_mapping = list(
            enumerate(
                (size, old_index)
                for old_index, size in enumerate(group_shape)
                if size != 1
            )
        )
        new_index = next(new for new, (_, old) in group_mapping if old == index)
        return cls(dim, tuple(item[1][0] for item in group_mapping), new_index)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


DimMap = tuple[DimSpec, ...]
Shape = tuple[int, ...]


def _explicit_unbacked_hint(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_factor_matches(expected: object, actual: object) -> bool:
    if expected is None:
        return False
    try:
        return int(expected) == int(actual)
    except (TypeError, ValueError):
        return expected == actual


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(
        InputDim(index) for index in range(ndim)
    )


def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    if ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    if ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    return tuple(InputDim(index) for index in range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    if len(shape) < len(input_shape):
        raise ValueError("expanded shape cannot have fewer dimensions")
    padded_input = dim_pad_left(len(input_shape), len(shape))
    result: list[DimSpec] = []
    for dim, desired in zip(padded_input, shape):
        desired = int(desired)
        if isinstance(dim, Singleton):
            if desired < 0:
                raise ValueError("new dimensions cannot use -1")
            result.append(NewDim.new(desired))
            continue
        actual = int(input_shape[dim.input_dim])
        if actual != 1 and desired not in (-1, actual):
            raise ValueError("expanded shape is incompatible with the input shape")
        result.append(
            dim
            if desired in (-1, 1, actual)
            else Broadcast.new(dim, desired)
        )
    return tuple(result)


def normalize_sizes(sizes: Sequence[Any]) -> Shape:
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = sizes[0]
    return tuple(int(value) for value in sizes)


def dim_flatten(ndim: int, start_dim: int = 0, end_dim: int = -1) -> DimMap:
    if ndim == 0:
        return (Singleton(),)
    start = normalize_dim(start_dim, ndim)
    end = normalize_dim(end_dim, ndim)
    if start > end:
        raise ValueError("start dimension must not exceed end dimension")
    return tuple(InputDim(index) for index in range(start)) + (
        Flatten.new(tuple(InputDim(index) for index in range(start, end + 1))),
    ) + tuple(InputDim(index) for index in range(end + 1, ndim))


def dim_movedim(
    ndim: int,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> DimMap:
    sources = normalize_dims(source, ndim)
    destinations = normalize_dims(destination, ndim)
    if len(sources) != len(destinations):
        raise ValueError("source and destination must have equal length")
    result = [-1] * ndim
    for source_dim, destination_dim in zip(sources, destinations):
        if result[destination_dim] != -1:
            raise ValueError("destination dimensions must be unique")
        result[destination_dim] = source_dim
    unused = iter(index for index in range(ndim) if index not in sources)
    for index, value in enumerate(result):
        if value == -1:
            result[index] = next(unused)
    return tuple(InputDim(index) for index in result)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    sizes = normalize_sizes(sizes)
    if len(sizes) < ndim:
        raise ValueError("repeat sizes cannot have fewer dimensions than the input")
    padding = len(sizes) - ndim
    return tuple(Repeat.new(Singleton(), size) for size in sizes[:padding]) + tuple(
        Repeat.new(InputDim(index), size) for index, size in enumerate(sizes[padding:])
    )


def infer_size(total_size: int, sizes: Shape) -> Shape:
    sizes = tuple(int(value) for value in sizes)
    unknown = [index for index, value in enumerate(sizes) if value == -1]
    if len(unknown) > 1:
        raise ValueError("only one dimension can be inferred")
    known_product = math.prod(value for value in sizes if value != -1)
    if unknown:
        if known_product == 0 or total_size % known_product:
            raise ValueError("inferred dimension is not integral")
        inferred = total_size // known_product
        return tuple(inferred if value == -1 else value for value in sizes)
    if known_product != total_size:
        raise ValueError("view sizes do not match the number of elements")
    return sizes


def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    source = tuple(int(value) for value in from_size)
    target = infer_size(math.prod(source), normalize_sizes(to_size))
    if math.prod(source) != math.prod(target):
        raise ValueError("view sizes do not match")
    if any(value == 0 for value in source + target):
        if source == target:
            return tuple(InputDim(index) for index in range(len(source)))
        raise ValueError("zero-sized reshape cannot be represented by a dimension map")

    source_index = target_index = 0
    result: list[DimSpec] = []
    while source_index < len(source) or target_index < len(target):
        source_group: list[int] = []
        target_group: list[int] = []
        source_product = source[source_index] if source_index < len(source) else 1
        if source_index < len(source):
            source_group.append(source_index)
            source_index += 1
        target_product = target[target_index] if target_index < len(target) else 1
        if target_index < len(target):
            target_group.append(target_index)
            target_index += 1
        if source_product == 1 and target_product != 1:
            target_index -= 1
            target_group.clear()
        elif source_product != 1 and target_product == 1:
            source_index -= 1
            source_group.clear()
        else:
            while source_product != target_product:
                if source_product < target_product:
                    if source_index >= len(source):
                        raise ValueError("view sizes cannot be grouped")
                    source_product *= source[source_index]
                    source_group.append(source_index)
                    source_index += 1
                else:
                    if target_index >= len(target):
                        raise ValueError("view sizes cannot be grouped")
                    target_product *= target[target_index]
                    target_group.append(target_index)
                    target_index += 1
        if target_group:
            flattened = Flatten.new(tuple(InputDim(index) for index in source_group))
            target_shape = tuple(target[index] for index in target_group)
            result.extend(
                Split.new(flattened, target_shape, index)
                for index in range(len(target_shape))
            )
    return tuple(result)


def dim_tile(ndim: int, dims: Shape) -> DimMap:
    dims = normalize_sizes(dims)
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    first = normalize_dim(dim1, ndim)
    second = normalize_dim(dim2, ndim)
    values = [InputDim(index) for index in range(ndim)]
    values[first], values[second] = values[second], values[first]
    return tuple(values)


def dim_squeeze(shape: Shape, dim: int | Sequence[int] | None = None) -> DimMap:
    if dim is None:
        target_dims = set(range(len(shape)))
    elif isinstance(dim, int):
        target_dims = {normalize_dim(dim, len(shape))}
    else:
        target_dims = set(normalize_dims(dim, len(shape)))
    return tuple(
        InputDim(index)
        for index, size in enumerate(shape)
        if int(size) != 1 or index not in target_dims
    )


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    position = int(dim)
    if position < 0:
        position += ndim + 1
    if position < 0 or position > ndim:
        raise IndexError("unsqueeze dimension is outside the output rank")
    values = tuple(InputDim(index) for index in range(ndim))
    return values[:position] + (Singleton(),) + values[position:]


def dim_view_as_real(shape: Shape) -> DimMap:
    if not shape:
        raise ValueError("view_as_real requires at least one dimension")
    ndim = len(shape)
    return tuple(InputDim(index) for index in range(ndim - 1)) + (
        Split(InputDim(ndim - 1), (shape[-1], 2), 0),
        Split(InputDim(ndim - 1), (shape[-1], 2), 1),
    )


def dim_reduction(
    ndim: int, dims: int | Sequence[int] | None, keepdim: bool
) -> DimMap:
    reduced = set(normalize_dims(dims, ndim))
    return tuple(
        InputDim(index) if index not in reduced else Singleton()
        for index in range(ndim)
        if index not in reduced or keepdim
    )


def _view_group_rule(value: Any, *sizes: Any, **kwargs: Any) -> DimMap:
    target = sizes
    if not target:
        target = (kwargs.get("shape", kwargs.get("size")),)
    return view_groups(tuple(value.shape), normalize_sizes(target))


def _expand_rule(value: Any, *sizes: Any, **kwargs: Any) -> DimMap:
    target = sizes
    if target and isinstance(target[-1], bool):
        target = target[:-1]
    if not target:
        target = (kwargs.get("size", kwargs.get("shape")),)
    return expand(tuple(value.shape), normalize_sizes(target))


def _repeat_rule(value: Any, *sizes: Any, **kwargs: Any) -> DimMap:
    target = sizes or (kwargs.get("repeats", kwargs.get("size")),)
    return dim_repeat(value.ndim, normalize_sizes(target))


def propagate_shape_and_sharding(
    input_src_placements: Sequence[Any],
    global_input_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
    strict_view: bool = False,
) -> tuple[Sequence[Any], Sequence[Any]]:
    propagator = _ViewShardingPropagator(
        input_src_placements, global_input_shape, rule, mesh_sizes, strict_view
    )
    input_tgt_placements, input_to_output_tensor_dims = propagator.analyze()
    output_placements = propagator.rewrite_output_placements(
        input_tgt_placements, input_to_output_tensor_dims
    )
    return input_tgt_placements, output_placements


class _ViewShardingPropagator:
    def __init__(
        self,
        input_src_placements: Sequence[Any],
        global_input_shape: Shape,
        rule: DimMap,
        mesh_sizes: Shape,
        strict_view: bool,
    ) -> None:
        self.input_src_placements = tuple(input_src_placements)
        self.global_input_shape = tuple(global_input_shape)
        self.rule = tuple(rule)
        self.mesh_sizes = tuple(mesh_sizes)
        self.strict_view = strict_view
        self.mesh_ndim = len(self.mesh_sizes)
        if len(self.input_src_placements) != self.mesh_ndim:
            raise ValueError("placement and mesh dimensions must have equal length")
        self.shard_allowed: dict[int, list[bool]] = {}
        self.matched_strided_mesh_dims: set[int] = set()
        self.strict_replicate_fallback: set[tuple[int, int]] = set()

    def analyze(self) -> tuple[tuple[Any, ...], dict[int, list[int]]]:
        input_dims_in_rule = self._input_dims_in_rule(self.rule)
        for dim in range(len(self.global_input_shape)):
            self.shard_allowed[dim] = [
                dim in input_dims_in_rule
            ] * self.mesh_ndim

        input_to_output_tensor_dims: dict[int, list[int]] = {}
        for output_dim, command in enumerate(self.rule):
            input_dims = self._analyze_dim(command)
            if isinstance(command, Flatten):
                for input_dim in input_dims:
                    if input_dim.input_dim in input_to_output_tensor_dims:
                        raise AssertionError(
                            f"input dimension {input_dim.input_dim} was mapped twice"
                        )
                    input_to_output_tensor_dims[input_dim.input_dim] = [output_dim]
            elif input_dims:
                input_dim = input_dims[0].input_dim
                input_to_output_tensor_dims.setdefault(input_dim, []).append(output_dim)
            elif isinstance(command, Split):
                root_spec: DimSpec = command.input_dim
                while isinstance(root_spec, (Flatten, Split)):
                    root_spec = (
                        root_spec.input_dims[0]
                        if isinstance(root_spec, Flatten)
                        else root_spec.input_dim
                    )
                if isinstance(root_spec, InputDim) and root_spec.input_dim in input_to_output_tensor_dims:
                    input_to_output_tensor_dims[root_spec.input_dim].append(output_dim)

        input_target: list[Any] = []
        for mesh_dim, placement in enumerate(self.input_src_placements):
            if not _is_shard_like(placement):
                input_target.append(placement)
                continue
            allowed = self.shard_allowed.get(placement.dim, [False] * self.mesh_ndim)
            if allowed[mesh_dim]:
                input_target.append(placement)
                continue
            if self.strict_view and (placement.dim, mesh_dim) not in self.strict_replicate_fallback:
                raise RuntimeError(
                    f"sharded dimension {placement.dim} cannot be changed by this view"
                )
            input_target.append(Replicate())
        return tuple(input_target), input_to_output_tensor_dims

    @staticmethod
    def _input_dims_in_rule(rule: DimMap) -> set[int]:
        result: set[int] = set()

        def walk(command: DimSpec) -> None:
            if isinstance(command, InputDim):
                result.add(command.input_dim)
            for child in command.inputs():
                walk(child)

        for command in rule:
            walk(command)
        return result

    def _find_plain_shard(
        self, input_dim: InputDim
    ) -> tuple[int | None, Shard | None]:
        for mesh_dim, placement in enumerate(self.input_src_placements):
            if type(placement) is Shard and placement.dim == input_dim.input_dim:
                return mesh_dim, placement
        return None, None

    def _find_shard_for_split(
        self,
        current_dim: int,
        command: Split,
        placements: Sequence[Any],
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        for mesh_dim, placement in enumerate(placements):
            if not _is_shard_like(placement) or placement.dim != current_dim:
                continue
            if mesh_dim in self.matched_strided_mesh_dims:
                continue
            if isinstance(placement, _StridedShard):
                expected = self._expected_split_factor(
                    command, current_dim, mesh_dim, placements
                )
                if _split_factor_matches(expected, placement.split_factor):
                    return mesh_dim, placement
            elif type(placement) is Shard:
                return mesh_dim, placement
        return None, None

    def _analyze_flatten(self, command: Flatten) -> list[InputDim]:
        sharded_dims: list[InputDim] = []
        for index, dim in enumerate(command.input_dims):
            if not isinstance(dim, InputDim):
                raise AssertionError(f"flatten dimension must be InputDim, got {type(dim)}")
            mesh_dim, placement = self._find_plain_shard(dim)
            if mesh_dim is None or placement is None:
                continue
            tensor_dim_size = int(self.global_input_shape[placement.dim])
            mesh_dim_size = int(self.mesh_sizes[mesh_dim])
            can_shard = True
            if self.strict_view:
                if index != len(command.input_dims) - 1 and tensor_dim_size % mesh_dim_size:
                    raise RuntimeError(
                        f"dimension {dim.input_dim} is unevenly sharded for flatten"
                    )
                sharded_dims.append(dim)
            elif index == 0:
                sharded_dims.append(dim)
                can_shard = tensor_dim_size % mesh_dim_size == 0
            else:
                can_shard = False
            self.shard_allowed[dim.input_dim] = [can_shard] * self.mesh_ndim
        if sharded_dims:
            return sharded_dims
        first = command.input_dims[0]
        if not isinstance(first, InputDim):
            raise AssertionError(f"flatten dimension must be InputDim, got {type(first)}")
        return [first]

    def _analyze_split(self, command: Split) -> list[InputDim]:
        input_dims = self._analyze_dim(command.input_dim)
        if not input_dims:
            return []
        input_dim = input_dims[0]
        if isinstance(command.input_dim, Flatten):
            for flat_dim in command.input_dim.input_dims[1:]:
                if not isinstance(flat_dim, InputDim):
                    raise AssertionError(
                        f"flatten dimension must be InputDim, got {type(flat_dim)}"
                    )
                for mesh_dim, placement in enumerate(self.input_src_placements):
                    if type(placement) is Shard and placement.dim == flat_dim.input_dim:
                        self.shard_allowed[flat_dim.input_dim][mesh_dim] = False
                        self.strict_replicate_fallback.add(
                            (flat_dim.input_dim, mesh_dim)
                        )
        output_size = int(command.group_shape[command.split_id])
        mesh_dim, source_placement = self._find_shard_for_split(
            input_dim.input_dim, command, self.input_src_placements
        )
        if command.split_id == 0:
            self.shard_allowed[input_dim.input_dim] = [
                output_size % int(size) == 0 for size in self.mesh_sizes
            ]
            plain_mesh_dim, _ = self._find_plain_shard(input_dim)
            if self.strict_view and plain_mesh_dim is not None:
                if not self.shard_allowed[input_dim.input_dim][plain_mesh_dim]:
                    raise RuntimeError(
                        f"output dimension {command.split_id} is unevenly sharded"
                    )
        if mesh_dim is not None and isinstance(source_placement, _StridedShard):
            is_last = command.split_id == len(command.group_shape) - 1
            if self.strict_view and not is_last and output_size % int(self.mesh_sizes[mesh_dim]):
                raise RuntimeError(
                    f"output dimension {command.split_id} is unevenly sharded"
                )
            self.matched_strided_mesh_dims.add(mesh_dim)
            if input_dim.input_dim in self.shard_allowed:
                self.shard_allowed[input_dim.input_dim][mesh_dim] = (
                    output_size % int(self.mesh_sizes[mesh_dim]) == 0 or is_last
                )
        return [input_dim] if command.split_id == 0 else []

    def _analyze_dim(self, command: DimSpec) -> list[InputDim]:
        if isinstance(command, InputDim):
            return [command]
        if isinstance(command, Flatten):
            return self._analyze_flatten(command)
        if isinstance(command, Split):
            return self._analyze_split(command)
        if isinstance(command, Repeat):
            input_dims = self._analyze_dim(command.input_dim)
            for input_dim in input_dims:
                self.shard_allowed[input_dim.input_dim] = [False] * self.mesh_ndim
            return []
        return []

    @staticmethod
    def _is_last_shard_in_flatten_range(
        mesh_dim: int,
        placements: Sequence[Any],
        flatten_start: int,
        flatten_end: int,
    ) -> bool:
        placement = placements[mesh_dim]
        if not _is_shard_like(placement):
            raise AssertionError("flatten range requires a sharded placement")
        return not any(
            _is_shard_like(other)
            and flatten_start <= other.dim < flatten_end
            and other.dim >= placement.dim
            for other in placements[mesh_dim + 1 :]
        )

    def _expected_split_factor(
        self,
        command: Split,
        sharded_dim: int,
        mesh_dim: int,
        placements: Sequence[Any],
    ) -> int | None:
        factor = math.prod(command.group_shape[: command.split_id])
        for index in range(mesh_dim):
            placement = placements[index]
            if _is_shard_like(placement) and placement.dim == sharded_dim:
                size = int(self.mesh_sizes[index])
                if factor % size:
                    return None
                factor //= size
        return int(factor)

    def _find_keep_ss_dim(
        self,
        target_dims: list[int],
        placement: _StridedShard,
        mesh_dim: int,
    ) -> int | None:
        total_shard = int(self.mesh_sizes[mesh_dim]) * int(placement.split_factor)
        input_dim_size = int(self.global_input_shape[placement.dim])
        if input_dim_size % total_shard:
            return None
        shard_size = input_dim_size // total_shard
        for target_dim in target_dims:
            command = self.rule[target_dim]
            if not isinstance(command, Split):
                continue
            inner_size = math.prod(command.group_shape[command.split_id + 1 :])
            trailing_size = 1
            if isinstance(command.input_dim, Flatten):
                found = False
                for flat_dim in command.input_dim.input_dims:
                    if not isinstance(flat_dim, InputDim):
                        raise AssertionError("flatten dimension must be InputDim")
                    if flat_dim.input_dim == placement.dim:
                        found = True
                    elif found:
                        trailing_size *= int(self.global_input_shape[flat_dim.input_dim])
            flattened_shard_size = shard_size * trailing_size
            if flattened_shard_size >= inner_size and flattened_shard_size % inner_size == 0:
                return target_dim
        return None

    def _rewrite_plain_shard(
        self,
        placement: Shard,
        mesh_dim: int,
        placements: Sequence[Any],
        claimed: set[ClaimedDim],
        local_shapes: list[int],
        input_to_output: dict[int, list[int]],
    ) -> tuple[Any, list[int]]:
        target_dims = [
            value
            for value in input_to_output[placement.dim]
            if ClaimedDim(placement.dim, value) not in claimed
        ]
        if not target_dims:
            raise AssertionError("no output dimension is available for the shard")
        if len(target_dims) == 1:
            target_dim = target_dims[0]
        else:
            target_dim = next(
                (
                    value
                    for value in target_dims
                    if isinstance(self.rule[value], Split)
                    and self.rule[value].split_id == 0
                ),
                None,
            )
            if target_dim is None:
                raise AssertionError("split output has no leading dimension")
        command = self.rule[target_dim]
        if isinstance(command, (Split, InputDim)):
            updated = list(local_shapes)
            updated[placement.dim] //= int(self.mesh_sizes[mesh_dim])
            return Shard(target_dim), updated
        if not isinstance(command, Flatten):
            raise AssertionError("view rule must contain a flatten or split")
        first = command.input_dims[0]
        last = command.input_dims[-1]
        if not isinstance(first, InputDim) or not isinstance(last, InputDim):
            raise AssertionError("flatten boundaries must be input dimensions")
        input_start = first.input_dim
        if placement.dim == input_start:
            output_placement: Any = Shard(target_dim)
        else:
            split_factor = math.prod(local_shapes[input_start : placement.dim])
            output_placement = _StridedShard(target_dim, split_factor=int(split_factor))
        flatten_end = last.input_dim + 1
        if (
            local_shapes[placement.dim] % int(self.mesh_sizes[mesh_dim])
            and not self._is_last_shard_in_flatten_range(
                mesh_dim, placements, input_start, flatten_end
            )
        ):
            raise RuntimeError("uneven sharding cannot be represented by this flatten")
        updated = list(local_shapes)
        updated[placement.dim] //= int(self.mesh_sizes[mesh_dim])
        return output_placement, updated

    def _rewrite_strided_shard(
        self,
        placement: _StridedShard,
        mesh_dim: int,
        placements: Sequence[Any],
        claimed: set[ClaimedDim],
        local_shapes: list[int],
        input_to_output: dict[int, list[int]],
    ) -> tuple[Any, list[int]]:
        target_dims = [
            value
            for value in input_to_output[placement.dim]
            if ClaimedDim(placement.dim, value) not in claimed
        ]
        for target_dim in target_dims:
            command = self.rule[target_dim]
            if isinstance(command, Split):
                expected = self._expected_split_factor(
                    command, placement.dim, mesh_dim, placements
                )
                if _split_factor_matches(expected, placement.split_factor):
                    claimed.add(ClaimedDim(placement.dim, target_dim))
                    updated = list(local_shapes)
                    updated[placement.dim] //= int(self.mesh_sizes[mesh_dim])
                    return Shard(target_dim), updated
        target_dim = self._find_keep_ss_dim(target_dims, placement, mesh_dim)
        if target_dim is None:
            if self.strict_view and any(isinstance(self.rule[value], Split) for value in target_dims):
                raise RuntimeError("strided sharding does not match the view split")
            if not target_dims:
                raise AssertionError("no output dimension is available for the strided shard")
            target_dim = target_dims[0]
        updated = list(local_shapes)
        updated[placement.dim] //= int(self.mesh_sizes[mesh_dim])
        return _StridedShard(target_dim, split_factor=int(placement.split_factor)), updated

    def rewrite_output_placements(
        self,
        input_tgt_placements: Sequence[Any],
        input_to_output: dict[int, list[int]],
    ) -> list[Any]:
        claimed: set[ClaimedDim] = set()
        local_shapes = [int(value) for value in self.global_input_shape]
        output: list[Any] = []
        for mesh_dim, placement in enumerate(input_tgt_placements):
            if isinstance(placement, _StridedShard):
                value, local_shapes = self._rewrite_strided_shard(
                    placement,
                    mesh_dim,
                    input_tgt_placements,
                    claimed,
                    local_shapes,
                    input_to_output,
                )
            elif type(placement) is Shard:
                value, local_shapes = self._rewrite_plain_shard(
                    placement,
                    mesh_dim,
                    input_tgt_placements,
                    claimed,
                    local_shapes,
                    input_to_output,
                )
            else:
                value = placement
            output.append(value)
        return output


dim_maps: dict[str, Callable[..., DimMap]] = {
    "atleast_1d": lambda value: dim_pad_left(value.ndim, 1),
    "atleast_2d": lambda value: dim_pad_left(value.ndim, 2),
    "atleast_3d": lambda value: dim_atleast_3d(value.ndim),
    "broadcast_to": _expand_rule,
    "expand": _expand_rule,
    "expand_copy": _expand_rule,
    "flatten": lambda value, start_dim=0, end_dim=-1: dim_flatten(
        value.ndim, start_dim, end_dim
    ),
    "movedim": lambda value, source, destination: dim_movedim(
        value.ndim, source, destination
    ),
    "permute": lambda value, dims: tuple(
        InputDim(index) for index in normalize_dims(dims, value.ndim)
    ),
    "permute_copy": lambda value, dims: tuple(
        InputDim(index) for index in normalize_dims(dims, value.ndim)
    ),
    "ravel": lambda value: dim_flatten(value.ndim),
    "repeat": _repeat_rule,
    "reshape": lambda value, shape: view_groups(tuple(value.shape), normalize_sizes((shape,))),
    "reshape_as": lambda value, other: view_groups(tuple(value.shape), tuple(other.shape)),
    "squeeze": lambda value, dim=None: dim_squeeze(tuple(value.shape), dim),
    "squeeze_": lambda value, dim=None: dim_squeeze(tuple(value.shape), dim),
    "squeeze_copy": lambda value, dim=None: dim_squeeze(tuple(value.shape), dim),
    "squeeze_dims": lambda value, dims: dim_squeeze(tuple(value.shape), dims),
    "squeeze_dims_": lambda value, dims: dim_squeeze(tuple(value.shape), dims),
    "tile": lambda value, dims: dim_tile(value.ndim, normalize_sizes((dims,))),
    "transpose": lambda value, dim0, dim1: dim_transpose(value.ndim, dim0, dim1),
    "transpose_copy": lambda value, dim0, dim1: dim_transpose(value.ndim, dim0, dim1),
    "unsqueeze": lambda value, dim: dim_unsqueeze(value.ndim, dim),
    "unsqueeze_copy": lambda value, dim: dim_unsqueeze(value.ndim, dim),
    "view": _view_group_rule,
    "view_copy": _view_group_rule,
    "_unsafe_view": _view_group_rule,
    "view_as": lambda value, other: view_groups(tuple(value.shape), tuple(other.shape)),
    "view_as_real": lambda value: dim_view_as_real(tuple(value.shape)),
    "view_as_real_copy": lambda value: dim_view_as_real(tuple(value.shape)),
}


def _view_strategy(
    mesh: Any,
    op_schema: OpSchema,
    dim_map: Callable[..., DimMap],
    strict_view: bool,
) -> OpStrategy:
    input_strategy = op_schema.args_schema[0]
    if not isinstance(input_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(input_strategy)}")
    rules = dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)
    global_shape = input_strategy.shape
    output_strategy = OpStrategy([])
    for input_strategy_item in input_strategy.strategies:
        input_src_spec = input_strategy_item.output_spec
        input_target, output_placements = propagate_shape_and_sharding(
            input_src_spec.placements,
            tuple(global_shape),
            rules,
            tuple(int(mesh.size(index)) for index in range(len(input_src_spec.placements))),
            strict_view,
        )
        input_target_spec = DTensorSpec(
            mesh,
            tuple(input_target),
            tensor_meta=input_src_spec.tensor_meta,
            use_strided_shard_as_shard_order=False,
        )
        output_spec = DTensorSpec(
            mesh,
            tuple(output_placements),
            use_strided_shard_as_shard_order=False,
        )
        output_strategy.strategies.append(
            OpSpec(
                output_specs=output_spec,
                input_specs=(input_target_spec,),
                redistribute_cost=[
                    generate_redistribute_costs(input_strategy, input_target_spec)
                ],
            )
        )
    return output_strategy


def register_op_strategy_map(
    operation: str,
    local_op_name: str,
    schema_info: RuntimeSchemaInfo | None = None,
    strict_view: bool = False,
) -> None:
    dim_map = dim_maps[local_op_name]

    def strategy(mesh: Any, op_schema: OpSchema) -> OpStrategy:
        return _view_strategy(mesh, op_schema, dim_map, strict_view)

    DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
        operation, strategy, schema_info
    )


def view_as_complex_single_dim_strategy(
    _operation: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Any]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    if ndim == 0:
        raise ValueError("view_as_complex requires at least one dimension")
    strategies: list[list[Any]] = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(ndim - 1)
    ]
    strategies.extend([[Partial("sum"), Partial("sum")], [Partial("avg"), Partial("avg")]])
    return strategies


_VIEW_OPS_READY = False


def register_view_ops() -> None:
    global _VIEW_OPS_READY
    if _VIEW_OPS_READY:
        return
    _VIEW_OPS_READY = True
    entries = (
        (("squeeze", "squeeze_", "squeeze_copy"), "squeeze", None, True),
        (("squeeze_dims", "squeeze_dims_"), "squeeze_dims", RuntimeSchemaInfo(1), True),
        (("view",), "view", RuntimeSchemaInfo(1), True),
        (("view_copy",), "view_copy", RuntimeSchemaInfo(1), False),
        (("_unsafe_view",), "_unsafe_view", RuntimeSchemaInfo(1), True),
        (("reshape",), "reshape", RuntimeSchemaInfo(1), False),
        (("reshape_as",), "reshape_as", RuntimeSchemaInfo(1), False),
        (("view_as",), "view_as", RuntimeSchemaInfo(1), False),
        (("unsqueeze", "unsqueeze_copy"), "unsqueeze", RuntimeSchemaInfo(1), False),
        (("expand", "expand_copy"), "expand", RuntimeSchemaInfo(1), False),
        (("broadcast_to",), "broadcast_to", RuntimeSchemaInfo(1), False),
        (("permute", "permute_copy"), "permute", RuntimeSchemaInfo(1), False),
        (("repeat",), "repeat", RuntimeSchemaInfo(1), False),
        (("flatten",), "flatten", RuntimeSchemaInfo(1), False),
        (("ravel",), "ravel", None, False),
        (("atleast_1d",), "atleast_1d", None, False),
        (("atleast_2d",), "atleast_2d", None, False),
        (("atleast_3d",), "atleast_3d", None, False),
        (("tile",), "tile", RuntimeSchemaInfo(1), False),
        (("transpose", "transpose_copy"), "transpose", RuntimeSchemaInfo(1), False),
        (("view_as_real", "view_as_real_copy"), "view_as_real", None, False),
    )
    for operations, local_name, schema_info, strict_view in entries:
        for operation in operations:
            register_op_strategy_map(
                operation, local_name, schema_info=schema_info, strict_view=strict_view
            )
    DTensor._op_dispatcher.sharding_propagator.register_single_dim_op_strategy(
        "view_as_complex", view_as_complex_single_dim_strategy
    )
