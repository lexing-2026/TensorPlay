"""Shape maps and placement propagation for view-like operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from ..placement_types import Replicate, Shard, _is_shard_like
from .utils import normalize_dim, normalize_dims, prod

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
    "dim_flatten",
    "dim_movedim",
    "dim_transpose",
    "propagate_shape_and_sharding",
]


class ClaimedDim(NamedTuple):
    input_dim: int
    output_dim: int


@dataclass(frozen=True)
class DimSpec:
    def inputs(self) -> Iterable["DimSpec"]:
        return ()


@dataclass(frozen=True)
class Singleton(DimSpec):
    pass


@dataclass(frozen=True)
class InputDim(DimSpec):
    index: int


@dataclass(frozen=True, init=False)
class Broadcast(DimSpec):
    dim: DimSpec
    dim_size: int

    def __init__(
        self,
        dim: DimSpec | int | None = None,
        dim_size: int | None = None,
        *,
        index: int | None = None,
    ) -> None:
        if index is not None:
            if dim is not None:
                raise TypeError("Broadcast accepts either dim or index")
            dim = index
        if dim is None:
            raise TypeError("Broadcast requires an input dimension")
        if isinstance(dim, int):
            dim = InputDim(dim)
        if not isinstance(dim, DimSpec):
            raise TypeError("Broadcast dim must be a dimension specification")
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "dim_size", 1 if dim_size is None else int(dim_size))

    @property
    def index(self) -> int:
        if not isinstance(self.dim, InputDim):
            raise AttributeError("broadcast dimension is not a direct input dimension")
        return self.dim.index

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return cls(dim, dim_size)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.dim,)


@dataclass(frozen=True)
class NewDim(DimSpec):
    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        return Singleton() if int(size) == 1 else cls(int(size))


@dataclass(frozen=True, init=False)
class Repeat(DimSpec):
    input_dim: DimSpec
    times: int

    def __init__(
        self,
        input_dim: DimSpec | int | None = None,
        times: int | None = None,
        *,
        size: int | None = None,
    ) -> None:
        if size is not None:
            if input_dim is not None or times is not None:
                raise TypeError("Repeat accepts either input_dim/times or size")
            input_dim = Singleton()
            times = size
        if input_dim is None or times is None:
            raise TypeError("Repeat requires an input dimension and repeat count")
        if isinstance(input_dim, int):
            input_dim = InputDim(input_dim)
        if not isinstance(input_dim, DimSpec):
            raise TypeError("Repeat input_dim must be a dimension specification")
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "times", int(times))

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        times = int(times)
        if times == 1:
            return dim
        if isinstance(dim, Singleton):
            return Broadcast(dim, times)
        return cls(dim, times)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


@dataclass(frozen=True, init=False)
class Flatten(DimSpec):
    input_dims: tuple[DimSpec, ...]

    def __init__(
        self,
        input_dims: Sequence[DimSpec] | None = None,
        end: int | None = None,
        *,
        start: int | None = None,
    ) -> None:
        if start is not None:
            if input_dims is not None or end is None:
                raise TypeError("Flatten(start, end) requires both bounds")
            input_dims = tuple(InputDim(index) for index in range(start, end + 1))
        if input_dims is None:
            raise TypeError("Flatten requires input dimensions")
        values = tuple(input_dims)
        if not all(isinstance(value, DimSpec) for value in values):
            raise TypeError("Flatten input_dims must contain dimension specifications")
        object.__setattr__(self, "input_dims", values)

    @property
    def start(self) -> int:
        if not self.input_dims or not isinstance(self.input_dims[0], InputDim):
            raise AttributeError("flatten does not start at a direct input dimension")
        return self.input_dims[0].index

    @property
    def end(self) -> int:
        if not self.input_dims or not isinstance(self.input_dims[-1], InputDim):
            raise AttributeError("flatten does not end at a direct input dimension")
        return self.input_dims[-1].index

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        values = tuple(dims)
        if len(values) == 0:
            return Singleton()
        if len(values) == 1:
            return values[0]
        return cls(values)

    def inputs(self) -> Iterable[DimSpec]:
        return self.input_dims


@dataclass(frozen=True, init=False)
class Split(DimSpec):
    input_dim: DimSpec
    group_shape: tuple[int, ...]
    split_id: int

    def __init__(
        self,
        input_dim: DimSpec | int | None = None,
        group_shape: Sequence[int] | None = None,
        split_id: int | None = None,
        *,
        factors: Sequence[int] | None = None,
    ) -> None:
        if factors is not None:
            if input_dim is not None or group_shape is not None or split_id is not None:
                raise TypeError("Split accepts either input_dim/group_shape/split_id or factors")
            input_dim = Singleton()
            group_shape = tuple(int(value) for value in factors)
            split_id = 0
        if input_dim is None or group_shape is None or split_id is None:
            raise TypeError("Split requires an input dimension, group shape, and index")
        if isinstance(input_dim, int):
            input_dim = InputDim(input_dim)
        shape = tuple(int(value) for value in group_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("Split group_shape must contain positive sizes")
        if split_id < 0 or split_id >= len(shape):
            raise IndexError("Split index is outside group_shape")
        if not isinstance(input_dim, DimSpec):
            raise TypeError("Split input_dim must be a dimension specification")
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "group_shape", shape)
        object.__setattr__(self, "split_id", int(split_id))

    @property
    def factors(self) -> tuple[int, ...]:
        return self.group_shape

    @classmethod
    def new(cls, dim: DimSpec, group_shape: Sequence[int], index: int) -> DimSpec:
        shape = tuple(int(value) for value in group_shape)
        if len(shape) == 1:
            if index != 0:
                raise IndexError("single-factor split has index zero only")
            return dim
        if shape[index] == 1:
            return Singleton()
        non_singletons = [(value, old_index) for old_index, value in enumerate(shape) if value != 1]
        new_index = next(new for new, (_, old) in enumerate(non_singletons) if old == index)
        return cls(dim, tuple(value for value, _ in non_singletons), new_index)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


DimMap = tuple[DimSpec, ...]


def _normalize_sizes(sizes: Sequence[Any]) -> tuple[int, ...]:
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = sizes[0]
    if not sizes:
        return ()
    if not all(isinstance(value, int) for value in sizes):
        raise TypeError("sizes must contain integers")
    return tuple(int(value) for value in sizes)


def _dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(InputDim(i) for i in range(ndim))


def _dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    if ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    if ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    return tuple(InputDim(i) for i in range(ndim))


def _expand(input_shape: Sequence[int], shape: Sequence[int]) -> DimMap:
    if len(shape) < len(input_shape):
        raise ValueError("expanded shape cannot have fewer dimensions")
    padded = _dim_pad_left(len(input_shape), len(shape))
    result: list[DimSpec] = []
    for dim, desired in zip(padded, shape):
        if isinstance(dim, Singleton):
            if desired < 0:
                raise ValueError("new dimensions cannot use -1")
            result.append(NewDim.new(desired))
            continue
        assert isinstance(dim, InputDim)
        actual = int(input_shape[dim.index])
        if actual != 1 and desired not in (-1, actual):
            raise ValueError("expanded shape is incompatible with the input shape")
        result.append(dim if desired in (-1, 1, actual) else Broadcast.new(dim, desired))
    return tuple(result)


def dim_flatten(ndim: int, start_dim: int = 0, end_dim: int = -1) -> DimMap:
    if ndim == 0:
        return (Singleton(),)
    start = normalize_dim(start_dim, ndim)
    end = normalize_dim(end_dim, ndim)
    if start > end:
        raise ValueError("start_dim must not exceed end_dim")
    return (
        tuple(InputDim(index) for index in range(start))
        + (Flatten.new(tuple(InputDim(index) for index in range(start, end + 1))),)
        + tuple(InputDim(index) for index in range(end + 1, ndim))
    )


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


def dim_repeat(ndim: int, sizes: Sequence[int]) -> DimMap:
    sizes = _normalize_sizes(sizes)
    if len(sizes) < ndim:
        raise ValueError("repeat sizes cannot have fewer dimensions than the input")
    padding = len(sizes) - ndim
    return tuple(Repeat.new(Singleton(), size) for size in sizes[:padding]) + tuple(
        Repeat.new(InputDim(index), size) for index, size in enumerate(sizes[padding:])
    )


def _infer_size(total_size: int, sizes: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(int(value) for value in sizes)
    unknown = [index for index, value in enumerate(sizes) if value == -1]
    if len(unknown) > 1:
        raise ValueError("only one dimension can be inferred")
    known_product = prod(value for value in sizes if value != -1)
    if unknown:
        if known_product == 0 or total_size % known_product:
            raise ValueError("inferred dimension is not integral")
        inferred = total_size // known_product
        return tuple(inferred if value == -1 else value for value in sizes)
    if known_product != total_size:
        raise ValueError("view sizes do not match the number of elements")
    return sizes


def _view_groups(from_size: Sequence[int], to_size: Sequence[int]) -> DimMap:
    source = tuple(int(value) for value in from_size)
    target = _infer_size(prod(source), to_size)
    if prod(source) != prod(target):
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
            target_group.append(target[target_index])
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
                    target_group.append(target[target_index])
                    target_index += 1
        if target_group:
            flattened = Flatten.new(tuple(InputDim(index) for index in source_group))
            result.extend(
                Split.new(flattened, tuple(target_group), index)
                for index in range(len(target_group))
            )
    return tuple(result)


def dim_tile(ndim: int, dims: Sequence[int]) -> DimMap:
    dims = _normalize_sizes(dims)
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    first, second = normalize_dim(dim1, ndim), normalize_dim(dim2, ndim)
    result = [InputDim(index) for index in range(ndim)]
    result[first], result[second] = result[second], result[first]
    return tuple(result)


def dim_squeeze(shape: Sequence[int], dim: int | None = None) -> DimMap:
    if dim is not None:
        selected = normalize_dim(dim, len(shape))
        if int(shape[selected]) != 1:
            return tuple(InputDim(index) for index in range(len(shape)))
        return tuple(InputDim(index) for index in range(len(shape)) if index != selected)
    return tuple(InputDim(index) for index, size in enumerate(shape) if int(size) != 1)


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    position = int(dim)
    if position < 0:
        position += ndim + 1
    if position < 0 or position > ndim:
        raise IndexError("unsqueeze dimension is outside the output rank")
    values = tuple(InputDim(index) for index in range(ndim))
    return values[:position] + (Singleton(),) + values[position:]


def dim_view_as_real(shape: Sequence[int]) -> DimMap:
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("view_as_real requires at least one dimension")
    values = [InputDim(index) for index in range(ndim - 1)]
    values.extend(
        (
            Split(InputDim(ndim - 1), (int(shape[-1]), 2), 0),
            Split(InputDim(ndim - 1), (int(shape[-1]), 2), 1),
        )
    )
    return tuple(values)


def dim_reduction(
    ndim: int, dims: int | Sequence[int] | None, keepdim: bool
) -> DimMap:
    reduced = set(normalize_dims(dims, ndim))
    return tuple(
        InputDim(index) if index not in reduced else Singleton()
        for index in range(ndim)
        if index not in reduced or keepdim
    )


def _collect_input_dims(spec: DimSpec, result: set[int]) -> None:
    if isinstance(spec, InputDim):
        result.add(spec.index)
    for child in spec.inputs():
        _collect_input_dims(child, result)


def _input_dim_for_sharding(
    spec: DimSpec,
    shardable_dims: dict[int, list[bool]],
    mesh_sizes: Sequence[int],
    input_placements: Sequence[Any],
) -> InputDim | None:
    if isinstance(spec, InputDim):
        return spec
    if isinstance(spec, Flatten):
        for child in spec.input_dims[1:]:
            if isinstance(child, InputDim):
                shardable_dims[child.index] = [False] * len(mesh_sizes)
        first = spec.input_dims[0] if spec.input_dims else None
        return first if isinstance(first, InputDim) else None
    if isinstance(spec, Split):
        input_dim = _input_dim_for_sharding(
            spec.input_dim, shardable_dims, mesh_sizes, input_placements
        )
        if spec.split_id == 0 and input_dim is not None:
            output_size = spec.group_shape[spec.split_id]
            shardable_dims[input_dim.index] = [
                output_size % int(size) == 0 for size in mesh_sizes
            ]
            submesh_size = 1
            for size, placement in zip(mesh_sizes, input_placements):
                if isinstance(placement, Shard) and placement.dim == input_dim.index:
                    submesh_size *= int(size)
            if output_size % submesh_size:
                raise ValueError("split output is not divisible by the current shard layout")
        return input_dim if spec.split_id == 0 else None
    if isinstance(spec, Repeat):
        input_dim = _input_dim_for_sharding(
            spec.input_dim, shardable_dims, mesh_sizes, input_placements
        )
        if input_dim is not None:
            shardable_dims[input_dim.index] = [False] * len(mesh_sizes)
        return None
    return None


def _propagate_layout(
    input_src_placements: Sequence[Any],
    local_in_shape: Sequence[int],
    rule: Sequence[DimSpec],
    mesh_sizes: Sequence[int],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    placements = tuple(input_src_placements)
    sizes = tuple(int(value) for value in mesh_sizes)
    shape = tuple(int(value) for value in local_in_shape)
    if len(placements) != len(sizes):
        raise ValueError("placement and mesh dimensions must have equal length")
    used_inputs: set[int] = set()
    for spec in rule:
        _collect_input_dims(spec, used_inputs)
    shardable = {
        index: [index in used_inputs] * len(sizes) for index in range(len(shape))
    }
    shard_map: dict[int, int] = {}
    for output_dim, spec in enumerate(rule):
        input_dim = _input_dim_for_sharding(spec, shardable, sizes, placements)
        if input_dim is not None:
            shard_map[input_dim.index] = output_dim
    input_target = tuple(
        Replicate()
        if isinstance(placement, Shard)
        and (
            placement.dim not in shardable
            or not all(shardable[placement.dim])
        )
        else placement
        for placement in placements
    )
    output: list[Any] = []
    for placement in input_target:
        if _is_shard_like(placement):
            output.append(
                Shard(shard_map[placement.dim])
                if placement.dim in shard_map
                else Replicate()
            )
        else:
            output.append(placement)
    return input_target, tuple(output)


def propagate_shape_and_sharding(
    first: Any = None, second: Any = None, *args: Any, **kwargs: Any
) -> Any:
    if "input_src_placements" in kwargs:
        if first is not None or second is not None or args:
            raise TypeError("layout propagation arguments were provided twice")
        first = kwargs.pop("input_src_placements")
        second = kwargs.pop("local_in_shape")
        args = (kwargs.pop("rule"), kwargs.pop("mesh_sizes"))
        if kwargs:
            raise TypeError("unexpected layout propagation arguments")
    if first is None or second is None:
        raise TypeError("layout propagation requires two leading arguments")
    if isinstance(first, DTensor):
        value = first
        operation: Callable[..., Any] = second
        result = operation(value.to_local(), *args, **kwargs)
        if hasattr(result, "shape"):
            shape = tuple(int(size) for size in result.shape)
            stride = tuple(int(size) for size in result.stride())
            meta = TensorMeta(shape, stride, result.dtype)
        else:
            meta = None
        return DTensorSpec(value.device_mesh, value.placements, meta)
    if len(args) != 2 or kwargs:
        raise TypeError("layout propagation requires placements, shape, rule, and mesh sizes")
    rule = tuple(
        value if isinstance(value, DimSpec) else InputDim(int(value))
        for value in args[0]
    )
    return _propagate_layout(first, second, rule, args[1])
