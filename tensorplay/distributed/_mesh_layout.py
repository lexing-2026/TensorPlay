import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import NoReturn, overload

from ._pycute import (
    as_tuple,
    coalesce,
    complement,
    composition,
    flatten,
    IntTuple,
    is_int,
    is_tuple,
    Layout,
    make_layout,
    match_structure,
    suffix_product,
)

__all__ = ["_FlatLayout", "_MeshLayout"]


@dataclass(frozen=True)
class _FlatLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __init__(self, shape: IntTuple, stride: IntTuple | None = None) -> None:
        if not is_tuple(shape) and not is_int(shape):
            raise TypeError(f"shape must be a tuple or int, got {type(shape)}")
        actual_stride = suffix_product(shape) if stride is None else stride
        if not is_tuple(actual_stride) and not is_int(actual_stride):
            raise TypeError(f"stride must be a tuple or int, got {type(actual_stride)}")
        if not match_structure(shape, actual_stride):
            raise ValueError(f"sizes {shape} and strides {actual_stride} don't match")
        normalized = coalesce(Layout(shape, actual_stride))
        flat_shape = flatten(normalized.shape)
        flat_stride = flatten(normalized.stride)
        if flat_shape == (1,) and flat_stride == (0,):
            flat_shape = ()
            flat_stride = ()
        object.__setattr__(self, "shape", flat_shape)
        object.__setattr__(self, "stride", flat_stride)

    def __len__(self) -> NoReturn:
        raise RuntimeError("the internal layout does not expose a sequence length")

    def __getitem__(self, index: int) -> NoReturn:
        raise RuntimeError("the internal layout does not expose indexed elements")

    def to_pycute(self) -> Layout:
        return Layout(1, 0) if not self.shape else Layout(self.shape, self.stride)

    def numel(self) -> int:
        return math.prod(self.shape)

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        result = composition(self.to_pycute(), layout.to_pycute())
        axes = [
            _FlatLayout(shape, stride)
            for shape, stride in zip(as_tuple(result.shape), as_tuple(result.stride))
        ]
        return _MeshLayout(axes)

    def complement(self, world_size: int) -> "_FlatLayout":
        result = complement(self.to_pycute(), world_size)
        return _FlatLayout(result.shape, result.stride)

    def all_ranks_from_zero(self) -> list[int]:
        return [
            sum(coordinate * stride for coordinate, stride in zip(coordinates, self.stride))
            for coordinates in product(*(range(size) for size in self.shape))
        ]

    def global_ranks(self, world_size: int) -> list[list[int]]:
        return [
            [offset + rank for rank in self.all_ranks_from_zero()]
            for offset in self.complement(world_size).all_ranks_from_zero()
        ]

    def check_sorted(self) -> bool:
        return tuple(sorted(self.stride, reverse=True)) == self.stride

    def check_orthogonal(self) -> bool:
        if len(self.shape) < 2:
            return True
        pairs = sorted(zip(self.stride, self.shape), reverse=True)
        strides, shapes = zip(*pairs)
        return all(
            strides[index] % (strides[index + 1] * shapes[index + 1]) == 0
            for index in range(len(strides) - 1)
        )

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        return zip(self.shape, self.stride)


@dataclass(frozen=True)
class _MeshLayout(Sequence[_FlatLayout]):
    axes: tuple[_FlatLayout, ...]

    def __init__(self, axes: Sequence[_FlatLayout]) -> None:
        object.__setattr__(self, "axes", tuple(axes))

    @classmethod
    def from_sizes_strides(
        cls, sizes: tuple[int, ...], strides: tuple[int, ...] | None = None
    ) -> "_MeshLayout":
        actual_strides = flatten(suffix_product(sizes)) if strides is None else strides
        if len(sizes) != len(actual_strides):
            raise ValueError(
                f"sizes and strides must have the same length, got {len(sizes)} and {len(actual_strides)}"
            )
        return cls(_FlatLayout(size, stride) for size, stride in zip(sizes, actual_strides))

    def __len__(self) -> int:
        return len(self.axes)

    @overload
    def __getitem__(self, index: int) -> _FlatLayout: ...

    @overload
    def __getitem__(self, index: slice) -> "_MeshLayout": ...

    def __getitem__(self, index: int | slice) -> "_FlatLayout | _MeshLayout":
        return _MeshLayout(self.axes[index]) if isinstance(index, slice) else self.axes[index]

    def __iter__(self) -> Iterator[_FlatLayout]:
        return iter(self.axes)

    def to_pycute(self) -> Layout:
        return Layout(1, 0) if not self.axes else make_layout(*(axis.to_pycute() for axis in self.axes))

    @property
    def top_level_sizes(self) -> tuple[int, ...]:
        return tuple(axis.numel() for axis in self.axes)

    def numel(self) -> int:
        return math.prod(self.top_level_sizes)

    def cosize(self) -> int:
        return self.to_pycute().cosize()

    def collapse(self) -> _FlatLayout:
        return _FlatLayout(
            tuple(axis.shape for axis in self.axes),
            tuple(axis.stride for axis in self.axes),
        )

    def splice(self, start: int, end: int, layout: "_MeshLayout") -> "_MeshLayout":
        axes = list(self.axes)
        axes[start:end] = list(layout.axes)
        return _MeshLayout(axes)

    def remap_to_tensor(self, rank_map):
        if rank_map.ndim != 1 or not rank_map.is_contiguous():
            raise AssertionError
        if rank_map.numel() < self.cosize():
            raise AssertionError
        collapsed = self.collapse()
        complement_layout = collapsed.complement(rank_map.numel())
        shape = complement_layout.shape + collapsed.shape
        stride = complement_layout.stride + collapsed.stride
        return rank_map.as_strided(shape, stride).reshape(-1, *self.top_level_sizes)
