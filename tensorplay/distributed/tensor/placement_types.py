"""Placement descriptions used by distributed tensor values."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

__all__ = ["Placement", "Shard", "Replicate", "Partial"]


class Placement:
    """Base class for a tensor layout on one mesh dimension."""

    def is_shard(self) -> bool:
        return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, Partial)

    def _short_repr(self) -> str:
        return repr(self)


class _StridedShardOffsetMode(IntEnum):
    FIRST = 0
    ALL = 1
    NONE = 2


class Shard(Placement):
    """Split one logical tensor dimension across a mesh dimension."""

    def __init__(self, dim: int) -> None:
        if type(dim) is not int:
            raise TypeError(f"Shard dim must be an integer, got {type(dim)!r}")
        self.dim = dim

    @staticmethod
    def _chunk_bounds(size: int, chunks: int, index: int) -> tuple[int, int]:
        if chunks <= 0:
            raise ValueError("number of chunks must be positive")
        if index < 0 or index >= chunks:
            raise IndexError(f"shard index {index} is outside {chunks} chunks")
        width = (size + chunks - 1) // chunks
        start = min(index * width, size)
        return start, min(start + width, size)

    @classmethod
    def _split_tensor_helper(
        cls,
        tensor: Any,
        num_chunks: int,
        with_padding: bool = True,
        contiguous: bool = True,
        dim: int = 0,
    ) -> tuple[list[Any], list[int]]:
        rank = tensor.dim()
        dim = dim if dim >= 0 else dim + rank
        if dim < 0 or dim >= rank:
            raise ValueError(f"shard dimension {dim} is outside tensor rank {rank}")
        size = int(tensor.shape[dim])
        width = (size + num_chunks - 1) // num_chunks
        shards: list[Any] = []
        pads: list[int] = []
        for index in range(num_chunks):
            start, end = cls._chunk_bounds(size, num_chunks, index)
            slices = [slice(None)] * rank
            slices[dim] = slice(start, end)
            shard = tensor[tuple(slices)]
            pad = width - (end - start)
            if with_padding and pad:
                shape = list(shard.shape)
                shape[dim] = pad
                padding = tensor.new_zeros(shape)
                shard = tensorplay_cat((shard, padding), dim)
            if contiguous and hasattr(shard, "contiguous"):
                shard = shard.contiguous()
            shards.append(shard)
            pads.append(pad)
        return shards, pads

    def _split_tensor(
        self,
        tensor: Any,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[Any], list[int]]:
        return self._split_tensor_helper(
            tensor, num_chunks, with_padding, contiguous, self.dim
        )

    def _select_split_tensor(
        self,
        tensor: Any,
        num_chunks: int,
        index: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> Any:
        shards, _ = self._split_tensor(
            tensor,
            num_chunks,
            with_padding=with_padding,
            contiguous=contiguous,
        )
        result = shards[index]
        return result.clone() if clone and hasattr(result, "clone") else result

    @staticmethod
    def local_shard_size_and_offset(
        curr_local_size: int, num_chunks: int, rank: int
    ) -> tuple[int, int]:
        start, end = Shard._chunk_bounds(int(curr_local_size), num_chunks, rank)
        return end - start, start

    def _local_shard_size_and_offset(
        self, curr_local_size: int, num_chunks: int, rank: int
    ) -> tuple[int, int]:
        return self.local_shard_size_and_offset(curr_local_size, num_chunks, rank)

    @staticmethod
    def _get_shard_pad_size(chunk_size: int, shard: Any, dim: int) -> int:
        return max(0, int(chunk_size) - int(shard.shape[dim]))

    @staticmethod
    def _maybe_unpad_tensor_with_sizes(
        dim: int, local_tensor: Any, pad_sizes: list[int], rank: int, make_contiguous: bool
    ) -> Any:
        pad = pad_sizes[rank]
        if pad:
            size = int(local_tensor.shape[dim]) - pad
            slices = [slice(None)] * local_tensor.dim()
            slices[dim] = slice(0, size)
            local_tensor = local_tensor[tuple(slices)]
            if make_contiguous and hasattr(local_tensor, "contiguous"):
                local_tensor = local_tensor.contiguous()
        return local_tensor

    def __hash__(self) -> int:
        return hash((type(self), self.dim))

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.dim == other.dim  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"Shard(dim={self.dim})"

    def __str__(self) -> str:
        return f"S({self.dim})"


class _StridedShard(Shard):
    """Shard placement with an explicit split factor for nested layouts."""

    def __init__(self, dim: int, split_factor: int = 1) -> None:
        super().__init__(dim)
        if type(split_factor) is not int or split_factor <= 0:
            raise ValueError("split_factor must be a positive integer")
        self._split_factor = split_factor

    @property
    def split_factor(self) -> int:
        return self._split_factor

    def __hash__(self) -> int:
        return hash((type(self), self.dim, self._split_factor))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _StridedShard)
            and self.dim == other.dim
            and self._split_factor == other._split_factor
        )

    def __repr__(self) -> str:
        return f"_StridedShard(dim={self.dim}, split_factor={self._split_factor})"


class Replicate(Placement):
    """Keep a complete copy of the logical tensor on every rank."""

    def __hash__(self) -> int:
        return -1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Replicate)

    def __repr__(self) -> str:
        return "Replicate()"

    def __str__(self) -> str:
        return "R"


class Partial(Placement):
    """Store values that still need a reduction across one mesh dimension."""

    ALL_REDUCE_OPS = ("sum", "avg", "min", "max", "product")
    LINEAR_REDUCE_OPS = ("sum", "avg")

    def __init__(self, reduce_op: str = "sum") -> None:
        if reduce_op not in self.ALL_REDUCE_OPS:
            raise ValueError(
                f"unsupported reduction {reduce_op!r}; expected one of {self.ALL_REDUCE_OPS}"
            )
        self.reduce_op = reduce_op

    def __hash__(self) -> int:
        return hash((type(self), self.reduce_op))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Partial) and self.reduce_op == other.reduce_op

    def __repr__(self) -> str:
        return f"Partial({self.reduce_op!r})"

    def __str__(self) -> str:
        return f"P({self.reduce_op})"


_Partial = Partial


def _is_shard_like(value: Placement) -> bool:
    return isinstance(value, (Shard, _StridedShard))


def tensorplay_cat(values: tuple[Any, ...], dim: int) -> Any:
    import tensorplay

    return tensorplay.cat(values, dim=dim)


__all__.extend(["_StridedShard", "_is_shard_like"])
