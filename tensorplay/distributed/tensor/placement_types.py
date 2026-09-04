"""Placement descriptions used by distributed tensor values."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Sequence

import tensorplay

__all__ = ["Placement", "Shard", "Replicate", "Partial"]


def _is_symbolic_int(value: Any) -> bool:
    checker = getattr(value, "is_symbolic", None)
    return bool(checker()) if callable(checker) else False


def _explicit_or_backed_hint(value: Any) -> int | None:
    if isinstance(value, int):
        return int(value)
    if not _is_symbolic_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    hint = getattr(value, "hint", None)
    if hint is not None:
        return int(hint)
    has_hint = getattr(value, "has_hint", None)
    if callable(has_hint) and has_hint():
        hinted = getattr(value, "hint", None)
        if hinted is not None:
            return int(hinted)
    return None


def _guarded_hint_int(value: Any, *, reason: str) -> int:
    if isinstance(value, int):
        return int(value)
    if not _is_symbolic_int(value):
        return int(value)
    hint = _explicit_or_backed_hint(value)
    if hint is None:
        raise RuntimeError(
            f"cannot specialize symbolic {reason} without a concrete hint: {value}"
        )
    guard = getattr(value, "guard_int", None)
    if callable(guard):
        return int(guard())
    return int(hint)


def _split_factor_key(value: Any) -> object:
    if _is_symbolic_int(value):
        expression = getattr(value, "expr", None)
        return expression if expression is not None else str(value)
    return int(value)


def _sym_mod(left: Any, right: Any) -> Any:
    return left % right


def _hint_proves_even_shard(size: Any, num_chunks: Any) -> bool:
    remainder = _sym_mod(size, num_chunks)
    guard_or_false = getattr(remainder == 0, "guard_or_false", None)
    if callable(guard_or_false):
        try:
            if bool(guard_or_false()):
                return True
        except (RuntimeError, TypeError, ValueError):
            pass
    size_hint = _explicit_or_backed_hint(size)
    chunks_hint = _explicit_or_backed_hint(num_chunks)
    return (
        size_hint is not None
        and chunks_hint is not None
        and chunks_hint != 0
        and size_hint % chunks_hint == 0
    )


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

    def __eq__(self, other: object) -> bool:
        del other
        raise RuntimeError(
            "base placement equality must be implemented by a concrete placement"
        )

    def __hash__(self) -> int:
        raise RuntimeError(
            "base placement hashing must be implemented by a concrete placement"
        )

    def __fx_repr__(self) -> tuple[str, dict[str, Any]]:
        raise RuntimeError(
            "placement FX representation must be implemented by a concrete placement"
        )


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
        tensor_list = cls._custom_chunk(tensor, num_chunks, dim)
        shards: list[Any] = []
        pads: list[int] = []
        for shard in tensor_list:
            pad = width - int(shard.shape[dim])
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

    @staticmethod
    def _custom_chunk(
        tensor: Any, num_chunks: int, dim: int
    ) -> list[Any]:
        if int(tensor.dim()) <= 0:
            raise AssertionError(
                f"expected a non-scalar tensor, got rank {tensor.dim()}"
            )
        num_chunks = _guarded_hint_int(num_chunks, reason="chunk count")
        if num_chunks <= 0:
            raise AssertionError(f"expected a positive chunk count, got {num_chunks}")
        rank = int(tensor.dim())
        dim = int(dim)
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise ValueError(f"shard dimension {dim} is outside tensor rank {rank}")
        chunks = list(tensorplay.chunk(tensor, num_chunks, dim=dim))
        if len(chunks) == num_chunks:
            return chunks
        if not chunks:
            shape = list(tensor.shape)
            shape[dim] = 0
            empty = tensor.new_zeros(tuple(shape))
            return [empty.clone() for _ in range(num_chunks)]
        from ._collective_utils import fill_empty_tensor_to_shards

        return fill_empty_tensor_to_shards(
            chunks, dim, num_chunks - len(chunks)
        )

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
        curr_local_size: Any, num_chunks: int, rank: int
    ) -> tuple[Any, Any]:
        num_chunks = _guarded_hint_int(num_chunks, reason="chunk count")
        rank = int(rank)
        if _hint_proves_even_shard(curr_local_size, num_chunks):
            full_chunk_size = curr_local_size // num_chunks
            return full_chunk_size, full_chunk_size * rank
        full_chunk_size = (curr_local_size + num_chunks - 1) // num_chunks
        shard_start = full_chunk_size * rank
        shard_end = tensorplay.sym_min(
            curr_local_size, shard_start + full_chunk_size
        )
        local_shard_size = tensorplay.sym_max(0, shard_end - shard_start)
        return local_shard_size, tensorplay.sym_min(curr_local_size, shard_start)

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

    def _shard_tensor(
        self,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> Any:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor.new_empty((0,), requires_grad=tensor.requires_grad)
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        rank = int(coordinate[mesh_dim])
        if src_data_rank is None:
            return self._select_split_tensor(
                tensor,
                num_chunks,
                rank,
                with_padding=False,
                contiguous=True,
            )
        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )
        first = scatter_list[0]
        if any(first.shape != item.shape for item in scatter_list[1:]):
            raise AssertionError("shard tensors must have equal shapes")
        output = tensor.new_empty(tuple(first.shape))
        from ._collective_utils import mesh_scatter

        mesh_scatter(
            output,
            scatter_list,
            mesh,
            mesh_dim=mesh_dim,
            group_src=src_data_rank,
        )
        return self._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, rank, True
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> Any:
        return cls(dim)._shard_tensor(tensor, mesh, mesh_dim, src_data_rank)

    def _reduce_shard_tensor(
        self,
        tensor: Any,
        mesh: Any,
        reduce_op: str,
        mesh_dim: int,
    ) -> Any:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        logical_dim_size = int(tensor.shape[self.dim])
        pad_sizes: list[int] | None = None
        if logical_dim_size % num_chunks:
            scatter_list, pad_sizes = self._split_tensor(
                tensor, num_chunks, with_padding=True, contiguous=True
            )
            tensor = tensorplay.cat(tuple(scatter_list), dim=self.dim)
        elif hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
            tensor = tensor.contiguous()
        from .. import _functional_collectives as funcol

        dim = self.dim if self.dim >= 0 else self.dim + int(tensor.dim())
        moved = tensor.movedim(dim, 0) if dim else tensor
        output = funcol.reduce_scatter_single(
            moved,
            reduce_op=reduce_op,
            scatter_dim=0,
            group=(mesh, mesh_dim),
        )
        if dim:
            output = funcol.wait_tensor(output).movedim(0, dim)
        if pad_sizes is not None:
            output = self._maybe_unpad_tensor_with_sizes(
                self.dim,
                output,
                pad_sizes,
                int(coordinate[mesh_dim]),
                False,
            )
        return output

    def _maybe_pad_tensor(
        self, local_tensor: Any, logical_dim_size: int, num_chunks: int
    ) -> Any:
        if int(logical_dim_size) % int(num_chunks):
            full_chunk_size = (int(logical_dim_size) + int(num_chunks) - 1) // int(
                num_chunks
            )
            from ._collective_utils import pad_tensor

            local_tensor = pad_tensor(
                local_tensor,
                self.dim,
                full_chunk_size - int(local_tensor.shape[self.dim]),
            )
        if hasattr(local_tensor, "is_contiguous") and not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    def _maybe_unpad_tensor(
        self, local_tensor: Any, logical_dim_size: int, num_chunks: int
    ) -> Any:
        if int(logical_dim_size) % int(num_chunks):
            full_chunk_size = (int(logical_dim_size) + int(num_chunks) - 1) // int(
                num_chunks
            )
            from ._collective_utils import unpad_tensor

            local_tensor = unpad_tensor(
                local_tensor,
                self.dim,
                full_chunk_size * int(num_chunks) - int(logical_dim_size),
            )
        if hasattr(local_tensor, "contiguous"):
            local_tensor = local_tensor.contiguous()
        return local_tensor

    def _to_replicate_tensor(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        current_logical_shape: Sequence[int],
    ) -> Any:
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        logical_dim_size = int(current_logical_shape[self.dim])
        local_tensor = self._maybe_pad_tensor(
            local_tensor, logical_dim_size, num_chunks
        )
        dim = self.dim if self.dim >= 0 else self.dim + int(local_tensor.dim())
        moved = local_tensor.movedim(dim, 0) if dim else local_tensor
        from .. import _functional_collectives as funcol

        result = funcol.all_gather_single(
            moved,
            gather_dim=0,
            group=(mesh, mesh_dim),
        )
        if dim:
            result = funcol.wait_tensor(result).movedim(0, dim)
        return self._maybe_unpad_tensor(result, logical_dim_size, num_chunks)

    def _replicate_to_shard(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        shard_index: int,
    ) -> Any:
        return self._select_split_tensor(
            local_tensor,
            int(mesh.size(mesh_dim=mesh_dim)),
            int(shard_index),
            with_padding=False,
            clone=True,
        )

    def _to_partial_tensor(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        current_logical_shape: Sequence[int],
    ) -> Any:
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        logical_dim_size = int(current_logical_shape[self.dim])
        rank = int(mesh.get_local_rank(mesh_dim))
        local_shard_size, local_offset = self.local_shard_size_and_offset(
            logical_dim_size, num_chunks, rank
        )
        if int(local_tensor.shape[self.dim]) != int(local_shard_size):
            raise ValueError("local shard shape does not match placement metadata")
        output_shape = list(local_tensor.shape)
        output_shape[self.dim] = logical_dim_size
        output = local_tensor.new_zeros(tuple(output_shape))
        indices = tensorplay.arange(
            int(local_shard_size), dtype=tensorplay.int64, device=local_tensor.device
        )
        return output.index_add(
            self.dim,
            indices + int(local_offset),
            local_tensor,
        )

    @staticmethod
    def _get_shard_pad_size(full_size: int, local_tensor: Any, dim: int) -> int:
        return int(full_size) - int(local_tensor.shape[dim])

    @staticmethod
    def _compute_padding_info(
        current_logical_shape: Sequence[int],
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> tuple[bool, int, int, bool, int, int]:
        result = []
        for shard_dim in (old_shard_dim, new_shard_dim):
            logical_size = int(current_logical_shape[shard_dim])
            full_chunk_size = (logical_size + int(num_chunks) - 1) // int(num_chunks)
            result.append(
                (logical_size % int(num_chunks) != 0, logical_size, full_chunk_size)
            )
        return result[0] + result[1]  # type: ignore[return-value]

    @staticmethod
    def _pad_for_new_shard_dim(
        current_logical_shape: Sequence[int],
        local_tensor: Any,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> Any:
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            _,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )
        from ._collective_utils import pad_tensor

        if old_dim_padding:
            local_tensor = pad_tensor(
                local_tensor,
                old_shard_dim,
                Shard._get_shard_pad_size(
                    old_dim_full_chunk_size, local_tensor, old_shard_dim
                ),
            )
        if new_dim_padding:
            local_tensor = pad_tensor(
                local_tensor,
                new_shard_dim,
                Shard._get_shard_pad_size(
                    new_dim_full_chunk_size * int(num_chunks),
                    local_tensor,
                    new_shard_dim,
                ),
            )
        if hasattr(local_tensor, "is_contiguous") and not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    @staticmethod
    def _unpad_for_new_shard_dim(
        current_logical_shape: Sequence[int],
        local_tensor: Any,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        local_rank: int,
    ) -> Any:
        (
            old_dim_padding,
            old_dim_logical_size,
            old_dim_full_chunk_size,
            new_dim_padding,
            new_dim_logical_size,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )
        from ._collective_utils import unpad_tensor

        if old_dim_padding:
            local_tensor = unpad_tensor(
                local_tensor,
                old_shard_dim,
                old_dim_full_chunk_size * int(num_chunks) - old_dim_logical_size,
            )
        if new_dim_padding:
            local_shard_size, _ = Shard.local_shard_size_and_offset(
                new_dim_logical_size, num_chunks, int(local_rank)
            )
            local_tensor = unpad_tensor(
                local_tensor,
                new_shard_dim,
                new_dim_full_chunk_size - int(local_shard_size),
            )
        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        current_logical_shape: Sequence[int],
        new_shard_dim: int,
    ) -> Any:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return local_tensor
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        local_tensor = Shard._pad_for_new_shard_dim(
            current_logical_shape,
            local_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
        )
        from ._collective_utils import shard_dim_alltoall

        result = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )
        return Shard._unpad_for_new_shard_dim(
            current_logical_shape,
            result,
            num_chunks,
            self.dim,
            new_shard_dim,
            int(coordinate[mesh_dim]),
        )

    def __hash__(self) -> int:
        return hash((type(self), self.dim))

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.dim == other.dim  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"Shard(dim={self.dim})"

    def __fx_repr__(self) -> tuple[str, dict[str, Any]]:
        return (
            f"tensorplay.distributed.tensor.placement_types.Shard(dim={self.dim})",
            {},
        )

    def __str__(self) -> str:
        return f"S({self.dim})"


class _StridedShard(Shard):
    """Shard placement with an explicit split factor for nested layouts."""

    def __init__(
        self,
        dim: int,
        split_factor: Any = 1,
        *,
        sf: Any | None = None,
    ) -> None:
        if sf is not None:
            if split_factor != 1:
                raise TypeError("_StridedShard received both split_factor and sf")
            split_factor = sf
        super().__init__(dim)
        split_factor_int = _guarded_hint_int(
            split_factor, reason="_StridedShard split factor"
        )
        if split_factor_int <= 0:
            raise ValueError("split_factor must be a positive integer")
        self._split_factor = split_factor_int
        if _is_symbolic_int(split_factor):
            self._symbolic_split_factor = split_factor

    @property
    def split_factor(self) -> Any:
        symbolic = getattr(self, "_symbolic_split_factor", None)
        if symbolic is not None:
            return symbolic
        return self._split_factor

    def _split_factor_int(self) -> int:
        return int(self._split_factor)

    def _split_tensor(
        self,
        tensor: Any,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[Any], list[int]]:
        first_split, _ = Shard._split_tensor_helper(
            tensor,
            self._split_factor_int(),
            with_padding=False,
            contiguous=False,
            dim=self.dim,
        )
        second_split = [
            Shard._split_tensor_helper(
                value,
                int(num_chunks),
                with_padding=False,
                contiguous=False,
                dim=self.dim,
            )[0]
            for value in first_split
        ]
        shards: list[Any] = []
        for index in range(int(num_chunks)):
            shard = tensorplay.cat(
                tuple(value[index] for value in second_split), dim=self.dim
            )
            if contiguous and hasattr(shard, "contiguous"):
                shard = shard.contiguous()
            shards.append(shard)
        if not with_padding:
            return shards, []
        dim = self.dim if self.dim >= 0 else self.dim + int(tensor.dim())
        width = max(int(shard.shape[dim]) for shard in shards)
        pads = [width - int(shard.shape[dim]) for shard in shards]
        return [
            tensorplay.cat(
                (shard, tensor.new_zeros(
                    tuple(
                        pad if axis == dim else int(size)
                        for axis, size in enumerate(shard.shape)
                    )
                )),
                dim=dim,
            ) if pad else shard
            for shard, pad in zip(shards, pads)
        ], pads

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
            contiguous=False,
        )
        result = shards[int(index)]
        if clone and hasattr(result, "clone"):
            result = result.clone()
        elif contiguous and hasattr(result, "contiguous"):
            result = result.contiguous()
        return result

    def _shard_tensor(
        self,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> Any:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor.new_empty((0,), requires_grad=tensor.requires_grad)
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        rank = int(coordinate[mesh_dim])
        if src_data_rank is None:
            return self._select_split_tensor(
                tensor,
                num_chunks,
                rank,
                with_padding=False,
                contiguous=True,
            )
        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )
        first = scatter_list[0]
        if any(first.shape != item.shape for item in scatter_list[1:]):
            raise AssertionError("shard tensors must have equal shapes")
        output = tensor.new_empty(tuple(first.shape))
        from ._collective_utils import mesh_scatter

        mesh_scatter(
            output,
            scatter_list,
            mesh,
            mesh_dim=mesh_dim,
            group_src=src_data_rank,
        )
        return self._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, rank, True
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        src_data_rank: int | None = 0,
        split_factor: int = 1,
    ) -> Any:
        placement = cls(dim, split_factor=split_factor)
        return placement._shard_tensor(tensor, mesh, mesh_dim, src_data_rank)

    def _to_replicate_tensor(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        current_logical_shape: Sequence[int],
    ) -> Any:
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        dim = self.dim if self.dim >= 0 else self.dim + int(local_tensor.dim())
        logical_dim_size = int(current_logical_shape[self.dim])
        indices_tensor = tensorplay.arange(
            logical_dim_size,
            dtype=tensorplay.int64,
            device=local_tensor.device,
        ).view([1] * dim + [logical_dim_size])
        sharded_indices, _ = self._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        sharded_indices = [value.view(-1) for value in sharded_indices]
        max_chunk_size = max(int(value.numel()) for value in sharded_indices)
        from ._collective_utils import pad_tensor

        local_tensor = pad_tensor(
            local_tensor,
            dim,
            max_chunk_size - int(local_tensor.shape[dim]),
        )
        if hasattr(local_tensor, "is_contiguous") and not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        from .. import _functional_collectives as funcol

        gathered = funcol.all_gather_single(
            local_tensor.movedim(dim, 0) if dim else local_tensor,
            gather_dim=0,
            group=(mesh, mesh_dim),
        )
        gathered = funcol.wait_tensor(gathered)
        if dim:
            gathered = gathered.movedim(0, dim)
        positions = [
            tensorplay.arange(
                index * max_chunk_size,
                index * max_chunk_size + int(value.numel()),
                dtype=tensorplay.int64,
                device=local_tensor.device,
            )
            for index, value in enumerate(sharded_indices)
        ]
        permutation = tensorplay.cat(tuple(sharded_indices))
        select_positions = tensorplay.cat(tuple(positions))
        select_indices = select_positions.index_select(
            0, tensorplay.argsort(permutation)
        )
        return gathered.index_select(dim, select_indices).contiguous()

    def _replicate_to_strided_shard(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        shard_index: int,
    ) -> Any:
        return self._select_split_tensor(
            local_tensor,
            int(mesh.size(mesh_dim=mesh_dim)),
            int(shard_index),
            with_padding=False,
            clone=True,
        )

    @staticmethod
    def _local_shard_size(sharded_indices: list[Any], rank: int) -> int:
        return int(sharded_indices[int(rank)].numel())

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: int,
        offset_mode: _StridedShardOffsetMode = _StridedShardOffsetMode.FIRST,
    ) -> tuple[int, int | list[int] | None]:
        return self.local_shard_size_and_offset(
            curr_local_size, num_chunks, rank, offset_mode
        )

    def local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: int,
        offset_mode: _StridedShardOffsetMode = _StridedShardOffsetMode.FIRST,
    ) -> tuple[int, int | list[int] | None]:
        mode = _StridedShardOffsetMode(offset_mode)
        from ._utils import _strided_shard_indices

        offsets = _strided_shard_indices(
            int(curr_local_size),
            int(num_chunks),
            int(rank),
            self._split_factor_int(),
        )
        if mode is _StridedShardOffsetMode.NONE:
            return len(offsets), None
        if mode is _StridedShardOffsetMode.ALL:
            return len(offsets), offsets
        return len(offsets), (offsets[0] if offsets else -1)

    @staticmethod
    def _compute_padding_info(
        logical_size_on_dim: int,
        num_chunks: int,
        shard_dim: int,
        split_factor: int = 1,
    ) -> tuple[bool, int]:
        del shard_dim
        logical_size_on_dim = int(logical_size_on_dim)
        num_chunks = int(num_chunks)
        split_factor = int(split_factor)
        ceil_div = lambda value, divisor: (value + divisor - 1) // divisor
        if split_factor != 1:
            first_chunk_size = ceil_div(logical_size_on_dim, split_factor)
            full_chunks = logical_size_on_dim // first_chunk_size if first_chunk_size else 0
            remainder = logical_size_on_dim - full_chunks * first_chunk_size
            partial = remainder > 0
            empty = split_factor - full_chunks - (1 if partial else 0)
            uneven = first_chunk_size % num_chunks != 0 if first_chunk_size else False
            max_chunk_size = ceil_div(first_chunk_size, num_chunks) * full_chunks
            if partial:
                max_chunk_size += ceil_div(remainder, num_chunks)
            return partial or empty > 0 or uneven, max_chunk_size
        return (
            logical_size_on_dim % num_chunks != 0,
            ceil_div(logical_size_on_dim, num_chunks),
        )

    def _pad_for_new_shard_dim(
        self,
        current_logical_shape: Sequence[int],
        local_tensor: Any,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        split_factor: int,
    ) -> Any:
        old_padding, old_width = self._compute_padding_info(
            current_logical_shape[old_shard_dim],
            num_chunks,
            old_shard_dim,
            split_factor,
        )
        new_padding, new_width = self._compute_padding_info(
            current_logical_shape[new_shard_dim],
            num_chunks,
            new_shard_dim,
        )
        from ._collective_utils import pad_tensor

        if old_padding:
            local_tensor = pad_tensor(
                local_tensor,
                old_shard_dim,
                old_width - int(local_tensor.shape[old_shard_dim]),
            )
        if new_padding:
            local_tensor = pad_tensor(
                local_tensor,
                new_shard_dim,
                new_width * int(num_chunks) - int(local_tensor.shape[new_shard_dim]),
            )
        return local_tensor.contiguous()

    def _unpad_for_new_shard_dim(
        self,
        current_logical_shape: Sequence[int],
        local_tensor: Any,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        split_factor: int,
        local_rank: int,
    ) -> Any:
        old_size = int(current_logical_shape[old_shard_dim])
        new_size = int(current_logical_shape[new_shard_dim])
        _, new_width = self._compute_padding_info(
            new_size, num_chunks, new_shard_dim
        )
        from ._collective_utils import unpad_tensor

        indices_tensor = tensorplay.arange(
            old_size, dtype=tensorplay.int64, device=local_tensor.device
        ).view([1] * old_shard_dim + [old_size])
        sharded_indices, _ = self._split_tensor(
            indices_tensor, num_chunks, with_padding=False, contiguous=False
        )
        sharded_indices = [value.view(-1) for value in sharded_indices]
        max_chunk_size = len(sharded_indices[0])
        positions = [
            tensorplay.arange(
                index * max_chunk_size,
                index * max_chunk_size + int(value.numel()),
                dtype=tensorplay.int64,
                device=local_tensor.device,
            )
            for index, value in enumerate(sharded_indices)
        ]
        permutation = tensorplay.cat(tuple(sharded_indices))
        select_positions = tensorplay.cat(tuple(positions))
        select_indices = select_positions.index_select(
            0, tensorplay.argsort(permutation)
        )
        local_tensor = local_tensor.index_select(old_shard_dim, select_indices)
        if new_size % int(num_chunks):
            local_size, _ = Shard.local_shard_size_and_offset(
                new_size, num_chunks, int(local_rank)
            )
            local_tensor = unpad_tensor(
                local_tensor, new_shard_dim, new_width - int(local_size)
            )
        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: Any,
        mesh: Any,
        mesh_dim: int,
        current_logical_shape: Sequence[int],
        new_shard_dim: int,
    ) -> Any:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return local_tensor
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        local_tensor = self._pad_for_new_shard_dim(
            current_logical_shape,
            local_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            self._split_factor_int(),
        )
        from ._collective_utils import shard_dim_alltoall

        result = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )
        return self._unpad_for_new_shard_dim(
            current_logical_shape,
            result,
            num_chunks,
            self.dim,
            new_shard_dim,
            self._split_factor_int(),
            int(coordinate[mesh_dim]),
        )

    def __hash__(self) -> int:
        return hash((type(self), self.dim, _split_factor_key(self.split_factor)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _StridedShard)
            and self.dim == other.dim
            and _split_factor_key(self.split_factor)
            == _split_factor_key(other.split_factor)
        )

    def __repr__(self) -> str:
        return f"_StridedShard(dim={self.dim}, sf={self.split_factor})"

    def __str__(self) -> str:
        return f"_S({self.dim}, {self._split_factor})"

    @staticmethod
    def _select_shard(shards: list[Any], shard_index: int) -> Any:
        return shards[int(shard_index)].clone()

    def __fx_repr__(self) -> tuple[str, dict[str, Any]]:
        return (
            "tensorplay.distributed.tensor.placement_types."
            f"_StridedShard(dim={self.dim}, sf={self._split_factor_int()})",
            {},
        )


class Replicate(Placement):
    """Keep a complete copy of the logical tensor on every rank."""

    def __hash__(self) -> int:
        return -1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Replicate)

    def __repr__(self) -> str:
        return "Replicate()"

    def __fx_repr__(self) -> tuple[str, dict[str, Any]]:
        return (
            "tensorplay.distributed.tensor.placement_types.Replicate()",
            {},
        )

    def __str__(self) -> str:
        return "R"

    @classmethod
    def _make_replicate_tensor(
        cls,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> Any:
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            return tensor.new_empty((0,), requires_grad=tensor.requires_grad)
        tensor = tensor.contiguous()
        if src_data_rank is not None:
            from ._collective_utils import mesh_broadcast

            mesh_broadcast(
                tensor,
                mesh,
                mesh_dim=mesh_dim,
                group_src=src_data_rank,
            )
        return tensor

    def _replicate_tensor(
        self,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> Any:
        return Replicate._make_replicate_tensor(
            tensor, mesh, mesh_dim, src_data_rank
        )


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

    def __fx_repr__(self) -> tuple[str, dict[str, Any]]:
        return (
            "tensorplay.distributed.tensor.placement_types."
            f"Partial({self.reduce_op!r})",
            {},
        )

    def __str__(self) -> str:
        return f"P({self.reduce_op})"

    def _reduce_value(self, tensor: Any, mesh: Any, mesh_dim: int) -> Any:
        from .. import _functional_collectives as funcol

        return funcol.all_reduce(
            tensor,
            reduce_op=self.reduce_op,
            group=(mesh, mesh_dim),
        )

    def _reduce_shard_value(
        self,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> Any:
        if not isinstance(shard_spec, Shard):
            raise TypeError("shard_spec must be a Shard placement")
        return shard_spec._reduce_shard_tensor(
            tensor, mesh, self.reduce_op, mesh_dim
        )

    def _partition_value(self, tensor: Any, mesh: Any, mesh_dim: int) -> Any:
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        if self.reduce_op == "sum":
            return tensor / num_chunks
        if self.reduce_op in ("avg", "min", "max"):
            return tensor
        raise ValueError(
            f"Replicate to Partial({self.reduce_op}) conversion is not supported."
        )


_Partial = Partial


class _MaskPartial(Partial):
    """Partial placement carrying the index mask for a row-sharded lookup."""

    def __init__(
        self,
        reduce_op: str | None = None,
        mask_buffer: Any = None,
        offset_shape: Sequence[int] | None = None,
        offset_dim: int = 0,
    ) -> None:
        super().__init__("sum" if reduce_op is None else reduce_op)
        if mask_buffer is None:
            from ._ops._mask_buffer import MaskBuffer

            mask_buffer = MaskBuffer()
        self.mask_buffer = mask_buffer
        self.offset_shape = None if offset_shape is None else tuple(offset_shape)
        self.offset_dim = int(offset_dim)

    @staticmethod
    def _mask_tensor(
        tensor: Any,
        local_offset_on_dim: int,
        local_shard_size: int,
    ) -> tuple[Any, Any]:
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        return mask, masked_tensor

    def _partition_value(self, tensor: Any, mesh: Any, mesh_dim: int) -> Any:
        if mesh.get_coordinate() is None:
            raise AssertionError("rank is not part of mesh")
        if self.offset_shape is None:
            raise AssertionError("offset_shape must be set for _MaskPartial")
        num_chunks = int(mesh.size(mesh_dim=mesh_dim))
        local_shard_size, local_offset = Shard.local_shard_size_and_offset(
            int(self.offset_shape[self.offset_dim]),
            num_chunks,
            int(mesh.get_local_rank(mesh_dim)),
        )
        mask, masked_tensor = self._mask_tensor(
            tensor, local_offset, local_shard_size
        )
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(self, tensor: Any, mesh: Any, mesh_dim: int) -> Any:
        if self.mask_buffer.data is None:
            raise AssertionError("mask buffer is not materialized")
        self.mask_buffer.apply_mask(tensor)
        self.mask_buffer.release_mask()
        from .. import _functional_collectives as funcol

        return funcol.all_reduce(
            tensor,
            reduce_op=self.reduce_op,
            group=(mesh, mesh_dim),
        )

    def _reduce_shard_value(
        self,
        tensor: Any,
        mesh: Any,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> Any:
        if self.mask_buffer.data is None:
            raise AssertionError("mask buffer is not materialized")
        self.mask_buffer.apply_mask(tensor)
        self.mask_buffer.release_mask()
        if not isinstance(shard_spec, Shard):
            raise TypeError("shard_spec must be a Shard placement")
        return shard_spec._reduce_shard_tensor(
            tensor, mesh, self.reduce_op, mesh_dim
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _MaskPartial)
            and self.reduce_op == other.reduce_op
            and self.offset_shape == other.offset_shape
            and self.offset_dim == other.offset_dim
            and self.mask_buffer is other.mask_buffer
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.reduce_op,
                self.offset_shape,
                self.offset_dim,
                id(self.mask_buffer),
            )
        )

    def __repr__(self) -> str:
        return (
            f"_MaskPartial(reduce_op={self.reduce_op!r}, "
            f"offset_shape={self.offset_shape!r}, offset_dim={self.offset_dim})"
        )

    def __str__(self) -> str:
        return f"MaskP({self.reduce_op}, {self.offset_shape}, {self.offset_dim})"

    def __fx_repr__(self) -> tuple[str, dict[str, Any]]:
        return (
            "tensorplay.distributed.tensor.placement_types."
            f"_MaskPartial(reduce_op={self.reduce_op!r}, "
            f"offset_shape={self.offset_shape!r}, offset_dim={self.offset_dim})",
            {},
        )


def _is_shard_like(value: Placement) -> bool:
    return isinstance(value, (Shard, _StridedShard))


def tensorplay_cat(values: tuple[Any, ...], dim: int) -> Any:
    import tensorplay

    return tensorplay.cat(values, dim=dim)


__all__.extend(["_MaskPartial", "_StridedShard", "_is_shard_like"])
