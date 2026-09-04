"""Microbatch splitting and merging utilities."""

import operator
from typing import Any, Iterable, Sequence

import tensorplay as tp
from ..tensor import DTensor
from tensorplay.utils._pytree import flatten_up_to, tree_flatten, tree_map, tree_unflatten

__all__ = ["TensorChunkSpec", "split_args_kwargs_into_chunks", "merge_chunks"]


class _CustomReducer:
    def __init__(self, init_value: Any, reduce_fn: Any) -> None:
        self.init_value = init_value
        self.reduce_fn = reduce_fn


class _LossReducer(_CustomReducer):
    pass


sum_reducer = _LossReducer(tp.tensor(0.0), operator.add)
DEFAULT_CHUNK_DIM = 0


class TensorChunkSpec:
    """Describes the dimension used to partition one tensor leaf."""

    def __init__(self, split_dim: int) -> None:
        if not isinstance(split_dim, int):
            raise TypeError("split_dim must be an integer")
        self.split_dim = split_dim

    @staticmethod
    def from_tuple(chunk_dims: tuple[int, ...]) -> tuple["TensorChunkSpec", ...]:
        return tree_map(TensorChunkSpec, chunk_dims)

    @staticmethod
    def from_dict(chunk_dims: dict[str, int]) -> dict[str, "TensorChunkSpec"]:
        return tree_map(TensorChunkSpec, chunk_dims)

    def __repr__(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}({self.split_dim})"

    def __str__(self) -> str:
        return f"TensorChunkSpec({self.split_dim})"


class _Replicate:
    pass


def _flatten_value_specs(value: Any, spec: Any) -> list[tuple[Any, Any]]:
    """Expand a possibly shallow specification to value leaves."""
    if _spec_is_replicate(spec):
        return [(leaf, spec) for leaf in tree_flatten(value)[0]]
    if isinstance(spec, TensorChunkSpec):
        if not _is_tensor_value(value) and not _is_block_mask(value):
            raise ValueError(
                "a tensor chunk specification must select a tensor leaf"
            )
        return [(value, spec)]

    spec_leaves, spec_tree = tree_flatten(spec)
    try:
        value_subtrees = flatten_up_to(value, spec_tree)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "chunk specification structure does not match the input"
        ) from error

    pairs: list[tuple[Any, Any]] = []
    for subtree, leaf_spec in zip(value_subtrees, spec_leaves):
        if _spec_is_replicate(leaf_spec):
            pairs.extend(
                (leaf, leaf_spec) for leaf in tree_flatten(subtree)[0]
            )
        elif isinstance(leaf_spec, TensorChunkSpec):
            if not _is_tensor_value(subtree) and not _is_block_mask(subtree):
                raise ValueError(
                    "a tensor chunk specification must select a tensor leaf"
                )
            pairs.append((subtree, leaf_spec))
        elif isinstance(leaf_spec, _CustomReducer):
            pairs.append((subtree, leaf_spec))
        else:
            raise ValueError(f"unsupported chunk specification: {leaf_spec!r}")
    return pairs


def _split_tensor(value: Any, spec: TensorChunkSpec, num_chunks: int) -> list[Any]:
    if not _is_tensor_value(value):
        raise TypeError(f"expected a tensor, got {type(value).__name__}")
    dim = spec.split_dim if spec.split_dim >= 0 else spec.split_dim + value.dim()
    if dim < 0 or dim >= value.dim():
        raise ValueError("split dimension is outside the tensor rank")
    if int(value.shape[dim]) < num_chunks:
        raise ValueError("tensor dimension is smaller than the requested chunk count")
    if isinstance(value, DTensor):
        local_tensor = value.to_local()
        local_chunks = tp.tensor_split(local_tensor, num_chunks, dim=dim)
        global_size = int(value.shape[dim])
        quotient, remainder = divmod(global_size, num_chunks)
        global_stride = value.stride()
        chunks = []
        for index, local_chunk in enumerate(local_chunks):
            chunk_shape = list(value.shape)
            chunk_shape[dim] = quotient + (1 if index < remainder else 0)
            chunks.append(
                DTensor.from_local(
                    local_chunk,
                    value.device_mesh,
                    value.placements,
                    shape=tuple(chunk_shape),
                    stride=global_stride,
                    run_check=False,
                )
            )
        return chunks
    return list(tp.tensor_split(value, num_chunks, dim=dim))


def _is_tensor_value(value: Any) -> bool:
    return isinstance(value, (tp.Tensor, DTensor))


def _spec_is_replicate(spec: Any) -> bool:
    return spec is None or spec is _Replicate or isinstance(spec, _Replicate)


def _is_block_mask(value: Any) -> bool:
    return (
        not _is_tensor_value(value)
        and all(
            hasattr(value, name)
            for name in (
                "kv_num_blocks",
                "kv_indices",
                "full_kv_num_blocks",
                "full_kv_indices",
                "BLOCK_SIZE",
                "mask_mod",
                "seq_lengths",
            )
        )
        and callable(getattr(type(value), "from_kv_blocks", None))
    )


def _split_block_mask(block_mask: Any, num_chunks: int) -> list[Any]:
    batch_size = int(block_mask.kv_num_blocks.size(0))
    if batch_size == 1:
        return [block_mask] * num_chunks
    if batch_size < num_chunks:
        raise AssertionError(
            "Block mask has fewer batch size than the number of chunks. "
        )

    batch_dim = 0
    kv_num_blocks_chunks = tp.tensor_split(
        block_mask.kv_num_blocks, num_chunks, batch_dim
    )
    kv_indices_chunks = tp.tensor_split(block_mask.kv_indices, num_chunks, batch_dim)
    full_kv_num_blocks_chunks = (
        tp.tensor_split(block_mask.full_kv_num_blocks, num_chunks, batch_dim)
        if block_mask.full_kv_num_blocks is not None
        else [None] * num_chunks
    )
    full_kv_indices_chunks = (
        tp.tensor_split(block_mask.full_kv_indices, num_chunks, batch_dim)
        if block_mask.full_kv_indices is not None
        else [None] * num_chunks
    )

    chunk_block_masks = []
    batch_offset = 0
    for chunk_idx in range(num_chunks):

        def create_mask_mod(idx: int):
            def batch_offset_mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any):
                b_offset = tp.full_like(b, idx)
                return block_mask.mask_mod(b + b_offset, h, q_idx, kv_idx)

            return batch_offset_mask_mod

        chunk_block_masks.append(
            type(block_mask).from_kv_blocks(
                kv_num_blocks=kv_num_blocks_chunks[chunk_idx],
                kv_indices=kv_indices_chunks[chunk_idx],
                full_kv_num_blocks=full_kv_num_blocks_chunks[chunk_idx],
                full_kv_indices=full_kv_indices_chunks[chunk_idx],
                BLOCK_SIZE=block_mask.BLOCK_SIZE,
                mask_mod=create_mask_mod(batch_offset),
                seq_lengths=block_mask.seq_lengths,
            )
        )
        batch_offset += int(kv_num_blocks_chunks[chunk_idx].size(0))
    return chunk_block_masks


def _leaf_chunks(value: Any, spec: Any, num_chunks: int) -> list[Any]:
    if _spec_is_replicate(spec):
        return [value] * num_chunks
    if isinstance(spec, TensorChunkSpec):
        if _is_block_mask(value):
            if spec.split_dim != 0:
                raise AssertionError("BlockMask only supports split_dim=0")
            return _split_block_mask(value, num_chunks)
        if not _is_tensor_value(value):
            return [value] * num_chunks
        return _split_tensor(value, spec, num_chunks)
    raise ValueError(f"unsupported value/spec pair: {type(value).__name__}, {spec!r}")


def _split_tree(value: Any, spec: Any, num_chunks: int) -> list[Any]:
    values, value_spec = tree_flatten(value)
    pairs = _flatten_value_specs(value, spec)
    if len(pairs) != len(values):
        raise ValueError("chunk specification structure does not match the input")
    chunks = [
        _leaf_chunks(item, item_spec, num_chunks)
        for item, item_spec in pairs
    ]
    return [tree_unflatten([chunk[index] for chunk in chunks], value_spec) for index in range(num_chunks)]


def _adjust_chunk_count(value: Any, spec: Any, requested: int) -> int:
    pairs = _flatten_value_specs(value, spec)
    count = requested
    found_tensor = False
    for item, item_spec in pairs:
        if not isinstance(item_spec, TensorChunkSpec):
            continue
        if _is_block_mask(item):
            if item_spec.split_dim != 0:
                raise AssertionError("BlockMask only supports split_dim=0")
            size = int(item.kv_num_blocks.size(0))
            if not found_tensor:
                found_tensor = True
                if size != 1:
                    count = min(count, size)
            elif size != 1 and size < count:
                raise ValueError(
                    "a later block mask has fewer batches than the effective chunk count"
                )
            continue
        if not _is_tensor_value(item):
            continue
        dim = item_spec.split_dim if item_spec.split_dim >= 0 else item_spec.split_dim + item.dim()
        if dim < 0 or dim >= item.dim():
            raise ValueError("split dimension is outside the tensor rank")
        size = int(item.shape[dim])
        if not found_tensor:
            found_tensor = True
            if size == 0:
                if requested != 1:
                    raise ValueError("cannot split an empty dimension into multiple chunks")
            elif size < count:
                count = size
        elif size < count:
            raise ValueError(
                "a later tensor has fewer elements on its chunking dimension "
                "than the effective chunk count"
            )
    if count <= 0:
        raise ValueError("effective chunk count must be positive")
    return count


def split_args_kwargs_into_chunks(args: tuple[Any, ...], kwargs: dict[str, Any], num_chunks: int, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if args_chunk_spec is None:
        args_chunk_spec = tree_map(
            lambda value: TensorChunkSpec(DEFAULT_CHUNK_DIM)
            if _is_tensor_value(value)
            else _Replicate(),
            args,
        )
    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = tree_map(
            lambda value: TensorChunkSpec(DEFAULT_CHUNK_DIM)
            if _is_tensor_value(value) or _is_block_mask(value)
            else _Replicate(),
            kwargs,
        )
    effective_chunks = _adjust_chunk_count(args, args_chunk_spec, num_chunks)
    arg_chunks = _split_tree(args, args_chunk_spec, effective_chunks)
    kwarg_chunks_count = _adjust_chunk_count(
        kwargs, kwargs_chunk_spec, effective_chunks
    )
    if kwarg_chunks_count != effective_chunks:
        effective_chunks = kwarg_chunks_count
        arg_chunks = _split_tree(args, args_chunk_spec, effective_chunks)
    kwarg_chunks = _split_tree(kwargs, kwargs_chunk_spec, effective_chunks)
    return [tuple(chunk) for chunk in arg_chunks], [dict(chunk) for chunk in kwarg_chunks]


def _merge_leaf(chunks: list[Any], spec: Any) -> Any:
    if _spec_is_replicate(spec):
        first = chunks[0]
        for value in chunks[1:]:
            if not _same_replicated_value(first, value):
                raise ValueError("replicated values differ between chunks")
        return first
    if isinstance(spec, TensorChunkSpec):
        if not all(_is_tensor_value(value) for value in chunks):
            raise TypeError("tensor chunk specification requires tensor values")
        dim = spec.split_dim if spec.split_dim >= 0 else spec.split_dim + chunks[0].dim()
        dtensor_flags = [isinstance(value, DTensor) for value in chunks]
        if any(dtensor_flags):
            if not all(dtensor_flags):
                raise ValueError("tensor chunks must use one tensor representation")
            placements = chunks[0].placements
            for index, value in enumerate(chunks[1:], 1):
                if value.placements != placements:
                    raise ValueError(f"tensor chunk placement mismatch at index {index}")
            first = chunks[0]
            local_value = tp.cat(
                [value.to_local() for value in chunks],
                dim=dim,
            )
            shape = list(first.shape)
            shape[dim] = sum(value.shape[dim] for value in chunks)
            return DTensor.from_local(
                local_value,
                first.device_mesh,
                placements,
                shape=tuple(shape),
                stride=first.stride(),
                run_check=False,
            )
        return tp.cat(tuple(chunks), dim=dim)
    if isinstance(spec, _CustomReducer):
        result = spec.init_value
        for value in chunks:
            result = spec.reduce_fn(result, value)
        return result
    raise ValueError(f"unsupported merge specification: {spec!r}")


def _same_replicated_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, tp.Tensor) or isinstance(right, tp.Tensor):
        return False
    try:
        result = left == right
        return result if isinstance(result, bool) else bool(result)
    except (TypeError, ValueError):
        return False


def merge_chunks(chunks: list[Any], chunk_spec: Any) -> Any:
    if not chunks:
        raise ValueError("cannot merge an empty chunk list")
    flat_chunks, value_spec = tree_flatten(chunks[0])
    if chunk_spec is None:
        flat_specs = [
            TensorChunkSpec(DEFAULT_CHUNK_DIM)
            if _is_tensor_value(value)
            else _Replicate()
            for value in flat_chunks
        ]
    else:
        spec_pairs = _flatten_value_specs(chunks[0], chunk_spec)
        if len(spec_pairs) != len(flat_chunks):
            raise ValueError(
                "merge specification structure does not match the chunks"
            )
        flat_specs = [item_spec for _, item_spec in spec_pairs]
    merged = []
    for chunk in chunks:
        if tree_flatten(chunk)[1] != value_spec:
            raise ValueError("all chunks must have the same structure")
    flattened_chunks = [tree_flatten(chunk)[0] for chunk in chunks]
    for index, spec in enumerate(flat_specs):
        merged.append(_merge_leaf([chunk[index] for chunk in flattened_chunks], spec))
    return tree_unflatten(merged, value_spec)


def _shard_dict_of_args(args_dict: dict[str, Any], args_chunk_spec: dict[str, Any], num_chunks: int) -> list[dict[str, Any]]:
    return _split_tree(args_dict, args_chunk_spec, num_chunks)
