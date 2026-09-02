"""Microbatch splitting and merging utilities."""

import operator
from typing import Any, Iterable, Sequence

import tensorplay as tp
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
        if not isinstance(value, tp.Tensor):
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
            if not isinstance(subtree, tp.Tensor):
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
    if not isinstance(value, tp.Tensor):
        raise TypeError(f"expected a tensor, got {type(value).__name__}")
    dim = spec.split_dim if spec.split_dim >= 0 else spec.split_dim + value.dim()
    if dim < 0 or dim >= value.dim():
        raise ValueError("split dimension is outside the tensor rank")
    if int(value.shape[dim]) < num_chunks:
        raise ValueError("tensor dimension is smaller than the requested chunk count")
    return list(value.tensor_split(num_chunks, dim=dim)) if hasattr(value, "tensor_split") else list(tp.tensor_split(value, num_chunks, dim=dim))


def _spec_is_replicate(spec: Any) -> bool:
    return spec is None or spec is _Replicate or isinstance(spec, _Replicate)


def _leaf_chunks(value: Any, spec: Any, num_chunks: int) -> list[Any]:
    if _spec_is_replicate(spec):
        return [value] * num_chunks
    if isinstance(spec, TensorChunkSpec):
        if not isinstance(value, tp.Tensor):
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
        if not isinstance(item_spec, TensorChunkSpec) or not isinstance(item, tp.Tensor):
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
            if isinstance(value, tp.Tensor)
            else _Replicate(),
            args,
        )
    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = tree_map(
            lambda value: TensorChunkSpec(DEFAULT_CHUNK_DIM)
            if isinstance(value, tp.Tensor)
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
        if not all(isinstance(value, tp.Tensor) for value in chunks):
            raise TypeError("tensor chunk specification requires tensor values")
        dim = spec.split_dim if spec.split_dim >= 0 else spec.split_dim + chunks[0].dim()
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
            if isinstance(value, tp.Tensor)
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
