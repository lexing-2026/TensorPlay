"""Microbatch splitting and merging utilities."""

import operator
from typing import Any, Iterable, Sequence

import tensorplay as tp
from tensorplay.utils._pytree import tree_flatten, tree_map, tree_unflatten

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
        return tuple(TensorChunkSpec(dim) for dim in chunk_dims)

    @staticmethod
    def from_dict(chunk_dims: dict[str, int]) -> dict[str, "TensorChunkSpec"]:
        return {key: TensorChunkSpec(dim) for key, dim in chunk_dims.items()}

    def __repr__(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}({self.split_dim})"

    def __str__(self) -> str:
        return f"TensorChunkSpec({self.split_dim})"


class _Replicate:
    pass


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
    if isinstance(value, tp.Tensor) and isinstance(spec, TensorChunkSpec):
        return _split_tensor(value, spec, num_chunks)
    raise ValueError(f"unsupported value/spec pair: {type(value).__name__}, {spec!r}")


def _split_tree(value: Any, spec: Any, num_chunks: int) -> list[Any]:
    values, value_spec = tree_flatten(value)
    specs, spec_spec = tree_flatten(spec)
    if value_spec != spec_spec:
        raise ValueError("chunk specification structure does not match the input")
    chunks = [_leaf_chunks(item, item_spec, num_chunks) for item, item_spec in zip(values, specs)]
    return [tree_unflatten([chunk[index] for chunk in chunks], value_spec) for index in range(num_chunks)]


def split_args_kwargs_into_chunks(args: tuple[Any, ...], kwargs: dict[str, Any], num_chunks: int, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if args_chunk_spec is None:
        args_chunk_spec = tuple(TensorChunkSpec(DEFAULT_CHUNK_DIM) if isinstance(value, tp.Tensor) else _Replicate() for value in args)
    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = {key: TensorChunkSpec(DEFAULT_CHUNK_DIM) if isinstance(value, tp.Tensor) else _Replicate() for key, value in kwargs.items()}
    arg_chunks = _split_tree(args, args_chunk_spec, num_chunks)
    kwarg_chunks = _split_tree(kwargs, kwargs_chunk_spec, num_chunks)
    return [tuple(chunk) for chunk in arg_chunks], [dict(chunk) for chunk in kwarg_chunks]


def _merge_leaf(chunks: list[Any], spec: Any) -> Any:
    if _spec_is_replicate(spec):
        return chunks[0]
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


def merge_chunks(chunks: list[Any], chunk_spec: Any) -> Any:
    if not chunks:
        raise ValueError("cannot merge an empty chunk list")
    flat_chunks, value_spec = tree_flatten(chunks[0])
    flat_specs, spec_spec = tree_flatten(chunk_spec)
    if value_spec != spec_spec:
        raise ValueError("merge specification structure does not match the chunks")
    merged = []
    for index, spec in enumerate(flat_specs):
        merged.append(_merge_leaf([tree_flatten(chunk)[0][index] for chunk in chunks], spec))
    return tree_unflatten(merged, value_spec)


def _shard_dict_of_args(args_dict: dict[str, Any], args_chunk_spec: dict[str, Any], num_chunks: int) -> list[dict[str, Any]]:
    return _split_tree(args_dict, args_chunk_spec, num_chunks)

