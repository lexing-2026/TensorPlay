"""Native batch-dimension execution for the vectorizing transform."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import itertools
from typing import Any, Callable, Optional, Union

import tensorplay
from tensorplay.utils._pytree import (
    TreeSpec,
    _broadcast_to_and_flatten,
    tree_flatten,
    tree_map_,
    tree_unflatten,
)

in_dims_t = Union[int, tuple]
out_dims_t = Union[int, tuple[int, ...], None]

_vmap_depth = contextvars.ContextVar("tensorplay_vmap_depth", default=0)


def doesnt_support_saved_tensors_hooks(func: Callable) -> Callable:
    """Reject transform execution while saved-value hooks are active."""

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from tensorplay.autograd.graph import _hook_stack

        if _hook_stack:
            raise RuntimeError(
                "Function transforms do not support saved tensor hooks."
            )
        return func(*args, **kwargs)

    return wrapped


def _get_name(func: Callable[..., Any]) -> str:
    if hasattr(func, "__name__"):
        return func.__name__
    if isinstance(func, functools.partial):
        return f"functools.partial({_get_name(func.func)}, ...)"
    return repr(func)


def _validate_and_get_batch_size(
    flat_in_dims: list[Optional[int]], flat_args: list[Any]
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    if not batch_sizes:
        raise ValueError("vmap: Expected at least one Tensor to vmap over")
    if any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            "vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    return batch_sizes[0]


def _process_batched_inputs(
    in_dims: in_dims_t, args: tuple[Any, ...], func: Callable[..., Any]
) -> tuple[int, list[Optional[int]], list[Any], TreeSpec]:
    """Resolve input dimensions and validate the native batch size."""

    if not isinstance(in_dims, (int, tuple)):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            "expected in_dims to be an int or a tuple matching inputs"
        )
    if not args:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs; "
            "a mapped call needs at least one input"
        )

    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            "in_dims is not compatible with the input structure"
        )

    for index, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if in_dim is not None and not isinstance(in_dim, int):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim}; every mapped dimension must be an integer "
                "or None"
            )
        if in_dim is None:
            continue
        if not isinstance(arg, tensorplay.Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input of type {type(arg)}; "
                "non-Tensor inputs must use in_dim=None"
            )
        if in_dim < -arg.dim() or in_dim >= arg.dim():
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for a Tensor with dimensionality "
                f"{arg.dim()}"
            )
        if in_dim < 0:
            flat_in_dims[index] = in_dim % arg.dim()

    return (
        _validate_and_get_batch_size(flat_in_dims, flat_args),
        flat_in_dims,
        flat_args,
        args_spec,
    )


def _create_batched_inputs(
    flat_in_dims: list[Optional[int]],
    flat_args: list[Any],
    level: int,
    args_spec: TreeSpec,
) -> tuple[Any, ...]:
    """Attach the current native transform level to mapped Tensor leaves."""

    wrapped = [
        tensorplay._C._transform_make_batched(arg, dim, level)
        if dim is not None
        else arg
        for arg, dim in zip(flat_args, flat_in_dims)
    ]
    return tree_unflatten(wrapped, args_spec)


@contextlib.contextmanager
def vmap_increment_nesting(batch_size: int, randomness: str):
    """Push and pop one native vectorizing layer."""

    level = tensorplay._C._transform_push_vmap(batch_size, randomness)
    token = _vmap_depth.set(_vmap_depth.get() + 1)
    try:
        yield level
    finally:
        try:
            tensorplay._C._transform_pop()
        finally:
            _vmap_depth.reset(token)


def _check_int_or_none(
    value: Any, func: Callable[..., Any], out_dims: out_dims_t
) -> None:
    if isinstance(value, int) or value is None:
        return
    raise ValueError(
        f"vmap({_get_name(func)}, ..., out_dims={out_dims}): out_dims must be "
        "an int, None, or a pytree of ints and None"
    )


def _check_out_dims_is_int_or_int_pytree(
    out_dims: out_dims_t, func: Callable[..., Any]
) -> None:
    if isinstance(out_dims, int):
        return
    tree_map_(
        functools.partial(_check_int_or_none, func=func, out_dims=out_dims),
        out_dims,
    )


def _flat_output_dims(
    output: Any, out_dims: out_dims_t, output_spec: TreeSpec
) -> Optional[list[Optional[int]]]:
    if isinstance(output, tensorplay.Tensor):
        if isinstance(out_dims, int) or out_dims is None:
            return [out_dims]
        if isinstance(out_dims, tuple) and len(out_dims) == 1:
            return [out_dims[0]]
        return None
    return _broadcast_to_and_flatten(out_dims, output_spec)


def _unwrap_batched(
    outputs: Any,
    out_dims: out_dims_t,
    level: int,
    batch_size: int,
    func: Callable[..., Any],
) -> Any:
    """Remove one native level and place its dimension at the public index."""

    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    flat_outputs, output_spec = tree_flatten(outputs)
    flat_out_dims = _flat_output_dims(outputs, out_dims, output_spec)
    if flat_out_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            "out_dims is not compatible with the output structure"
        )

    rebuilt = []
    for output, out_dim in zip(flat_outputs, flat_out_dims):
        if not isinstance(output, tensorplay.Tensor):
            if out_dim is not None:
                raise ValueError(
                    f"vmap({_get_name(func)}, ...): function returned a "
                    f"non-Tensor value of type {type(output)} with out_dim={out_dim}"
                )
            rebuilt.append(output)
            continue

        unwrapped, batch_dim = tensorplay._C._transform_unwrap(output, level)
        if batch_dim is None:
            if out_dim is None:
                rebuilt.append(unwrapped)
                continue
            # The output carries no tag for the current layer: it is a
            # per-sample constant, so re-materialize the batch dimension by
            # unit-expansion at the requested position.
            target_dim = out_dim if out_dim >= 0 else out_dim + unwrapped.dim() + 1
            if target_dim < 0 or target_dim > unwrapped.dim():
                raise ValueError(
                    f"vmap({_get_name(func)}, ...): out_dim={out_dim} is out of range "
                    f"for an unbatched output with {unwrapped.dim()} dimensions"
                )
            shape = list(unwrapped.shape)
            shape.insert(target_dim, batch_size)
            rebuilt.append(unwrapped.unsqueeze(target_dim).expand(shape))
            continue
        if out_dim is None:
            raise ValueError(
                f"vmap({_get_name(func)}, ...): out_dim=None was specified for "
                "a mapped output"
            )

        ndim = unwrapped.dim()
        target_dim = out_dim if out_dim >= 0 else out_dim + ndim
        if target_dim < 0 or target_dim >= ndim:
            raise ValueError(
                f"vmap({_get_name(func)}, ...): out_dim={out_dim} is out of range "
                f"for an output with {ndim} dimensions including the mapped one"
            )
        rebuilt.append(tensorplay.movedim(unwrapped, batch_dim, target_dim))

    return tree_unflatten(rebuilt, output_spec)


def _flat_vmap(
    func: Callable[..., Any],
    batch_size: int,
    flat_in_dims: list[Optional[int]],
    flat_args: list[Any],
    args_spec: TreeSpec,
    out_dims: out_dims_t,
    randomness: str,
    **kwargs: Any,
) -> Any:
    """Execute one callback under one native vectorizing layer."""

    with vmap_increment_nesting(batch_size, randomness) as level:
        batched_args = _create_batched_inputs(
            flat_in_dims, flat_args, level, args_spec
        )
        outputs = func(*batched_args, **kwargs)
        return _unwrap_batched(outputs, out_dims, level, batch_size, func)


def get_chunk_sizes(total_elems: int, chunk_size: int) -> list[int]:
    """Return positive chunk lengths and preserve an empty batch as one chunk."""

    if total_elems < 0:
        raise ValueError("total_elems must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if total_elems == 0:
        return [0]
    n_chunks, remainder = divmod(total_elems, chunk_size)
    sizes = [chunk_size] * n_chunks
    if remainder:
        sizes.append(remainder)
    return sizes


def _get_chunked_inputs(
    flat_args: list[Any],
    flat_in_dims: list[Optional[int]],
    batch_size: int,
    chunk_size: int,
):
    """Split mapped leaves into native-call chunks."""

    chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
    if len(chunk_sizes) == 1:
        return iter(
            [
                tuple(flat_args),
            ]
        )

    split_points = tuple(itertools.accumulate(chunk_sizes[:-1]))
    flat_args_chunks = tuple(
        a.tensor_split(split_points, dim=in_dim)
        if in_dim is not None
        else [a] * len(chunk_sizes)
        for a, in_dim in zip(flat_args, flat_in_dims)
    )
    return zip(*flat_args_chunks)


def _join_chunk_outputs(
    chunk_outputs: list[Any], out_dims: out_dims_t, output_spec: TreeSpec, func: Callable
) -> Any:
    flat_out_dims = _flat_output_dims(chunk_outputs[0], out_dims, output_spec)
    if flat_out_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            "out_dims is not compatible with the output structure"
        )

    flat_chunks = [tree_flatten(result)[0] for result in chunk_outputs]
    joined = []
    for leaf_index, out_dim in enumerate(flat_out_dims):
        pieces = [chunk[leaf_index] for chunk in flat_chunks]
        if out_dim is None:
            reference = pieces[0]
            if isinstance(reference, tensorplay.Tensor):
                for other in pieces[1:]:
                    if not bool((other == reference).all().item()):
                        raise ValueError(
                            f"vmap({_get_name(func)}, chunk_size=...): "
                            "an out_dim=None value differs between chunks"
                        )
            elif any(other != reference for other in pieces[1:]):
                raise ValueError(
                    f"vmap({_get_name(func)}, chunk_size=...): "
                    "an out_dim=None value differs between chunks"
                )
            joined.append(reference)
            continue
        if not isinstance(pieces[0], tensorplay.Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, ...): mapped outputs must be Tensors"
            )
        joined.append(tensorplay.cat(pieces, dim=out_dim))
    return tree_unflatten(joined, output_spec)


def _chunked_vmap(
    func: Callable[..., Any],
    flat_in_dims: list[Optional[int]],
    chunks_flat_args,
    args_spec: TreeSpec,
    out_dims: out_dims_t,
    randomness: str,
    **kwargs: Any,
) -> Any:
    """Run independent native layers for bounded-size chunks."""

    chunk_outputs = []
    output_spec = None
    for flat_args in chunks_flat_args:
        chunk_batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
        result = _flat_vmap(
            func,
            chunk_batch_size,
            flat_in_dims,
            list(flat_args),
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )
        _, result_spec = tree_flatten(result)
        if output_spec is not None and result_spec != output_spec:
            raise ValueError(
                "vmap(..., chunk_size=...): every chunk must return the same "
                "pytree structure"
            )
        output_spec = result_spec
        chunk_outputs.append(result)

    if not chunk_outputs or output_spec is None:
        raise ValueError("vmap: no chunks were produced")
    if len(chunk_outputs) == 1:
        return chunk_outputs[0]
    return _join_chunk_outputs(chunk_outputs, out_dims, output_spec, func)


def vmap_impl(
    func: Callable[..., Any],
    in_dims: in_dims_t,
    out_dims: out_dims_t,
    randomness: str,
    chunk_size: Optional[int],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Entry point used by the public vectorizing API."""

    _check_randomness_arg(randomness)
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
        in_dims, args, func
    )
    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(
                f"vmap: chunk_size should be None or > 0. Got {chunk_size}"
            )
        return _chunked_vmap(
            func,
            flat_in_dims,
            _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size),
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )
    return _flat_vmap(
        func,
        batch_size,
        flat_in_dims,
        flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )


def _restore_outputs(outputs: Any, level: int) -> tuple[Any, Any]:
    flat_outputs, output_spec = tree_flatten(outputs)
    values = []
    dims = []
    for output in flat_outputs:
        if not isinstance(output, tensorplay.Tensor):
            values.append(output)
            dims.append(None)
            continue
        value, batch_dim = tensorplay._C._transform_unwrap(output, level)
        if batch_dim is None:
            values.append(value)
            dims.append(None)
            continue
        values.append(tensorplay.movedim(value, batch_dim, 0))
        dims.append(0)
    return tree_unflatten(values, output_spec), tree_unflatten(dims, output_spec)


def restore_vmap(
    func: Callable,
    in_dims: in_dims_t,
    batch_size: int,
    randomness: str,
) -> Callable:
    """Run a callable once under a pre-sized native batch layer."""

    _check_randomness_arg(randomness)

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        actual_size, flat_dims, flat_args, args_spec = _process_batched_inputs(
            in_dims, args, func
        )
        if actual_size != batch_size:
            raise ValueError(
                f"restore_vmap: expected batch_size={batch_size}, got {actual_size}"
            )
        with vmap_increment_nesting(batch_size, randomness) as level:
            batched_args = _create_batched_inputs(
                flat_dims, flat_args, level, args_spec
            )
            outputs = func(*batched_args, **kwargs)
            return _restore_outputs(outputs, level)

    return wrapped


def wrap_batched(
    args: tuple[Any, ...], bdims: in_dims_t, level: int
) -> tuple[Any, ...]:
    """Attach a supplied native level to a batched input pytree."""

    flat_args, spec = tree_flatten(args)
    flat_dims = _broadcast_to_and_flatten(bdims, spec)
    if flat_dims is None:
        raise ValueError("batch dimensions are not compatible with the inputs")
    return _create_batched_inputs(flat_dims, flat_args, level, spec)


def unwrap_batched(args: Any, level: int) -> tuple[Any, Any]:
    """Remove one native level and return its dimension pytree."""

    flat_args, spec = tree_flatten(args)
    values = []
    dims = []
    for arg in flat_args:
        if not isinstance(arg, tensorplay.Tensor):
            values.append(arg)
            dims.append(None)
            continue
        value, dim = tensorplay._C._transform_unwrap(arg, level)
        values.append(value)
        dims.append(dim)
    return tree_unflatten(values, spec), tree_unflatten(dims, spec)


def _check_randomness_arg(randomness: str) -> None:
    if randomness not in ("error", "different", "same"):
        raise RuntimeError(
            "Only allowed values for randomness are 'error', 'different', or "
            f"'same'. Got {randomness}"
        )
