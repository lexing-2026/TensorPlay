"""The public entry points of the function-transform layer.

These are thin: argument validation happens here so that errors point at the
call the user wrote, and the work is handed to the implementations in
:mod:`tensorplay._transforms.vmap` and
:mod:`tensorplay._transforms.eager_transforms`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Union

from .eager_transforms import grad_and_value_impl, grad_impl
from .utils import argnums_t, exposed_in
from .vmap import (
    _check_out_dims_is_int_or_int_pytree,
    _check_randomness_arg,
    _chunked_vmap,
    _get_chunked_inputs,
    _process_batched_inputs,
    in_dims_t,
    out_dims_t,
    vmap_impl,
)

__all__ = ["vmap", "chunk_vmap", "grad", "grad_and_value"]


def _wraps_without_dynamo_attrs(func: Callable) -> Callable:
    """Copies callable metadata without leaking compiler-only attributes."""

    def decorator(wrapper: Callable) -> Callable:
        wrapped = functools.wraps(func)(wrapper)
        for name in tuple(wrapped.__dict__):
            if name.startswith("_tensorplay_compile"):
                wrapped.__dict__.pop(name, None)
        return wrapped

    return decorator


@exposed_in("tensorplay.func")
def vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size: Optional[int] = None,
) -> Callable:
    """Returns a function that maps ``func`` over an added batch dimension.

    Write the function for a single sample; ``vmap`` handles the batch.  That
    keeps the single-sample logic readable and removes the reshaping and
    unsqueezing that hand-batching otherwise scatters through it.

    Args:
        func (Callable): a function taking one or more arguments, returning one
            or more tensors.
        in_dims (int or nested structure): which dimension of each input to map
            over.  ``None`` marks an argument that is not batched and is passed
            through whole.  The structure must be a prefix of the argument
            structure.  Default: 0.
        out_dims (int or python collection): where the mapped dimension should
            appear in each output.  Default: 0.
        randomness (str): how random operations inside ``func`` behave.  With
            ``"error"`` (the default) they raise, because the intent is
            ambiguous; ``"different"`` draws fresh values per sample, and
            ``"same"`` replays the same values for every sample.
        chunk_size (int, optional): process the batch ``chunk_size`` samples at
            a time to bound peak memory.  ``None`` processes it in one go.

    Example:

        >>> def dot(x, y):
        ...     return (x * y).sum()
        >>> x, y = tensorplay.randn(4, 3), tensorplay.randn(4, 3)
        >>> vmap(dot)(x, y).shape
        tensorplay.Size(4)
    """
    if not callable(func):
        raise TypeError(f"vmap expected a callable, got {type(func)!r}")
    _check_randomness_arg(randomness)
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError(
            f"vmap: chunk_size should be None or greater than 0. Got {chunk_size}"
        )

    @_wraps_without_dynamo_attrs(func)
    def wrapped(*args, **kwargs):
        return vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)

    return wrapped


@exposed_in("tensorplay.func")
def chunk_vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    chunks: int = 2,
) -> Callable:
    """:func:`vmap` splitting the batch into ``chunks`` pieces.

    Prefer ``vmap(..., chunk_size=...)``, which states the bound in samples
    rather than in pieces and so does not change meaning with the batch size.
    """
    _check_randomness_arg(randomness)
    if chunks < 1:
        raise ValueError(f"chunk_vmap: chunks should be greater than 0. Got {chunks}")
    if chunks == 1:
        return vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness)

    @_wraps_without_dynamo_attrs(func)
    def wrapped_with_chunks(*args, **kwargs):
        _check_out_dims_is_int_or_int_pytree(out_dims, func)
        batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
            in_dims, args, func
        )
        # ``chunks`` counts pieces; the chunking machinery counts samples.
        chunk_size = max(1, -(-batch_size // chunks))
        chunks_flat_args = _get_chunked_inputs(
            flat_args, flat_in_dims, batch_size, chunk_size
        )
        return _chunked_vmap(
            func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs
        )

    return wrapped_with_chunks


@exposed_in("tensorplay.func")
def grad(
    func: Callable, argnums: argnums_t = 0, has_aux: bool = False
) -> Callable:
    """Returns a function computing the gradient of ``func``.

    ``func`` must return a scalar tensor; the returned function has the same
    signature and returns the gradient with respect to ``argnums``.  Because it
    is again an ordinary function of the same inputs, ``grad(grad(f))`` is the
    second derivative.

    Args:
        func (Callable): a function returning a single-element tensor.
        argnums (int or Tuple[int]): which positional arguments to
            differentiate with respect to.  Default: 0.
        has_aux (bool): whether ``func`` returns ``(output, aux)``, where
            ``aux`` is carried through undifferentiated.

    Example:

        >>> x = tensorplay.randn([])
        >>> grad(tensorplay.sin)(x)
    """
    if not callable(func):
        raise TypeError(f"grad expected a callable, got {type(func)!r}")

    @_wraps_without_dynamo_attrs(func)
    def wrapper(*args, **kwargs):
        return grad_impl(func, argnums, has_aux, args, kwargs)

    return wrapper


@exposed_in("tensorplay.func")
def grad_and_value(
    func: Callable, argnums: argnums_t = 0, has_aux: bool = False
) -> Callable:
    """Returns a function computing both the gradient of ``func`` and its value.

    The value comes from the same forward pass the gradient needs, so this
    costs no more than :func:`grad` alone and saves evaluating ``func`` twice.

    Returns a function producing ``(gradient, value)``, or
    ``(gradient, (value, aux))`` when ``has_aux`` is set.
    """
    if not callable(func):
        raise TypeError(f"grad_and_value expected a callable, got {type(func)!r}")

    @_wraps_without_dynamo_attrs(func)
    def wrapper(*args, **kwargs):
        return grad_and_value_impl(func, argnums, has_aux, args, kwargs)

    return wrapper
