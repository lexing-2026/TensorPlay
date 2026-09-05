"""Shared argument normalization for the frequency-domain transforms.

Holds the axis / transform-size / normalization bookkeeping that every
transform family needs, so the 1-D, n-D and Hermitian drivers in
:mod:`tensorplay.fft._transforms` only carry their own math.
"""
from tensorplay import (
    cat,
    fft_fft as _c2c_fwd,
    fft_ifft as _c2c_inv,
    view_as_complex as _view_as_complex,
    view_as_real as _view_as_real,
)

__all__ = [
    "apply_c2c",
    "conj",
    "default_dims",
    "norm_mode",
    "normalize_dims",
    "split_last_dim",
    "transform_size",
    "transform_sizes",
]


def norm_mode(norm):
    """Maps the optional ``norm`` argument onto the kernel-level tag."""
    value = "backward" if norm is None else norm
    if value not in ("backward", "forward", "ortho"):
        raise ValueError(
            "norm must be None, 'backward', 'forward', or 'ortho'"
        )
    return value


def transform_size(n):
    """Maps an optional signal length onto the kernel sentinel for "unset"."""
    if n is None:
        return -1
    value = int(n)
    if value == -1:
        return value
    if value <= 0:
        raise ValueError(f"transform size must be positive, got {value}")
    return value


def conj(input):
    """Complex conjugate built from the real/imaginary interleaved view."""
    if not input.is_complex():
        return input
    vr = _view_as_real(input)
    return _view_as_complex(cat([vr[..., :1], vr[..., 1:] * -1], dim=-1).contiguous())


def normalize_dims(dim, ndim):
    """Resolves negative axes and rejects duplicates/out-of-range entries."""
    if ndim <= 0:
        raise ValueError(f"a transformed dimension requires a non-empty input, got {ndim}-D")
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = [int(d) for d in dim]
    if len(dims) == 0:
        raise ValueError("at least one transformed dimension must be specified")
    seen = set()
    out = []
    for d in dims:
        d = d + ndim if d < 0 else d
        if not 0 <= d < ndim:
            raise ValueError(f"dimension {d} out of range for {ndim}-D input")
        if d in seen:
            raise ValueError("duplicate dimensions are not allowed")
        seen.add(d)
        out.append(d)
    return out


def default_dims(input, s):
    """Trailing axes used when ``dim`` is omitted: as many as ``s`` requests."""
    ndim = input.dim()
    if s is None:
        k = ndim
    elif isinstance(s, int):
        k = 1
    else:
        k = len(s)
    if k <= 0:
        raise ValueError("at least one transformed dimension must be specified")
    if k > ndim:
        raise ValueError(
            f"requested {k} transformed dimensions for a {ndim}-D input"
        )
    return list(range(ndim - k, ndim))


def transform_sizes(s, k):
    """Normalizes an optional sequence of transform sizes to a per-dim list."""
    if k <= 0:
        raise ValueError("at least one transformed dimension must be specified")
    if s is None:
        return [None] * k
    if isinstance(s, int):
        if k != 1:
            raise ValueError(
                f"a scalar transform size requires exactly one dimension, got {k}"
            )
        sizes = [int(s)]
    else:
        sizes = [int(v) for v in s]
    if len(sizes) != k:
        raise ValueError(
            f"s ({list(sizes)}) must have the same length as dim ({k})"
        )
    for size in sizes:
        if size != -1 and size <= 0:
            raise ValueError(f"transform size must be positive, got {size}")
    return sizes


def split_last_dim(input, s, dim):
    """Splits ``(dim, s)`` into the leading complex-to-complex axes plus the
    single axis that carries the one-sided (half-spectrum) transform."""
    dims = normalize_dims(dim, input.dim())
    sizes = transform_sizes(s, len(dims))
    return dims[:-1], dims[-1], sizes[:-1], sizes[-1]


def apply_c2c(input, dims, sizes, norm, forward):
    """Chains 1-D complex-to-complex transforms over ``dims``."""
    op = _c2c_fwd if forward else _c2c_inv
    out = input
    for d, n in zip(dims, sizes):
        out = op(out, transform_size(n), d, norm)
    return out
