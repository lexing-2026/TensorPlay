"""Fourier transforms and related frequency-domain utilities.

The one-dimensional and two-dimensional transforms are dispatched to the
compiled spectral kernels. Higher-dimensional and Hermitian helpers retain
the same axis and normalization conventions while using the available native
building blocks.
"""
from tensorplay import (
    Tensor,
    arange,
    cat,
    fft_fft as _c2c_fwd,
    fft_fft2 as _fft2_native,
    fft_ifft as _c2c_inv,
    fft_ifft2 as _ifft2_native,
    fft_irfft as _c2r,
    fft_irfft2 as _irfft2_native,
    fft_rfft as _r2c,
    fft_rfft2 as _rfft2_native,
    view_as_complex as _view_as_complex,
    view_as_real as _view_as_real,
)
from tensorplay._C import DType

__all__ = [
    "Tensor",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]


def _norm(norm):
    return "backward" if norm is None else norm


def _n(n):
    return -1 if n is None else int(n)


def _conj(input):
    if not input.is_complex():
        return input
    vr = _view_as_real(input)
    return _view_as_complex(cat([vr[..., :1], vr[..., 1:] * -1], dim=-1).contiguous())


def _normalize_dims(dim, ndim):
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = [int(d) for d in dim]
    if len(dims) == 0:
        raise ValueError("at least one transformed dimension must be specified")
    seen = set()
    out = []
    for d in dims:
        d = d % ndim if d < 0 else d
        if not 0 <= d < ndim:
            raise ValueError(f"dimension {d} out of range for {ndim}-D input")
        if d in seen:
            raise ValueError("duplicate dimensions are not allowed")
        seen.add(d)
        out.append(d)
    return out


def _default_dims(input, s):
    ndim = input.dim()
    k = len(s) if s is not None else ndim
    return list(range(ndim - k, ndim))


def _sizes(s, k):
    # Normalize an optional sequence of transform sizes to a per-dim list.
    if s is None:
        return [None] * k
    if isinstance(s, int):
        return [int(s)] + [None] * (k - 1)
    sizes = [int(v) for v in s]
    if len(sizes) != k:
        raise ValueError(
            f"s ({list(sizes)}) must have the same length as dim ({k})"
        )
    return sizes


# ---------------------------------------------------------------------------
# 1-D transforms
# ---------------------------------------------------------------------------

def fft(input, n=None, dim=-1, norm=None):
    """Computes the one-dimensional discrete Fourier transform.

    Args:
        input (Tensor): the input tensor
        n (int, optional): signal length; zero-pads/truncates :attr:`dim`
        dim (int, optional): the dimension to transform. Default: -1
        norm (str, optional): ``"backward"``, ``"forward"`` or ``"ortho"``.
            Default: ``None`` (= ``"backward"``)
    """
    return _c2c_fwd(input, _n(n), dim, _norm(norm))


def ifft(input, n=None, dim=-1, norm=None):
    """Computes the one-dimensional inverse discrete Fourier transform."""
    return _c2c_inv(input, _n(n), dim, _norm(norm))


def rfft(input, n=None, dim=-1, norm=None):
    """Computes the one-dimensional FFT of real input, one-sided output."""
    return _r2c(input, _n(n), dim, _norm(norm))


def irfft(input, n=None, dim=-1, norm=None):
    """Computes the inverse of :func:`rfft`; :attr:`n` is the output length."""
    return _c2r(input, _n(n), dim, _norm(norm))


def hfft(input, n=None, dim=-1, norm=None):
    """Computes the 1-D FFT of a Hermitian-symmetric spectrum; real output.

    Equivalent to :func:`irfft` applied to ``input.conj()``; :attr:`n` is the
    output length (default ``2 * (input.size(dim) - 1)``).
    """
    return _c2r(_conj(input), _n(n), dim, _norm(norm))


def ihfft(input, n=None, dim=-1, norm=None):
    """Computes the inverse of :func:`hfft`; one-sided complex output.

    Equivalent to :func:`rfft` applied to ``input.conj()``; :attr:`n`
    zero-pads/truncates the real input along :attr:`dim`.
    """
    return _r2c(_conj(input), _n(n), dim, _norm(norm))


# ---------------------------------------------------------------------------
# n-D composition helpers
# ---------------------------------------------------------------------------

def _apply_c2c(input, dims, sizes, norm, forward):
    op = _c2c_fwd if forward else _c2c_inv
    out = input
    for d, n in zip(dims, sizes):
        if n is not None:
            out = op(out, int(n), d, norm)
        else:
            out = op(out, -1, d, norm)
    return out


# ---------------------------------------------------------------------------
# 2-D transforms
# ---------------------------------------------------------------------------

def fft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the two-dimensional discrete Fourier transform."""
    return _fft2_native(input, None if s is None else list(s), list(dim), _norm(norm))


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the two-dimensional inverse discrete Fourier transform."""
    return _ifft2_native(input, None if s is None else list(s), list(dim), _norm(norm))


def fftn(input, s=None, dim=None, norm=None):
    """Computes the N-dimensional discrete Fourier transform over :attr:`dim`."""
    if dim is None:
        dim = _default_dims(input, s)
    if len(dim) == 2:
        return fft2(input, s, tuple(dim), norm)
    dims = _normalize_dims(list(dim), input.dim())
    sizes = _sizes(s, len(dims))
    return _apply_c2c(input, dims, sizes, _norm(norm), forward=True)


def ifftn(input, s=None, dim=None, norm=None):
    """Computes the N-dimensional inverse discrete Fourier transform."""
    if dim is None:
        dim = _default_dims(input, s)
    if len(dim) == 2:
        return ifft2(input, s, tuple(dim), norm)
    dims = _normalize_dims(list(dim), input.dim())
    sizes = _sizes(s, len(dims))
    return _apply_c2c(input, dims, sizes, _norm(norm), forward=False)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real n-D families (one-sided on the last dim)
# ---------------------------------------------------------------------------

def _split_last_dim(input, s, dim):
    dims = _normalize_dims(list(dim), input.dim())
    sizes = _sizes(s, len(dims))
    return dims[:-1], dims[-1], sizes[:-1], sizes[-1]


def rfft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the two-dimensional FFT of real input."""
    return _rfft2_native(input, None if s is None else list(s), list(dim), _norm(norm))


def irfft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the inverse of :func:`rfft2`."""
    return _irfft2_native(input, None if s is None else list(s), list(dim), _norm(norm))


def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    """N-dimensional FFT of real input; one-sided along the last listed dim."""
    if out is not None:
        raise NotImplementedError("rfftn: out= is not supported")
    if dim is None:
        dim = _default_dims(input, s)
    if len(dim) == 2:
        return rfft2(input, s, tuple(dim), norm)
    dims = _normalize_dims(list(dim), input.dim())
    sizes = _sizes(s, len(dims))
    rest_dims, last_dim = dims[:-1], dims[-1]
    rest_sizes, last_size = sizes[:-1], sizes[-1]
    out_t = _r2c(input, _n(last_size), last_dim, _norm(norm))
    return _apply_c2c(out_t, rest_dims, rest_sizes, _norm(norm), forward=True)


def irfftn(input, s=None, dim=None, norm=None, *, out=None):
    """Inverse of :func:`rfftn`; :attr:`s[-1]` is the real output size."""
    if out is not None:
        raise NotImplementedError("irfftn: out= is not supported")
    if dim is None:
        dim = _default_dims(input, s)
    if len(dim) == 2:
        return irfft2(input, s, tuple(dim), norm)
    rest_dims, last_dim, rest_sizes, last_size = _split_last_dim(input, s, dim)
    out_t = _c2r(input, _n(last_size), last_dim, _norm(norm))
    return _apply_c2c(out_t, rest_dims, rest_sizes, _norm(norm), forward=False)


# ---------------------------------------------------------------------------
# Hermitian n-D families
# ---------------------------------------------------------------------------

def hfft2(input, s=None, dim=(-2, -1), norm=None):
    """Two-dimensional inverse of a Hermitian-symmetric spectrum; real output."""
    return hfftn(input, s, list(dim), norm)


def ihfft2(input, s=None, dim=(-2, -1), norm=None):
    """Two-dimensional counterpart of :func:`ihfft`."""
    return ihfftn(input, s, list(dim), norm)


def hfftn(input, s=None, dim=None, norm=None):
    """N-dimensional FFT of a Hermitian-symmetric spectrum; real output.

    Applies :func:`hfft` (conjugate + complex-to-real) along the final
    transformed dimension, then :func:`ifft` over the remaining dimensions.
    """
    if dim is None:
        dim = _default_dims(input, s)
    rest_dims, last_dim, rest_sizes, last_size = _split_last_dim(input, s, dim)
    out = _c2r(_conj(input), _n(last_size), last_dim, _norm(norm))
    return _apply_c2c(out, rest_dims, rest_sizes, _norm(norm), forward=False)


def ihfftn(input, s=None, dim=None, norm=None):
    """Inverse of :func:`hfftn`: :func:`ihfft` along the final transformed
    dimension, then :func:`fft` over the remaining dimensions."""
    if dim is None:
        dim = _default_dims(input, s)
    rest_dims, last_dim, rest_sizes, last_size = _split_last_dim(input, s, dim)
    out = _r2c(_conj(input), _n(last_size), last_dim, _norm(norm))
    return _apply_c2c(out, rest_dims, rest_sizes, _norm(norm), forward=True)


# ---------------------------------------------------------------------------
# Frequency / shift helpers
# ---------------------------------------------------------------------------

def fftfreq(n, d=1.0, *, dtype=DType.float32, device=None):
    """DFT sample frequencies (cycles/unit): ``[0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)``.

    Args:
        n (int): window length
        d (float, optional): sample spacing. Default: 1.0
        dtype / device: forwarded to the factory ops. Default: float32/CPU
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    pos = arange((n + 1) // 2, dtype=dtype, device=device)
    neg = arange(-(n // 2), 0, dtype=dtype, device=device)
    step = d / (n * d)
    return cat([pos, neg]) * step


def rfftfreq(n, d=1.0, *, dtype=DType.float32, device=None):
    """Sample frequencies for :func:`rfft`/one-sided transforms: ``[0..n//2] / (n*d)``."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    val = 1.0 / (n * d)
    return arange(n // 2 + 1, dtype=dtype, device=device) * val


def fftshift(input, dim=None):
    """Re-orders an N-D FFT output so the zero-frequency term is centered.

    Shifts by ``+n // 2`` along each (or the given) dimension(s).
    """
    ndim = input.dim()
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim % ndim]
    else:
        dims = [d % ndim for d in dim]
    out = input
    for d in dims:
        n = out.size(d)
        if n < 2:
            continue
        k = n // 2
        out = cat([out.narrow(d, k, n - k), out.narrow(d, 0, k)], dim=d)
    return out


def ifftshift(input, dim=None):
    """Inverse of :func:`fftshift`; shifts by ``-(n // 2)`` (odd-safe)."""
    ndim = input.dim()
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim % ndim]
    else:
        dims = [d % ndim for d in dim]
    out = input
    for d in dims:
        n = out.size(d)
        if n < 2:
            continue
        k = n // 2
        out = cat([out.narrow(d, n - k, k), out.narrow(d, 0, n - k)], dim=d)
    return out
