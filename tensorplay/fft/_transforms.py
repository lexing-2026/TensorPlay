"""Forward and inverse discrete Fourier transforms.

The 1-D and 2-D transforms dispatch straight to the compiled spectral
kernels.  The n-D and Hermitian families compose those kernels while keeping
the same axis and normalization conventions.
"""
from tensorplay import (
    fft_fft as _c2c_fwd,
    fft_fft2 as _fft2_native,
    fft_ifft as _c2c_inv,
    fft_ifft2 as _ifft2_native,
    fft_irfft as _c2r,
    fft_irfft2 as _irfft2_native,
    fft_rfft as _r2c,
    fft_rfft2 as _rfft2_native,
)

from ._helpers import (
    apply_c2c,
    conj,
    default_dims,
    norm_mode,
    normalize_dims,
    split_last_dim,
    transform_size,
    transform_sizes,
)

__all__ = [
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
]


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
    return _c2c_fwd(input, transform_size(n), dim, norm_mode(norm))


def ifft(input, n=None, dim=-1, norm=None):
    """Computes the one-dimensional inverse discrete Fourier transform."""
    return _c2c_inv(input, transform_size(n), dim, norm_mode(norm))


def rfft(input, n=None, dim=-1, norm=None):
    """Computes the one-dimensional FFT of real input, one-sided output."""
    return _r2c(input, transform_size(n), dim, norm_mode(norm))


def irfft(input, n=None, dim=-1, norm=None):
    """Computes the inverse of :func:`rfft`; :attr:`n` is the output length."""
    return _c2r(input, transform_size(n), dim, norm_mode(norm))


def hfft(input, n=None, dim=-1, norm=None):
    """Computes the 1-D FFT of a Hermitian-symmetric spectrum; real output.

    Equivalent to :func:`irfft` applied to ``input.conj()``; :attr:`n` is the
    output length (default ``2 * (input.size(dim) - 1)``).
    """
    return _c2r(conj(input), transform_size(n), dim, norm_mode(norm))


def ihfft(input, n=None, dim=-1, norm=None):
    """Computes the inverse of :func:`hfft`; one-sided complex output.

    Equivalent to :func:`rfft` applied to ``input.conj()``; :attr:`n`
    zero-pads/truncates the real input along :attr:`dim`.
    """
    return _r2c(conj(input), transform_size(n), dim, norm_mode(norm))


# ---------------------------------------------------------------------------
# Complex-to-complex 2-D / n-D transforms
# ---------------------------------------------------------------------------

def fft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the two-dimensional discrete Fourier transform."""
    return _fft2_native(input, None if s is None else list(s), list(dim), norm_mode(norm))


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the two-dimensional inverse discrete Fourier transform."""
    return _ifft2_native(input, None if s is None else list(s), list(dim), norm_mode(norm))


def fftn(input, s=None, dim=None, norm=None):
    """Computes the N-dimensional discrete Fourier transform over :attr:`dim`."""
    if dim is None:
        dim = default_dims(input, s)
    if len(dim) == 2:
        return fft2(input, s, tuple(dim), norm)
    dims = normalize_dims(list(dim), input.dim())
    sizes = transform_sizes(s, len(dims))
    return apply_c2c(input, dims, sizes, norm_mode(norm), forward=True)


def ifftn(input, s=None, dim=None, norm=None):
    """Computes the N-dimensional inverse discrete Fourier transform."""
    if dim is None:
        dim = default_dims(input, s)
    if len(dim) == 2:
        return ifft2(input, s, tuple(dim), norm)
    dims = normalize_dims(list(dim), input.dim())
    sizes = transform_sizes(s, len(dims))
    return apply_c2c(input, dims, sizes, norm_mode(norm), forward=False)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real families (one-sided on the last dim)
# ---------------------------------------------------------------------------

def rfft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the two-dimensional FFT of real input."""
    return _rfft2_native(input, None if s is None else list(s), list(dim), norm_mode(norm))


def irfft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the inverse of :func:`rfft2`."""
    return _irfft2_native(input, None if s is None else list(s), list(dim), norm_mode(norm))


def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    """N-dimensional FFT of real input; one-sided along the last listed dim."""
    if out is not None:
        raise NotImplementedError("rfftn: out= is not supported")
    if dim is None:
        dim = default_dims(input, s)
    if len(dim) == 2:
        return rfft2(input, s, tuple(dim), norm)
    dims = normalize_dims(list(dim), input.dim())
    sizes = transform_sizes(s, len(dims))
    rest_dims, last_dim = dims[:-1], dims[-1]
    rest_sizes, last_size = sizes[:-1], sizes[-1]
    out_t = _r2c(input, transform_size(last_size), last_dim, norm_mode(norm))
    return apply_c2c(out_t, rest_dims, rest_sizes, norm_mode(norm), forward=True)


def irfftn(input, s=None, dim=None, norm=None, *, out=None):
    """Inverse of :func:`rfftn`; :attr:`s[-1]` is the real output size."""
    if out is not None:
        raise NotImplementedError("irfftn: out= is not supported")
    if dim is None:
        dim = default_dims(input, s)
    if len(dim) == 2:
        return irfft2(input, s, tuple(dim), norm)
    rest_dims, last_dim, rest_sizes, last_size = split_last_dim(input, s, dim)
    out_t = _c2r(input, transform_size(last_size), last_dim, norm_mode(norm))
    return apply_c2c(out_t, rest_dims, rest_sizes, norm_mode(norm), forward=False)


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
        dim = default_dims(input, s)
    rest_dims, last_dim, rest_sizes, last_size = split_last_dim(input, s, dim)
    out = _c2r(conj(input), transform_size(last_size), last_dim, norm_mode(norm))
    return apply_c2c(out, rest_dims, rest_sizes, norm_mode(norm), forward=False)


def ihfftn(input, s=None, dim=None, norm=None):
    """Inverse of :func:`hfftn`: :func:`ihfft` along the final transformed
    dimension, then :func:`fft` over the remaining dimensions."""
    if dim is None:
        dim = default_dims(input, s)
    rest_dims, last_dim, rest_sizes, last_size = split_last_dim(input, s, dim)
    out = _r2c(conj(input), transform_size(last_size), last_dim, norm_mode(norm))
    return apply_c2c(out, rest_dims, rest_sizes, norm_mode(norm), forward=True)
