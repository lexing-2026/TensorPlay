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


def _copy_to_out(result, out):
    if out is not None:
        if out.dtype != result.dtype:
            raise TypeError(
                f"out has dtype {out.dtype}, but the result has dtype {result.dtype}"
            )
        if out.device != result.device:
            raise RuntimeError(
                f"out is on {out.device}, but the result is on {result.device}"
            )
        if tuple(out.shape) != tuple(result.shape):
            out.resize_(result.shape)
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# 1-D transforms
# ---------------------------------------------------------------------------

def fft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the one-dimensional discrete Fourier transform.

    Args:
        input (Tensor): the input tensor
        n (int, optional): signal length; zero-pads/truncates :attr:`dim`
        dim (int, optional): the dimension to transform. Default: -1
        norm (str, optional): ``"backward"``, ``"forward"`` or ``"ortho"``.
            Default: ``None`` (= ``"backward"``)
    """
    result = _c2c_fwd(input, transform_size(n), dim, norm_mode(norm))
    return _copy_to_out(result, out)


def ifft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the one-dimensional inverse discrete Fourier transform."""
    result = _c2c_inv(input, transform_size(n), dim, norm_mode(norm))
    return _copy_to_out(result, out)


def rfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the one-dimensional FFT of real input, one-sided output."""
    result = _r2c(input, transform_size(n), dim, norm_mode(norm))
    return _copy_to_out(result, out)


def irfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the inverse of :func:`rfft`; :attr:`n` is the output length."""
    result = _c2r(input, transform_size(n), dim, norm_mode(norm))
    return _copy_to_out(result, out)


def hfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the 1-D FFT of a Hermitian-symmetric spectrum; real output.

    Equivalent to :func:`irfft` applied to ``input.conj()``; :attr:`n` is the
    output length (default ``2 * (input.size(dim) - 1)``).
    """
    result = _c2r(conj(input), transform_size(n), dim, norm_mode(norm))
    return _copy_to_out(result, out)


def ihfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the inverse of :func:`hfft`; one-sided complex output.

    Equivalent to :func:`rfft` applied to ``input.conj()``; :attr:`n`
    zero-pads/truncates the real input along :attr:`dim`.
    """
    result = _r2c(conj(input), transform_size(n), dim, norm_mode(norm))
    return _copy_to_out(result, out)


# ---------------------------------------------------------------------------
# Complex-to-complex 2-D / n-D transforms
# ---------------------------------------------------------------------------

def fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the two-dimensional discrete Fourier transform."""
    dims = normalize_dims(dim, input.dim())
    if len(dims) != 2:
        raise ValueError("fft2 expects exactly two transformed dimensions")
    sizes = transform_sizes(s, 2) if s is not None else None
    result = _fft2_native(
        input, sizes, dims, norm_mode(norm))
    return _copy_to_out(result, out)


def ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the two-dimensional inverse discrete Fourier transform."""
    dims = normalize_dims(dim, input.dim())
    if len(dims) != 2:
        raise ValueError("ifft2 expects exactly two transformed dimensions")
    sizes = transform_sizes(s, 2) if s is not None else None
    result = _ifft2_native(
        input, sizes, dims, norm_mode(norm))
    return _copy_to_out(result, out)


def fftn(input, s=None, dim=None, norm=None, *, out=None):
    """Computes the N-dimensional discrete Fourier transform over :attr:`dim`."""
    dims = default_dims(input, s) if dim is None else normalize_dims(dim, input.dim())
    if len(dims) == 2:
        result = fft2(input, s, dims, norm)
    else:
        sizes = transform_sizes(s, len(dims))
        result = apply_c2c(input, dims, sizes, norm_mode(norm), forward=True)
    return _copy_to_out(result, out)


def ifftn(input, s=None, dim=None, norm=None, *, out=None):
    """Computes the N-dimensional inverse discrete Fourier transform."""
    dims = default_dims(input, s) if dim is None else normalize_dims(dim, input.dim())
    if len(dims) == 2:
        result = ifft2(input, s, dims, norm)
    else:
        sizes = transform_sizes(s, len(dims))
        result = apply_c2c(input, dims, sizes, norm_mode(norm), forward=False)
    return _copy_to_out(result, out)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real families (one-sided on the last dim)
# ---------------------------------------------------------------------------

def rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the two-dimensional FFT of real input."""
    dims = normalize_dims(dim, input.dim())
    if len(dims) != 2:
        raise ValueError("rfft2 expects exactly two transformed dimensions")
    sizes = transform_sizes(s, 2) if s is not None else None
    result = _rfft2_native(
        input, sizes, dims, norm_mode(norm))
    return _copy_to_out(result, out)


def irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the inverse of :func:`rfft2`."""
    dims = normalize_dims(dim, input.dim())
    if len(dims) != 2:
        raise ValueError("irfft2 expects exactly two transformed dimensions")
    sizes = transform_sizes(s, 2) if s is not None else None
    result = _irfft2_native(
        input, sizes, dims, norm_mode(norm))
    return _copy_to_out(result, out)


def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    """N-dimensional FFT of real input; one-sided along the last listed dim."""
    dims = default_dims(input, s) if dim is None else normalize_dims(dim, input.dim())
    if len(dims) == 2:
        result = rfft2(input, s, dims, norm)
    else:
        sizes = transform_sizes(s, len(dims))
        rest_dims, last_dim = dims[:-1], dims[-1]
        rest_sizes, last_size = sizes[:-1], sizes[-1]
        out_t = _r2c(input, transform_size(last_size), last_dim, norm_mode(norm))
        result = apply_c2c(out_t, rest_dims, rest_sizes, norm_mode(norm), forward=True)
    return _copy_to_out(result, out)


def irfftn(input, s=None, dim=None, norm=None, *, out=None):
    """Inverse of :func:`rfftn`; :attr:`s[-1]` is the real output size."""
    dims = default_dims(input, s) if dim is None else normalize_dims(dim, input.dim())
    if len(dims) == 2:
        result = irfft2(input, s, dims, norm)
    else:
        rest_dims, last_dim, rest_sizes, last_size = split_last_dim(input, s, dims)
        out_t = _c2r(input, transform_size(last_size), last_dim, norm_mode(norm))
        result = apply_c2c(out_t, rest_dims, rest_sizes, norm_mode(norm), forward=False)
    return _copy_to_out(result, out)


# ---------------------------------------------------------------------------
# Hermitian n-D families
# ---------------------------------------------------------------------------

def hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Two-dimensional inverse of a Hermitian-symmetric spectrum; real output."""
    return hfftn(input, s, dim, norm, out=out)


def ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Two-dimensional counterpart of :func:`ihfft`."""
    return ihfftn(input, s, dim, norm, out=out)


def hfftn(input, s=None, dim=None, norm=None, *, out=None):
    """N-dimensional FFT of a Hermitian-symmetric spectrum; real output.

    Applies :func:`hfft` (conjugate + complex-to-real) along the final
    transformed dimension, then :func:`ifft` over the remaining dimensions.
    """
    dims = default_dims(input, s) if dim is None else normalize_dims(dim, input.dim())
    rest_dims, last_dim, rest_sizes, last_size = split_last_dim(input, s, dims)
    result = _c2r(
        conj(input), transform_size(last_size), last_dim, norm_mode(norm))
    result = apply_c2c(
        result, rest_dims, rest_sizes, norm_mode(norm), forward=False)
    return _copy_to_out(result, out)


def ihfftn(input, s=None, dim=None, norm=None, *, out=None):
    """Inverse of :func:`hfftn`: :func:`ihfft` along the final transformed
    dimension, then :func:`fft` over the remaining dimensions."""
    dims = default_dims(input, s) if dim is None else normalize_dims(dim, input.dim())
    rest_dims, last_dim, rest_sizes, last_size = split_last_dim(input, s, dims)
    result = _r2c(
        conj(input), transform_size(last_size), last_dim, norm_mode(norm))
    result = apply_c2c(
        result, rest_dims, rest_sizes, norm_mode(norm), forward=True)
    return _copy_to_out(result, out)
