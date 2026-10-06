"""Forward and inverse discrete Fourier transforms.

Every transform is a native op: the 1-D and 2-D families have their own
kernels, and the n-D and Hermitian families compose them natively, so
gradients, ``out=`` and compiled capture behave the same for all of them.
"""
from tensorplay import (
    fft_fft as _fft,
    fft_fft2 as _fft2,
    fft_fftn as _fftn,
    fft_hfft as _hfft,
    fft_hfft2 as _hfft2,
    fft_hfftn as _hfftn,
    fft_ifft as _ifft,
    fft_ifft2 as _ifft2,
    fft_ifftn as _ifftn,
    fft_ihfft as _ihfft,
    fft_ihfft2 as _ihfft2,
    fft_ihfftn as _ihfftn,
    fft_irfft as _irfft,
    fft_irfft2 as _irfft2,
    fft_irfftn as _irfftn,
    fft_rfft as _rfft,
    fft_rfft2 as _rfft2,
    fft_rfftn as _rfftn,
)

from ._helpers import int_list, norm_arg, signal_length

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

def fft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the one-dimensional discrete Fourier transform.

    Args:
        input (Tensor): the input tensor
        n (int, optional): signal length; zero-pads/truncates :attr:`dim`
        dim (int, optional): the dimension to transform. Default: -1
        norm (str, optional): ``"backward"``, ``"forward"`` or ``"ortho"``.
            Default: ``None`` (= ``"backward"``)
    """
    return _fft(input, signal_length(n), dim, norm_arg(norm), out=out)


def ifft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the one-dimensional inverse discrete Fourier transform."""
    return _ifft(input, signal_length(n), dim, norm_arg(norm), out=out)


def rfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the one-dimensional FFT of real input, one-sided output."""
    return _rfft(input, signal_length(n), dim, norm_arg(norm), out=out)


def irfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the inverse of :func:`rfft`; :attr:`n` is the output length."""
    return _irfft(input, signal_length(n), dim, norm_arg(norm), out=out)


def hfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the 1-D FFT of a Hermitian-symmetric spectrum; real output.

    Equivalent to :func:`irfft` applied to ``input.conj()``; :attr:`n` is the
    output length (default ``2 * (input.size(dim) - 1)``).
    """
    return _hfft(input, n, dim, norm, out=out)


def ihfft(input, n=None, dim=-1, norm=None, *, out=None):
    """Computes the inverse of :func:`hfft`; one-sided complex output.

    Equivalent to the conjugate of :func:`rfft` of the real input, scaled
    as an inverse transform; :attr:`n` zero-pads/truncates the input along
    :attr:`dim`.
    """
    return _ihfft(input, n, dim, norm, out=out)


# ---------------------------------------------------------------------------
# Complex-to-complex 2-D / n-D transforms
# ---------------------------------------------------------------------------

def fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the two-dimensional discrete Fourier transform."""
    return _fft2(input, int_list(s), int_list(dim), norm_arg(norm), out=out)


def ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the two-dimensional inverse discrete Fourier transform."""
    return _ifft2(input, int_list(s), int_list(dim), norm_arg(norm), out=out)


def fftn(input, s=None, dim=None, norm=None, *, out=None):
    """Computes the N-dimensional discrete Fourier transform over :attr:`dim`.

    :attr:`dim` defaults to the last ``len(s)`` dimensions when :attr:`s` is
    given and to every dimension otherwise; an entry of ``-1`` in :attr:`s`
    keeps that dimension's length.
    """
    return _fftn(input, int_list(s), int_list(dim), norm, out=out)


def ifftn(input, s=None, dim=None, norm=None, *, out=None):
    """Computes the N-dimensional inverse discrete Fourier transform."""
    return _ifftn(input, int_list(s), int_list(dim), norm, out=out)


# ---------------------------------------------------------------------------
# Real-to-complex / complex-to-real families (one-sided on the last dim)
# ---------------------------------------------------------------------------

def rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the two-dimensional FFT of real input."""
    return _rfft2(input, int_list(s), int_list(dim), norm_arg(norm), out=out)


def irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Computes the inverse of :func:`rfft2`."""
    return _irfft2(input, int_list(s), int_list(dim), norm_arg(norm), out=out)


def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    """N-dimensional FFT of real input; one-sided along the last listed dim."""
    return _rfftn(input, int_list(s), int_list(dim), norm, out=out)


def irfftn(input, s=None, dim=None, norm=None, *, out=None):
    """Inverse of :func:`rfftn`; :attr:`s[-1]` is the real output size
    (default ``2 * (input.size(dim[-1]) - 1)``)."""
    return _irfftn(input, int_list(s), int_list(dim), norm, out=out)


# ---------------------------------------------------------------------------
# Hermitian 2-D / n-D families
# ---------------------------------------------------------------------------

def hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Two-dimensional FFT of a Hermitian-symmetric spectrum; real output."""
    return _hfft2(input, int_list(s), int_list(dim), norm, out=out)


def ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Two-dimensional counterpart of :func:`ihfft`."""
    return _ihfft2(input, int_list(s), int_list(dim), norm, out=out)


def hfftn(input, s=None, dim=None, norm=None, *, out=None):
    """N-dimensional FFT of a Hermitian-symmetric spectrum; real output.

    The forward transform over the leading transformed dimensions, then
    :func:`hfft` (conjugate + complex-to-real) along the final one.
    """
    return _hfftn(input, int_list(s), int_list(dim), norm, out=out)


def ihfftn(input, s=None, dim=None, norm=None, *, out=None):
    """Inverse of :func:`hfftn`: :func:`ihfft` along the final transformed
    dimension, then the inverse transform over the remaining dimensions."""
    return _ihfftn(input, int_list(s), int_list(dim), norm, out=out)
