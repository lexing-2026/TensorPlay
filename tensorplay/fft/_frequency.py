"""Frequency-grid construction and spectrum re-ordering helpers."""
from tensorplay import (
    fft_fftfreq as _fftfreq,
    fft_fftshift as _fftshift,
    fft_ifftshift as _ifftshift,
    fft_rfftfreq as _rfftfreq,
)

from ._helpers import int_list

__all__ = ["fftfreq", "rfftfreq", "fftshift", "ifftshift"]


def fftfreq(n, d=1.0, *, out=None, dtype=None, layout=None, device=None,
            requires_grad=False):
    """DFT sample frequencies (cycles/unit): ``[0, 1, ..., (n-1)//2, -(n//2), ..., -1] / (n*d)``.

    Args:
        n (int): window length
        d (float, optional): sample spacing. Default: 1.0
        dtype / layout / device: of the result. Default: the default
            floating dtype, on the CPU
    """
    result = _fftfreq(n, d, dtype=dtype, layout=layout, device=device, out=out)
    return result.requires_grad_() if requires_grad else result


def rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=None, device=None,
             requires_grad=False):
    """Sample frequencies for :func:`rfft`/one-sided transforms: ``[0..n//2] / (n*d)``."""
    result = _rfftfreq(n, d, dtype=dtype, layout=layout, device=device, out=out)
    return result.requires_grad_() if requires_grad else result


def fftshift(input, dim=None):
    """Re-orders an N-D FFT output so the zero-frequency term is centered.

    Rolls by ``n // 2`` along each (or the given) dimension(s).
    """
    return _fftshift(input, int_list(dim))


def ifftshift(input, dim=None):
    """Inverse of :func:`fftshift`; rolls by ``(n + 1) // 2`` (odd-safe)."""
    return _ifftshift(input, int_list(dim))
