"""Frequency-grid construction and spectrum re-ordering helpers."""
from tensorplay import arange, cat
from tensorplay._C import DType

from ._helpers import normalize_dims

__all__ = ["fftfreq", "rfftfreq", "fftshift", "ifftshift"]


def _copy_to_out(result, out):
    if out is None:
        return result
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


def fftfreq(n, d=1.0, *, dtype=DType.float32, device=None, out=None):
    """DFT sample frequencies (cycles/unit): ``[0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)``.

    Args:
        n (int): window length
        d (float, optional): sample spacing. Default: 1.0
        dtype / device: forwarded to the factory ops. Default: float32/CPU
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if d == 0:
        raise ValueError("d must be non-zero")
    pos = arange((n + 1) // 2, dtype=dtype, device=device)
    neg = arange(-(n // 2), 0, dtype=dtype, device=device)
    result = cat([pos, neg]) * (1.0 / (n * d))
    return _copy_to_out(result, out)


def rfftfreq(n, d=1.0, *, dtype=DType.float32, device=None, out=None):
    """Sample frequencies for :func:`rfft`/one-sided transforms: ``[0..n//2] / (n*d)``."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if d == 0:
        raise ValueError("d must be non-zero")
    val = 1.0 / (n * d)
    result = arange(n // 2 + 1, dtype=dtype, device=device) * val
    return _copy_to_out(result, out)


def _shift_dims(input, dim):
    ndim = input.dim()
    if dim is None:
        return list(range(ndim))
    return normalize_dims(dim, ndim)


def fftshift(input, dim=None):
    """Re-orders an N-D FFT output so the zero-frequency term is centered.

    Shifts by ``+n // 2`` along each (or the given) dimension(s).
    """
    out = input
    for d in _shift_dims(input, dim):
        n = out.size(d)
        if n < 2:
            continue
        k = n // 2
        out = cat([out.narrow(d, k, n - k), out.narrow(d, 0, k)], dim=d)
    return out


def ifftshift(input, dim=None):
    """Inverse of :func:`fftshift`; shifts by ``-(n // 2)`` (odd-safe)."""
    out = input
    for d in _shift_dims(input, dim):
        n = out.size(d)
        if n < 2:
            continue
        k = n // 2
        out = cat([out.narrow(d, n - k, k), out.narrow(d, 0, n - k)], dim=d)
    return out
