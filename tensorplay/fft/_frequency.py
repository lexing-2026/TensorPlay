"""Frequency-grid construction and spectrum re-ordering helpers."""
from tensorplay import arange, cat
from tensorplay._C import DType

__all__ = ["fftfreq", "rfftfreq", "fftshift", "ifftshift"]


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


def _shift_dims(input, dim):
    ndim = input.dim()
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        return [dim % ndim]
    return [d % ndim for d in dim]


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
