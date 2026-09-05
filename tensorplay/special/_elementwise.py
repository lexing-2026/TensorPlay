"""Pointwise transforms and the normalized-exponential reductions."""
import tensorplay as tp
from tensorplay import exp, exp2, expm1, log, log1p, sinc

__all__ = [
    "exp2",
    "expm1",
    "log1p",
    "log_softmax",
    "logsumexp",
    "round",
    "sinc",
    "softmax",
]


def _normalize_dims(dim, ndim):
    if dim is None:
        return list(range(ndim))
    dims = [dim] if isinstance(dim, int) else [int(d) for d in dim]
    result = []
    for value in dims:
        if ndim <= 0:
            raise ValueError("a reduction dimension requires a non-empty input")
        value = value + ndim if value < 0 else value
        if value < 0 or value >= ndim:
            raise ValueError(f"dimension {value} out of range for {ndim}-D input")
        if value in result:
            raise ValueError("reduction dimensions must be unique")
        result.append(value)
    return result


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


def round(input):
    """Rounds to nearest even integer (native ``Tensor.round``)."""
    return input.round()


def softmax(input, dim=None, *, dtype=None):
    """Normalized exponential along :attr:`dim` (default: the last dim)."""
    from tensorplay.nn.functional import softmax as _sm
    return _sm(input, dim=dim if dim is not None else -1, dtype=dtype)


def log_softmax(input, dim=None, *, dtype=None):
    """``log(softmax(input, dim))``, evaluated without the intermediate exp."""
    from tensorplay.nn.functional import log_softmax as _ls
    return _ls(input, dim=dim if dim is not None else -1, dtype=dtype)


def logsumexp(input, dim=None, keepdim=False, *, out=None):
    """Log-sum-exp along :attr:`dim`, computed with the max-shift trick."""
    dims = _normalize_dims(dim, input.dim())
    if not dims:
        return _copy_to_out(input, out)

    reduce_all = dim is None
    work = input.reshape([-1]) if reduce_all else input
    reduce_dims = [0] if reduce_all else dims
    m = tp.amax(work, dim=reduce_dims, keepdim=True)
    safe_m = tp.where(tp.isinf(m), tp.zeros_like(m), m)
    s = (exp(work - safe_m)).sum(dim=reduce_dims, keepdim=True)
    lse = log(s) + m
    lse = tp.where(tp.isposinf(m), m, lse)
    if reduce_all:
        lse = lse.reshape([1] * input.dim() if keepdim else [])
    elif not keepdim:
        lse = lse.reshape(
            [size for i, size in enumerate(lse.shape) if i not in dims]
        )
    return _copy_to_out(lse, out)
