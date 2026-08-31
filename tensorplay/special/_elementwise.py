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


def logsumexp(input, dim, keepdim=False, *, out=None):
    """Log-sum-exp along :attr:`dim`, computed with the max-shift trick."""
    if out is not None:
        raise NotImplementedError("logsumexp: out= is not supported")
    dims = [dim] if isinstance(dim, int) else list(dim)
    m = tp.amax(input, dim=dims, keepdim=True)
    s = (exp(input - m)).sum(dim=dims, keepdim=True)
    lse = log(s) + m
    if not keepdim:
        reduced = {d % input.dim() for d in dims}
        lse = lse.reshape(
            [size for i, size in enumerate(lse.shape) if i not in reduced]
        )
    return lse
