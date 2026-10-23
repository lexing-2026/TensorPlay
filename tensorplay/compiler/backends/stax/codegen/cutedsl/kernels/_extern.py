"""Row kernels reached from a graph as one operation each.

Each of these answers a whole call: it reads every row of a value, reduces the
row, and writes the result, which is more than a loop over positions can say.
What is left is to put the call in the shape the kernels take -- a contiguous
matrix whose last axis is the one being reduced -- and to say what the answer
looks like without computing it while a graph is only being read.
"""

from __future__ import annotations

import tensorplay as tp

from .rmsnorm import rmsnorm_fwd
from .softmax import softmax_bwd, softmax_bwd_uniform, softmax_fwd


def standing_in(*values) -> bool:
    """Whether any of these values only describes the value it stands for."""

    from tensorplay._higher_order_ops._hop_base import is_fake_tensor

    return any(is_fake_tensor(value) for value in values)


def _matrix(value):
    """The rows and the row extent of a contiguous value."""

    n = int(value.shape[-1])
    return int(value.numel()) // n, n


def _row_shape(rows: int, rank: int) -> tuple:
    """A per-row result laid out the way a reduction over the last axes is."""

    return (rows,) + (1,) * (rank - 1)


def softmax_forward(x):
    """Normalised exponentials over the last axis."""

    rows, n = _matrix(x)
    if standing_in(x):
        return tp.empty_like(x)
    return softmax_fwd(x.reshape(rows, n)).reshape(x.shape)


def softmax_backward(grad, output):
    """Gradient of :func:`softmax_forward`, given the forward's own result."""

    rows, n = _matrix(output)
    if standing_in(output, grad):
        return tp.empty_like(output)
    dx = softmax_bwd(output.reshape(rows, n), grad.reshape(rows, n))
    return dx.reshape(output.shape)


def softmax_backward_uniform(output, grad):
    """Gradient of :func:`softmax_forward` when every incoming gradient is equal.

    ``grad`` holds the one value that every position of the incoming gradient
    has, which is what a graph ending in a reduction hands back.
    """

    rows, n = _matrix(output)
    if standing_in(output, grad):
        return tp.empty_like(output)
    dx = softmax_bwd_uniform(output.reshape(rows, n), grad.reshape(1))
    return dx.reshape(output.shape)


def rms_norm(x, weight, eps):
    """The normalised value and the per-row scale it was normalised by."""

    rows, n = _matrix(x)
    if standing_in(x):
        out = tp.empty_like(x)
        rstd = tp.empty(_row_shape(rows, len(x.shape)), dtype=tp.float32,
                        device=x.device)
        return out, rstd
    out, rstd = rmsnorm_fwd(x.reshape(rows, n), weight.reshape(n), float(eps))
    return out.reshape(x.shape), rstd.reshape(_row_shape(rows, len(x.shape)))
