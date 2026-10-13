"""Spike encoders: convert real-valued data into spike trains.

Two classic schemes are provided:

* :func:`rate_encode` -- Bernoulli sampling: each step emits a spike with
  probability equal to the (clipped) input value;
* :func:`temporal_encode` -- time-to-first-spike: larger inputs fire earlier
  within the ``T``-step window.
"""

import tensorplay as tp
from tensorplay import Tensor

__all__ = ["rate_encode", "poisson_encode", "temporal_encode"]


def rate_encode(x: Tensor, T: int, generator=None) -> Tensor:
    """Encode values in ``[0, 1]`` as Bernoulli spikes over ``T`` steps.

    Returns a tensor of shape ``[T, *x.shape]`` in ``x.dtype`` where each
    entry is an independent spike with probability equal to the input value.
    """
    x = x.clamp(0, 1)
    probs = x.unsqueeze(0).expand((T,) + tuple(x.shape))
    return tp.bernoulli(probs, generator=generator).to(x.dtype)


def poisson_encode(x: Tensor, T: int, generator=None) -> Tensor:
    """Poisson-process encoding.

    Equivalent to :func:`rate_encode`: the number of spikes in a Poisson
    process with rate ``x`` follows ``Binomial(T, x)`` when each step is an
    independent Bernoulli draw.
    """
    return rate_encode(x, T, generator=generator)


def temporal_encode(x: Tensor, T: int, tau: float = 1.0) -> Tensor:
    """Encode values in ``(0, 1]`` as time-to-first-spike.

    The spike time is ``ceil(-tau * ln(x))`` clamped to ``[0, T-1]``, so a
    larger input fires earlier.  Returns a tensor of shape ``[T, *x.shape]``
    in ``x.dtype`` with exactly one spike per element.
    """
    x = x.clamp(min=1e-7)
    spike_time = tp.ceil(-tau * tp.log(x)).to(tp.int64).clamp(0, T - 1)
    shape = (T,) + tuple(x.shape)
    out = tp.zeros(shape, dtype=x.dtype, device=x.device)
    flat_time = spike_time.reshape(-1)
    flat_out = out.reshape(T, -1)
    src = tp.ones((1, flat_time.numel()), dtype=out.dtype, device=out.device)
    flat_out.scatter_(0, flat_time.unsqueeze(0), src)
    return flat_out.reshape(shape)