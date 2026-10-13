"""Spike encoders: convert real-valued data into spike trains.

Provided schemes:

* :func:`rate_encode` -- Bernoulli sampling: each step emits a spike with
  probability equal to the (clipped) input value;
* :func:`poisson_encode` -- alias of :func:`rate_encode`;
* :func:`temporal_encode` -- time-to-first-spike: larger inputs fire earlier
  within the ``T``-step window;
* :class:`PeriodicEncoder` -- repeats a fixed periodic spike pattern;
* :class:`GaussianTuningCurveEncoder` -- population encoding where each input
  feature drives a bank of Gaussian-tuned Bernoulli neurons.
"""

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import Module

__all__ = [
    "rate_encode",
    "poisson_encode",
    "temporal_encode",
    "PeriodicEncoder",
    "GaussianTuningCurveEncoder",
]


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
    in ``x.dtype`` with exactly one spike per element.  The one-hot
    placement uses a single fused ``scatter_`` on the flat [T, N] buffer.
    """
    x = x.clamp(min=1e-7)
    spike_time = tp.ceil(-tau * tp.log(x)).to(tp.int64).clamp(0, T - 1)
    shape = (T,) + tuple(x.shape)
    out = tp.zeros(shape, dtype=x.dtype, device=x.device)
    flat_time = spike_time.reshape(-1)
    flat_out = out.reshape(T, -1)
    flat_out.scatter_(0, flat_time.unsqueeze(0), 1)
    return flat_out.reshape(shape)


class PeriodicEncoder(Module):
    """Encode input-independent periodic spike patterns.

    Args:
        encoding: a tensor of shape ``[period, ...]`` giving one period of
            the spike pattern.

    Calling ``forward(T)`` repeats the pattern along time and returns a
    ``[T, ...]`` tensor.  If ``T`` is not a multiple of ``period`` the
    output is truncated to the first ``T`` steps.
    """

    def __init__(self, encoding: Tensor) -> None:
        super().__init__()
        if encoding.dim() < 1:
            raise ValueError("PeriodicEncoder: encoding must have a time dim")
        self.register_buffer("encoding", encoding)

    def forward(self, T: int) -> Tensor:
        period = self.encoding.shape[0]
        reps = (T + period - 1) // period
        out = self.encoding.repeat((reps,) + (1,) * (self.encoding.dim() - 1))
        return out[:T]


class GaussianTuningCurveEncoder(Module):
    """Population encoding with Gaussian tuning curves.

    Each of the ``n_features`` input features drives ``n_neurons`` Bernoulli
    neurons whose firing probability follows a Gaussian bump centered at
    ``mu[f, k]`` with width ``sigma``:

    .. math::
        p_{f,k} = \\exp\\left(-\\frac{(x_f - \\mu_{f,k})^2}{2 \\sigma^2}\\right)

    Args:
        mu: centers, shape ``[n_features, n_neurons]``.
        sigma: width (scalar or broadcastable against ``mu``).

    ``forward(x, T)`` with ``x`` of shape ``[..., n_features]`` returns a
    ``[T, ..., n_features * n_neurons]`` binary spike train.
    """

    def __init__(self, mu: Tensor, sigma: float = 1.0) -> None:
        super().__init__()
        if mu.dim() != 2:
            raise ValueError(
                "GaussianTuningCurveEncoder: mu must have shape "
                "[n_features, n_neurons]"
            )
        self.register_buffer("mu", mu)
        self.sigma = sigma

    def forward(self, x: Tensor, T: int, generator=None) -> Tensor:
        n_features, n_neurons = self.mu.shape
        if x.shape[-1] != n_features:
            raise ValueError(
                f"GaussianTuningCurveEncoder: expected {n_features} input "
                f"features, got {x.shape[-1]}"
            )
        diff = x.unsqueeze(-1) - self.mu  # [..., n_features, n_neurons]
        prob = tp.exp(-(diff * diff) / (2.0 * self.sigma**2))
        flat = prob.reshape(x.shape[:-1] + (n_features * n_neurons,))
        return rate_encode(flat, T, generator=generator)
