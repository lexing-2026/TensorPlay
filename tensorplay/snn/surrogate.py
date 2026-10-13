"""Surrogate gradient selectors for the spike step.

The spike function is the Heaviside step, whose derivative is zero almost
everywhere.  Surrogate-gradient training replaces that derivative with the
gradient of a smooth approximation so that error can flow through the spike.
Each selector names a native gradient formula (see the ``snn_surrogate_grad``
operator) and carries its scale ``alpha``.
"""

from tensorplay import Tensor

__all__ = [
    "StraightThroughSurrogate",
    "SigmoidSurrogate",
    "FastSigmoidSurrogate",
    "PiecewiseLinearSurrogate",
    "ATanSurrogate",
    "ErfSurrogate",
]

# Surrogate ids must match the native snn_surrogate_grad kernel dispatch.
_SURROGATE_STRAIGHT_THROUGH = 0
_SURROGATE_SIGMOID = 1
_SURROGATE_FAST_SIGMOID = 2
_SURROGATE_PIECEWISE_LINEAR = 3
_SURROGATE_ATAN = 4
_SURROGATE_ERF = 5


class _Surrogate:
    """Base class for surrogate gradient selectors.

    Forward always emits the Heaviside spike; backward replaces the step
    derivative with the smooth surrogate gradient selected here.
    """

    id = _SURROGATE_STRAIGHT_THROUGH
    alpha = 1.0

    def __call__(self, x: Tensor) -> Tensor:
        from .functional import spike

        return spike(x, self)

    def grad(self, x: Tensor) -> Tensor:
        from .functional import surrogate_grad

        return surrogate_grad(x, surrogate=self.id, alpha=self.alpha)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha})"


class StraightThroughSurrogate(_Surrogate):
    """Straight-through estimator: backward passes the gradient unchanged."""

    id = _SURROGATE_STRAIGHT_THROUGH


class SigmoidSurrogate(_Surrogate):
    """Sigmoid surrogate.

    Backward weight: ``alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))``.
    """

    id = _SURROGATE_SIGMOID

    def __init__(self, alpha: float = 4.0) -> None:
        self.alpha = alpha


class FastSigmoidSurrogate(_Surrogate):
    """Fast sigmoid surrogate.

    Backward weight: ``alpha / (1 + alpha*|x|)^2``.
    """

    id = _SURROGATE_FAST_SIGMOID

    def __init__(self, alpha: float = 4.0) -> None:
        self.alpha = alpha


class PiecewiseLinearSurrogate(_Surrogate):
    """Piecewise-linear (triangular) surrogate.

    Backward weight: 1 inside ``(-alpha, alpha)``, 0 outside.
    """

    id = _SURROGATE_PIECEWISE_LINEAR

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha


class ATanSurrogate(_Surrogate):
    """Arctangent surrogate.

    Backward weight: ``alpha / (1 + (pi*alpha*x)^2)``.
    """

    id = _SURROGATE_ATAN

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha


class ErfSurrogate(_Surrogate):
    """Gaussian error-function surrogate.

    Backward weight: ``alpha / sqrt(pi) * exp(-(alpha*x)^2)``.
    """

    id = _SURROGATE_ERF

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha