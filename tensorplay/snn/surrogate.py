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
    "PiecewiseQuadraticSurrogate",
    "PiecewiseExpSurrogate",
    "SuperSpikeSurrogate",
    "NonzeroSignLogAbsSurrogate",
    "PiecewiseLeakyRELUSurrogate",
    "LeakyKRELUSurrogate",
    "S2NNSurrogate",
    "QPseudoSpikeSurrogate",
    "SquarewaveFourierSeriesSurrogate",
]

# Surrogate ids must match the native snn_surrogate_grad kernel dispatch.
_SURROGATE_STRAIGHT_THROUGH = 0
_SURROGATE_SIGMOID = 1
_SURROGATE_FAST_SIGMOID = 2
_SURROGATE_PIECEWISE_LINEAR = 3
_SURROGATE_ATAN = 4
_SURROGATE_ERF = 5
_SURROGATE_PIECEWISE_QUADRATIC = 6
_SURROGATE_PIECEWISE_EXP = 7
_SURROGATE_SUPER_SPIKE = 8
_SURROGATE_NONZERO_SIGN_LOG_ABS = 9
_SURROGATE_PIECEWISE_LEAKY_RELU = 10
_SURROGATE_LEAKY_K_RELU = 11
_SURROGATE_S2NN = 12
_SURROGATE_Q_PSEUDO_SPIKE = 13
_SURROGATE_SQUAREWAVE_FOURIER = 14


class _Surrogate:
    """Base class for surrogate gradient selectors.

    Forward always emits the Heaviside spike; backward replaces the step
    derivative with the smooth surrogate gradient selected here.

    The extra attributes (``beta``/``w``/``c``/``n``/``T_period``) feed the
    native gradient kernel for the multi-parameter surrogates; the
    single-parameter families leave them at their defaults.
    """

    id = _SURROGATE_STRAIGHT_THROUGH
    alpha = 1.0
    beta = 1.0
    w = 1.0
    c = 0.0
    n = 1
    T_period = 1.0

    def __call__(self, x: Tensor) -> Tensor:
        from .functional import spike

        return spike(x, self)

    def grad(self, x: Tensor) -> Tensor:
        from .functional import surrogate_grad

        return surrogate_grad(
            x,
            surrogate=self.id,
            alpha=self.alpha,
            beta=self.beta,
            w=self.w,
            c=self.c,
            n=self.n,
            T_period=self.T_period,
        )

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


class PiecewiseQuadraticSurrogate(_Surrogate):
    """Piecewise-quadratic surrogate.

    Backward weight: ``-alpha^2*|x| + alpha`` inside ``|x| <= 1/alpha`` and
    zero outside.
    """

    id = _SURROGATE_PIECEWISE_QUADRATIC

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha


class PiecewiseExpSurrogate(_Surrogate):
    """Piecewise-exponential surrogate.

    Backward weight: ``alpha/2 * exp(-alpha*|x|)``.
    """

    id = _SURROGATE_PIECEWISE_EXP

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha


class SuperSpikeSurrogate(_Surrogate):
    """Super-spike surrogate.

    Backward weight: ``alpha / (1 + |x|)^2``.
    """

    id = _SURROGATE_SUPER_SPIKE

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha


class NonzeroSignLogAbsSurrogate(_Surrogate):
    """Nonzero-sign-log-abs surrogate.

    Backward weight: ``alpha / (1 + alpha*|x|)``.
    """

    id = _SURROGATE_NONZERO_SIGN_LOG_ABS

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha


class PiecewiseLeakyRELUSurrogate(_Surrogate):
    """Piecewise-leaky-ReLU surrogate.

    Backward weight: ``1/(2w)`` inside ``|x| < w`` and ``c`` outside.
    """

    id = _SURROGATE_PIECEWISE_LEAKY_RELU

    def __init__(self, w: float = 1.0, c: float = 0.01) -> None:
        self.w = w
        self.c = c


class LeakyKRELUSurrogate(_Surrogate):
    """Leaky-K-ReLU surrogate.

    Backward weight: ``k`` for ``x >= 0`` and ``leak`` for ``x < 0``.
    """

    id = _SURROGATE_LEAKY_K_RELU

    def __init__(self, leak: float = 0.0, k: float = 1.0) -> None:
        self.beta = leak
        self.w = k


class S2NNSurrogate(_Surrogate):
    """S2NN surrogate.

    Backward weight: sigmoid gradient on the negative half-axis and
    ``beta / (x + 1)`` on the non-negative half-axis.
    """

    id = _SURROGATE_S2NN

    def __init__(self, alpha: float = 4.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta


class QPseudoSpikeSurrogate(_Surrogate):
    """q-Pseudo-Spike surrogate.

    Backward weight: ``(1 + 2/(alpha-1)*|x|)^(-alpha)``.
    """

    id = _SURROGATE_Q_PSEUDO_SPIKE

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha


class SquarewaveFourierSeriesSurrogate(_Surrogate):
    """Square-wave Fourier-series surrogate.

    Backward weight: ``4/T * sum_i cos((2i-1)*2*pi/T*x)`` for
    ``i = 1..n-1``.
    """

    id = _SURROGATE_SQUAREWAVE_FOURIER

    def __init__(self, n: int = 2, T_period: float = 2.0) -> None:
        self.n = n
        self.T_period = T_period