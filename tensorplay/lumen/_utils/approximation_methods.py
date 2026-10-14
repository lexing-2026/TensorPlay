#!/usr/bin/env python3

"""Integral approximation methods used by path-based attribution.

Supports the Riemann-sum variants and the Gauss-Legendre quadrature rule,
returning step sizes and alpha coefficients for a given number of steps.
"""

from enum import Enum
from typing import Callable, cast, List, Tuple

import tensorplay as tensorplay


class Riemann(Enum):
    left = 1
    right = 2
    middle = 3
    trapezoid = 4


SUPPORTED_RIEMANN_METHODS = [
    "riemann_left",
    "riemann_right",
    "riemann_middle",
    "riemann_trapezoid",
]

SUPPORTED_METHODS: List[str] = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]


def approximation_parameters(
    method: str,
) -> Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]:
    r"""Retrieve parameters for the input approximation `method`.

    Args:
        method (str): The name of the approximation method. Currently only the
                riemann variants and gauss legendre are supported.
    """
    if method in SUPPORTED_RIEMANN_METHODS:
        return riemann_builders(method=Riemann[method.split("_")[-1]])
    if method == "gausslegendre":
        return gauss_legendre_builders()
    raise ValueError("Invalid integral approximation method name: {}".format(method))


def riemann_builders(
    method: Riemann = Riemann.trapezoid,
) -> Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]:
    r"""Step sizes are identical and alphas are scaled in [0, 1].

    Args:
        method (Riemann): `left`, `right`, `middle` and `trapezoid` riemann

    Returns:
        2-element tuple of **step_sizes**, **alphas**:
        - **step_sizes** (*Callable*):
                    `step_sizes` takes the number of steps as an
                    input argument and returns an array of steps sizes which
                    sum is smaller than or equal to one.

        - **alphas** (*Callable*):
                    `alphas` takes the number of steps as an input argument
                    and returns the multipliers/coefficients for the inputs
                    of integrand in the range of [0, 1]
    """

    def step_sizes(n: int) -> List[float]:
        assert n > 1, "The number of steps has to be larger than one"
        deltas = [1 / n] * n
        if method == Riemann.trapezoid:
            deltas[0] /= 2
            deltas[-1] /= 2
        return deltas

    def alphas(n: int) -> List[float]:
        assert n > 1, "The number of steps has to be larger than one"
        if method == Riemann.trapezoid:
            return tensorplay.linspace(0, 1, n).tolist()
        elif method == Riemann.left:
            return tensorplay.linspace(0, 1 - 1 / n, n).tolist()
        elif method == Riemann.middle:
            return tensorplay.linspace(1 / (2 * n), 1 - 1 / (2 * n), n).tolist()
        elif method == Riemann.right:
            return tensorplay.linspace(1 / n, 1, n).tolist()
        else:
            raise AssertionError("Provided Reimann approximation method is not valid.")

    return step_sizes, alphas


def gauss_legendre_builders() -> (
    Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]
):
    r"""Compute step sizes and alpha coefficients using the Gauss-Legendre
    quadrature rule.

    The Gauss-Legendre quadrature rule approximates integrals by evaluating the
    integrand at the roots of the Legendre polynomial of degree ``n`` and
    weighting the evaluations accordingly. The nodes and weights returned by
    the quadrature lie in [-1, 1]; they are rescaled here to [0, 1] and the
    weights are halved to account for the change of interval.

    Returns:
        2-element tuple of **step_sizes**, **alphas**:
        - **step_sizes** (*Callable*):
                    `step_sizes` takes the number of steps as an
                    input argument and returns an array of steps sizes which
                    sum is smaller than or equal to one.

        - **alphas** (*Callable*):
                    `alphas` takes the number of steps as an input argument
                    and returns the multipliers/coefficients for the inputs
                    of integrand in the range of [0, 1]
    """

    # allow using riemann even without np
    import numpy as np
    from numpy.typing import NDArray

    def step_sizes(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from 2 to 1
        return cast(
            NDArray[np.float64], 0.5 * np.polynomial.legendre.leggauss(n)[1]
        ).tolist()

    def alphas(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from [-1, 1] to [0, 1]
        return cast(
            NDArray[np.float64], 0.5 * (1 + np.polynomial.legendre.leggauss(n)[0])
        ).tolist()

    return step_sizes, alphas