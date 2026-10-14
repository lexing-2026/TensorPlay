"""Functional interface for the Minkowski distance metric."""

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.exceptions import MetricsUserError


def _minkowski_distance_update(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    """Update and return variables required to compute the Minkowski distance.

    Checks for same shape of input tensors.

    Args:
        preds: predicted tensor
        targets: ground truth tensor
        p: non-negative number acting as the exponent to the errors

    """
    _check_same_shape(preds, targets)

    if not (isinstance(p, (float, int)) and p >= 1):
        raise MetricsUserError(f"Argument ``p`` must be a float or int greater than 1, but got {p}")

    difference = tp.abs(preds - targets)
    return tp.sum(tp.pow(difference, p))


def _minkowski_distance_compute(distance: Tensor, p: float) -> Tensor:
    """Compute the Minkowski distance.

    Args:
        distance: sum of the p-th powers of errors over all observations
        p: the non-negative numeric power the errors are to be raised to

    """
    return tp.pow(distance, 1.0 / p)


def minkowski_distance(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    """Compute the Minkowski distance.

    The Minkowski distance is a generalization of the Euclidean distance (``p=2``) and the
    Manhattan distance (``p=1``).

    Args:
        preds: estimated labels of type Tensor
        targets: ground truth labels of type Tensor
        p: int or float larger than 1, exponent to which the difference between preds and target is to be raised

    Return:
        Tensor with the Minkowski distance

    """
    minkowski_dist_sum = _minkowski_distance_update(preds, targets, p)
    return _minkowski_distance_compute(minkowski_dist_sum, p)