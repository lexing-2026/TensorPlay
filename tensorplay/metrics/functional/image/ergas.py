import tensorplay
from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.distributed import reduce


def _ergas_update(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Erreur Relative Globale Adimensionnelle de Synthèse.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got preds: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(
            f"Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _ergas_compute(
    preds: Tensor,
    target: Tensor,
    ratio: float = 4,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Erreur Relative Globale Adimensionnelle de Synthèse.

    Args:
        preds: estimated image
        target: ground truth image
        ratio: ratio of high resolution to low resolution
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> from tensorplay import rand
        >>> preds = rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> preds, target = _ergas_update(preds, target)
        >>> tensorplay.round(_ergas_compute(preds, target))
        tensor(10.)

    """
    b, c, h, w = preds.shape
    preds = preds.reshape(b, c, h * w)
    target = target.reshape(b, c, h * w)

    diff = preds - target
    sum_squared_error = tensorplay.sum(diff * diff, dim=2)
    rmse_per_band = tensorplay.sqrt(sum_squared_error / (h * w))
    mean_target = tensorplay.mean(target, dim=2)

    ergas_score = 100 / ratio * tensorplay.sqrt(tensorplay.sum((rmse_per_band / mean_target) ** 2, dim=1) / c)
    return reduce(ergas_score, reduction)


def error_relative_global_dimensionless_synthesis(
    preds: Tensor,
    target: Tensor,
    ratio: float = 4,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Calculates `Error relative global dimensionless synthesis`_ (ERGAS) metric.

    Args:
        preds: estimated image
        target: ground truth image
        ratio: ratio of high resolution to low resolution
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor with RelativeG score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.

    Example:
        >>> from tensorplay import rand
        >>> from tensorplay.metrics.functional.image import error_relative_global_dimensionless_synthesis
        >>> preds = rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> error_relative_global_dimensionless_synthesis(preds, target)
        tensor(9.6193)

    """
    preds, target = _ergas_update(preds, target)
    return _ergas_compute(preds, target, ratio, reduction)