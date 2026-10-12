"""Numeric computation helpers shared by metrics."""

from typing import Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities import rank_zero_warn


def _safe_matmul(x: Tensor, y: Tensor) -> Tensor:
    """Matrix multiplication that avoids half precision accumulation errors."""
    if x.dtype == tensorplay.float16 or y.dtype == tensorplay.float16:
        return (x.float() @ y.T.float()).half()
    return x @ y.T


def _safe_xlogy(x: Tensor, y: Tensor) -> Tensor:
    """Compute ``x * log(y)``, returning 0 when ``x`` is 0."""
    res = x * tensorplay.log(y)
    res[x == 0] = 0.0
    return res


def _safe_divide(
    num: Tensor,
    denom: Tensor,
    zero_division: Union[float, str] = 0.0,
) -> Tensor:
    """Divide ``num`` by ``denom``, replacing zero denominators.

    Args:
        num: numerator tensor
        denom: denominator tensor, which may contain zeros
        zero_division: value used to replace elements divided by zero; the
            string ``"warn"`` behaves like ``0.0`` but emits a warning

    """
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    if isinstance(zero_division, (float, int)) or zero_division == "warn":
        if zero_division == "warn" and tensorplay.any(denom == 0):
            rank_zero_warn("Detected zero division in _safe_divide. Setting 0/0 to 0.0")
        zero_division = 0.0 if zero_division == "warn" else zero_division
        zero_division_tensor = tensorplay.full((), zero_division, dtype=num.dtype, device=num.device)
        return tensorplay.where(denom != 0, num / denom, zero_division_tensor)
    return tensorplay.true_divide(num, denom)


def _safe_sum_dim(x: Tensor, dim: int) -> Tensor:
    """Sum along ``dim``; scalar inputs are returned unchanged."""
    if x.ndim == 0:
        return x
    return tensorplay.sum(x, dim=dim)


def _adjust_weights_safe_divide(
    score: Tensor,
    average: Optional[str],
    multilabel: bool,
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    top_k: int = 1,
) -> Tensor:
    """Weight a per-class score by its support and reduce it."""
    if average is None or average == "none":
        return score
    if average == "weighted":
        weights = tp + fn
    else:
        weights = tensorplay.ones_like(score)
        if not multilabel:
            weights[tp + fp + fn == 0 if top_k == 1 else tp + fn == 0] = 0.0
    return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)


def _auc_format_inputs(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Validate inputs for AUC computation."""
    x = x.squeeze() if x.ndim > 1 else x
    y = y.squeeze() if y.ndim > 1 else y

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _auc_compute_without_check(x: Tensor, y: Tensor, direction: float, axis: int = -1) -> Tensor:
    """Compute area under the curve with the trapezoidal rule."""
    with tensorplay.no_grad():
        auc_score: Tensor = tensorplay.trapz(y, x, dim=axis) * direction
    return auc_score


def _auc_compute(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Compute area under the curve with the trapezoidal rule."""
    with tensorplay.no_grad():
        if reorder:
            x, x_idx = tensorplay.sort(x, stable=True)
            y = y[x_idx]

        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx <= 0).all():
                direction = -1.0
            else:
                raise ValueError(
                    "The `x` tensor is neither increasing or decreasing. Try setting the reorder argument to `True`."
                )
        else:
            direction = 1.0
        return _auc_compute_without_check(x, y, direction)


def auc(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing

    Return:
        Tensor containing the AUC score

    """
    x, y = _auc_format_inputs(x, y)
    return _auc_compute(x, y, reorder=reorder)


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Args:
        x: x-coordinates at which to evaluate the interpolated values
        xp: x-coordinates of the data points, must be increasing
        fp: y-coordinates of the data points, same length as ``xp``

    Returns:
        interpolated values, same size as ``x``

    """
    m = _safe_divide(fp[1:] - fp[:-1], xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indices = tensorplay.sum(tensorplay.ge(x[:, None], xp[None, :]), 1) - 1
    indices = tensorplay.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]


def normalize_logits_if_needed(tensor: Tensor, normalization: Optional[str]) -> Tensor:
    """Normalize logits if needed.

    If the input tensor contains values outside ``[0, 1]`` it is treated as
    logits and ``sigmoid``/``softmax`` is applied according to ``normalization``.

    """
    # if not specified, do nothing.
    if not normalization:
        return tensor
    # decrease sigmoid on cpu.
    if tensor.device == tensorplay.device("cpu"):
        if not tensorplay.all((tensor >= 0) * (tensor <= 1)):
            tensor = tensorplay.sigmoid(tensor) if normalization == "sigmoid" else tensorplay.softmax(tensor, dim=1)
        return tensor

    # decrease device-host sync on device.
    condition = ((tensor < 0) | (tensor > 1)).any()
    return tensorplay.where(
        condition,
        tensorplay.sigmoid(tensor) if normalization == "sigmoid" else tensorplay.softmax(tensor, dim=1),
        tensor,
    )