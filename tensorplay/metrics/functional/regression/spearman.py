"""Functional Spearman rank correlation coefficient."""

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.utils import _check_data_shape_to_num_outputs
from tensorplay.metrics.utilities.checks import _check_same_shape


def _rank_data(data: Tensor) -> Tensor:
    """Calculate the rank for each element of a tensor.

    The rank refers to the indices of an element in the corresponding sorted tensor (starting from 1). Duplicates of
    the same value will be assigned the mean of their rank.

    """
    n = data.numel()
    rank = tensorplay.empty_like(data, dtype=tensorplay.int32)
    idx = tensorplay.argsort(data)
    rank[idx[:n]] = tensorplay.arange(1, n + 1, dtype=tensorplay.int32, device=data.device)
    uniq, inv, counts = tensorplay.unique(data, sorted=True, return_inverse=True, return_counts=True)
    sum_ranks = tensorplay.zeros_like(uniq, dtype=tensorplay.int32)
    tensorplay.scatter_add_(sum_ranks, 0, inv, rank.to(tensorplay.int32))
    mean_ranks = sum_ranks / counts
    return mean_ranks[inv]


def _spearman_corrcoef_update(preds: Tensor, target: Tensor, num_outputs: int) -> tuple[Tensor, Tensor]:
    """Update and return variables required to compute Spearman Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    if not (tensorplay.is_floating_point(preds) and tensorplay.is_floating_point(target)):
        raise TypeError(
            "Expected `preds` and `target` both to be floating point tensors, but got {preds.dtype} and {target.dtype}"
        )
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)

    return preds, target


def _spearman_corrcoef_compute(preds: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute Spearman Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        eps: Avoids ``ZeroDivisionError``.

    """
    if preds.ndim == 1:
        preds = _rank_data(preds)
        target = _rank_data(target)
    else:
        preds = tensorplay.stack([_rank_data(p) for p in preds.T]).T
        target = tensorplay.stack([_rank_data(t) for t in target.T]).T

    preds_diff = preds - preds.mean(0)
    target_diff = target - target.mean(0)

    cov = (preds_diff * target_diff).mean(0)
    preds_std = tensorplay.sqrt((preds_diff * preds_diff).mean(0))
    target_std = tensorplay.sqrt((target_diff * target_diff).mean(0))

    corrcoef = cov / (preds_std * target_std + eps)
    return tensorplay.clamp(corrcoef, -1.0, 1.0)


def spearman_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """Compute spearman rank correlation coefficient.

    Args:
        preds: estimated scores
        target: ground truth scores

    Return:
        Tensor with spearman correlation coefficient

    """
    preds, target = _spearman_corrcoef_update(preds, target, num_outputs=1)
    return _spearman_corrcoef_compute(preds, target)