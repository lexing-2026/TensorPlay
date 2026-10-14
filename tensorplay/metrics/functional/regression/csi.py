"""Critical success index metric."""

from typing import Optional

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.compute import _safe_divide

__all__ = ["critical_success_index"]


def _critical_success_index_update(
    preds: Tensor, target: Tensor, threshold: float, keep_sequence_dim: Optional[int] = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Return hits, misses and false alarms after thresholding."""
    _check_same_shape(preds, target)

    if keep_sequence_dim is None:
        sum_dims = None
    elif not 0 <= keep_sequence_dim < preds.ndim:
        raise ValueError(f"Expected keep_sequence dim to be in range [0, {preds.ndim}] but got {keep_sequence_dim}")
    else:
        sum_dims = tuple(i for i in range(preds.ndim) if i != keep_sequence_dim)

    # binarize the tensors with the threshold
    preds_bin = (preds >= threshold).bool()
    target_bin = (target >= threshold).bool()

    if keep_sequence_dim is None:
        hits = tp.sum(preds_bin & target_bin).int()
        misses = tp.sum((preds_bin ^ target_bin) & target_bin).int()
        false_alarms = tp.sum((preds_bin ^ target_bin) & preds_bin).int()
    else:
        hits = tp.sum(preds_bin & target_bin, dim=sum_dims).int()
        misses = tp.sum((preds_bin ^ target_bin) & target_bin, dim=sum_dims).int()
        false_alarms = tp.sum((preds_bin ^ target_bin) & preds_bin, dim=sum_dims).int()
    return hits, misses, false_alarms


def critical_success_index(
    preds: Tensor, target: Tensor, threshold: float, keep_sequence_dim: Optional[int] = None
) -> Tensor:
    """Compute the critical success index.

    Args:
        preds: predicted tensor
        target: ground truth tensor
        threshold: values above or equal to the threshold count as positive
        keep_sequence_dim: if provided, score each image in a sequence separately

    Returns:
        scalar (or per-sequence) critical success index
    """
    hits, misses, false_alarms = _critical_success_index_update(preds, target, threshold, keep_sequence_dim)
    return _critical_success_index_compute(hits, misses, false_alarms)


def _critical_success_index_compute(hits: Tensor, misses: Tensor, false_alarms: Tensor) -> Tensor:
    """Compute the critical success index from hits, misses and false alarms."""
    return _safe_divide(hits, hits + misses + false_alarms)
