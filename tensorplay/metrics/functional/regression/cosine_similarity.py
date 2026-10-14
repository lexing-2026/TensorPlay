"""Functional interface for the cosine similarity metric."""

from typing import Optional

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape


def _cosine_similarity_update(
    preds: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Update and return variables required to compute the cosine similarity.

    Checks for same shape of input tensors.

    Args:
        preds: predicted tensor
        target: ground truth tensor

    Raises:
        ValueError: if the input tensors are not 2D

    """
    _check_same_shape(preds, target)
    if preds.ndim != 2:
        raise ValueError(
            "Expected input to cosine similarity to be 2D tensors of shape `[N,D]` where `N` is the number of samples"
            f" and `D` is the number of dimensions, but got tensor of shape {preds.shape}"
        )
    preds = preds.float()
    target = target.float()

    return preds, target


def _cosine_similarity_compute(preds: Tensor, target: Tensor, reduction: Optional[str] = "sum") -> Tensor:
    """Compute the cosine similarity.

    Args:
        preds: predicted tensor
        target: ground truth tensor
        reduction: the method of reducing along the batch dimension using ``sum``, ``mean`` or
            taking the individual scores (``"none"``/``None``)

    """
    dot_product = (preds * target).sum(dim=-1)
    preds_norm = preds.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    similarity = dot_product / (preds_norm * target_norm)
    reduction_mapping = {
        "sum": tp.sum,
        "mean": tp.mean,
        "none": lambda x: x,
        None: lambda x: x,
    }
    return reduction_mapping[reduction](similarity)  # type: ignore[operator]


def cosine_similarity(preds: Tensor, target: Tensor, reduction: Optional[str] = "sum") -> Tensor:
    """Compute the cosine similarity.

    The cosine similarity between two vectors ``x`` and ``y`` is defined as their dot product
    divided by the product of their norms.

    Args:
        preds: predicted tensor with shape ``(N, d)``
        target: ground truth tensor with shape ``(N, d)``
        reduction: the method of reducing along the batch dimension using ``sum``, ``mean`` or
            taking the individual scores (``"none"``/``None``)

    """
    preds, target = _cosine_similarity_update(preds, target)
    return _cosine_similarity_compute(preds, target, reduction)