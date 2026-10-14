"""Functional interface for the edit distance metric."""

from collections.abc import Sequence
from typing import Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.text.helper import _LevenshteinEditDistance as _LE_distance


def _edit_distance_update(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    substitution_cost: int = 1,
) -> Tensor:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    if not all(isinstance(x, str) for x in preds):
        raise ValueError(f"Expected all values in argument `preds` to be string type, but got {preds}")
    if not all(isinstance(x, str) for x in target):
        raise ValueError(f"Expected all values in argument `target` to be string type, but got {target}")
    if len(preds) != len(target):
        raise ValueError(
            f"Expected argument `preds` and `target` to have same length, but got {len(preds)} and {len(target)}"
        )

    distance = [
        _LE_distance(t, op_substitute=substitution_cost)(p)[0]  # type: ignore[arg-type]
        for p, t in zip(preds, target)
    ]
    return tp.tensor(distance, dtype=tp.int32)


def _edit_distance_compute(
    edit_scores: Tensor,
    num_elements: Union[Tensor, int],
    reduction: Optional[str] = "mean",
) -> Tensor:
    """Compute the final edit distance reduced over the batch."""
    if edit_scores.numel() == 0:
        return tp.tensor(0, dtype=tp.int32)
    if reduction == "mean":
        return edit_scores.sum() / num_elements
    if reduction == "sum":
        return edit_scores.sum()
    if reduction is None or reduction == "none":
        return edit_scores
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")


def edit_distance(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    substitution_cost: int = 1,
    reduction: Optional[str] = "mean",
) -> Tensor:
    """Calculate the Levenshtein edit distance between two sequences.

    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to
    transform the predicted text into the reference text. The lower the distance, the more accurate the
    model is considered to be.

    Args:
        preds: an iterable of predicted texts (strings)
        target: an iterable of reference texts (strings)
        substitution_cost: the cost of substituting one character for another
        reduction: a method to reduce metric score over samples:

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Raises:
        ValueError: if ``preds`` and ``target`` do not have the same length
        ValueError: if ``preds`` or ``target`` contain non-string values

    """
    distance = _edit_distance_update(preds, target, substitution_cost)
    return _edit_distance_compute(distance, num_elements=distance.numel(), reduction=reduction)