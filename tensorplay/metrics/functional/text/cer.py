"""Functional interface for the character error rate metric."""

from typing import Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.text.helper import _edit_distance


def _cer_update(
    preds: Union[str, list[str]],
    target: Union[str, list[str]],
) -> tuple[Tensor, Tensor]:
    """Update the cer score with the current set of references and predictions.

    Args:
        preds: transcription(s) to score as a string or list of strings
        target: reference(s) for each speech input as a string or list of strings

    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of characters overall references

    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tp.tensor(0, dtype=tp.float32)
    total = tp.tensor(0, dtype=tp.float32)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred
        tgt_tokens = tgt
        errors += _edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)
    return errors, total


def _cer_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the character error rate.

    Args:
        errors: number of edit operations to get from the reference to the prediction, summed over all samples
        total: number of characters over all references

    Returns:
        Character error rate score

    """
    return errors / total


def char_error_rate(preds: Union[str, list[str]], target: Union[str, list[str]]) -> Tensor:
    """Compute the character error rate (CER).

    The metric indicates the percentage of characters that were incorrectly predicted. The lower the value, the
    better the performance of the ASR system, with a CER of 0 being a perfect score.

    Args:
        preds: transcription(s) to score as a string or list of strings
        target: reference(s) for each speech input as a string or list of strings

    Returns:
        Character error rate score

    """
    errors, total = _cer_update(preds, target)
    return _cer_compute(errors, total)