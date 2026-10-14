"""Functional interface for the hinge loss metric."""

from typing import Optional

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
)
from tensorplay.metrics.utilities.compute import normalize_logits_if_needed
from tensorplay.metrics.utilities.data import to_onehot
from tensorplay.metrics.utilities.enums import ClassificationTaskNoMultilabel


def _hinge_loss_compute(measure: Tensor, total: Tensor) -> Tensor:
    return measure / total


def _binary_hinge_loss_arg_validation(squared: bool, ignore_index: Optional[int] = None) -> None:
    if not isinstance(squared, bool):
        raise ValueError(f"Expected argument `squared` to be an bool but got {squared}")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _binary_hinge_loss_tensor_validation(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> None:
    _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def _binary_hinge_loss_update(
    preds: Tensor,
    target: Tensor,
    squared: bool,
) -> tuple[Tensor, Tensor]:
    target = target.bool()
    margin = tp.zeros_like(preds)
    margin[target] = preds[target]
    margin[~target] = -preds[~target]

    measures = 1 - margin
    measures = tp.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tp.tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def binary_hinge_loss(
    preds: Tensor,
    target: Tensor,
    squared: bool = False,
    ignore_index: Optional[int] = None,
    validate_args: bool = False,
) -> Tensor:
    """Compute the mean hinge loss for binary tasks.

    The hinge loss is defined as ``max(0, 1 - y * y_hat)`` where ``y`` is the target in
    ``{-1, 1}`` and ``y_hat`` is the prediction.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0, 1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0, 1} values (except if ``ignore_index`` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        squared: if ``True``, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if validate_args:
        _binary_hinge_loss_arg_validation(squared, ignore_index)
        _binary_hinge_loss_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False
    )
    measures, total = _binary_hinge_loss_update(preds, target, squared)
    return _hinge_loss_compute(measures, total)


def _multiclass_hinge_loss_arg_validation(
    num_classes: int,
    squared: bool = False,
    multiclass_mode: str = "crammer-singer",
    ignore_index: Optional[int] = None,
) -> None:
    _binary_hinge_loss_arg_validation(squared, ignore_index)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    allowed_mm = ("crammer-singer", "one-vs-all")
    if multiclass_mode not in allowed_mm:
        raise ValueError(f"Expected argument `multiclass_mode` to be one of {allowed_mm}, but got {multiclass_mode}.")


def _multiclass_hinge_loss_tensor_validation(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> None:
    _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def _multiclass_hinge_loss_update(
    preds: Tensor,
    target: Tensor,
    squared: bool,
    multiclass_mode: str = "crammer-singer",
) -> tuple[Tensor, Tensor]:
    preds = normalize_logits_if_needed(preds, "softmax")
    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= tp.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = tp.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]

    measures = 1 - margin
    measures = tp.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tp.tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def multiclass_hinge_loss(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    squared: bool = False,
    multiclass_mode: str = "crammer-singer",
    ignore_index: Optional[int] = None,
    validate_args: bool = False,
) -> Tensor:
    """Compute the mean hinge loss for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used, or
    the one-vs-all approach where each class is valued against all other classes in a binary fashion.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0, 1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if ``ignore_index`` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        num_classes: integer specifying the number of classes
        squared: if ``True``, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode: determines how to compute the metric
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _multiclass_hinge_loss_update(preds, target, squared, multiclass_mode)
    return _hinge_loss_compute(measures, total)


def hinge_loss(
    preds: Tensor,
    target: Tensor,
    task: str,
    num_classes: Optional[int] = None,
    squared: bool = False,
    multiclass_mode: str = "crammer-singer",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the mean hinge loss.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``.

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_hinge_loss(preds, target, squared, ignore_index, validate_args)
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_hinge_loss(preds, target, num_classes, squared, multiclass_mode, ignore_index, validate_args)
    raise ValueError(f"Not handled value: {task}")