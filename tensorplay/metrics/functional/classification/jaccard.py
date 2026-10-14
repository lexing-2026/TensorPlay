"""Functional interface for the Jaccard index metric."""

from typing import Optional

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from tensorplay.metrics.utilities.compute import _safe_divide
from tensorplay.metrics.utilities.enums import ClassificationTask


def _jaccard_index_reduce(
    confmat: Tensor,
    average: Optional[str],
    ignore_index: Optional[int] = None,
    zero_division: float = 0.0,
) -> Tensor:
    """Perform reduction of an un-normalized confusion matrix into a Jaccard score.

    Args:
        confmat: tensor with un-normalized confusion matrix
        average: reduction method

            - ``'binary'``: binary reduction, expects a 2x2 matrix
            - ``'macro'``: calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class)
            - ``'micro'``: calculate the metric globally, across all samples and classes
            - ``'weighted'``: calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``)
            - ``'none'`` or ``None``: calculate the metric for each class separately, and return
              the metric for every class

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """
    allowed_average = ["binary", "micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")
    confmat = confmat.float()
    if average == "binary":
        return _safe_divide(confmat[1, 1], (confmat[0, 1] + confmat[1, 0] + confmat[1, 1]), zero_division=zero_division)

    ignore_index_cond = ignore_index is not None and 0 <= ignore_index < confmat.shape[0]
    multilabel = confmat.ndim == 3
    if multilabel:
        num = confmat[:, 1, 1]
        denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
    else:  # multiclass
        num = tp.diag(confmat)
        denom = confmat.sum(0) + confmat.sum(1) - num

    if average == "micro":
        num = num.sum()
        denom = denom.sum() - (denom[ignore_index] if ignore_index_cond else 0.0)

    jaccard = _safe_divide(num, denom, zero_division=zero_division)

    if average is None or average == "none" or average == "micro":
        return jaccard
    if average == "weighted":
        weights = confmat[:, 1, 1] + confmat[:, 1, 0] if confmat.ndim == 3 else confmat.sum(1)
    else:
        weights = tp.ones_like(jaccard)
        if ignore_index_cond:
            weights[ignore_index] = 0.0
        if not multilabel:
            weights[confmat.sum(1) + confmat.sum(0) == 0] = 0.0
    return ((weights * jaccard) / weights.sum()).sum()


def binary_jaccard_index(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Tensor:
    """Calculate the Jaccard index for binary tasks.

    The Jaccard index (also known as the intersection over union or Jaccard similarity coefficient)
    is defined as the size of the intersection divided by the size of the union of two sample sets.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0, 1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        threshold: threshold for transforming probability to binary (0, 1) predictions
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _jaccard_index_reduce(confmat, average="binary", zero_division=zero_division)


def _multiclass_jaccard_index_arg_validation(
    num_classes: int,
    ignore_index: Optional[int] = None,
    average: Optional[str] = None,
) -> None:
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}.")


def multiclass_jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Tensor:
    """Calculate the Jaccard index for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        num_classes: integer specifying the number of classes
        average: defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: sum statistics over all labels
            - ``macro``: calculate statistics for each label and average them
            - ``weighted``: calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: calculate statistic for each label and apply no reduction

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """
    if validate_args:
        _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _jaccard_index_reduce(confmat, average=average, ignore_index=ignore_index, zero_division=zero_division)


def _multilabel_jaccard_index_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    average: Optional[str] = "macro",
) -> None:
    _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}.")


def multilabel_jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Tensor:
    """Calculate the Jaccard index for multilabel tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0, 1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        num_labels: integer specifying the number of labels
        threshold: threshold for transforming probability to binary (0, 1) predictions
        average: defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: sum statistics over all labels
            - ``macro``: calculate statistics for each label and average them
            - ``weighted``: calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: calculate statistic for each label and apply no reduction

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """
    if validate_args:
        _multilabel_jaccard_index_arg_validation(num_labels, threshold, ignore_index)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _jaccard_index_reduce(confmat, average=average, ignore_index=ignore_index, zero_division=zero_division)


def jaccard_index(
    preds: Tensor,
    target: Tensor,
    task: str,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[str] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Tensor:
    """Calculate the Jaccard index.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_jaccard_index(preds, target, threshold, ignore_index, validate_args, zero_division)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_jaccard_index(preds, target, num_classes, average, ignore_index, validate_args, zero_division)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_jaccard_index(
            preds, target, num_labels, threshold, average, ignore_index, validate_args, zero_division
        )
    raise ValueError(f"Not handled value: {task}")