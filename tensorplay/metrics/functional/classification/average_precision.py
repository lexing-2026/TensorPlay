"""Functional versions of average precision (AP) computation."""

from typing import List, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from tensorplay.metrics.utilities.compute import _safe_divide
from tensorplay.metrics.utilities.data import _bincount
from tensorplay.metrics.utilities.enums import ClassificationTask
from tensorplay.metrics.utilities.prints import rank_zero_warn


def _reduce_average_precision(
    precision: Union[Tensor, List[Tensor]],
    recall: Union[Tensor, List[Tensor]],
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Reduce multiple average precision scores into one number."""
    if isinstance(precision, Tensor) and isinstance(recall, Tensor):
        precision = tensorplay.where(tensorplay.isnan(precision), tensorplay.zeros_like(precision), precision)
        recall = tensorplay.where(tensorplay.isnan(recall), tensorplay.zeros_like(recall), recall)
        res = -tensorplay.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, :-1], 1)
    else:
        res = tensorplay.stack([-tensorplay.sum((r[1:] - r[:-1]) * p[:-1]) for p, r in zip(precision, recall)])
    if average is None or average == "none":
        return res
    if tensorplay.isnan(res).any():
        rank_zero_warn(
            f"Average precision score for one or more classes was `nan`. Ignoring these classes in {average}-average",
            UserWarning,
        )
    idx = ~tensorplay.isnan(res)
    if average == "macro":
        return res[idx].mean()
    if average == "weighted" and weights is not None:
        weights = _safe_divide(weights[idx], weights[idx].sum())
        return (res[idx] * weights).sum()
    raise ValueError("Received an incompatible combinations of inputs to make reduction.")


def _binary_average_precision_compute(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
) -> Tensor:
    """Compute the average precision for the binary case."""
    precision, recall, _ = _binary_precision_recall_curve_compute(state, thresholds)
    precision = tensorplay.where(tensorplay.isnan(precision), tensorplay.zeros_like(precision), precision)
    recall = tensorplay.where(tensorplay.isnan(recall), tensorplay.zeros_like(recall), recall)
    return -tensorplay.sum((recall[1:] - recall[:-1]) * precision[:-1])


def binary_average_precision(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the average precision (AP) score for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        A single scalar with the average precision score

    """
    if validate_args:
        _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_average_precision_compute(state, thresholds)


def _multiclass_average_precision_arg_validation(
    num_classes: int,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
    allowed_average = ("macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average} but got {average}")


def _multiclass_average_precision_compute(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    num_classes: int,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Tensor] = None,
) -> Tensor:
    """Compute the average precision for the multiclass case."""
    precision, recall, _ = _multiclass_precision_recall_curve_compute(state, num_classes, thresholds)
    return _reduce_average_precision(
        precision,
        recall,
        average,
        weights=_bincount(state[1], minlength=num_classes).float() if thresholds is None else state[0][:, 1, :].sum(-1),
    )


def multiclass_average_precision(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the average precision (AP) score for multiclass tasks.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``macro``: Calculate score for each class and average them
            - ``weighted``: calculates score for each class and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each class and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with AP score per class.
        If `average="macro"|"weighted"` then a single scalar is returned.

    """
    if validate_args:
        _multiclass_average_precision_arg_validation(num_classes, average, thresholds, ignore_index)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds)
    return _multiclass_average_precision_compute(state, num_classes, average, thresholds)


def _multilabel_average_precision_arg_validation(
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average} but got {average}")


def _multilabel_average_precision_compute(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    thresholds: Optional[Tensor],
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Compute the average precision for the multilabel case."""
    if average == "micro":
        if isinstance(state, Tensor) and thresholds is not None:
            state = state.sum(1)
        else:
            preds, target = state[0].flatten(), state[1].flatten()
            if ignore_index is not None:
                idx = target == ignore_index
                preds = preds[~idx]
                target = target[~idx]
            state = (preds, target)
        return _binary_average_precision_compute(state, thresholds)

    precision, recall, _ = _multilabel_precision_recall_curve_compute(state, num_labels, thresholds, ignore_index)
    return _reduce_average_precision(
        precision,
        recall,
        average,
        weights=(state[1] == 1).sum(dim=0).float() if thresholds is None else state[0][:, 1, :].sum(-1),
    )


def multilabel_average_precision(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the average precision (AP) score for multilabel tasks.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with AP score per class.
        If `average="micro|macro"|"weighted"` then a single scalar is returned.

    """
    if validate_args:
        _multilabel_average_precision_arg_validation(num_labels, average, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_average_precision_compute(state, num_labels, average, thresholds, ignore_index)


def average_precision(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Optional[Tensor]:
    """Compute the average precision (AP) score.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_average_precision(preds, target, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_average_precision(
            preds, target, num_classes, average, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_average_precision(preds, target, num_labels, average, thresholds, ignore_index, validate_args)
    return None