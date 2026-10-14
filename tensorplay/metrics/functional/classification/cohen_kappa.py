"""Functional versions of Cohen's kappa score computation."""

from typing import Optional

import tensorplay as tensorplay
from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
)
from tensorplay.metrics.utilities.enums import ClassificationTaskNoMultilabel


def _cohen_kappa_reduce(confmat: Tensor, weights: Optional[Literal["linear", "quadratic", "none"]] = None) -> Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the cohen kappa score."""
    confmat = confmat.float() if not confmat.is_floating_point() else confmat
    num_classes = confmat.shape[0]
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()  # outer product

    if weights is None or weights == "none":
        w_mat = tensorplay.ones_like(confmat).flatten()
        w_mat[:: num_classes + 1] = 0
        w_mat = w_mat.reshape(num_classes, num_classes)
    elif weights in ("linear", "quadratic"):
        w_mat = tensorplay.zeros_like(confmat)
        w_mat += tensorplay.arange(num_classes, dtype=w_mat.dtype, device=w_mat.device)
        w_mat = tensorplay.abs(w_mat - w_mat.T) if weights == "linear" else tensorplay.pow(w_mat - w_mat.T, 2.0)
    else:
        raise ValueError(
            f"Received {weights} for argument ``weights`` but should be either None, 'linear' or 'quadratic'"
        )
    k = tensorplay.sum(w_mat * confmat) / tensorplay.sum(w_mat * expected)
    return 1 - k


def _binary_cohen_kappa_arg_validation(
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``ignore_index`` has to be None or int
    - ``weights`` has to be "linear" | "quadratic" | "none" | None

    """
    _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize=None)
    allowed_weights = ("linear", "quadratic", "none", None)
    if weights not in allowed_weights:
        raise ValueError(f"Expected argument `weight` to be one of {allowed_weights}, but got {weights}.")


def binary_cohen_kappa(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Calculate Cohen's kappa score that measures inter-annotator agreement for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if validate_args:
        _binary_cohen_kappa_arg_validation(threshold, ignore_index, weights)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _cohen_kappa_reduce(confmat, weights)


def _multiclass_cohen_kappa_arg_validation(
    num_classes: int,
    ignore_index: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``ignore_index`` has to be None or int
    - ``weights`` has to be "linear" | "quadratic" | "none" | None

    """
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize=None)
    allowed_weights = ("linear", "quadratic", "none", None)
    if weights not in allowed_weights:
        raise ValueError(f"Expected argument `weight` to be one of {allowed_weights}, but got {weights}.")


def multiclass_cohen_kappa(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Calculate Cohen's kappa score that measures inter-annotator agreement for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``tensorplay.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if validate_args:
        _multiclass_cohen_kappa_arg_validation(num_classes, ignore_index, weights)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _cohen_kappa_reduce(confmat, weights)


def cohen_kappa(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Calculate Cohen's kappa score that measures inter-annotator agreement.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``.

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_cohen_kappa(preds, target, threshold, weights, ignore_index, validate_args)
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_cohen_kappa(preds, target, num_classes, weights, ignore_index, validate_args)
    raise ValueError(f"Not handled value: {task}")