"""Functional versions of confusion matrix computation."""

from typing import Optional

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.compute import normalize_logits_if_needed
from tensorplay.metrics.utilities.data import _bincount
from tensorplay.metrics.utilities.enums import ClassificationTask
from tensorplay.metrics.utilities.prints import rank_zero_warn


def _confusion_matrix_reduce(
    confmat: Tensor, normalize: Optional[str] = None
) -> Tensor:
    """Reduce an un-normalized confusion matrix.

    Args:
        confmat: un-normalized confusion matrix
        normalize: normalization method:

            - ``"true"`` will divide by the sum of the column dimension.
            - ``"pred"`` will divide by the sum of the row dimension.
            - ``"all"`` will divide by the sum of the full matrix
            - ``"none"`` or ``None`` will apply no reduction.

    Returns:
        Normalized confusion matrix

    """
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(f"Argument `normalize` needs to one of the following: {allowed_normalize}")
    if normalize is not None and normalize != "none":
        confmat = confmat.float() if not tensorplay.is_floating_point(confmat) else confmat
        if normalize == "true":
            confmat = confmat / confmat.sum(dim=-1, keepdim=True)
        elif normalize == "pred":
            confmat = confmat / confmat.sum(dim=-2, keepdim=True)
        elif normalize == "all":
            confmat = confmat / confmat.sum(dim=[-2, -1], keepdim=True)

        nan_elements = tensorplay.isnan(confmat).sum().item()
        if nan_elements:
            confmat[tensorplay.isnan(confmat)] = 0
            rank_zero_warn(f"{nan_elements} NaN values found in confusion matrix have been replaced with zeros.")
    return confmat


def _binary_confusion_matrix_arg_validation(
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    normalize: Optional[str] = None,
) -> None:
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _binary_confusion_matrix_tensor_validation(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
) -> None:
    if preds.ndim < 1:
        raise ValueError("Expected input to binary confusion matrix to be at least a 1d tensor.")
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected input to binary confusion matrix to have the same shape, but got {preds.shape} and {target.shape}"
        )
    if not tensorplay.is_floating_point(preds):
        preds_uniq = tensorplay.unique(preds)
        if ignore_index is not None:
            preds_uniq = preds_uniq[preds_uniq != ignore_index]
        if tensorplay.any((preds_uniq != 0) & (preds_uniq != 1)):
            raise ValueError(
                f"Detected the following values in `preds`: {preds_uniq} but expected only"
                " the following values [0,1] since `preds` is a label tensor."
            )
    target_uniq = tensorplay.unique(target)
    if ignore_index is not None:
        target_uniq = target_uniq[target_uniq != ignore_index]
    if tensorplay.any((target_uniq != 0) & (target_uniq != 1)):
        raise ValueError(
            f"Detected the following values in `target`: {target_uniq} but expected only"
            " the following values [0,1]."
        )


def _binary_confusion_matrix_format(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    convert_to_labels: bool = True,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    if tensorplay.is_floating_point(preds):
        preds = normalize_logits_if_needed(preds, "sigmoid")
        if convert_to_labels:
            preds = preds > threshold

    return preds, target


def _binary_confusion_matrix_update(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (target * 2 + preds).to(tensorplay.long)
    bins = _bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)


def _binary_confusion_matrix_compute(
    confmat: Tensor, normalize: Optional[str] = None
) -> Tensor:
    """Reduce the confusion matrix to its final form."""
    return _confusion_matrix_reduce(confmat, normalize)


def binary_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the confusion matrix for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
        normalize: Normalization mode, one of ``"true"``, ``"pred"``, ``"all"``, ``"none"`` or ``None``
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The confusion matrix of shape ``(2, 2)``

    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _binary_confusion_matrix_compute(confmat, normalize)


def _multiclass_confusion_matrix_arg_validation(
    num_classes: Optional[int],
    ignore_index: Optional[int] = None,
    normalize: Optional[str] = None,
) -> None:
    if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 2):
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _multiclass_confusion_matrix_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int],
    ignore_index: Optional[int] = None,
) -> None:
    if preds.ndim == target.ndim + 1:
        if not tensorplay.is_floating_point(preds):
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if num_classes is not None and preds.shape[1] != num_classes:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds.shape[1]` should be"
                " equal to number of classes."
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )
    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,"
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}."
            )
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )
    if num_classes is not None:
        check_value = num_classes if ignore_index is None else num_classes + 1
        for t, name in ((target, "target"),) + ((preds, "preds"),) if not tensorplay.is_floating_point(preds) else ():
            num_unique_values = len(tensorplay.unique(t, dim=None))
            if num_unique_values > check_value:
                raise RuntimeError(
                    f"Detected more unique values in `{name}` than expected. Expected only {check_value} but found"
                    f" {num_unique_values} in `{name}`. Found values: {tensorplay.unique(t, dim=None)}."
                )


def _multiclass_confusion_matrix_format(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
    convert_to_labels: bool = True,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1 and convert_to_labels:
        preds = tensorplay.argmax(preds, dim=1)

    preds = preds.flatten() if convert_to_labels else tensorplay.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
    target = target.flatten()

    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    return preds, target


def _multiclass_confusion_matrix_update(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = target.to(tensorplay.long) * num_classes + preds.to(tensorplay.long)
    bins = _bincount(unique_mapping, minlength=num_classes**2)
    return bins.reshape(num_classes, num_classes)


def _multiclass_confusion_matrix_compute(
    confmat: Tensor, normalize: Optional[str] = None
) -> Tensor:
    """Reduce the confusion matrix to its final form."""
    return _confusion_matrix_reduce(confmat, normalize)


def multiclass_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the confusion matrix for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        normalize: Normalization mode, one of ``"true"``, ``"pred"``, ``"all"``, ``"none"`` or ``None``
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The confusion matrix of shape ``(num_classes, num_classes)``

    """
    if validate_args:
        _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _multiclass_confusion_matrix_compute(confmat, normalize)


def _multilabel_confusion_matrix_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    normalize: Optional[str] = None,
) -> None:
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}")
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _multilabel_confusion_matrix_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    ignore_index: Optional[int] = None,
) -> None:
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected input to multilabel confusion matrix to have the same shape, but got {preds.shape} and {target.shape}"
        )
    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels"
            f" but got {preds.shape[1]} and expected {num_labels}"
        )
    if not tensorplay.is_floating_point(preds):
        preds_uniq = tensorplay.unique(preds)
        if ignore_index is not None:
            preds_uniq = preds_uniq[preds_uniq != ignore_index]
        if tensorplay.any((preds_uniq != 0) & (preds_uniq != 1)):
            raise ValueError(
                f"Detected the following values in `preds`: {preds_uniq} but expected only"
                " the following values [0,1] since `preds` is a label tensor."
            )
    target_uniq = tensorplay.unique(target)
    if ignore_index is not None:
        target_uniq = target_uniq[target_uniq != ignore_index]
    if tensorplay.any((target_uniq != 0) & (target_uniq != 1)):
        raise ValueError(
            f"Detected the following values in `target`: {target_uniq} but expected only"
            " the following values [0,1]."
        )


def _multilabel_confusion_matrix_format(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    should_threshold: bool = True,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    if tensorplay.is_floating_point(preds):
        preds = normalize_logits_if_needed(preds, "sigmoid")
        if should_threshold:
            preds = preds > threshold
    preds = tensorplay.movedim(preds, 1, -1).reshape(-1, num_labels)
    target = tensorplay.movedim(target, 1, -1).reshape(-1, num_labels)

    if ignore_index is not None:
        preds = preds.clone()
        target = target.clone()
        # Make sure that when we map, it will always result in a negative number that we can filter away
        # Each label correspond to a 2x2 matrix = 4 elements per label
        idx = target == ignore_index
        preds[idx] = -4 * num_labels
        target[idx] = -4 * num_labels

    return preds, target


def _multilabel_confusion_matrix_update(preds: Tensor, target: Tensor, num_labels: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = ((2 * target + preds) + 4 * tensorplay.arange(num_labels, device=preds.device)).flatten()
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels)
    return bins.reshape(num_labels, 2, 2)


def _multilabel_confusion_matrix_compute(
    confmat: Tensor, normalize: Optional[str] = None
) -> Tensor:
    """Reduce the confusion matrix to its final form."""
    return _confusion_matrix_reduce(confmat, normalize)


def multilabel_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the confusion matrix for multilabel tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        normalize: Normalization mode, one of ``"true"``, ``"pred"``, ``"all"``, ``"none"`` or ``None``
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The confusion matrix of shape ``(num_labels, 2, 2)``

    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _multilabel_confusion_matrix_compute(confmat, normalize)


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    task: str,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the confusion matrix for binary, multiclass or multilabel tasks.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_confusion_matrix(preds, target, threshold, normalize, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_confusion_matrix(preds, target, num_classes, normalize, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_confusion_matrix(preds, target, num_labels, threshold, normalize, ignore_index, validate_args)
    raise ValueError(f"Task {task} not supported.")