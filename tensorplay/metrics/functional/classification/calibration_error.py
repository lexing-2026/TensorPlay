"""Functional interface for the calibration error metric."""

from typing import Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
)
from tensorplay.metrics.utilities.compute import normalize_logits_if_needed
from tensorplay.metrics.utilities.enums import ClassificationTaskNoMultilabel


def _binning_bucketize(
    confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute calibration bins using ``bucketize``.

    Args:
        confidences: the confidence (i.e. predicted probability) of the top-1 prediction
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise
        bin_boundaries: bin boundaries separating the ``linspace`` from 0 to 1

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities

    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = tp.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    conf_bin = tp.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    count_bin = tp.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)

    indices = tp.bucketize(confidences, bin_boundaries, right=True) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=tp.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = tp.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = tp.nan_to_num(acc_bin / count_bin)

    prop_bin = count_bin / count_bin.sum()
    return acc_bin, conf_bin, prop_bin


def _ce_compute(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Union[Tensor, int],
    norm: str = "l1",
    debias: bool = False,
) -> Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: the confidence (i.e. predicted probability) of the top-1 prediction
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise
        bin_boundaries: bin boundaries separating the ``linspace`` from 0 to 1
        norm: norm function to use when computing calibration error. Defaults to ``"l1"``.
        debias: apply debiasing to L2 norm computation. Defaults to ``False``.

    Raises:
        ValueError: if an unsupported norm function is provided.

    Returns:
        Tensor: calibration error scalar

    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = tp.linspace(0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device)

    if norm not in {"l1", "l2", "max"}:
        raise ValueError(f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}")

    with tp.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(confidences, accuracies, bin_boundaries)

    if norm == "l1":
        return tp.sum(tp.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "max":
        ce = tp.max(tp.abs(acc_bin - conf_bin))
    if norm == "l2":
        ce = tp.sum(tp.pow(acc_bin - conf_bin, 2) * prop_bin)
        if debias:
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (prop_bin * accuracies.size()[0] - 1)
            ce += tp.sum(tp.nan_to_num(debias_bins))  # replace nans with zeros if nothing appeared in a bin
        return tp.sqrt(ce) if ce > 0 else tp.tensor(0)
    return ce


def _binary_calibration_error_arg_validation(
    n_bins: int,
    norm: str = "l1",
    ignore_index: Optional[int] = None,
) -> None:
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"Expected argument `n_bins` to be an integer larger than 0, but got {n_bins}")
    allowed_norm = ("l1", "l2", "max")
    if norm not in allowed_norm:
        raise ValueError(f"Expected argument `norm` to be one of {allowed_norm}, but got {norm}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _binary_calibration_error_tensor_validation(
    preds: Tensor, target: Tensor, ignore_index: Optional[int] = None
) -> None:
    _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def _binary_calibration_error_update(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    confidences, accuracies = preds, target
    return confidences, accuracies


def binary_calibration_error(
    preds: Tensor,
    target: Tensor,
    n_bins: int = 15,
    norm: str = "l1",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the top-label calibration error for binary tasks.

    The expected calibration error can be used to quantify how well a given model is calibrated,
    i.e. how well the predicted output probabilities match the actual probabilities of the ground
    truth distribution. Three different norms are implemented, each corresponding to a variation of
    the calibration error metric:

    - L1 norm: Expected Calibration Error (ECE)
    - Infinity norm: Maximum Calibration Error (MCE)
    - L2 norm: Root Mean Square Calibration Error (RMSCE)

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
        n_bins: number of bins to use when computing the metric
        norm: norm used to compare empirical and expected probability bins
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if validate_args:
        _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        _binary_calibration_error_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False
    )
    confidences, accuracies = _binary_calibration_error_update(preds, target)
    return _ce_compute(confidences, accuracies, n_bins, norm)


def _multiclass_calibration_error_arg_validation(
    num_classes: int,
    n_bins: int,
    norm: str = "l1",
    ignore_index: Optional[int] = None,
) -> None:
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"Expected argument `n_bins` to be an integer larger than 0, but got {n_bins}")
    allowed_norm = ("l1", "l2", "max")
    if norm not in allowed_norm:
        raise ValueError(f"Expected argument `norm` to be one of {allowed_norm}, but got {norm}.")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _multiclass_calibration_error_tensor_validation(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> None:
    _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def _multiclass_calibration_error_update(
    preds: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    preds = normalize_logits_if_needed(preds, "softmax")
    confidences, predictions = preds.max(dim=1)
    accuracies = predictions.eq(target)
    return confidences.float(), accuracies.float()


def multiclass_calibration_error(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    n_bins: int = 15,
    norm: str = "l1",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the top-label calibration error for multiclass tasks.

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
        n_bins: number of bins to use when computing the metric
        norm: norm used to compare empirical and expected probability bins
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if validate_args:
        _multiclass_calibration_error_arg_validation(num_classes, n_bins, norm, ignore_index)
        _multiclass_calibration_error_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    confidences, accuracies = _multiclass_calibration_error_update(preds, target)
    return _ce_compute(confidences, accuracies, n_bins, norm)


def calibration_error(
    preds: Tensor,
    target: Tensor,
    task: str,
    n_bins: int = 15,
    norm: str = "l1",
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the top-label calibration error.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``.

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_calibration_error(preds, target, n_bins, norm, ignore_index, validate_args)
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_calibration_error(preds, target, num_classes, n_bins, norm, ignore_index, validate_args)
    raise ValueError(f"Expected argument `task` to either be `'binary'` or `'multiclass'` but got {task}")