"""Functional versions of classification statistics (TP/FP/TN/FN and support)."""

from typing import Optional

import tensorplay as tensorplay
from tensorplay import Tensor
import tensorplay.nn.functional as F  # noqa: N812

from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.compute import normalize_logits_if_needed
from tensorplay.metrics.utilities.data import _bincount, select_topk
from tensorplay.metrics.utilities.enums import ClassificationTask


def _binary_stat_scores_arg_validation(
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input."""
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}.")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    if zero_division not in [0, 1]:
        raise ValueError(f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}.")


def _binary_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input."""
    _check_same_shape(preds, target)

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = tensorplay.unique(target, dim=None)
    if ignore_index is None:
        check = tensorplay.any((unique_values != 0) & (unique_values != 1))
    else:
        check = tensorplay.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0, 1] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not tensorplay.is_floating_point(preds):
        unique_values = tensorplay.unique(preds, dim=None)
        if tensorplay.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since `preds` is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 2:
        raise ValueError("Expected input to be at least 2D when multidim_average is set to `samplewise`")


def _binary_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    if tensorplay.is_floating_point(preds):
        preds = normalize_logits_if_needed(preds, "sigmoid")
        preds = preds > threshold

    preds = tensorplay.reshape(preds, (preds.shape[0], -1))
    target = tensorplay.reshape(target, (target.shape[0], -1))

    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1

    return preds, target


def _binary_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    multidim_average: str = "global",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics."""
    sum_dim = [0, 1] if multidim_average == "global" else [1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _binary_stat_scores_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    multidim_average: str = "global",
) -> Tensor:
    """Stack statistics and compute support also."""
    return tensorplay.stack([tp, fp, tn, fn, tp + fn], dim=0 if multidim_average == "global" else 1).squeeze()


def binary_stat_scores(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``).

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _binary_stat_scores_compute(tp, fp, tn, fn, multidim_average)


def _multiclass_stat_scores_arg_validation(
    num_classes: Optional[int],
    top_k: int = 1,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input."""
    if num_classes is None and average != "micro":
        raise ValueError(
            f"Argument `num_classes` can only be `None` for `average='micro'`, but got `average={average}`."
        )
    if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 2):
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"Expected argument `top_k` to be an integer larger than or equal to 1, but got {top_k}")
    if top_k > (num_classes if num_classes is not None else 1):
        raise ValueError(
            f"Expected argument `top_k` to be smaller or equal to `num_classes` but got {top_k} and {num_classes}"
        )
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    if zero_division not in [0, 1]:
        raise ValueError(f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}.")


def _multiclass_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int],
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input."""
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
        if multidim_average != "global" and preds.ndim < 3:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " at least 3D when multidim_average is set to `samplewise`"
            )

    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,"
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}."
            )
        if multidim_average != "global" and preds.ndim < 2:
            raise ValueError(
                "When `preds` and `target` have the same shape, the shape of `preds` should be"
                " at least 2D when multidim_average is set to `samplewise`"
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


def _multiclass_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    top_k: int = 1,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1."""
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = tensorplay.argmax(preds, dim=1)
    preds = tensorplay.reshape(preds, (*preds.shape[:2], -1)) if top_k != 1 else tensorplay.reshape(preds, (preds.shape[0], -1))
    target = tensorplay.reshape(target, (target.shape[0], -1))
    return preds, target


def _refine_preds_oh(preds: Tensor, preds_oh: Tensor, target: Tensor, top_k: int) -> Tensor:
    """Refine prediction one-hot encodings by replacing entries with the target when there is an intersection.

    When no intersection is found between the top-k predictions and target, uses the top-1 prediction.
    """
    if preds.dim() == 1:  # Handle 1D tensor case (single sample)
        preds = preds.unsqueeze(0)  # Add batch dimension
        target = target.unsqueeze(0) if target.dim() == 0 else target

    top_k_indices = tensorplay.topk(preds, k=top_k, dim=1).indices
    top_1_indices = top_k_indices[:, 0]
    target_in_topk = tensorplay.any(top_k_indices == target.unsqueeze(1), dim=1)
    result = tensorplay.where(target_in_topk, target, top_1_indices)
    return tensorplay.zeros_like(preds_oh, dtype=tensorplay.int32).scatter_(-1, result.unsqueeze(-1), 1)


def _multiclass_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    top_k: int = 1,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics."""
    if multidim_average == "samplewise" or top_k != 1:
        ignore_in = 0 <= ignore_index <= num_classes - 1 if ignore_index is not None else None
        if ignore_index is not None and not ignore_in:
            preds = preds.clone()
            target = target.clone()
            idx = target == ignore_index
            target[idx] = num_classes
            idx = idx.unsqueeze(1).repeat(1, num_classes, 1) if preds.ndim > target.ndim else idx
            preds[idx] = num_classes

        if top_k > 1:
            preds_oh = tensorplay.movedim(select_topk(preds, topk=top_k, dim=1), 1, -1)
            preds_oh = _refine_preds_oh(preds, preds_oh, target, top_k)
        else:
            preds_oh = F.one_hot(
                preds.long(), num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
            )

        target_oh = F.one_hot(
            target.long(), num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
        )

        if ignore_index is not None:
            if 0 <= ignore_index <= num_classes - 1:
                target_oh[target == ignore_index, :] = -1
            else:
                preds_oh = preds_oh[..., :-1] if top_k == 1 else preds_oh
                target_oh = target_oh[..., :-1]
                target_oh[target == num_classes, :] = -1
        sum_dim = [0, 1] if multidim_average == "global" else [1]
        tp = ((target_oh == preds_oh) & (target_oh == 1)).sum(sum_dim)
        fn = ((target_oh != preds_oh) & (target_oh == 1)).sum(sum_dim)
        fp = ((target_oh != preds_oh) & (target_oh == 0)).sum(sum_dim)
        tn = ((target_oh == preds_oh) & (target_oh == 0)).sum(sum_dim)
    elif average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        tp = (preds == target).sum()
        fp = (preds != target).sum()
        fn = (preds != target).sum()
        tn = num_classes * preds.numel() - (fp + fn + tp)
    else:
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        unique_mapping = target.to(tensorplay.long) * num_classes + preds.to(tensorplay.long)
        bins = _bincount(unique_mapping, minlength=num_classes**2)
        confmat = bins.reshape(num_classes, num_classes)
        tp = tensorplay.diagonal(confmat)
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
    return tp, fp, tn, fn


def _multiclass_stat_scores_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
) -> Tensor:
    """Stack statistics and compute support also, applying the average strategy."""
    res = tensorplay.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim) if res.ndim > 1 else res
    if average == "macro":
        return res.float().mean(sum_dim)
    if average == "weighted":
        weight = tp + fn
        if multidim_average == "global":
            return (res * (weight / weight.sum()).reshape(*weight.shape, 1)).sum(sum_dim)
        return (res * (weight / weight.sum(-1, keepdim=True)).reshape(*weight.shape, 1)).sum(sum_dim)
    if average is None or average == "none":
        return res
    return None


def multiclass_stat_scores(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    top_k: int = 1,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``).

    """
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, average, multidim_average, ignore_index
    )
    return _multiclass_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)


def _multilabel_stat_scores_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input."""
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}")
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    if zero_division not in [0, 1]:
        raise ValueError(f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}.")


def _multilabel_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input."""
    _check_same_shape(preds, target)

    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels"
            f" but got {preds.shape[1]} and expected {num_labels}"
        )

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = tensorplay.unique(target, dim=None)
    if ignore_index is None:
        check = tensorplay.any((unique_values != 0) & (unique_values != 1))
    else:
        check = tensorplay.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0, 1] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not tensorplay.is_floating_point(preds):
        unique_values = tensorplay.unique(preds, dim=None)
        if tensorplay.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since preds is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 3:
        raise ValueError("Expected input to be at least 3D when multidim_average is set to `samplewise`")


def _multilabel_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    if tensorplay.is_floating_point(preds):
        preds = normalize_logits_if_needed(preds, "sigmoid")
        preds = preds > threshold
    preds = tensorplay.reshape(preds, (*preds.shape[:2], -1))
    target = tensorplay.reshape(target, (*target.shape[:2], -1))

    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1

    return preds, target


def _multilabel_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    multidim_average: str = "global",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics."""
    sum_dim = [0, -1] if multidim_average == "global" else [-1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _multilabel_stat_scores_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
) -> Tensor:
    """Stack statistics and compute support also, applying the average strategy."""
    res = tensorplay.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim)
    if average == "macro":
        return res.float().mean(sum_dim)
    if average == "weighted":
        w = tp + fn
        return (res * (w / w.sum()).reshape(*w.shape, 1)).sum(sum_dim)
    if average is None or average == "none":
        return res
    return None


def multilabel_stat_scores(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for multilabel tasks.

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
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``).

    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _multilabel_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)


def stat_scores(
    preds: Tensor,
    target: Tensor,
    task: str,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[str] = "micro",
    multidim_average: Optional[str] = "global",
    top_k: Optional[int] = 1,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute the number of true positives, false positives, true negatives, false negatives and the support.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None  # noqa: S101  # needed for mypy
    if task == ClassificationTask.BINARY:
        return binary_stat_scores(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        if not isinstance(top_k, int):
            raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
        return multiclass_stat_scores(
            preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_stat_scores(
            preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args
        )
    raise ValueError(f"Unsupported task `{task}`")