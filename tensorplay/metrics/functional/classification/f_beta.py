"""Functional versions of F-beta and F-1 scores."""

from typing import Optional

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)
from tensorplay.metrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide, _safe_sum_dim
from tensorplay.metrics.utilities.enums import ClassificationTask


def _fbeta_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    beta: float,
    average: Optional[str],
    multidim_average: str = "global",
    multilabel: bool = False,
    zero_division: float = 0,
) -> Tensor:
    """Reduce classification statistics into an F-beta score."""
    beta2 = beta**2
    if average == "binary":
        return _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp, zero_division)
    if average == "micro":
        tp = _safe_sum_dim(tp, 0 if multidim_average == "global" else 1)
        fn = _safe_sum_dim(fn, 0 if multidim_average == "global" else 1)
        fp = _safe_sum_dim(fp, 0 if multidim_average == "global" else 1)
        return _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp, zero_division)

    fbeta_score = _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp, zero_division)
    return _adjust_weights_safe_divide(fbeta_score, average, multilabel, tp, fp, fn)


def _binary_fbeta_score_arg_validation(
    beta: float,
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    if not (isinstance(beta, float) and beta > 0):
        raise ValueError(f"Expected argument `beta` to be a float larger than 0, but got {beta}.")
    _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)


def binary_fbeta_score(
    preds: Tensor,
    target: Tensor,
    beta: float,
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-beta score for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to replace zero divisions with

    Returns:
        F-beta score

    """
    if validate_args:
        _binary_fbeta_score_arg_validation(beta, threshold, multidim_average, ignore_index, zero_division)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _fbeta_reduce(tp, fp, tn, fn, beta, average="binary", multidim_average=multidim_average, zero_division=zero_division)


def _multiclass_fbeta_score_arg_validation(
    beta: float,
    num_classes: int,
    top_k: int = 1,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    if not (isinstance(beta, float) and beta > 0):
        raise ValueError(f"Expected argument `beta` to be a float larger than 0, but got {beta}.")
    _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index, zero_division)


def multiclass_fbeta_score(
    preds: Tensor,
    target: Tensor,
    beta: float,
    num_classes: int,
    average: Optional[str] = "macro",
    top_k: int = 1,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-beta score for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
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

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to replace zero divisions with

    Returns:
        F-beta score

    """
    if validate_args:
        _multiclass_fbeta_score_arg_validation(
            beta, num_classes, top_k, average, multidim_average, ignore_index, zero_division
        )
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, average, multidim_average, ignore_index
    )
    return _fbeta_reduce(
        tp, fp, tn, fn, beta, average=average, multidim_average=multidim_average, zero_division=zero_division
    )


def _multilabel_fbeta_score_arg_validation(
    beta: float,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    if not (isinstance(beta, float) and beta > 0):
        raise ValueError(f"Expected argument `beta` to be a float larger than 0, but got {beta}.")
    _multilabel_stat_scores_arg_validation(
        num_labels, threshold, average, multidim_average, ignore_index, zero_division
    )


def multilabel_fbeta_score(
    preds: Tensor,
    target: Tensor,
    beta: float,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-beta score for multilabel tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
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

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to replace zero divisions with

    Returns:
        F-beta score

    """
    if validate_args:
        _multilabel_fbeta_score_arg_validation(
            beta, num_labels, threshold, average, multidim_average, ignore_index, zero_division
        )
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _fbeta_reduce(
        tp,
        fp,
        tn,
        fn,
        beta,
        average=average,
        multidim_average=multidim_average,
        multilabel=True,
        zero_division=zero_division,
    )


def fbeta_score(
    preds: Tensor,
    target: Tensor,
    task: str,
    beta: float = 1.0,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[str] = "micro",
    multidim_average: Optional[str] = "global",
    top_k: Optional[int] = 1,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-beta score for binary, multiclass or multilabel tasks.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None  # noqa: S101  # needed for mypy
    if task == ClassificationTask.BINARY:
        return binary_fbeta_score(
            preds, target, beta, threshold, multidim_average, ignore_index, validate_args, zero_division
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        if not isinstance(top_k, int):
            raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
        return multiclass_fbeta_score(
            preds,
            target,
            beta,
            num_classes,
            average,
            top_k,
            multidim_average,
            ignore_index,
            validate_args,
            zero_division,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_fbeta_score(
            preds,
            target,
            beta,
            num_labels,
            threshold,
            average,
            multidim_average,
            ignore_index,
            validate_args,
            zero_division,
        )
    raise ValueError(f"Unsupported task `{task}` passed.")


def f1_score(
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
    zero_division: float = 0,
) -> Tensor:
    """Compute F-1 score for binary, multiclass or multilabel tasks.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    return fbeta_score(
        preds,
        target,
        task,
        beta=1.0,
        threshold=threshold,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average,
        multidim_average=multidim_average,
        top_k=top_k,
        ignore_index=ignore_index,
        validate_args=validate_args,
        zero_division=zero_division,
    )


def binary_f1_score(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-1 score for binary tasks."""
    return binary_fbeta_score(
        preds,
        target,
        beta=1.0,
        threshold=threshold,
        multidim_average=multidim_average,
        ignore_index=ignore_index,
        validate_args=validate_args,
        zero_division=zero_division,
    )


def multiclass_f1_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    top_k: int = 1,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-1 score for multiclass tasks."""
    return multiclass_fbeta_score(
        preds,
        target,
        beta=1.0,
        num_classes=num_classes,
        average=average,
        top_k=top_k,
        multidim_average=multidim_average,
        ignore_index=ignore_index,
        validate_args=validate_args,
        zero_division=zero_division,
    )


def multilabel_f1_score(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> Tensor:
    """Compute F-1 score for multilabel tasks."""
    return multilabel_fbeta_score(
        preds,
        target,
        beta=1.0,
        num_labels=num_labels,
        threshold=threshold,
        average=average,
        multidim_average=multidim_average,
        ignore_index=ignore_index,
        validate_args=validate_args,
        zero_division=zero_division,
    )