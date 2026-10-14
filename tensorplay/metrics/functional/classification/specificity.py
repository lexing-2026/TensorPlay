"""Functional interface for the specificity metric."""

from typing import Optional

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
from tensorplay.metrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from tensorplay.metrics.utilities.enums import ClassificationTask


def _specificity_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[str],
    multidim_average: str = "global",
    multilabel: bool = False,
) -> Tensor:
    if average == "binary":
        return _safe_divide(tn, tn + fp)
    if average == "micro":
        tn = tn.sum(dim=0 if multidim_average == "global" else 1)
        fp = fp.sum(dim=0 if multidim_average == "global" else 1)
        return _safe_divide(tn, tn + fp)

    specificity_score = _safe_divide(tn, tn + fp)
    return _adjust_weights_safe_divide(specificity_score, average, multilabel, tp, fp, fn)


def binary_specificity(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute specificity for binary tasks.

    Specificity is defined as the ratio between true negatives and the sum of true negatives and
    false positives.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0, 1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        threshold: threshold for transforming probability to binary {0, 1} predictions
        multidim_average: defines how additionally dimensions ``...`` should be handled. Should be one of the
            following:

            - ``global``: additional dimensions are flattened along the batch dimension
            - ``samplewise``: statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        If ``multidim_average`` is set to ``global``, the metric returns a scalar value. If ``multidim_average``
        is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar value per sample.

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _specificity_reduce(tp, fp, tn, fn, average="binary", multidim_average=multidim_average)


def multiclass_specificity(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    top_k: int = 1,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute specificity for multiclass tasks.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: tensor with predictions
        target: tensor with true labels
        num_classes: integer specifying the number of classes
        average: defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: sum statistics over all labels
            - ``macro``: calculate statistics for each label and average them
            - ``weighted``: calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: calculate statistic for each label and apply no reduction

        top_k: number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average: defines how additionally dimensions ``...`` should be handled. Should be one of the
            following:

            - ``global``: additional dimensions are flattened along the batch dimension
            - ``samplewise``: statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    """
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, average, multidim_average, ignore_index
    )
    return _specificity_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average)


def multilabel_specificity(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[str] = "macro",
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute specificity for multilabel tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0, 1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

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

        multidim_average: defines how additionally dimensions ``...`` should be handled. Should be one of the
            following:

            - ``global``: additional dimensions are flattened along the batch dimension
            - ``samplewise``: statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _specificity_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average, multilabel=True)


def specificity(
    preds: Tensor,
    target: Tensor,
    task: str,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[str] = "micro",
    multidim_average: str = "global",
    top_k: Optional[int] = 1,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    """Compute specificity.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_specificity(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        if not isinstance(top_k, int):
            raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
        return multiclass_specificity(
            preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_specificity(
            preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args
        )
    raise ValueError(f"Not handled value: {task}")