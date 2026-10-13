"""Class-based precision-recall curve metrics."""

from typing import Any, List, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
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
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.enums import ClassificationTask


class BinaryPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for binary tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): An int tensor of shape ``(N, ...)`` containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive
      class.

    As output to ``forward`` and ``compute`` the metric returns a tuple of 3 tensors:

    - ``precision``: an 1d tensor of size ``(n_thresholds+1, )`` with precision values
    - ``recall``: an 1d tensor of size ``(n_thresholds+1, )`` with recall values
    - ``thresholds``: an 1d tensor of size ``(n_thresholds, )`` with increasing threshold values

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]
    confmat: Tensor

    def __init__(
        self,
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        normalization: Optional[Literal["sigmoid", "softmax"]] = "sigmoid",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)

        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.normalization = normalization

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.add_state(
                "confmat", default=tensorplay.zeros(len(thresholds), 2, 2, dtype=tensorplay.long), dist_reduce_fx="sum"
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(preds, target, self.ignore_index)
        preds, target, _ = _binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
            self.normalization,
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _binary_precision_recall_curve_compute(state, self.thresholds)


class MulticlassPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, C, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): An int tensor of shape ``(N, ...)`` containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 3 tensors or 3 lists containing:

    - ``precision``: if `thresholds=None` a list for each class is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with precision values (length may differ between classes). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with precision values.
    - ``recall``: if `thresholds=None` a list for each class is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with recall values (length may differ between classes). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with recall values.
    - ``thresholds``: if `thresholds=None` a list for each class is returned with an 1d tensor of size
      ``(n_thresholds, )`` with increasing threshold values (length may differ between classes). If `threshold` is set
      to something else, then a single 1d tensor of size ``(n_thresholds, )`` with shared threshold values.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        average: Optional[Literal["micro", "macro"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index, average)

        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.add_state(
                "confmat",
                default=tensorplay.zeros(len(thresholds), num_classes, 2, 2, dtype=tensorplay.long),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multiclass_precision_recall_curve_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target, _ = _multiclass_precision_recall_curve_format(
            preds, target, self.num_classes, self.thresholds, self.ignore_index, self.average
        )
        state = _multiclass_precision_recall_curve_update(
            preds, target, self.num_classes, self.thresholds, self.average
        )
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Union[tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _multiclass_precision_recall_curve_compute(state, self.num_classes, self.thresholds, self.average)


class MultilabelPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for multilabel tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, C, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 3 tensors or 3 lists containing:

    - ``precision``: if `thresholds=None` a list for each label is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with precision values (length may differ between labels). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with precision values.
    - ``recall``: if `thresholds=None` a list for each label is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with recall values (length may differ between labels). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with recall values.
    - ``thresholds``: if `thresholds=None` a list for each label is returned with an 1d tensor of size
      ``(n_thresholds, )`` with increasing threshold values (length may differ between labels). If `threshold` is set
      to something else, then a single 1d tensor of size ``(n_thresholds, )`` with shared threshold values.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]
    confmat: Tensor

    def __init__(
        self,
        num_labels: int,
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)

        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.add_state(
                "confmat",
                default=tensorplay.zeros(len(thresholds), num_labels, 2, 2, dtype=tensorplay.long),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multilabel_precision_recall_curve_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target, _ = _multilabel_precision_recall_curve_format(
            preds, target, self.num_labels, self.thresholds, self.ignore_index
        )
        state = _multilabel_precision_recall_curve_update(preds, target, self.num_labels, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Union[tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _multilabel_precision_recall_curve_compute(state, self.num_labels, self.thresholds, self.ignore_index)


class PrecisionRecallCurve(_ClassificationTaskWrapper):
    """Compute the precision-recall curve.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["PrecisionRecallCurve"],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"thresholds": thresholds, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryPrecisionRecallCurve(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassPrecisionRecallCurve(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelPrecisionRecallCurve(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")