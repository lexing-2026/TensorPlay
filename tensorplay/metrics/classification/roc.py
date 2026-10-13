"""Class-based receiver operating characteristic (ROC) metrics."""

from typing import Any, List, Optional, Union

from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from tensorplay.metrics.functional.classification.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.enums import ClassificationTask


class BinaryROC(BinaryPrecisionRecallCurve):
    """Compute the receiver operating characteristic (ROC) for binary tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): An int tensor of shape ``(N, ...)`` containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive
      class.

    As output to ``forward`` and ``compute`` the metric returns a tuple of 3 tensors containing:

    - ``fpr``: an 1d tensor of size ``(n_thresholds+1, )`` with false positive rate values
    - ``tpr``: an 1d tensor of size ``(n_thresholds+1, )`` with true positive rate values
    - ``thresholds``: an 1d tensor of size ``(n_thresholds, )`` with decreasing threshold values

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        """Compute metric."""
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        return _binary_roc_compute(state, self.thresholds)  # type: ignore[arg-type]


class MulticlassROC(MulticlassPrecisionRecallCurve):
    """Compute the receiver operating characteristic (ROC) for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, C, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): An int tensor of shape ``(N, ...)`` containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 3 tensors or 3 lists containing:

    - ``fpr``: if `thresholds=None` a list for each class is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with false positive rate values (length may differ between classes). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with false positive
      rate values.
    - ``tpr``: if `thresholds=None` a list for each class is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with true positive rate values (length may differ between classes). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with true positive rate
      values.
    - ``thresholds``: if `thresholds=None` a list for each class is returned with an 1d tensor of size
      ``(n_thresholds, )`` with decreasing threshold values (length may differ between classes). If `threshold` is
      set to something else, then a single 1d tensor of size ``(n_thresholds, )`` with shared threshold values.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> Union[tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric."""
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        return _multiclass_roc_compute(state, self.num_classes, self.thresholds, self.average)  # type: ignore[arg-type]


class MultilabelROC(MultilabelPrecisionRecallCurve):
    """Compute the receiver operating characteristic (ROC) for multilabel tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, C, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 3 tensors or 3 lists containing:

    - ``fpr``: if `thresholds=None` a list for each label is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with false positive rate values (length may differ between labels). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with false positive
      rate values.
    - ``tpr``: if `thresholds=None` a list for each label is returned with an 1d tensor of size
      ``(n_thresholds+1, )`` with true positive rate values (length may differ between labels). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with true positive rate
      values.
    - ``thresholds``: if `thresholds=None` a list for each label is returned with an 1d tensor of size
      ``(n_thresholds, )`` with decreasing threshold values (length may differ between labels). If `threshold` is
      set to something else, then a single 1d tensor of size ``(n_thresholds, )`` with shared threshold values.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def compute(self) -> Union[tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric."""
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        return _multilabel_roc_compute(state, self.num_labels, self.thresholds, self.ignore_index)  # type: ignore[arg-type]


class ROC(_ClassificationTaskWrapper):
    """Compute the receiver operating characteristic (ROC).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["ROC"],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["micro", "macro"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"thresholds": thresholds, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryROC(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassROC(num_classes, average=average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelROC(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")