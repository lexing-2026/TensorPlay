"""Class-based average precision (AP) metrics."""

from typing import Any, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from tensorplay.metrics.functional.classification.average_precision import (
    _binary_average_precision_compute,
    _multiclass_average_precision_arg_validation,
    _multiclass_average_precision_compute,
    _multilabel_average_precision_arg_validation,
    _multilabel_average_precision_compute,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.enums import ClassificationTask
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryAveragePrecision(BinaryPrecisionRecallCurve):
    """Compute the average precision (AP) score for binary tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): An int tensor of shape ``(N, ...)`` containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive
      class.

    As output to ``forward`` and ``compute`` the metric returns a single scalar with the average precision score.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _binary_average_precision_compute(state, self.thresholds)

    def plot(  # type: ignore[override]
        self, val: Optional[Union[Tensor, list[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these results.
                If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        """
        return self._plot(val, ax)


class MulticlassAveragePrecision(MulticlassPrecisionRecallCurve):
    """Compute the average precision (AP) score for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, C, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): An int tensor of shape ``(N, ...)`` containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns:

    - If ``average=None|"none"`` then a 1d tensor of shape (n_classes, ) with AP score per class.
    - If ``average="macro"|"weighted"`` then a single scalar.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_average_precision_arg_validation(num_classes, average, thresholds, ignore_index)
        self.average = average  # type: ignore[assignment]
        self.validate_args = validate_args

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _multiclass_average_precision_compute(
            state,
            self.num_classes,
            self.average,  # type: ignore[arg-type]
            self.thresholds,
        )

    def plot(  # type: ignore[override]
        self, val: Optional[Union[Tensor, list[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these results.
                If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        """
        return self._plot(val, ax)


class MultilabelAveragePrecision(MultilabelPrecisionRecallCurve):
    """Compute the average precision (AP) score for multilabel tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): A tensor of shape ``(N, C, ...)`` containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns:

    - If ``average=None|"none"`` then a 1d tensor of shape (n_classes, ) with AP score per class.
    - If ``average="micro|macro"|"weighted"`` then a single scalar.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multilabel_average_precision_arg_validation(num_labels, average, thresholds, ignore_index)
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _multilabel_average_precision_compute(state, self.num_labels, self.average, self.thresholds, self.ignore_index)

    def plot(  # type: ignore[override]
        self, val: Optional[Union[Tensor, list[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these results.
                If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        """
        return self._plot(val, ax)


class AveragePrecision(_ClassificationTaskWrapper):
    """Compute the average precision (AP) score.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["AveragePrecision"],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"thresholds": thresholds, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryAveragePrecision(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassAveragePrecision(num_classes, average=average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelAveragePrecision(num_labels, average=average, **kwargs)
        raise ValueError(f"Task {task} not supported!")