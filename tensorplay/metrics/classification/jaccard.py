"""Jaccard index metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from tensorplay.metrics.functional.classification.jaccard import (
    _jaccard_index_reduce,
    _multiclass_jaccard_index_arg_validation,
    _multilabel_jaccard_index_arg_validation,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.enums import ClassificationTask
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryJaccardIndex(BinaryConfusionMatrix):
    """Calculate the Jaccard index for binary tasks.

    The Jaccard index (also known as the intersection over union or Jaccard similarity coefficient)
    is defined as the size of the intersection divided by the size of the union of two sample sets.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): shape ``(N, ...)``. If preds is a floating point tensor with values outside
      [0, 1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally, we
      convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): shape ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        threshold: threshold for transforming probability to binary (0, 1) predictions
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, ignore_index=ignore_index, normalize=None, validate_args=validate_args, **kwargs
        )
        self.zero_division = zero_division

    def compute(self) -> Tensor:
        """Compute the Jaccard index."""
        return _jaccard_index_reduce(self.confmat, average="binary", zero_division=self.zero_division)

    def plot(  # type: ignore[override]
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        """
        return self._plot(val, ax)


class MulticlassJaccardIndex(MulticlassConfusionMatrix):
    """Calculate the Jaccard index for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int tensor): shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``. If preds is a floating
      point tensor we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): shape ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: integer specifying the number of classes
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        average: defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: sum statistics over all labels
            - ``macro``: calculate statistics for each label and average them
            - ``weighted``: calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: calculate statistic for each label and apply no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, ignore_index=ignore_index, normalize=None, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        self.validate_args = validate_args
        self.average = average
        self.zero_division = zero_division

    def compute(self) -> Tensor:
        """Compute the Jaccard index."""
        return _jaccard_index_reduce(
            self.confmat, average=self.average, ignore_index=self.ignore_index, zero_division=self.zero_division
        )

    def plot(  # type: ignore[override]
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        """
        return self._plot(val, ax)


class MultilabelJaccardIndex(MultilabelConfusionMatrix):
    """Calculate the Jaccard index for multilabel tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): shape ``(N, C, ...)``. If preds is a floating point tensor with values
      outside [0, 1] range we consider the input to be logits and will auto apply sigmoid per element.
      Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): shape ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_labels: integer specifying the number of labels
        threshold: threshold for transforming probability to binary (0, 1) predictions
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        average: defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: sum statistics over all labels
            - ``macro``: calculate statistics for each label and average them
            - ``weighted``: calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: calculate statistic for each label and apply no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: value to replace when there is a division by zero. Should be ``0`` or ``1``.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[str] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
            normalize=None,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multilabel_jaccard_index_arg_validation(num_labels, threshold, ignore_index, average)
        self.validate_args = validate_args
        self.average = average
        self.zero_division = zero_division

    def compute(self) -> Tensor:
        """Compute the Jaccard index."""
        return _jaccard_index_reduce(self.confmat, average=self.average, zero_division=self.zero_division)

    def plot(  # type: ignore[override]
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        """
        return self._plot(val, ax)


class JaccardIndex(_ClassificationTaskWrapper):
    """Calculate the Jaccard index.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["JaccardIndex"],
        task: str,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[str] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryJaccardIndex(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassJaccardIndex(num_classes, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelJaccardIndex(num_labels, threshold, average, **kwargs)
        raise ValueError(f"Task {task} not supported!")