"""Specificity metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from tensorplay.metrics.functional.classification.specificity import _specificity_reduce
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.enums import ClassificationTask
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinarySpecificity(BinaryStatScores):
    """Compute specificity for binary tasks.

    Specificity is defined as the ratio between true negatives and the sum of true negatives and
    false positives. The metric is only properly defined when ``TN + FP != 0``. If this case is
    encountered a score of 0 is returned.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): shape ``(N, ...)``. If preds is a floating point tensor with values outside
      [0, 1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally, we
      convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): shape ``(N, ...)``

    Args:
        threshold: threshold for transforming probability to binary {0, 1} predictions
        multidim_average: defines how additionally dimensions ``...`` should be handled. Should be one of the
            following:

            - ``global``: additional dimensions are flattened along the batch dimension
            - ``samplewise``: statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """

    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute specificity."""
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average)

    def plot(
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


class MulticlassSpecificity(MulticlassStatScores):
    """Compute specificity for multiclass tasks.

    The metric is only properly defined when ``TN + FP != 0``. If this case is encountered for any
    class, the metric for that class will be set to 0 and the overall metric may therefore be affected
    in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int tensor): shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``. If preds is a floating
      point tensor we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): shape ``(N, ...)``

    Args:
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

    """

    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def compute(self) -> Tensor:
        """Compute specificity."""
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)

    def plot(
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


class MultilabelSpecificity(MultilabelStatScores):
    """Compute specificity for multilabel tasks.

    The metric is only properly defined when ``TN + FP != 0``. If this case is encountered for any
    label, the metric for that label will be set to 0 and the overall metric may therefore be affected
    in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): shape ``(N, C, ...)``. If preds is a floating point tensor with values
      outside [0, 1] range we consider the input to be logits and will auto apply sigmoid per element.
      Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): shape ``(N, C, ...)``

    Args:
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

    """

    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def compute(self) -> Tensor:
        """Compute specificity."""
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(
            tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average, multilabel=True
        )

    def plot(
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


class Specificity(_ClassificationTaskWrapper):
    """Compute specificity.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["Specificity"],
        task: str,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[str] = "micro",
        multidim_average: Optional[str] = "global",
        top_k: Optional[int] = 1,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update(
            {"multidim_average": multidim_average, "ignore_index": ignore_index, "validate_args": validate_args}
        )
        if task == ClassificationTask.BINARY:
            return BinarySpecificity(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            if not isinstance(top_k, int):
                raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
            return MulticlassSpecificity(num_classes, top_k, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelSpecificity(num_labels, threshold, average, **kwargs)
        raise ValueError(f"Task {task} not supported!")