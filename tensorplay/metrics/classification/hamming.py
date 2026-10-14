"""Class-based Hamming distance metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from tensorplay.metrics.functional.classification.hamming import _hamming_distance_reduce
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.enums import ClassificationTask
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryHammingDistance(BinaryStatScores):
    """Compute the average Hamming distance (also known as Hamming loss) for binary tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns:

    - If ``multidim_average`` is set to ``global``, the metric returns a scalar value.
    - If ``multidim_average`` is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar
      value per sample.

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)


class MulticlassHammingDistance(MulticlassStatScores):
    """Compute the average Hamming distance (also known as Hamming loss) for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns:

    - If ``multidim_average`` is set to ``global``:

      - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
      - If ``average=None/'none'``, the shape will be ``(C,)``

    - If ``multidim_average`` is set to ``samplewise``:

      - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
      - If ``average=None/'none'``, the shape will be ``(N, C)``

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)


class MultilabelHammingDistance(MultilabelStatScores):
    """Compute the average Hamming distance (also known as Hamming loss) for multilabel tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    As output to ``forward`` and ``compute`` the metric returns:

    - If ``multidim_average`` is set to ``global``:

      - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
      - If ``average=None/'none'``, the shape will be ``(C,)``

    - If ``multidim_average`` is set to ``samplewise``:

      - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
      - If ``average=None/'none'``, the shape will be ``(N, C)``

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(
            tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average, multilabel=True
        )

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)


class HammingDistance(_ClassificationTaskWrapper):
    """Compute the average Hamming distance (also known as Hamming loss).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["HammingDistance"],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        multidim_average: Literal["global", "samplewise"] = "global",
        top_k: Optional[int] = 1,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryHammingDistance(threshold, multidim_average, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            if not isinstance(top_k, int):
                raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
            return MulticlassHammingDistance(num_classes, average, top_k, multidim_average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelHammingDistance(num_labels, threshold, average, multidim_average, **kwargs)
        raise ValueError(f"Task {task} not supported!")