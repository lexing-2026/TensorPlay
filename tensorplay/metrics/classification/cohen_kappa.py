"""Class-based Cohen's kappa score metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor
from typing import Literal

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.classification.confusion_matrix import BinaryConfusionMatrix, MulticlassConfusionMatrix
from tensorplay.metrics.functional.classification.cohen_kappa import (
    _binary_cohen_kappa_arg_validation,
    _cohen_kappa_reduce,
    _multiclass_cohen_kappa_arg_validation,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.enums import ClassificationTaskNoMultilabel
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryCohenKappa(BinaryConfusionMatrix):
    """Calculate Cohen's kappa score that measures inter-annotator agreement for binary tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns a scalar value representing the Cohen's kappa score.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        weights: Optional[Literal["linear", "quadratic", "none"]] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold, ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _binary_cohen_kappa_arg_validation(threshold, ignore_index, weights)
        self.weights = weights
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        """Compute metric."""
        return _cohen_kappa_reduce(self.confmat, self.weights)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)


class MulticlassCohenKappa(MulticlassConfusionMatrix):
    """Calculate Cohen's kappa score that measures inter-annotator agreement for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns a scalar value representing the Cohen's kappa score.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        weights: Optional[Literal["linear", "quadratic", "none"]] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _multiclass_cohen_kappa_arg_validation(num_classes, ignore_index, weights)
        self.weights = weights
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        """Compute metric."""
        return _cohen_kappa_reduce(self.confmat, self.weights)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)


class CohenKappa(_ClassificationTaskWrapper):
    """Calculate Cohen's kappa score that measures inter-annotator agreement.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["CohenKappa"],
        task: Literal["binary", "multiclass"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        weights: Optional[Literal["linear", "quadratic", "none"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({"ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCohenKappa(threshold, weights=weights, **kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassCohenKappa(num_classes, weights=weights, **kwargs)
        raise ValueError(f"Task {task} not supported!")