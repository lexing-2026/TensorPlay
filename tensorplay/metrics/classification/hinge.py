"""Hinge loss metrics."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.functional.classification.hinge import (
    _binary_hinge_loss_arg_validation,
    _binary_hinge_loss_tensor_validation,
    _binary_hinge_loss_update,
    _binary_confusion_matrix_format,
    _hinge_loss_compute,
    _multiclass_hinge_loss_arg_validation,
    _multiclass_hinge_loss_tensor_validation,
    _multiclass_hinge_loss_update,
    _multiclass_confusion_matrix_format,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.enums import ClassificationTaskNoMultilabel
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryHingeLoss(Metric):
    """Compute the mean hinge loss for binary tasks.

    The hinge loss is defined as ``max(0, 1 - y * y_hat)`` where ``y`` is the target in ``{-1, 1}``
    and ``y_hat`` is the prediction.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): shape ``(N, ...)``. Preds should be a tensor containing probabilities or logits for
      each observation. If preds has values outside [0, 1] range we consider the input to be logits and will auto
      apply sigmoid per element.
    - ``target`` (int tensor): shape ``(N, ...)``. Target should be a tensor containing ground truth labels, and
      therefore only contain {0, 1} values (except if ``ignore_index`` is specified). The value 1 always encodes
      the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        squared: if ``True``, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    measures: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = False,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_hinge_loss_arg_validation(squared, ignore_index)
        self.validate_args = validate_args
        self.squared = squared
        self.ignore_index = ignore_index
        self.add_state("measures", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state."""
        if self.validate_args:
            _binary_hinge_loss_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(
            preds, target, threshold=0.0, ignore_index=self.ignore_index, convert_to_labels=False
        )
        measures, total = _binary_hinge_loss_update(preds, target, self.squared)
        self.measures += measures
        self.total += total

    def compute(self) -> Tensor:
        """Compute the hinge loss."""
        return _hinge_loss_compute(self.measures, self.total)

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


class MulticlassHingeLoss(Metric):
    """Compute the mean hinge loss for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used, or
    the one-vs-all approach where each class is valued against all other classes in a binary fashion.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): shape ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits
      for each observation. If preds has values outside [0, 1] range we consider the input to be logits and will
      auto apply softmax per sample.
    - ``target`` (int tensor): shape ``(N, ...)``. Target should be a tensor containing ground truth labels, and
      therefore only contain values in the [0, n_classes-1] range (except if ``ignore_index`` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: integer specifying the number of classes
        squared: if ``True``, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode: determines how to compute the metric
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"
    measures: Tensor
    total: Tensor

    def __init__(
        self,
        num_classes: int,
        squared: bool = False,
        multiclass_mode: str = "crammer-singer",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.squared = squared
        self.multiclass_mode = multiclass_mode
        self.ignore_index = ignore_index
        self.add_state(
            "measures",
            default=tp.tensor(0.0)
            if self.multiclass_mode == "crammer-singer"
            else tp.zeros(
                num_classes,
            ),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state."""
        if self.validate_args:
            _multiclass_hinge_loss_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(preds, target, self.ignore_index, convert_to_labels=False)
        measures, total = _multiclass_hinge_loss_update(preds, target, self.squared, self.multiclass_mode)
        self.measures += measures
        self.total += total

    def compute(self) -> Tensor:
        """Compute the hinge loss."""
        return _hinge_loss_compute(self.measures, self.total)

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


class HingeLoss(_ClassificationTaskWrapper):
    """Compute the mean hinge loss.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["HingeLoss"],
        task: str,
        num_classes: Optional[int] = None,
        squared: bool = False,
        multiclass_mode: Optional[str] = "crammer-singer",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({"ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryHingeLoss(squared, **kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            if multiclass_mode not in ("crammer-singer", "one-vs-all"):
                raise ValueError(
                    f"`multiclass_mode` is expected to be one of 'crammer-singer' or 'one-vs-all' but "
                    f"`{multiclass_mode}` was passed."
                )
            return MulticlassHingeLoss(num_classes, squared, multiclass_mode, **kwargs)
        raise ValueError(f"Unsupported task `{task}`")