"""Calibration error metrics."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.classification.base import _ClassificationTaskWrapper
from tensorplay.metrics.functional.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _binary_calibration_error_tensor_validation,
    _binary_calibration_error_update,
    _binary_confusion_matrix_format,
    _ce_compute,
    _multiclass_calibration_error_arg_validation,
    _multiclass_calibration_error_tensor_validation,
    _multiclass_calibration_error_update,
    _multiclass_confusion_matrix_format,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.enums import ClassificationTaskNoMultilabel
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryCalibrationError(Metric):
    """Compute the top-label calibration error for binary tasks.

    The expected calibration error can be used to quantify how well a given model is calibrated,
    i.e. how well the predicted output probabilities match the actual probabilities of the ground
    truth distribution. Three different norms are implemented, each corresponding to a variation
    of the calibration error metric.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): shape ``(N, ...)``. Preds should be a tensor containing probabilities or logits for
      each observation. If preds has values outside [0, 1] range we consider the input to be logits and will auto
      apply sigmoid per element.
    - ``target`` (int tensor): shape ``(N, ...)``. Target should be a tensor containing ground truth labels, and
      therefore only contain {0, 1} values (except if ``ignore_index`` is specified). The value 1 always encodes
      the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        n_bins: number of bins to use when computing the metric
        norm: norm used to compare empirical and expected probability bins
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(
        self,
        n_bins: int = 15,
        norm: str = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.validate_args = validate_args
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _binary_calibration_error_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(
            preds, target, threshold=0.0, ignore_index=self.ignore_index, convert_to_labels=False
        )
        confidences, accuracies = _binary_calibration_error_update(preds, target)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Compute the calibration error."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

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


class MulticlassCalibrationError(Metric):
    """Compute the top-label calibration error for multiclass tasks.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): shape ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits
      for each observation. If preds has values outside [0, 1] range we consider the input to be logits and will
      auto apply softmax per sample.
    - ``target`` (int tensor): shape ``(N, ...)``. Target should be a tensor containing ground truth labels, and
      therefore only contain values in the [0, n_classes-1] range (except if ``ignore_index`` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: integer specifying the number of classes
        n_bins: number of bins to use when computing the metric
        norm: norm used to compare empirical and expected probability bins
        ignore_index: specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 15,
        norm: str = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_calibration_error_arg_validation(num_classes, n_bins, norm, ignore_index)
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _multiclass_calibration_error_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(
            preds, target, ignore_index=self.ignore_index, convert_to_labels=False
        )
        confidences, accuracies = _multiclass_calibration_error_update(preds, target)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Compute the calibration error."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

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


class CalibrationError(_ClassificationTaskWrapper):
    """Compute the top-label calibration error.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``.

    """

    def __new__(  # type: ignore[misc]
        cls: type["CalibrationError"],
        task: str,
        n_bins: int = 15,
        norm: str = "l1",
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({"n_bins": n_bins, "norm": norm, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCalibrationError(**kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassCalibrationError(num_classes, **kwargs)
        raise ValueError(f"Not handled value: {task}")