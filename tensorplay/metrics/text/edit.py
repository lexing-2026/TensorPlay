"""Edit distance metric."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.text.edit import _edit_distance_compute, _edit_distance_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class EditDistance(Metric):
    """Calculate the Levenshtein edit distance between two sequences.

    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to
    transform the predicted text into the reference text. The lower the distance, the more accurate the
    model is considered to be.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (sequence of str): an iterable of hypothesis corpus
    - ``target`` (sequence of str or sequence of sequences of str): an iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``eed`` (float tensor): a tensor with the extended edit distance score. If ``reduction`` is set to
      ``'none'`` or ``None``, this has shape ``(N, )`` where ``N`` is the batch size. Otherwise, this is a scalar.

    Args:
        substitution_cost: the cost of substituting one character for another
        reduction: a method to reduce metric score over samples:

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    edit_scores_list: List[Tensor]
    edit_scores: Tensor
    num_elements: Tensor

    def __init__(
        self, substitution_cost: int = 1, reduction: Optional[str] = "mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not (isinstance(substitution_cost, int) and substitution_cost >= 0):
            raise ValueError(
                f"Expected argument `substitution_cost` to be a positive integer, but got {substitution_cost}"
            )
        self.substitution_cost = substitution_cost
        allowed_reduction = (None, "mean", "sum", "none")
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction}, but got {reduction}")
        self.reduction = reduction
        if self.reduction == "none" or self.reduction is None:
            self.add_state("edit_scores_list", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("edit_scores", default=tp.tensor(0), dist_reduce_fx="sum")
            self.add_state("num_elements", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        distance = _edit_distance_update(preds, target, self.substitution_cost)
        if self.reduction == "none" or self.reduction is None:
            self.edit_scores_list.append(distance)
        else:
            self.edit_scores += distance.sum()
            self.num_elements += distance.shape[0]

    def compute(self) -> Tensor:
        """Compute the edit distance over state."""
        if self.reduction == "none" or self.reduction is None:
            return _edit_distance_compute(dim_zero_cat(self.edit_scores_list), 1, self.reduction)
        return _edit_distance_compute(self.edit_scores, self.num_elements, self.reduction)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        """
        return self._plot(val, ax)