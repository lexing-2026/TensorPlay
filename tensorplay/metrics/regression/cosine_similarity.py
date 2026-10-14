"""Cosine similarity metric."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.functional.regression.cosine_similarity import _cosine_similarity_compute, _cosine_similarity_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class CosineSimilarity(Metric):
    """Compute the cosine similarity.

    The cosine similarity between two vectors ``x`` and ``y`` is defined as their dot product
    divided by the product of their norms.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predicted float tensor with shape ``(N, d)``
    - ``target`` (float tensor): ground truth float tensor with shape ``(N, d)``

    As output of ``forward`` and ``compute`` the metric returns a float tensor with the cosine similarity.

    Args:
        reduction: how to reduce over the batch dimension using ``'sum'``, ``'mean'`` or ``'none'``
            (taking the individual scores)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        reduction: str = "sum",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}")
        self.reduction = reduction
        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states with predictions and targets."""
        preds, target = _cosine_similarity_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Compute the cosine similarity."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _cosine_similarity_compute(preds, target, self.reduction)

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