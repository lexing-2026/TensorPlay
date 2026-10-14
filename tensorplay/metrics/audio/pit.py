"""Permutation invariant training (PIT) metric."""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.audio.pit import permutation_invariant_training
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from typing import Literal

__all__ = ["PermutationInvariantTraining"]


class PermutationInvariantTraining(Metric):
    """Calculate permutation invariant training (PIT).

    This metric evaluates models for speaker independent multi-talker speech
    separation in a permutation invariant way.

    Args:
        metric_func: a metric function accepting a batch of targets and
            estimates. For ``mode='speaker-wise'`` it is called as
            ``metric_func(preds[:, i, ...], target[:, j, ...])`` and must return
            a batch of metric tensors of shape ``(batch,)``. For
            ``mode='permutation-wise'`` it is called as
            ``metric_func(preds[:, p, ...], target[:, :, ...])`` for each
            permutation ``p``.
        mode: whether to evaluate speaker-wise or permutation-wise.
        eval_func: ``'max'`` or ``'min'`` depending on whether larger or
            smaller metric values are better.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    full_state_update: bool = False
    is_differentiable: bool = True
    sum_pit_metric: Tensor
    total: Tensor
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(
        self,
        metric_func: Callable,
        mode: Literal["speaker-wise", "permutation-wise"] = "speaker-wise",
        eval_func: Literal["max", "min"] = "max",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric_func = metric_func
        self.mode = mode
        self.eval_func = eval_func
        self.kwargs = kwargs

        self.add_state("sum_pit_metric", default=tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        pit_metric = permutation_invariant_training(
            preds, target, self.metric_func, self.mode, self.eval_func, **self.kwargs
        )[0]

        self.sum_pit_metric += pit_metric.sum()
        self.total += pit_metric.numel()

    def compute(self) -> Tensor:
        """Compute the metric."""
        return self.sum_pit_metric / self.total

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or
                ``metric.compute`` or a list of these results. If no value is
                provided, ``metric.compute`` is called and that result plotted.
            ax: a matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object
        """
        return self._plot(val, ax)