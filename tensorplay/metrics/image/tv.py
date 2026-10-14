"""Total variation metric."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.image.tv import _total_variation_compute, _total_variation_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class TotalVariation(Metric):
    """Compute the total variation loss.

    The total variation measures the sum of absolute differences between neighboring pixel values,
    providing a smoothness prior for images.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``img`` (float tensor): a tensor of shape ``(N, C, H, W)`` consisting of images

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``tv`` (float tensor): if ``reduction!='none'`` returns a float scalar tensor with the total
      variation value over samples else returns tensor of shape ``(N,)`` with TV values per sample

    Args:
        reduction: a method to reduce metric score over samples:

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Raises:
        ValueError: if ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``

    """

    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = False
    plot_lower_bound: float = 0.0
    num_elements: Tensor
    score_list: List[Tensor]
    score: Tensor

    def __init__(self, reduction: Optional[str] = "sum", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if reduction is not None and reduction not in ("sum", "mean", "none"):
            raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")
        self.reduction = reduction
        self.add_state("score_list", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=tp.tensor(0, dtype=tp.float32), dist_reduce_fx="sum")
        self.add_state("num_elements", default=tp.tensor(0, dtype=tp.int32), dist_reduce_fx="sum")

    def update(self, img: Tensor) -> None:
        """Update current score with batch of input images."""
        score, num_elements = _total_variation_update(img)
        if self.reduction is None or self.reduction == "none":
            self.score_list.append(score)
        else:
            self.score += score.sum()
        self.num_elements += num_elements

    def compute(self) -> Tensor:
        """Compute the final total variation."""
        score = dim_zero_cat(self.score_list) if self.reduction is None or self.reduction == "none" else self.score
        return _total_variation_compute(score, self.num_elements, self.reduction)

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