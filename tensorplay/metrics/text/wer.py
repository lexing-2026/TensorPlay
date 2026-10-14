"""Word error rate metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.text.wer import _wer_compute, _wer_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class WordErrorRate(Metric):
    """Compute the word error rate (WER).

    The metric is a common metric of the performance of an automatic speech recognition system. It
    indicates the percentage of words that were incorrectly predicted. The lower the value, the better
    the performance of the ASR system, with a WER of 0 being a perfect score.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (str or list of str): transcription(s) to score
    - ``target`` (str or list of str): reference(s) for each speech input

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``wer`` (float tensor): a tensor with the word error rate score

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    errors: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("errors", tp.tensor(0, dtype=tp.float32), dist_reduce_fx="sum")
        self.add_state("total", tp.tensor(0, dtype=tp.float32), dist_reduce_fx="sum")

    def update(self, preds: Union[str, list[str]], target: Union[str, list[str]]) -> None:
        """Update state with predictions and targets."""
        errors, total = _wer_update(preds, target)
        self.errors += errors
        self.total += total

    def compute(self) -> Tensor:
        """Calculate the word error rate."""
        return _wer_compute(self.errors, self.total)

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