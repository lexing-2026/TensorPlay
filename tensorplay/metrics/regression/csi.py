"""Critical success index metric."""

from typing import Any, Optional, Union
from collections.abc import Sequence

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.csi import _critical_success_index_compute, _critical_success_index_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = ["CriticalSuccessIndex"]


class CriticalSuccessIndex(Metric):
    """Compute the critical success index.

    Args:
        threshold: values above or equal to the threshold count as positive.
        keep_sequence_dim: if provided, score each image in a sequence separately.
        kwargs: additional keyword arguments for :class:`Metric`.
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    hits: Tensor
    misses: Tensor
    false_alarms: Tensor

    def __init__(self, threshold: float, keep_sequence_dim: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.keep_sequence_dim = keep_sequence_dim
        self.add_state("hits", default=tp.tensor(0), dist_reduce_fx="sum")
        self.add_state("misses", default=tp.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_alarms", default=tp.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        hits, misses, false_alarms = _critical_success_index_update(
            preds, target, self.threshold, self.keep_sequence_dim
        )
        self.hits += hits
        self.misses += misses
        self.false_alarms += false_alarms

    def compute(self) -> Tensor:
        """Compute the metric."""
        return _critical_success_index_compute(self.hits, self.misses, self.false_alarms)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot metric values."""
        return self._plot(val, ax)
