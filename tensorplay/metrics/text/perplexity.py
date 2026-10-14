from collections.abc import Sequence
from typing import Any, Optional, Union

from tensorplay import Tensor, tensor

from tensorplay.metrics.functional.text.perplexity import _perplexity_compute, _perplexity_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["Perplexity.plot"]


class Perplexity(Metric):
    r"""Perplexity measures how well a language model predicts a text sample.

    It's calculated as the average number of bits per word a model needs to represent the sample.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~tensorplay.Tensor`): Logits or a unnormalized score assigned to each token in a sequence with shape
      [batch_size, seq_len, vocab_size], which is the output of a language model. Scores will be normalized internally
      using softmax.
    - ``target`` (:class:`~tensorplay.Tensor`): Ground truth values with a shape [batch_size, seq_len]

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``perp`` (:class:`~tensorplay.Tensor`): A tensor with the perplexity score

    Args:
        ignore_index: Integer specifying a target class to ignore.
            If given, this class index does not contribute to the returned score.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Examples:
        >>> from tensorplay import rand, randint
        >>> from tensorplay.metrics.text import Perplexity
        >>> preds = rand(2, 8, 5)
        >>> target = randint(5, (2, 8))
        >>> target[0, 6:] = -100
        >>> perp = Perplexity(ignore_index=-100)
        >>> perp(preds, target)
        tensor(5.8540)

    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(f"Argument `ignore_index` expected to either be `None` or an `int` but got {ignore_index}")
        self.ignore_index = ignore_index
        self.add_state("total_log_probs", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        total_log_probs, count = _perplexity_update(preds, target, self.ignore_index)
        self.total_log_probs += total_log_probs
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity."""
        return _perplexity_compute(self.total_log_probs, self.count)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from tensorplay.metrics.text import Perplexity
            >>> metric = Perplexity()
            >>> metric.update(tensorplay.rand(2, 8, 5), tensorplay.randint(5, (2, 8)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from tensorplay.metrics.text import Perplexity
            >>> metric = Perplexity()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(tensorplay.rand(2, 8, 5), tensorplay.randint(5, (2, 8))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)