"""BLEU score metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update, _tokenize_fn
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BLEUScore(Metric):
    """Calculate the BLEU score of machine translated text with one or more references.

    BLEU evaluates the quality of machine-translated text by computing the geometric mean of clipped
    n-gram precisions, adjusted with a brevity penalty.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (sequence of str): an iterable of machine translated corpus
    - ``target`` (sequence of sequences of str): an iterable of iterables of reference corpus

    As output of ``forward`` and ``update`` the metric returns the following output:

    - ``bleu`` (float tensor): a tensor with the BLEU score

    Args:
        n_gram: gram value ranged from 1 to 4
        smooth: whether or not to apply smoothing
        weights: weights used for unigrams, bigrams, etc. to calculate the BLEU score. If not provided,
            uniform weights are used.

    Raises:
        ValueError: if a length of a list of weights is not ``None`` and not equal to ``n_gram``

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds_len: Tensor
    target_len: Tensor
    numerator: Tensor
    denominator: Tensor

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.smooth = smooth
        if weights is not None and len(weights) != n_gram:
            raise ValueError(f"List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}")
        self.weights = weights if weights is not None else [1.0 / n_gram] * n_gram

        self.add_state("preds_len", tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_len", tp.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numerator", tp.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", tp.zeros(self.n_gram), dist_reduce_fx="sum")

    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        self.preds_len, self.target_len = _bleu_score_update(
            preds,
            target,
            self.numerator,
            self.denominator,
            self.preds_len,
            self.target_len,
            self.n_gram,
            _tokenize_fn,
        )

    def compute(self) -> Tensor:
        """Calculate the BLEU score."""
        return _bleu_score_compute(
            self.preds_len, self.target_len, self.numerator, self.denominator, self.n_gram, self.weights, self.smooth
        )

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