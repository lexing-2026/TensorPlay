"""ROUGE score metric."""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.functional.text.rouge import (
    ALLOWED_ACCUMULATE_VALUES,
    ALLOWED_ROUGE_KEYS,
    _rouge_score_compute,
    _rouge_score_update,
)
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.imports import _NLTK_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class ROUGEScore(Metric):
    """Calculate the ROUGE score, used for automatic summarization.

    ROUGE measures the overlap of n-grams, the longest common subsequence, or the longest common
    subsequence over sentences between a predicted summary and reference summaries.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (sequence of str): an iterable of predicted sentences or a single predicted sentence
    - ``target`` (sequence of str or sequence of sequences of str): an iterable of target sentences or an
      iterable of interables of target sentences or a single target sentence

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``rouge`` (dict): a dictionary of tensor rouge scores for each input str rouge key

    Args:
        use_stemmer: use Porter stemmer to strip word suffixes to improve matching
        normalizer: a user's own normalizer function. If this is ``None``, replacing any non-alpha-numeric
            characters with spaces is default. This function must take a ``str`` and return a ``str``.
        tokenizer: a user's own tokenizer function. If this is ``None``, splitting by spaces is default. This
            function must take a ``str`` and return ``Sequence[str]``
        accumulate: useful in case of multi-reference rouge score:

            - ``avg`` takes the average of all references with respect to predictions
            - ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references

        rouge_keys: a list of rouge types to calculate. Keys that are allowed are ``rougeL``, ``rougeLsum``, and
            ``rouge1`` through ``rouge9``

    Raises:
        ValueError: if the python package ``nltk`` is not installed
        ValueError: if any of the ``rouge_keys`` does not belong to the allowed set of keys

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        use_stemmer: bool = False,
        normalizer: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
        accumulate: str = "best",
        rouge_keys: Union[str, tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if use_stemmer or "rougeLsum" in rouge_keys:
            if not _NLTK_AVAILABLE:
                raise ModuleNotFoundError(
                    "Stemmer and/or `rougeLsum` requires that `nltk` is installed. Use `pip install nltk`."
                )
            import nltk
        if not isinstance(rouge_keys, tuple):
            rouge_keys = (rouge_keys,)
        for key in rouge_keys:
            if key not in ALLOWED_ROUGE_KEYS:
                raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {ALLOWED_ROUGE_KEYS}")
        if accumulate not in ALLOWED_ACCUMULATE_VALUES:
            raise ValueError(
                f"Got unknown accumulate value {accumulate}. Expected to be one of {ALLOWED_ACCUMULATE_VALUES}"
            )
        self.rouge_keys = rouge_keys
        self.rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]
        self.stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.accumulate = accumulate
        # Adding stated dynamically to prevent IndexError during sync function as some lists can be empty.
        for rouge_key in self.rouge_keys:
            for score in ["fmeasure", "precision", "recall"]:
                self.add_state(f"{rouge_key}_{score}", [], dist_reduce_fx=None)

    def update(
        self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str], Sequence[Sequence[str]]]
    ) -> None:
        """Update state with predictions and targets."""
        if isinstance(target, list) and all(isinstance(tgt, str) for tgt in target):
            target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [[target]]
        output: dict[Union[int, str], list[dict[str, Tensor]]] = _rouge_score_update(
            preds,
            target,
            self.rouge_keys_values,
            stemmer=self.stemmer,
            normalizer=self.normalizer,
            tokenizer=self.tokenizer,
            accumulate=self.accumulate,
        )
        for rouge_key, metrics in output.items():
            for metric in metrics:
                for score_type, value in metric.items():
                    getattr(self, f"rouge{rouge_key}_{score_type}").append(value.to(self.device))

    def compute(self) -> dict[str, Tensor]:
        """Calculate (aggregate) the ROUGE score."""
        update_output = {}
        for rouge_key in self.rouge_keys_values:
            for tp in ["fmeasure", "precision", "recall"]:
                update_output[f"rouge{rouge_key}_{tp}"] = getattr(self, f"rouge{rouge_key}_{tp}")
        return _rouge_score_compute(update_output)

    def __hash__(self) -> int:
        """Return a unique hash for the specific instance of this metric."""
        # override to hash list objects
        hash_vals = [self.__class__.__name__]
        for key in self._defaults:
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)
        return hash(tuple(hash_vals))

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