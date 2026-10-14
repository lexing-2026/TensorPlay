"""Functional interface for the ROUGE score metric."""

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.imports import _NLTK_AVAILABLE

ALLOWED_ROUGE_KEYS: dict[str, Union[int, str]] = {
    "rouge1": 1,
    "rouge2": 2,
    "rouge3": 3,
    "rouge4": 4,
    "rouge5": 5,
    "rouge6": 6,
    "rouge7": 7,
    "rouge8": 8,
    "rouge9": 9,
    "rougeL": "L",
    "rougeLsum": "Lsum",
}
ALLOWED_ACCUMULATE_VALUES = ("avg", "best")


def _ensure_nltk_punkt_is_downloaded() -> None:
    """Check whether ``nltk`` ``punkt`` is downloaded.

    If not, try to download if a machine is connected to the internet.

    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True, force=False, halt_on_error=False, raise_on_error=True)
        except ValueError as err:
            raise OSError(
                "`nltk` resource `punkt` is not available on a disk and cannot be downloaded as a machine is not "
                "connected to the internet."
            ) from err


def _split_sentence(x: str) -> Sequence[str]:
    """Split sentence to get ``rougeLsum`` scores matching published rougeL scores."""
    if not _NLTK_AVAILABLE:
        raise ModuleNotFoundError("ROUGE-Lsum calculation requires that `nltk` is installed. Use `pip install nltk`.")
    import nltk

    _ensure_nltk_punkt_is_downloaded()

    re.sub("<n>", "", x)  # remove pegasus newline char
    return nltk.sent_tokenize(x)


def _compute_metrics(hits_or_lcs: int, pred_len: int, target_len: int) -> dict[str, Tensor]:
    """Compute overall metrics.

    This function computes precision, recall and F1 score based on hits/lcs, the length of lists of tokenized
    predicted and target sentences.

    Args:
        hits_or_lcs: a number of matches or a length of the longest common subsequence
        pred_len: a length of a tokenized predicted sentence
        target_len: a length of a tokenized target sentence

    """
    precision = hits_or_lcs / pred_len
    recall = hits_or_lcs / target_len
    if precision == recall == 0.0:
        return {"precision": tp.tensor(0.0), "recall": tp.tensor(0.0), "fmeasure": tp.tensor(0.0)}

    fmeasure = 2 * precision * recall / (precision + recall)
    return {"precision": tp.tensor(precision), "recall": tp.tensor(recall), "fmeasure": tp.tensor(fmeasure)}


def _lcs(
    pred_tokens: Sequence[str], target_tokens: Sequence[str], return_full_table: bool = False
) -> Union[int, Sequence[Sequence[int]]]:
    """Run a DP algorithm to compute the length of the longest common subsequence.

    Args:
        pred_tokens: a tokenized predicted sentence
        target_tokens: a tokenized target sentence
        return_full_table: if the full table of the longest common subsequence should be returned or just the largest

    """
    lcs = [[0] * (len(pred_tokens) + 1) for _ in range(len(target_tokens) + 1)]
    for i in range(1, len(target_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if target_tokens[i - 1] == pred_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    if return_full_table:
        return lcs
    return lcs[-1][-1]


def _backtracked_lcs(
    lcs_table: Sequence[Sequence[int]], pred_tokens: Sequence[str], target_tokens: Sequence[str]
) -> Sequence[int]:
    """Backtrack the LCS table.

    Args:
        lcs_table: a table containing information for the calculation of the longest common subsequence
        pred_tokens: a tokenized predicted sentence
        target_tokens: a tokenized target sentence

    """
    i = len(pred_tokens)
    j = len(target_tokens)
    backtracked_lcs: list[int] = []
    while i > 0 and j > 0:
        if pred_tokens[i - 1] == target_tokens[j - 1]:
            backtracked_lcs.insert(0, j - 1)
            i -= 1
            j -= 1
        elif lcs_table[j][i - 1] > lcs_table[j - 1][i]:
            i -= 1
        else:
            j -= 1
    return backtracked_lcs


def _union_lcs(pred_tokens_list: Sequence[Sequence[str]], target_tokens: Sequence[str]) -> Sequence[str]:
    r"""Find the union LCS between a target sentence and an iterable of predicted tokens.

    Args:
        pred_tokens_list: a tokenized predicted sentence split by ``'\n'``
        target_tokens: a tokenized single part of target sentence split by ``'\n'``

    """

    def lcs_ind(pred_tokens: Sequence[str], target_tokens: Sequence[str]) -> Sequence[int]:
        """Return one of the longest of longest common subsequence via backtracked lcs table."""
        lcs_table: Sequence[Sequence[int]] = _lcs(pred_tokens, target_tokens, return_full_table=True)  # type: ignore
        return _backtracked_lcs(lcs_table, pred_tokens, target_tokens)

    def find_union(lcs_tables: Sequence[Sequence[int]]) -> Sequence[int]:
        """Find the union LCS given a list of LCS."""
        return sorted(set().union(*lcs_tables))

    lcs_tables = [lcs_ind(pred_tokens, target_tokens) for pred_tokens in pred_tokens_list]
    return [target_tokens[i] for i in find_union(lcs_tables)]


def _normalize_and_tokenize_text(
    text: str,
    stemmer: Optional[Any] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
) -> Sequence[str]:
    """Normalize and tokenize input text.

    The rouge score should be calculated only over lowercased words and digits. Optionally, a Porter stemmer can
    be used to strip word suffixes to improve matching.

    Args:
        text: an input sentence
        stemmer: Porter stemmer instance to strip word suffixes to improve matching
        normalizer: a user's own normalizer function. If this is ``None``, replacing any non-alpha-numeric
            characters with spaces is default. This function must take a ``str`` and return a ``str``.
        tokenizer: a user's own tokenizer function. If this is ``None``, splitting by spaces is default. This
            function must take a ``str`` and return ``Sequence[str]``

    """
    # If normalizer is none, replace any non-alpha-numeric characters with spaces.
    text = normalizer(text) if callable(normalizer) else re.sub(r"[^a-z0-9]+", " ", text.lower())

    # If tokenizer is none, splitting by spaces
    tokens = tokenizer(text) if callable(tokenizer) else re.split(r"\s+", text)

    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]

    # One final check to drop any empty or invalid tokens.
    return [x for x in tokens if (isinstance(x, str) and len(x) > 0)]


def _rouge_n_score(pred: Sequence[str], target: Sequence[str], n_gram: int) -> dict[str, Tensor]:
    """Compute precision, recall and F1 score for the Rouge-N metric.

    Args:
        pred: a predicted sentence
        target: a target sentence
        n_gram: n-gram overlap

    """

    def _create_ngrams(tokens: Sequence[str], n: int) -> Counter:
        ngrams: Counter = Counter()
        for ngram in (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    pred_ngrams, target_ngrams = _create_ngrams(pred, n_gram), _create_ngrams(target, n_gram)
    pred_len, target_len = sum(pred_ngrams.values()), sum(target_ngrams.values())
    if 0 in (pred_len, target_len):
        return {"precision": tp.tensor(0.0), "recall": tp.tensor(0.0), "fmeasure": tp.tensor(0.0)}

    # It is sufficient to take a set(pred_tokenized) for hits count as we consider intersection of pred & target
    hits = sum(min(pred_ngrams[w], target_ngrams[w]) for w in set(pred_ngrams))
    return _compute_metrics(hits, max(pred_len, 1), max(target_len, 1))


def _rouge_l_score(pred: Sequence[str], target: Sequence[str]) -> dict[str, Tensor]:
    """Compute precision, recall and F1 score for the Rouge-L metric.

    Args:
        pred: a predicted sentence
        target: a target sentence

    """
    pred_len, target_len = len(pred), len(target)
    if 0 in (pred_len, target_len):
        return {"precision": tp.tensor(0.0), "recall": tp.tensor(0.0), "fmeasure": tp.tensor(0.0)}

    lcs: int = _lcs(pred, target)  # type: ignore
    return _compute_metrics(lcs, pred_len, target_len)


def _rouge_lsum_score(pred: Sequence[Sequence[str]], target: Sequence[Sequence[str]]) -> dict[str, Tensor]:
    r"""Compute precision, recall and F1 score for the Rouge-LSum metric.

    Args:
        pred: an iterable of predicted sentence split by ``'\n'``
        target: an iterable target sentence split by ``'\n'``

    """
    pred_len = sum(map(len, pred))
    target_len = sum(map(len, target))
    if 0 in (pred_len, target_len):
        return {"precision": tp.tensor(0.0), "recall": tp.tensor(0.0), "fmeasure": tp.tensor(0.0)}

    # Get token counts
    def _get_token_counts(sentences: Sequence[Sequence[str]]) -> Counter:
        ngrams: Counter = Counter()
        for sentence in sentences:
            ngrams.update(sentence)
        return ngrams

    pred_tokens_count = _get_token_counts(pred)
    target_tokens_count = _get_token_counts(target)

    # Calculate hits
    hits = 0
    for tgt in target:
        lcs = _union_lcs(pred, tgt)
        for token in lcs:
            if pred_tokens_count[token] > 0 and target_tokens_count[token] > 0:
                hits += 1
                pred_tokens_count[token] -= 1
                target_tokens_count[token] -= 1

    return _compute_metrics(hits, pred_len, target_len)


def _rouge_score_update(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    rouge_keys_values: list[Union[int, str]],
    accumulate: str,
    stemmer: Optional[Any] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
) -> dict[Union[int, str], list[dict[str, Tensor]]]:
    """Update the rouge score with the current set of predicted and target sentences.

    Args:
        preds: an iterable of predicted sentences
        target: an iterable of iterable of target sentences
        rouge_keys_values: list of N-grams/'L'/'Lsum' arguments
        accumulate: useful in case of multi-reference rouge score. ``avg`` takes the average of all references with
            respect to predictions, ``best`` takes the best fmeasure score obtained between prediction and multiple
            corresponding references. Allowed values are ``avg`` and ``best``.
        stemmer: Porter stemmer instance to strip word suffixes to improve matching
        normalizer: a user's own normalizer function. If this is ``None``, replacing any non-alpha-numeric characters
            with spaces is default. This function must take a ``str`` and return a ``str``.
        tokenizer: a user's own tokenizer function. If this is ``None``, splitting by spaces is default. This function
            must take a ``str`` and return ``Sequence[str]``

    """
    results: dict[Union[int, str], list[dict[str, Tensor]]] = {rouge_key: [] for rouge_key in rouge_keys_values}

    for pred_raw, target_raw in zip(preds, target):
        result_inner: dict[Union[int, str], dict[str, Tensor]] = {rouge_key: {} for rouge_key in rouge_keys_values}
        result_avg: dict[Union[int, str], list[dict[str, Tensor]]] = {rouge_key: [] for rouge_key in rouge_keys_values}
        list_results = []
        pred = _normalize_and_tokenize_text(pred_raw, stemmer, normalizer, tokenizer)
        if "Lsum" in rouge_keys_values:
            pred_lsum = [
                _normalize_and_tokenize_text(pred_sentence, stemmer, normalizer, tokenizer)
                for pred_sentence in _split_sentence(pred_raw)
            ]

        for target_raw_inner in target_raw:
            tgt = _normalize_and_tokenize_text(target_raw_inner, stemmer, normalizer, tokenizer)

            if "Lsum" in rouge_keys_values:
                target_lsum = [
                    _normalize_and_tokenize_text(tgt_sentence, stemmer, normalizer, tokenizer)
                    for tgt_sentence in _split_sentence(target_raw_inner)
                ]

            for rouge_key in rouge_keys_values:
                if isinstance(rouge_key, int):
                    score = _rouge_n_score(pred, tgt, rouge_key)
                elif rouge_key == "L":
                    score = _rouge_l_score(pred, tgt)
                elif rouge_key == "Lsum":
                    score = _rouge_lsum_score(pred_lsum, target_lsum)
                result_inner[rouge_key] = score
                result_avg[rouge_key].append(score)
            list_results.append(result_inner.copy())

        if accumulate == "best":
            for k in rouge_keys_values:
                index = tp.argmax(tp.stack([s[k]["fmeasure"] for s in list_results]))
                results[k].append(list_results[index][k])

        elif accumulate == "avg":
            new_result_avg: dict[Union[int, str], dict[str, Tensor]] = {
                rouge_key: {} for rouge_key in rouge_keys_values
            }
            for rouge_key, metrics in result_avg.items():
                _dict_metric_score_batch: dict[str, List[Tensor]] = {}
                for metric in metrics:
                    for _type, value in metric.items():
                        if _type not in _dict_metric_score_batch:
                            _dict_metric_score_batch[_type] = []
                        _dict_metric_score_batch[_type].append(value)

                new_result_avg[rouge_key] = {
                    _type: tp.stack(_dict_metric_score_batch[_type]).mean() for _type in _dict_metric_score_batch
                }

            for rouge_key in rouge_keys_values:
                results[rouge_key].append(new_result_avg[rouge_key])

    return results


def _rouge_score_compute(sentence_results: dict[str, List[Tensor]]) -> dict[str, Tensor]:
    """Compute the combined ROUGE metric for all the input set of predicted and target sentences.

    Args:
        sentence_results: Rouge-N/Rouge-L/Rouge-LSum metrics calculated for single sentence

    """
    results: dict[str, Tensor] = {}
    # Obtain mean scores for individual rouge metrics
    if sentence_results == {}:
        return results

    for rouge_key, scores in sentence_results.items():
        results[rouge_key] = tp.stack(scores).mean()

    return results


def rouge_score(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str], Sequence[Sequence[str]]],
    accumulate: str = "best",
    use_stemmer: bool = False,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
    rouge_keys: Union[str, tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
) -> dict[str, Tensor]:
    """Calculate the ROUGE score, used for automatic summarization.

    ROUGE measures the overlap of n-grams, the longest common subsequence, or the longest common
    subsequence over sentences between a predicted summary and reference summaries.

    Args:
        preds: an iterable of predicted sentences or a single predicted sentence
        target: an iterable of iterables of target sentences or an iterable of target sentences or a single target
            sentence
        accumulate: useful in case of multi-reference rouge score:

            - ``avg`` takes the average of all references with respect to predictions
            - ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references

        use_stemmer: use Porter stemmer to strip word suffixes to improve matching
        normalizer: a user's own normalizer function. If this is ``None``, replacing any non-alpha-numeric characters
            with spaces is default. This function must take a ``str`` and return a ``str``.
        tokenizer: a user's own tokenizer function. If this is ``None``, splitting by spaces is default. This function
            must take a ``str`` and return ``Sequence[str]``
        rouge_keys: a list of rouge types to calculate. Keys that are allowed are ``rougeL``, ``rougeLsum``, and
            ``rouge1`` through ``rouge9``.

    Return:
        Python dictionary of rouge scores for each input rouge key

    Raises:
        ModuleNotFoundError: if the python package ``nltk`` is not installed
        ValueError: if any of the ``rouge_keys`` does not belong to the allowed set of keys

    """
    if use_stemmer:
        if not _NLTK_AVAILABLE:
            raise ModuleNotFoundError("Stemmer requires that `nltk` is installed. Use `pip install nltk`.")
        import nltk

    stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None

    if not isinstance(rouge_keys, tuple):
        rouge_keys = (rouge_keys,)
    for key in rouge_keys:
        if key not in ALLOWED_ROUGE_KEYS:
            raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {list(ALLOWED_ROUGE_KEYS.keys())}")
    rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]

    if isinstance(target, list) and all(isinstance(tgt, str) for tgt in target):
        target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]

    if isinstance(preds, str):
        preds = [preds]

    if isinstance(target, str):
        target = [[target]]

    sentence_results: dict[Union[int, str], list[dict[str, Tensor]]] = _rouge_score_update(
        preds,
        target,
        rouge_keys_values,
        stemmer=stemmer,
        normalizer=normalizer,
        tokenizer=tokenizer,
        accumulate=accumulate,
    )

    output: dict[str, List[Tensor]] = {
        f"rouge{rouge_key}_{tp}": [] for rouge_key in rouge_keys_values for tp in ["fmeasure", "precision", "recall"]
    }
    for rouge_key, metrics in sentence_results.items():
        for metric in metrics:
            for score_type, value in metric.items():
                output[f"rouge{rouge_key}_{score_type}"].append(value)

    return _rouge_score_compute(output)