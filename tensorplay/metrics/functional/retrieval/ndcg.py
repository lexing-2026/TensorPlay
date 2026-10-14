from typing import Optional

import tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_retrieval_functional_inputs


def _tie_average_dcg(target: Tensor, preds: Tensor, discount_cumsum: Tensor) -> Tensor:
    """Translated version of sklearns `_tie_average_dcg` function.

    Args:
        target: ground truth about each document relevance.
        preds: estimated probabilities of each document to be relevant.
        discount_cumsum: cumulative sum of the discount.

    Returns:
        The cumulative gain of the tied elements.

    """
    _, inv, counts = tensorplay.unique(-preds, return_inverse=True, return_counts=True)
    ranked = tensorplay.zeros_like(counts, dtype=tensorplay.float32)
    ranked.scatter_add_(0, inv, target.to(dtype=ranked.dtype))
    ranked = ranked / counts
    groups = counts.cumsum(dim=0) - 1
    discount_sums = tensorplay.zeros_like(counts, dtype=tensorplay.float32)
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = discount_cumsum[groups].diff()
    return (ranked * discount_sums).sum()


def _dcg_sample_scores(target: Tensor, preds: Tensor, top_k: int, ignore_ties: bool) -> Tensor:
    """Translated version of sklearns `_dcg_sample_scores` function.

    Args:
        target: ground truth about each document relevance.
        preds: estimated probabilities of each document to be relevant.
        top_k: consider only the top k elements
        ignore_ties: If True, ties are ignored. If False, ties are averaged.

    Returns:
        The cumulative gain

    """
    discount = 1.0 / (tensorplay.log2(tensorplay.arange(target.shape[-1], device=target.device) + 2.0))
    discount[top_k:] = 0.0

    if ignore_ties:
        ranking = preds.argsort(descending=True)
        ranked = target[ranking]
        cumulative_gain = (discount * ranked).sum()
    else:
        discount_cumsum = discount.cumsum(dim=-1)
        cumulative_gain = _tie_average_dcg(target, preds, discount_cumsum)
    return cumulative_gain


def retrieval_normalized_dcg(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Compute `Normalized Discounted Cumulative Gain`_ (for information retrieval).

    ``preds`` and ``target`` should be of the same shape and live on the same device.
    ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document relevance.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        A single-value tensor with the nDCG of the predictions ``preds`` w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` parameter is not `None` or an integer larger than 0

    Example:
        >>> from tensorplay.metrics.functional.retrieval import retrieval_normalized_dcg
        >>> preds = tensorplay.tensor([.1, .2, .3, 4, 70])
        >>> target = tensorplay.tensor([10, 0, 0, 1, 5])
        >>> retrieval_normalized_dcg(preds, target)
        tensor(0.6957)

    """
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    top_k = preds.shape[-1] if top_k is None else top_k

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    gain = _dcg_sample_scores(target, preds, top_k, ignore_ties=False)
    normalized_gain = _dcg_sample_scores(target, target, top_k, ignore_ties=True)

    # filter undefined scores
    all_irrelevant = normalized_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalized_gain[~all_irrelevant]

    return gain.mean()