import tensorplay
from tensorplay import Tensor
from typing import Literal


def _fleiss_kappa_update(ratings: Tensor, mode: Literal["counts", "probs"] = "counts") -> Tensor:
    """Updates the counts for fleiss kappa metric.

    Args:
        ratings: ratings matrix
        mode: whether ratings are provided as counts or probabilities

    """
    if mode == "probs":
        if ratings.ndim != 3 or not ratings.is_floating_point():
            raise ValueError(
                "If argument ``mode`` is 'probs', ratings must have 3 dimensions with the format"
                " [n_samples, n_categories, n_raters] and be floating point."
            )
        ratings = ratings.argmax(dim=1)
        one_hot = tensorplay.nn.functional.one_hot(ratings, num_classes=ratings.shape[1]).permute(0, 2, 1)
        ratings = one_hot.sum(dim=-1)
    elif mode == "counts" and (ratings.ndim != 2 or ratings.is_floating_point()):
        raise ValueError(
            "If argument ``mode`` is `counts`, ratings must have 2 dimensions with the format"
            " [n_samples, n_categories] and be none floating point."
        )
    return ratings


def _fleiss_kappa_compute(counts: Tensor) -> Tensor:
    """Computes fleiss kappa from counts matrix.

    Args:
        counts: counts matrix of shape [n_samples, n_categories]

    """
    total = counts.shape[0]
    num_raters = counts.sum(1).max()

    p_i = counts.sum(dim=0) / (total * num_raters)
    p_j = ((counts**2).sum(dim=1) - num_raters) / (num_raters * (num_raters - 1))
    p_bar = p_j.mean()
    pe_bar = (p_i**2).sum()
    return (p_bar - pe_bar) / (1 - pe_bar + 1e-5)


def fleiss_kappa(ratings: Tensor, mode: Literal["counts", "probs"] = "counts") -> Tensor:
    r"""Calculatees `Fleiss kappa`_ a statistical measure for inter agreement between raters.

    .. math::
        \kappa = \frac{\bar{p} - \bar{p_e}}{1 - \bar{p_e}}

    where :math:`\bar{p}` is the mean of the agreement probability over all raters and :math:`\bar{p_e}` is the mean
    agreement probability over all raters if they were randomly assigned. If the raters are in complete agreement then
    the score 1 is returned, if there is no agreement among the raters (other than what would be expected by chance)
    then a score smaller than 0 is returned.

    Args:
        ratings: Ratings of shape [n_samples, n_categories] or [n_samples, n_categories, n_raters] depedenent on `mode`.
            If `mode` is `counts`, `ratings` must be integer and contain the number of raters that chose each category.
            If `mode` is `probs`, `ratings` must be floating point and contain the probability/logits that each rater
            chose each category.
        mode: Whether `ratings` will be provided as counts or probabilities.

    Example:
        >>> # Ratings are provided as counts
        >>> from tensorplay import randint
        >>> from tensorplay.metrics.functional.nominal import fleiss_kappa
        >>> ratings = randint(0, 10, size=(100, 5)).long()  # 100 samples, 5 categories, 10 raters
        >>> fleiss_kappa(ratings)
        tensor(0.0089)

    Example:
        >>> # Ratings are provided as probabilities
        >>> from tensorplay import randn
        >>> from tensorplay.metrics.functional.nominal import fleiss_kappa
        >>> ratings = randn(100, 5, 10).softmax(dim=1)  # 100 samples, 5 categories, 10 raters
        >>> fleiss_kappa(ratings, mode='probs')
        tensor(-0.0075)

    """
    if mode not in ["counts", "probs"]:
        raise ValueError("Argument ``mode`` must be one of ['counts', 'probs'].")
    counts = _fleiss_kappa_update(ratings, mode)
    return _fleiss_kappa_compute(counts)