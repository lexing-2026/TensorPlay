"""Functional interface for the Jensen-Shannon divergence metric."""

from typing import Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.kl_divergence import kl_divergence
from tensorplay.metrics.utilities.checks import _check_same_shape


def _jsd_update(p: Tensor, q: Tensor, log_prob: bool) -> tuple[Tensor, int]:
    """Update and return Jensen-Shannon divergence scores for each observation and the total number of observations.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributions sum to 1

    """
    _check_same_shape(p, q)
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError(f"Expected both p and q distribution to be 2D but got {p.ndim} and {q.ndim} respectively")

    total = p.shape[0]
    if log_prob:
        mean = tp.logsumexp(tp.stack([p, q]), dim=0) - tp.log(tp.tensor(2.0))
        measures = 0.5 * kl_divergence(p, mean, log_prob=log_prob, reduction=None) + 0.5 * kl_divergence(
            q, mean, log_prob=log_prob, reduction=None
        )
    else:
        p = p / p.sum(axis=-1, keepdim=True)
        q = q / q.sum(axis=-1, keepdim=True)
        mean = (p + q) / 2
        measures = 0.5 * kl_divergence(p, mean, log_prob=log_prob, reduction=None) + 0.5 * kl_divergence(
            q, mean, log_prob=log_prob, reduction=None
        )
    return measures, total


def _jsd_compute(
    measures: Tensor, total: Union[int, Tensor], reduction: str = "mean"
) -> Tensor:
    """Compute and reduce the Jensen-Shannon divergence based on the type of reduction."""
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def jensen_shannon_divergence(
    p: Tensor, q: Tensor, log_prob: bool = False, reduction: str = "mean"
) -> Tensor:
    """Compute the Jensen-Shannon divergence.

    The Jensen-Shannon divergence is a symmetrical measure of similarity between two probability
    distributions, defined as the average of the KL divergence of each distribution to their mean.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributions sum to 1
        reduction: determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: averages score across samples
            - ``'sum'``: sums score across samples
            - ``'none'`` or ``None``: returns score per sample

    """
    measures, total = _jsd_update(p, q, log_prob)
    return _jsd_compute(measures, total, reduction)