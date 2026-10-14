"""Functional interface for the Kullback-Leibler divergence metric."""

from typing import Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.compute import _safe_xlogy


def _kld_update(p: Tensor, q: Tensor, log_prob: bool) -> tuple[Tensor, int]:
    """Update and return KL divergence scores for each observation and the total number of observations.

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
        measures = tp.sum(p.exp() * (p - q), axis=-1)
    else:
        p = p / p.sum(axis=-1, keepdim=True)
        q = q / q.sum(axis=-1, keepdim=True)
        measures = _safe_xlogy(p, p / q).sum(axis=-1)

    return measures, total


def _kld_compute(
    measures: Tensor, total: Union[int, Tensor], reduction: str = "mean"
) -> Tensor:
    """Compute the KL divergence based on the type of reduction.

    Args:
        measures: tensor of KL divergence scores for each observation
        total: number of observations
        reduction: determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: averages score across samples
            - ``'sum'``: sums score across samples
            - ``'none'`` or ``None``: returns score per sample

    """
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def kl_divergence(p: Tensor, q: Tensor, log_prob: bool = False, reduction: str = "mean") -> Tensor:
    """Compute the Kullback-Leibler (KL) divergence.

    The KL divergence is a non-symmetrical measure of how one probability distribution ``P``
    diverges from a second distribution ``Q``.

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
    measures, total = _kld_update(p, q, log_prob)
    return _kld_compute(measures, total, reduction)