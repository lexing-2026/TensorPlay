"""Kendall rank correlation coefficient metric."""

from typing import List, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.regression.utils import _check_data_shape_to_num_outputs
from tensorplay.metrics.utilities.checks import _check_same_shape
from tensorplay.metrics.utilities.data import _bincount, dim_zero_cat

from typing import Literal

__all__ = ["kendall_rank_corrcoef"]


def _sort_on_first_sequence(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Sort sequences in ascending order according to the sequence ``x``."""
    y = tp.clone(y)
    x, y = x.T, y.T
    x, perm = x.sort()
    for i in range(x.shape[0]):
        y[i] = y[i][perm[i]]
    return x.T, y.T


def _concordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
    """Count the total number of concordant pairs in a single sequence."""
    return tp.logical_and(x[i] < x[(i + 1) :], y[i] < y[(i + 1) :]).sum(0).unsqueeze(0)


def _count_concordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count the total number of concordant pairs in the given sequences."""
    return tp.cat([_concordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


def _discordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
    """Count the total number of discordant pairs in a single sequence."""
    return (
        tp.logical_or(
            tp.logical_and(x[i] > x[(i + 1) :], y[i] < y[(i + 1) :]),
            tp.logical_and(x[i] < x[(i + 1) :], y[i] > y[(i + 1) :]),
        )
        .sum(0)
        .unsqueeze(0)
    )


def _count_discordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count the total number of discordant pairs in the given sequences."""
    return tp.cat([_discordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


def _convert_sequence_to_dense_rank(x: Tensor, sort: bool = False) -> Tensor:
    """Convert a sequence to the dense rank tensor."""
    if sort:
        x, _ = x.sort(dim=0)
    _ones = tp.zeros(1, x.shape[1], dtype=tp.int32, device=x.device)
    return tp.cumsum(tp.cat([_ones, (x[1:] != x[:-1]).int()], dim=0), dim=0)


def _get_ties(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the number of ties and statistics for p-value calculation."""
    ties = tp.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    ties_p1 = tp.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    ties_p2 = tp.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    for dim in range(x.shape[1]):
        n_ties = _bincount(x[:, dim])
        n_ties = n_ties[n_ties > 1]
        ties[dim] = (n_ties * (n_ties - 1) // 2).sum()
        ties_p1[dim] = (n_ties * (n_ties - 1.0) * (n_ties - 2)).sum()
        ties_p2[dim] = (n_ties * (n_ties - 1.0) * (2 * n_ties + 5)).sum()

    return ties, ties_p1, ties_p2


def _get_metric_metadata(
    preds: Tensor, target: Tensor, variant: str
) -> tuple[
    Tensor,
    Tensor,
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Tensor,
]:
    """Obtain statistics needed to calculate the metric value."""
    preds, target = _sort_on_first_sequence(preds, target)

    concordant_pairs = _count_concordant_pairs(preds, target)
    discordant_pairs = _count_discordant_pairs(preds, target)

    n_total = tp.tensor(preds.shape[0], device=preds.device)
    preds_ties = target_ties = None
    preds_ties_p1 = preds_ties_p2 = target_ties_p1 = target_ties_p2 = None
    if variant != "a":
        preds = _convert_sequence_to_dense_rank(preds)
        target = _convert_sequence_to_dense_rank(target, sort=True)
        preds_ties, preds_ties_p1, preds_ties_p2 = _get_ties(preds)
        target_ties, target_ties_p1, target_ties_p2 = _get_ties(target)
    return (
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        preds_ties_p1,
        preds_ties_p2,
        target_ties,
        target_ties_p1,
        target_ties_p2,
        n_total,
    )


def _calculate_tau(
    preds: Tensor,
    target: Tensor,
    concordant_pairs: Tensor,
    discordant_pairs: Tensor,
    con_min_dis_pairs: Tensor,
    n_total: Tensor,
    preds_ties: Optional[Tensor],
    target_ties: Optional[Tensor],
    variant: str,
) -> Tensor:
    """Calculate Kendall's tau from the metric metadata."""
    if variant == "a":
        return con_min_dis_pairs / (concordant_pairs + discordant_pairs)
    if variant == "b":
        total_combinations: Tensor = n_total * (n_total - 1) // 2
        if preds_ties is None:
            preds_ties = tp.tensor(0.0, dtype=total_combinations.dtype, device=total_combinations.device)
        if target_ties is None:
            target_ties = tp.tensor(0.0, dtype=total_combinations.dtype, device=total_combinations.device)
        denominator = (total_combinations - preds_ties) * (total_combinations - target_ties)
        return con_min_dis_pairs / tp.sqrt(denominator)

    preds_unique = tp.tensor([len(p.unique()) for p in preds.T], dtype=preds.dtype, device=preds.device)
    target_unique = tp.tensor([len(t.unique()) for t in target.T], dtype=target.dtype, device=target.device)
    min_classes = tp.minimum(preds_unique, target_unique)
    return 2 * con_min_dis_pairs / ((min_classes - 1) / min_classes * n_total**2)


def _get_p_value_for_t_value_from_dist(t_value: Tensor) -> Tensor:
    """Obtain the p-value for a tensor of t-values."""
    normal_dist = tp.distributions.Normal(tp.tensor([0.0]).to(t_value), tp.tensor([1.0]).to(t_value))

    is_nan = t_value.isnan()
    t_value = t_value.nan_to_num()
    p_value = normal_dist.cdf(t_value)
    return p_value.where(~is_nan, tp.tensor(float("nan"), dtype=p_value.dtype, device=p_value.device))


def _calculate_p_value(
    con_min_dis_pairs: Tensor,
    n_total: Tensor,
    preds_ties: Optional[Tensor],
    preds_ties_p1: Optional[Tensor],
    preds_ties_p2: Optional[Tensor],
    target_ties: Optional[Tensor],
    target_ties_p1: Optional[Tensor],
    target_ties_p2: Optional[Tensor],
    variant: str,
    alternative: Optional[str],
) -> Tensor:
    """Calculate the p-value for Kendall's tau."""
    t_value_denominator_base = n_total * (n_total - 1) * (2 * n_total + 5)
    if variant == "a":
        t_value = 3 * con_min_dis_pairs / tp.sqrt(t_value_denominator_base / 2)
    else:
        m = n_total * (n_total - 1)
        t_value_denominator: Tensor = (
            t_value_denominator_base - (preds_ties_p2 if preds_ties_p2 is not None else 0) - (
                target_ties_p2 if target_ties_p2 is not None else 0
            )
        ) / 18
        t_value_denominator += (
            2 * (preds_ties if preds_ties is not None else 0) * (target_ties if target_ties is not None else 0)
        ) / m
        t_value_denominator += (
            (preds_ties_p1 if preds_ties_p1 is not None else 0)
            * (target_ties_p1 if target_ties_p1 is not None else 0)
            / (9 * m * (n_total - 2))
        )
        t_value = con_min_dis_pairs / tp.sqrt(t_value_denominator)

    if alternative == "two-sided":
        t_value = tp.abs(t_value)
    if alternative in ["two-sided", "greater"]:
        t_value *= -1
    p_value = _get_p_value_for_t_value_from_dist(t_value)
    if alternative == "two-sided":
        p_value *= 2
    return p_value


def _kendall_corrcoef_compute(
    preds: Tensor,
    target: Tensor,
    variant: str,
    alternative: Optional[str] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """Compute Kendall's rank correlation coefficient and optionally its p-value."""
    (
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        preds_ties_p1,
        preds_ties_p2,
        target_ties,
        target_ties_p1,
        target_ties_p2,
        n_total,
    ) = _get_metric_metadata(preds, target, variant)
    con_min_dis_pairs = concordant_pairs - discordant_pairs

    tau = _calculate_tau(
        preds, target, concordant_pairs, discordant_pairs, con_min_dis_pairs, n_total, preds_ties, target_ties, variant
    )
    p_value = (
        _calculate_p_value(
            con_min_dis_pairs,
            n_total,
            preds_ties,
            preds_ties_p1,
            preds_ties_p2,
            target_ties,
            target_ties_p1,
            target_ties_p2,
            variant,
            alternative,
        )
        if alternative
        else None
    )

    if tau.shape[0] == 1:
        tau = tau.squeeze()
        p_value = p_value.squeeze() if p_value is not None else None

    return tau.clamp(-1, 1), p_value


def kendall_rank_corrcoef(
    preds: Tensor,
    target: Tensor,
    variant: Literal["a", "b", "c"] = "b",
    t_test: bool = False,
    alternative: Optional[Literal["two-sided", "less", "greater"]] = "two-sided",
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    r"""Compute the Kendall rank correlation coefficient.

    .. math::
        tau_a = \frac{C - D}{C + D}

    .. math::
        tau_b = \frac{C - D}{\sqrt{(C + D + T_{preds}) * (C + D + T_{target})}}

    .. math::
        tau_c = 2 * \frac{C - D}{n^2 * \frac{m - 1}{m}}

    Args:
        preds: sequence of data of shape ``(N,)`` or ``(N, d)``
        target: sequence of data of shape ``(N,)`` or ``(N, d)``
        variant: which variant of Kendall's tau to use (``'a'``, ``'b'`` or ``'c'``)
        t_test: whether to run a t-test and return the p-value
        alternative: alternative hypothesis for the t-test, one of ``'two-sided'``,
            ``'less'`` or ``'greater'``

    Returns:
        tau statistic (and optionally the p-value)
    """
    if not isinstance(t_test, bool):
        raise ValueError(f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}.")
    if t_test and alternative is None:
        raise ValueError("Argument `alternative` is required if `t_test=True` but got `None`.")

    _check_same_shape(preds, target)
    num_outputs = 1 if preds.ndim == 1 else preds.shape[-1]
    _check_data_shape_to_num_outputs(preds, target, num_outputs)

    if num_outputs == 1:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)

    tau, p_value = _kendall_corrcoef_compute(preds, target, str(variant), alternative if t_test else None)

    if p_value is not None:
        return tau, p_value
    return tau