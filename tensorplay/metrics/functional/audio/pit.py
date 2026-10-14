"""Permutation invariant training (PIT) for speech separation."""

from itertools import permutations
from typing import Any, Callable, Dict, Literal

import tensorplay as tp
from tensorplay import Tensor

__all__ = ["permutation_invariant_training", "pit_permutate"]

# cache of permutations: str(spk_num) -> tensor of shape [perm_num, spk_num]
_ps_dict: Dict[str, Tensor] = {}


def _gen_permutations(spk_num: int) -> Tensor:
    """Return all permutations of ``range(spk_num)`` as a tensor of shape ``[perm_num, spk_num]``."""
    key = str(spk_num)
    if key not in _ps_dict:
        _ps_dict[key] = tp.tensor(list(permutations(range(spk_num))))
    return _ps_dict[key]


def _find_best_perm_by_exhaustive_method(metric_mtx: Tensor, eval_func: Callable) -> tuple[Tensor, Tensor]:
    """Solve the permutation assignment by exhaustively evaluating all permutations."""
    # create/read the cached permutations
    batch_size, spk_num = metric_mtx.shape[:2]
    ps = _gen_permutations(spk_num=spk_num)  # [perm_num, spk_num]

    # find the metric of each permutation
    perm_num = ps.shape[0]
    # shape of [batch_size, spk_num, perm_num]
    bps = ps.T[None, ...].expand(batch_size, spk_num, perm_num)
    # shape of [batch_size, spk_num, perm_num]
    metric_of_ps_details = tp.gather(metric_mtx, 2, bps)
    # shape of [batch_size, perm_num]
    metric_of_ps = metric_of_ps_details.mean(dim=1)

    # find the best metric and best permutation
    best_metric, best_indexes = eval_func(metric_of_ps, dim=1)
    best_indexes = best_indexes.detach()
    best_perm = ps[best_indexes, :]
    return best_metric, best_perm  # shape [batch], shape [batch, spk]


def permutation_invariant_training(
    preds: Tensor,
    target: Tensor,
    metric_func: Callable,
    mode: Literal["speaker-wise", "permutation-wise"] = "speaker-wise",
    eval_func: Literal["max", "min"] = "max",
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Calculate permutation invariant training (PIT).

    This metric evaluates models for multi-talker speech separation in a
    permutation invariant way: the estimate is scored against the target under
    the best matching speaker permutation.

    Args:
        preds: float tensor with shape ``(batch_size, num_speakers, ...)``.
        target: float tensor with shape ``(batch_size, num_speakers, ...)``.
        metric_func: a metric function accepting a batch of targets and
            estimates. For ``mode='speaker-wise'`` it is called as
            ``metric_func(preds[:, i, ...], target[:, j, ...])`` and must return
            a batch of metric tensors of shape ``(batch,)``. For
            ``mode='permutation-wise'`` it is called as
            ``metric_func(preds[:, p, ...], target[:, :, ...])`` for each
            permutation ``p``.
        mode: whether to evaluate speaker-wise or permutation-wise.
        eval_func: ``'max'`` or ``'min'`` depending on whether larger or
            smaller metric values are better.
        kwargs: additional keyword arguments for ``metric_func``.

    Returns:
        tuple of the best metric values of shape ``(batch,)`` and the best
        permutations of shape ``(batch, num_speakers)``.
    """
    if preds.shape[0:2] != target.shape[0:2]:
        raise RuntimeError("Predictions and targets are expected to have the same shape at the batch and speaker dimensions")
    if eval_func not in ["max", "min"]:
        raise ValueError(f'eval_func can only be "max" or "min" but got {eval_func}')
    if mode not in ["speaker-wise", "permutation-wise"]:
        raise ValueError(f'mode can only be "speaker-wise" or "permutation-wise" but got {mode}')
    if target.ndim < 2:
        raise ValueError(f"Inputs must be of shape [batch, spk, ...], got {target.shape} and {preds.shape} instead")

    eval_op = tp.max if eval_func == "max" else tp.min

    # calculate the metric matrix
    batch_size, spk_num = target.shape[0:2]

    if mode == "permutation-wise":
        perms = _gen_permutations(spk_num=spk_num)  # [perm_num, spk_num]
        perm_num = perms.shape[0]
        # shape of ppreds and ptarget: [batch_size*perm_num, spk_num, ...]
        ppreds = tp.index_select(preds, dim=1, index=perms.reshape(-1)).reshape(batch_size * perm_num, *preds.shape[1:])
        ptarget = target.repeat_interleave(repeats=perm_num, dim=0)
        # shape of metric_of_ps [batch_size*perm_num] or [batch_size*perm_num, spk_num]
        metric_of_ps = metric_func(ppreds, ptarget, **kwargs)
        metric_of_ps = tp.mean(metric_of_ps.reshape(batch_size, len(perms), -1), dim=-1)
        # find the best metric and best permutation
        best_metric, best_indexes = eval_op(metric_of_ps, dim=1)
        best_indexes = best_indexes.detach()
        best_perm = perms[best_indexes, :]
        return best_metric, best_perm

    # speaker-wise
    first_ele = metric_func(preds[:, 0, ...], target[:, 0, ...], **kwargs)  # needed for dtype and device
    metric_mtx = tp.empty((batch_size, spk_num, spk_num), dtype=first_ele.dtype, device=first_ele.device)
    metric_mtx[:, 0, 0] = first_ele
    for target_idx in range(spk_num):  # we have spk_num speeches in target in each sample
        for preds_idx in range(spk_num):  # we have spk_num speeches in preds in each sample
            if target_idx == 0 and preds_idx == 0:  # already calculated
                continue
            metric_mtx[:, target_idx, preds_idx] = metric_func(
                preds[:, preds_idx, ...], target[:, target_idx, ...], **kwargs
            )

    # find best
    best_metric, best_perm = _find_best_perm_by_exhaustive_method(metric_mtx, eval_op)
    return best_metric, best_perm


def pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """Permutate estimates according to a permutation.

    Args:
        preds: the estimates to permutate, shape ``[batch, spk, ...]``.
        perm: the permutation returned from
            :func:`permutation_invariant_training`, shape ``[batch, spk]``.

    Returns:
        the permutated version of the estimates.
    """
    return tp.stack([tp.index_select(pred, 0, p) for pred, p in zip(preds, perm)])