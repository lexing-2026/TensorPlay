"""Distributed reduction helpers for metrics."""

from typing import Any, List, Optional

import tensorplay as tensorplay
from tensorplay import Tensor
import tensorplay.nn.functional as F  # noqa: N812


def reduce(x: Tensor, reduction: Optional[str]) -> Tensor:
    """Reduce a tensor with a given reduction method.

    Args:
        x: tensor to reduce
        reduction: one of ``'elementwise_mean'``, ``'sum'``, ``'none'`` or ``None``

    Raises:
        ValueError: if ``reduction`` is not one of the allowed values

    """
    if reduction == "elementwise_mean":
        return tensorplay.mean(x)
    if reduction == "none" or reduction is None:
        return x
    if reduction == "sum":
        return tensorplay.sum(x)
    raise ValueError("Reduction parameter unknown.")


def class_reduce(
    num: Tensor,
    denom: Tensor,
    weights: Tensor,
    class_reduction: Optional[str] = "none",
) -> Tensor:
    """Reduce classification metrics of the form ``num / denom * weights``.

    Args:
        num: numerator tensor
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems:

            - ``'micro'``: calculate metrics globally
            - ``'macro'``: calculate metrics for each label, then average them
            - ``'weighted'``: calculate metrics for each label, then compute a
              weighted average using the support
            - ``'none'`` or ``None``: return the metric per class

    """
    valid_reduction = ("micro", "macro", "weighted", "none", None)
    fraction = tensorplay.sum(num) / tensorplay.sum(denom) if class_reduction == "micro" else num / denom

    fraction[fraction != fraction] = 0

    if class_reduction == "micro":
        return fraction
    if class_reduction == "macro":
        return tensorplay.mean(fraction)
    if class_reduction == "weighted":
        return tensorplay.sum(fraction * (weights.float() / tensorplay.sum(weights)))
    if class_reduction == "none" or class_reduction is None:
        return fraction

    raise ValueError(f"Reduction parameter {class_reduction} unknown. Choose between one of these: {valid_reduction}")


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    with tensorplay.no_grad():
        gathered_result = [tensorplay.zeros_like(result) for _ in range(world_size)]
        tensorplay.distributed.all_gather(gathered_result, result, group)
    # to propagate autograd graph from local rank
    gathered_result[tensorplay.distributed.get_rank(group)] = result
    return gathered_result


def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Gather tensors from all processes onto a list that is broadcast to all processes.

    Works on tensors with the same number of dimensions but differing sizes: the
    tensors are padded, gathered and then trimmed to equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        list with size equal to the process group where element ``i`` corresponds to
        the result tensor from process ``i``

    """
    if group is None:
        group = tensorplay.distributed.GroupMember.WORLD

    result = result.contiguous()

    world_size = tensorplay.distributed.get_world_size(group)
    tensorplay.distributed.barrier(group=group)

    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = tensorplay.tensor(result.shape, device=result.device)
    local_sizes = [tensorplay.zeros_like(local_size) for _ in range(world_size)]
    tensorplay.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = tensorplay.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, pad each local tensor to maximum size, gather and then truncate
    with tensorplay.no_grad():
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = F.pad(result, pad_dims)
        gathered_result = [tensorplay.zeros_like(result_padded) for _ in range(world_size)]
        tensorplay.distributed.all_gather(gathered_result, result_padded, group)
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][tuple(slice_param)]
    # to propagate autograd graph from local rank
    gathered_result[tensorplay.distributed.get_rank(group)] = result
    return gathered_result