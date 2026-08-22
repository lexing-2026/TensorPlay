# mypy: allow-untyped-defs
# Ported from torch/distributed/algorithms/ddp_comm_hooks/ddp_zero_hook.py.
import weakref
from collections.abc import Callable
from typing import Any

import tensorplay as tp

import tensorplay.distributed as dist
from tensorplay.distributed.optim import ZeroRedundancyOptimizer
from tensorplay.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from tensorplay.nn.parallel.distributed import DistributedDataParallel


__all__ = ["hook_with_zero_step", "hook_with_zero_step_interleaved"]

# Functional optimizers require passing a list of gradients to their `step()`
# method, and ZeRO requires a functional optimizer to overlap with DDP
# Passing a `None` instead of an actual gradient indicates to the optimizer
# to not update the corresponding parameter
_NO_PARAM_UPDATE: None = None


def _perform_local_step(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
):
    r"""
    Perform a local optimizer step using the gradients provided by ``bucket``.

    Arguments:
        bucket (dist.GradBucket): the bucket providing the gradients.
        zero (ZeroRedundancyOptimizer): the ZeRO instance to step.
        rank (int): the calling process's rank.

    .. warning::
        This function assumes that appropriate synchronization has taken place
        so that the bucket's gradients can be used.
    """
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    if len(zero.optim.param_groups) != 1:
        raise AssertionError(
            "Overlapping DDP with ZeRO only supports a single parameter group"
        )

    # Construct the `gradients` input for the local optimizer step, which
    # expects `None` in a list position to indicate that the corresponding
    # parameter should not be updated
    num_local_optim_params = len(zero.optim.param_groups[0]["params"])
    gradients: list[tp.Tensor | None] = [
        _NO_PARAM_UPDATE for _ in range(num_local_optim_params)
    ]
    if bucket_index not in overlap_info.offsets:
        raise AssertionError(
            f"Bucket index {bucket_index} was not assigned to rank {rank}"
        )
    gradients_offset = overlap_info.offsets[bucket_index]
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    bucket_gradients = bucket.gradients()[bucket_offset : bucket_offset + length]
    for i, grad in enumerate(bucket_gradients):
        gradients[gradients_offset + i] = grad

    zero._local_step(gradients)


def _broadcast_bucket(
    bucket_index: int,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Broadcasts a bucket's parameters.

    Arguments:
        bucket_index (int): the index of the bucket corresponding to the
            parameters to broadcast.
        zero (ZeroRedundancyOptimizer): the calling process's ZeRO instance.
    """
    overlap_info = zero._overlap_info
    if len(overlap_info.assigned_ranks_per_bucket) <= bucket_index:
        raise AssertionError("`assigned_ranks_per_bucket` is not fully constructed")
    # Sort to ensure the same ordering across ranks
    assigned_ranks = sorted(overlap_info.assigned_ranks_per_bucket[bucket_index])
    if len(assigned_ranks) <= 0:
        raise AssertionError(
            f"Bucket {bucket_index} should be assigned to at least one rank"
        )
    for assigned_rank in assigned_ranks:
        bucket_assignments = zero._bucket_assignments_per_rank[assigned_rank]
        if bucket_index in bucket_assignments:
            send_tensor = bucket_assignments[bucket_index].tensor
            if send_tensor is None:
                raise AssertionError
            overlap_info.broadcast_handles.append(
                dist.broadcast(
                    send_tensor,
                    src=dist.get_global_rank(zero.process_group, assigned_rank),
                    group=zero.process_group,
                    async_op=True,
                )
            )


def _save_ddp_bucket_info(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Save DDP gradient bucket information for the ZeRO instance ``zero``.
    """
    overlap_info = zero._overlap_info
    bucket_params = bucket.parameters()
    if len(bucket_params) <= 0:
        raise AssertionError("Empty bucket")

    # Save the parameters in the bucket
    overlap_info.params_per_bucket.append(bucket_params)
    if overlap_info.shard_buckets:
        # Additionally save the bucket size for the assignment heuristic to use
        bucket_size = 0
        for param in bucket_params:
            bucket_size += param.numel()
        if overlap_info.total_size is None:
            raise AssertionError
        overlap_info.total_size += bucket_size


def _hook_with_zero_step_setup(
    ddp_ref: weakref.ReferenceType,
    zero: ZeroRedundancyOptimizer,
    bucket: dist.GradBucket,
):
    r"""
    Encapsulate the setup logic shared by both overlapping hooks.
    """
    # Proceed as normal until the DDP buckets have been rebuilt; tp's DDP
    # does not rebuild buckets, so this is always satisfied after init.
    if not getattr(ddp_ref(), "_has_rebuilt_buckets", False) and \
            not ddp_ref()._lazy_init_ran:
        if zero._overlap_info.status != _OverlapStatus.UNINITIALIZED:
            raise AssertionError
        return

    bucket_index = bucket.index()
    overlap_info = zero._overlap_info
    if overlap_info.status == _OverlapStatus.UNINITIALIZED:
        overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

    if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
        if bucket_index == 0 and len(overlap_info.params_per_bucket) > 0:
            # This corresponds to the first bucket of the backward pass
            # immediately after all information has been saved, so we
            # can perform the delayed ZeRO initialization
            zero._init_zero_for_overlap()
        else:
            # Once DDP buckets have been rebuilt but ZeRO has not been
            # properly initialized yet, save the information needed
            _save_ddp_bucket_info(bucket, zero)


def hook_with_zero_step(
    hook: Callable[[Any, dist.GradBucket], Any],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], Any]:
    r"""
    Modify ``hook`` to overlap ZeRO's optimizer step with the DDP backward pass.

    The optimizer computation follows the backward computation, overlapping
    with outstanding backward communication. May be preferred over
    :func:`hook_with_zero_step_interleaved` when communication is relatively
    slow compared to computation.

    Arguments:
        hook: the hook to modify.
        ddp: the DDP instance to use.
        zero: the ZeRO instance to use.
        shard_buckets (bool): if ``True``, each DDP bucket assignment is
            partitioned across possibly multiple ranks.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.

    .. warning::
        The first two or three training iterations do not perform parameter
        updates while DDP bucketing information is being collected.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    ddp_ref = weakref.ref(ddp)

    # NOTE: Gloo may hang with this overlapping approach
    pg = dist.get_backend(ddp_ref().process_group)
    if pg == dist.Backend.GLOO:
        raise RuntimeError(
            "Gloo backend using Overlapping DDP with ZeRO may meet hangs"
        )

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_fn(
        state: Any,
        bucket: dist.GradBucket,
    ) -> Any:
        r"""
        Return a Future that runs the optimizer step on the last gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank

        if len(overlap_info.assigned_ranks_per_bucket) <= bucket_index:
            raise AssertionError("`assigned_ranks_per_bucket` is not fully constructed")
        assigned_to_bucket = (
            rank in overlap_info.assigned_ranks_per_bucket[bucket_index]
        )

        # Save the bucket reference and all-reduce future for the final bucket
        if assigned_to_bucket:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut

        # Check that buckets are indexed incrementally starting from 0 in the
        # order of their autograd hooks firing
        if len(overlap_info.bucket_indices_seen) > 0:
            if overlap_info.bucket_indices_seen[-1] != bucket_index - 1:
                raise AssertionError("Bucket indices are not in incremental order")
        else:
            if bucket_index != 0:
                raise AssertionError("Bucket indices do not start from 0")
        overlap_info.bucket_indices_seen.append(bucket_index)

        # Directly return the future without any optimizer computation if this
        # is not the last bucket
        num_buckets = len(overlap_info.params_per_bucket)
        is_last_bucket = bucket_index == num_buckets - 1
        if not is_last_bucket:
            return fut

        # Perform partial optimizer step on all buckets after the final
        # bucket has been computed
        for bucket_index in range(num_buckets):
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            if rank in assigned_ranks:
                # Wait on the bucket's all-reduce future to ensure correct
                # gradients
                if bucket_index not in overlap_info.bucket_index_to_future:
                    raise AssertionError(
                        f"All-reduce future for bucket {bucket_index} not saved "
                        f"on rank {rank}"
                    )
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()

                # Perform the partial optimizer step
                curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
                _perform_local_step(curr_bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero)

        # Ensure that all parameter updates are finished before the
        # next forward pass
        overlap_info.wait_for_broadcasts()
        overlap_info.clear_per_iter_info()

        return fut

    return hook_with_zero_fn


def hook_with_zero_step_interleaved(
    hook: Callable[[Any, dist.GradBucket], Any],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], Any]:
    r"""
    Modify ``hook`` to overlap ZeRO's optimizer step with the DDP backward pass.

    Once a bucket's gradients have been computed, the optimizer computation
    using those gradients launches, yielding an interleaving of all-reduces
    and broadcasts in the communication stream. Preferred over
    :func:`hook_with_zero_step` when communication is relatively fast.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    ddp_ref = weakref.ref(ddp)

    pg = dist.get_backend(ddp_ref().process_group)
    if pg == dist.Backend.GLOO:
        raise RuntimeError(
            "Gloo backend using Overlapping DDP with ZeRO may meet hangs"
        )

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_interleaved_fn(
        state,
        bucket: dist.GradBucket,
    ) -> Any:
        r"""
        Return a Future giving the gradient bucket tensor and performing a partial ZeRO step.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        def zero_step(_fut) -> tp.Tensor:
            r"""Perform partial ZeRO :meth:`step` using this bucket's gradients."""
            overlap_info = zero._overlap_info
            bucket_index = bucket.index()
            rank = zero.global_rank

            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            overlap_info.bucket_indices_seen.append(bucket_index)
            if rank in assigned_ranks:
                _perform_local_step(bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero)

            num_buckets = len(overlap_info.params_per_bucket)
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                # Ensure that all parameter updates are finished before the
                # next forward pass
                overlap_info.wait_for_broadcasts()
                overlap_info.clear_per_iter_info()

            return bucket.buffer()

        return fut.then(zero_step)

    return hook_with_zero_interleaved_fn
