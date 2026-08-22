# Ported from torch/distributed/optim/zero_redundancy_optimizer.py.
#
# Adaptations for tp: object broadcast uses pickle (tp.Tensor supports
# pickling), ``dist.group.WORLD`` maps to ``dist.GroupMember.WORLD``, and
# typename checks compare dtype/is_sparse directly.

import collections.abc
import copy
import enum
import inspect
import io
import logging
import pickle as _pickle
from itertools import chain
from typing import Any

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.futures
from tensorplay import optim
from tensorplay.optim.optimizer import Optimizer

from tensorplay.distributed.algorithms.join import Join, JoinHook, Joinable
from tensorplay.distributed.optim.utils import functional_optim_map


logger = logging.getLogger(__name__)


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device,
) -> Any:
    r"""
    Recursively searches lists, tuples, dicts and copies tensors to device if possible.

    Non-tensor values are passed as-is in the result.

    .. note::
        These are all copies, so if there are two objects that reference
        the same object, then after this call, there will be two different objects
        referenced on the device.
    """
    if isinstance(value, tp.Tensor):
        return value.to(device)

    if isinstance(value, (list, tuple)):
        values = [
            _recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            for val in value
        ]
        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )
            for key, val in value.items()
        }

    return value


def _is_trainable(param: tp.Tensor) -> bool:
    r"""Return if a parameter is trainable, where trainability is equivalent to requiring a gradient."""
    return param.requires_grad


def _broadcast_object(
    obj: Any,
    src_rank: int,
    group=None,
    device=None,
) -> Any:
    r"""
    Broadcasts an object to the given group.

    It will be sending the object if called from the source rank and receiving
    the object otherwise.

    Arguments:
        obj: object to broadcast; only used if called on the source rank.
        src_rank (int): source rank.
        group (``ProcessGroup``, optional): group used for the broadcast
            (default: ``dist.GroupMember.WORLD``).
        device: device to send from or receive to (default: CPU).

    Returns:
        The broadcasted object.
    """
    group = group if group is not None else dist.GroupMember.WORLD
    device = device or "cpu"
    if dist.get_rank() == src_rank:
        # Send the object
        buffer = io.BytesIO()
        _pickle.dump(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = tp.tensor([len(data)], dtype=tp.int64, device=device)
        data_send_tensor = tp.as_tensor(
            bytearray(data), dtype=tp.uint8).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group,
                       async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group,
                       async_op=False)
    else:
        # Receive the object
        length_tensor = tp.tensor([0], dtype=tp.int64, device=device)
        dist.broadcast(length_tensor, src=src_rank, group=group,
                       async_op=False)
        data_recv_tensor = tp.empty(
            int(length_tensor.item()), dtype=tp.uint8, device=device
        )
        dist.broadcast(data_recv_tensor, src=src_rank, group=group,
                       async_op=False)
        buffer = io.BytesIO(bytes(bytearray(
            data_recv_tensor.cpu().numpy().tobytes())))
        obj = _pickle.load(buffer)
    return obj


class _ZeROJoinHook(JoinHook):
    def __init__(self, zero):
        if not isinstance(zero, ZeroRedundancyOptimizer):
            raise AssertionError(
                "ZeRO join hook requires passing in a "
                "ZeroRedundancyOptimizer instance as the state"
            )
        self.zero = zero
        super().__init__()

    def main_hook(self):
        """
        Perform an optimizer step.

        This step updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """
        self.zero.step()


class _DDPBucketAssignment:
    r"""
    Represent a :class:`DistributedDataParallel` bucket assignment.

    This means that a (possibly non-strict) subset of the parameters corresponding to
    a DDP bucket assigned to a rank to update.

    Attributes:
        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        parameters (List[Tensor]): model parameters in the bucket
            assigned to this rank.
        offset (int): offset into the :class:`GradBucket` 's :meth:`parameters`
            giving the index of the first element in the passed-in
            ``parameters``; this equivalently indexes into the
            :class:`GradBucket` 's :meth:`gradients`.
        device: device on which the parameters are stored.
        tensor (Tensor): flattened tensor giving the data of the
            parameter subset assigned to the rank.
    """

    def __init__(
        self,
        bucket_index: int,
        parameters: list[tp.Tensor],
        offset: int,
    ):
        self.bucket_index = bucket_index
        self.parameters = parameters
        self.offset = offset
        if len(self.parameters) == 0:
            raise ValueError("Empty bucket assignment")
        # DDP guarantees all parameters in the bucket have the same device
        self.device = self.parameters[0].device
        self.tensor: tp.Tensor | None = None


class _OverlapStatus(enum.IntEnum):
    r"""
    Define possible statuses that :class:`ZeroRedundancyOptimizer` can be in when overlapping with :class:`DistributedDataParallel`.

    Attributes:
        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """

    UNINITIALIZED = 0
    DDP_HAS_REBUILT_BUCKETS = 1
    INITIALIZED = 2


class _OverlapInfo:
    r"""
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.

    Attributes:
        shard_buckets (bool): if ``True``, then the assignment of each
            DDP bucket is partitioned across possibly multiple ZeRO instances.
        status (_OverlapStatus): current status.
        params_per_bucket (List[List[Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]``.
        num_bucket_assignments (int): total number of bucket assignments.
        total_size (int, optional): total size of all buckets.
        broadcast_handles (List[Work]): async work handles for broadcasts.
        bucket_indices_seen (List[int]): bucket indices seen this iteration.
        bucket_index_to_future / bucket_index_to_bucket: per-iteration maps
            used by ``hook_with_zero_step()``.
    """

    def __init__(self, world_size) -> None:
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED
        self.shard_buckets: bool = False

        # Modified per bucket reconstruction
        self.params_per_bucket: list[list[tp.Tensor]] = []
        self.params_per_rank: list[list[tp.Tensor]] = [
            [] for _ in range(world_size)]
        self.offsets: dict[int, int] = {}
        # Group Ranks
        self.assigned_ranks_per_bucket: list[set[int]] = []
        self.num_bucket_assignments: int = 0
        self.total_size: int | None = None

        # Modified per iteration
        self.broadcast_handles: list[Any] = []
        self.bucket_indices_seen: list[int] = []
        self.bucket_index_to_future: dict[int, Any] = {}
        self.bucket_index_to_bucket: dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self) -> None:
        r"""
        Wait for all parameter broadcasts.
        """
        if len(self.broadcast_handles) != self.num_bucket_assignments:
            raise AssertionError(
                f"Missing at least one broadcast handle on rank {dist.get_rank()}"
            )
        _ = [x.wait() for x in self.broadcast_handles]
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        r"""Clear the data structures that are modified per-iteration."""
        self.bucket_indices_seen.clear()
        self.bucket_index_to_future.clear()
        self.bucket_index_to_bucket.clear()


class ZeroRedundancyOptimizer(Optimizer, Joinable):
    r"""
    Wrap an arbitrary :class:`optim.Optimizer` and shards its states across ranks in the group.

    The sharing is done as described by `ZeRO <https://arxiv.org/abs/1910.02054>`_.

    The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks.

    Arguments:
        params: an ``Iterable`` of tensors or dicts giving all parameters.

    Keyword Args:
        optimizer_class: the class of the local optimizer.
        process_group: ``ProcessGroup``
            (default: ``dist.GroupMember.WORLD``).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters
            are packed into buckets to speed up communication.
        overlap_with_ddp (bool, optional): if ``True``, requires a functional
            optimizer and registering one of the DDP communication hooks from
            ``ddp_zero_hook.py``.
        **defaults: forwarded to the local optimizer.

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.
    """

    def __init__(
        self,
        params,
        optimizer_class: type[Optimizer],
        process_group=None,
        parameters_as_bucket_view: bool = False,
        overlap_with_ddp: bool = False,
        **defaults: Any,
    ):
        r"""Init."""
        logger.info("Instantiating ZeroRedundancyOptimizer")

        # Perform type and assumption checks on the input parameters
        params = self._verify_and_init_params(params)
        self._verify_same_dense_param_type()

        # NOTE: The parent constructor uses `add_param_group()` which is
        # partially overloaded in ZeroRedundancyOptimizer, so we use the
        # `initialized` flag to dissociate the behaviour of `add_param_group()`
        # between the parent and child.
        self.initialized = False

        Optimizer.__init__(self, params, defaults)
        Joinable.__init__(self)
        # Now, all parameters are held in both `self._all_params` and
        # `self.param_groups`

        # Internal data structures (`_cache` indicates lazily evaluated)
        self._param_to_rank_cache: dict[Any, int] = {}
        self._param_to_index_cache: dict[Any, int] = {}
        self._partition_parameters_cache: list[list[dict]] = []
        self._index_to_param_cache: list[tp.Tensor] = []
        self._device_to_params_per_rank_cache: dict = {}
        self._bucket_assignments_per_rank_cache: list[
            dict[int, _DDPBucketAssignment]
        ] = []
        self._is_trainable_mask = self._get_is_trainable_mask()

        # Default device for collective communication and buckets
        self._default_device = self._all_params[0].device

        self.process_group = (
            process_group if process_group is not None else dist.GroupMember.WORLD
        )
        self.world_size: int = dist.get_world_size(self.process_group)
        self.rank: int = dist.get_rank(self.process_group)
        self.global_rank: int = dist.get_global_rank(
            self.process_group,
            self.rank,
        )

        self._overlap_with_ddp: bool = overlap_with_ddp
        self._optim_defaults = defaults
        self._optim_constructor = self._get_optimizer_constructor(optimizer_class)

        # If `overlap_with_ddp=True`, local optimizer initialization is delayed
        # to run time after the necessary information has been collected
        if not overlap_with_ddp:
            self._init_local_optimizer()
        else:
            self._overlap_info: _OverlapInfo = _OverlapInfo(self.world_size)
            if parameters_as_bucket_view:
                logger.warning(
                    "`parameters_as_bucket_view=True` will be ignored since "
                    "`overlap_with_ddp=True`; instead, a different bucketing "
                    "strategy will be used"
                )

        # `self._buckets` is used if `parameters_as_bucket_view=True`, in
        # which case parameter data is flattened into contiguous bucket tensors
        self.parameters_as_bucket_view = parameters_as_bucket_view
        self._buckets: list[list[tp.Tensor]] = []
        self._build_param_buckets()

        # Optional consolidated optimizer state, only populated if this rank
        # is the target in `consolidate_state_dict()`
        self._all_state_dicts: list[dict[str, Any]] = []

        self.initialized = True

    def _clear_cache(self) -> None:
        r"""Clear the cached data structures giving partition information."""
        self._partition_parameters_cache.clear()
        self._param_to_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()
        self._device_to_params_per_rank_cache.clear()
        self._bucket_assignments_per_rank_cache.clear()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        r"""
        Add a parameter group to the ``Optimizer``'s ``param_groups``.

        .. warning:: This method handles updating the shards on all partitions
            but needs to be called on all ranks.
        """
        if self.initialized and self._overlap_with_ddp:
            raise RuntimeError(
                "ZeroRedundancyOptimizer with `overlap_with_ddp=True` only "
                "supports a single parameter group"
            )

        super().add_param_group(param_group)
        # NOTE: The rest of the method assumes that the call to the parent's
        # `add_param_group()` appends the new parameter group and preserves
        # the previous parameter-group ordering

        if self.initialized:
            # Force a re-partitioning of the parameters
            self._clear_cache()
            param_groups = self._partition_parameters()[self.rank]
            # NOTE: All parameters in the old parameter groups should be
            # assigned to the same ranks so that the local optimizers do not
            # need to be reinitialized

            # Add the parameters assigned to this rank from the new parameter
            # group to the local optimizer, if any
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            if self.parameters_as_bucket_view:
                self._build_param_buckets()

    def consolidate_state_dict(self, to: int = 0) -> None:
        r"""
        Consolidate a list of ``state_dict`` s (one per rank) on the target rank.

        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).

        .. warning:: This needs to be called on all ranks.
        """
        self._check_overlap_initialized()

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Pull the sharded state from all ranks and store them in rank order
        empty_messenger = tp.tensor([0], dtype=tp.uint8,
                                    device=self._default_device)

        # NOTE: We wastefully use `broadcast()` (e.g. instead of `gather()`)
        # due to compatibility issues with NCCL backend.
        self._all_state_dicts = []
        for rank in range(self.world_size):
            global_rank = dist.get_global_rank(self.process_group, rank)
            if self.rank == to:
                # Consolidate all local `state_dict`s on this rank, storing on
                # CPU to save GPU memory
                if rank == self.rank:
                    # Directly append own optimizer state
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            self.optim.state_dict(),
                            non_blocking=True,
                            device="cpu",
                        )
                    )
                else:
                    # Receive the optimizer state from the source rank
                    local_state_dict = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            local_state_dict,
                            non_blocking=True,
                            device="cpu",
                        )
                    )
            else:
                if rank == self.rank:
                    # Send the optimizer state to the target rank
                    _ = _broadcast_object(
                        self.optim.state_dict(),
                        src_rank=self.global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                elif rank != to:
                    # Discard the received object; `broadcast()` is used for
                    # compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )

    def _verify_params_per_rank(
        self,
        params_per_rank,
    ) -> None:
        r"""
        Verify ``params_per_rank`` for :meth:`_partition_parameters`.
        """
        if len(params_per_rank) != self.world_size:
            raise ValueError(
                "`params_per_rank` must have length equal to the world size"
            )
        all_params_set = set(self._all_params)
        for params in params_per_rank:
            for param in params:
                if param not in all_params_set:
                    raise ValueError(
                        "Passing a new parameter in `params_per_rank` that "
                        "was not passed into the ZeroRedundancyOptimizer "
                        "constructor"
                    )

    def _partition_param_group(
        self, param_group: dict[str, Any], params_per_rank
    ) -> None:
        r"""Partition the parameter group according to ``params_per_rank``."""
        for rank, params in enumerate(params_per_rank):
            rank_param_group = copy.copy(param_group)
            rank_param_group["params"] = params
            self._partition_parameters_cache[rank].append(rank_param_group)

    def _partition_parameters(
        self,
        params_per_rank=None,
    ) -> list[list[dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Arguments:
            params_per_rank: optional manual partition; see torch docs.

        Returns:
            A list where element i contains the ``param_groups`` for rank i.
        """
        if params_per_rank is None:
            # Partition the parameters optimizing for uniformity
            if len(self._partition_parameters_cache) == 0:
                self._partition_parameters_cache = [[] for _ in range(self.world_size)]
                sizes = [0] * self.world_size
                for param_group in self.param_groups:
                    param_group_params_per_rank: list[list] = [
                        [] for _ in range(self.world_size)
                    ]
                    # Sort the parameters by size (largest first)
                    params_sorted = sorted(
                        param_group["params"], key=lambda t: t.numel(), reverse=True
                    )
                    for param in params_sorted:
                        # Greedily add the parameter to rank with smallest size so far
                        rank = self._get_min_index(sizes)
                        param_group_params_per_rank[rank].append(param)
                        sizes[rank] += param.numel()
                    # Apply the constructed partition of the parameter group
                    self._partition_param_group(
                        param_group, param_group_params_per_rank
                    )

            return self._partition_parameters_cache

        # Partition the parameters according to `params_per_rank`
        if len(self._partition_parameters_cache) != 0:
            raise AssertionError(
                "Specifying `params_per_rank` should only be done when the "
                "parameters have not been partitioned yet"
            )
        if len(self.param_groups) != 1:
            raise RuntimeError(
                "Specifying `params_per_rank` only supports a single parameter group"
            )
        self._verify_params_per_rank(params_per_rank)
        self._partition_parameters_cache = [[] for _ in range(self.world_size)]

        # Apply the passed-in partition of the parameter group
        param_group = self.param_groups[0]
        self._partition_param_group(param_group, params_per_rank)

        return self._partition_parameters_cache

    @property
    def _param_to_rank(self) -> dict:
        r""":class:`dict` mapping parameters to their assigned rank in the partition."""
        if len(self._param_to_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_to_rank_cache[param] = rank
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> dict:
        r""":class:`dict` mapping parameters to their global state indices."""
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {
                p: i
                for i, p in enumerate(
                    chain.from_iterable(g["params"] for g in self.param_groups)
                )
            }
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> list[tp.Tensor]:
        r"""List mapping parameter indices in the global scheme to params."""
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = list(
                chain.from_iterable(g["params"] for g in self.param_groups)
            )
        return self._index_to_param_cache

    def _broadcast_params_from_rank(self, rank: int):
        r"""
        Broadcast the shard of parameters from a given rank asynchronously.
        """
        if self._overlap_with_ddp:
            raise AssertionError(
                "`_broadcast_params_from_rank()` should not be used if "
                "`overlap_with_ddp=True`; instead, the broadcasting should "
                "happen in the DDP communication hook"
            )
        handles = []
        if self.parameters_as_bucket_view:
            for dev_i_buckets in self._buckets:
                bucket = dev_i_buckets[rank]
                global_rank = dist.get_global_rank(self.process_group, rank)
                handles.append(
                    dist.broadcast(
                        tensor=bucket,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                )
        else:
            param_groups = self._partition_parameters()[rank]
            global_rank = dist.get_global_rank(self.process_group, rank)
            for param_group in param_groups:
                handles.extend(
                    dist.broadcast(
                        tensor=param.data,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                    for param in param_group["params"]
                )
        return handles

    def _sync_params(self):
        r"""
        Sync all parameter shards across the ranks using ``broadcast()``.
        """
        handles = []
        for rank in range(self.world_size):
            handles.extend(self._broadcast_params_from_rank(rank))
        _ = [x.wait() for x in handles]

    @property
    def _device_to_params_per_rank(self):
        r"""Return device parameters assigned per rank."""
        if not self.parameters_as_bucket_view:
            raise AssertionError(
                "`_device_to_params_per_rank` should only be used if "
                "`parameters_as_bucket_view=True`"
            )
        if len(self._device_to_params_per_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        device = str(param.device)
                        if device not in self._device_to_params_per_rank_cache:
                            self._device_to_params_per_rank_cache[device] = [
                                [] for _ in range(self.world_size)
                            ]
                        self._device_to_params_per_rank_cache[device][rank].append(
                            param
                        )
        return self._device_to_params_per_rank_cache

    def _get_min_index(
        self,
        values: list[int],
        disallowed_indices: set[int] | None = None,
    ) -> int:
        r"""Return ``values.index(min(values))`` in one pass, excluding disallowed indices."""
        min_index = -1
        min_value = float("inf")
        for i, value in enumerate(values):
            if disallowed_indices and i in disallowed_indices:
                continue
            if value < min_value:
                min_value = value
                min_index = i
        if min_index < 0:
            raise AssertionError("All indices are disallowed")
        return min_index

    def _assign_bucket_subset_to_rank(
        self,
        bucket_index: int,
        bucket_params: list[tp.Tensor],
        bucket_offset: int,
        assigned_rank: int,
        assigned_ranks_per_bucket: list[set[int]],
    ) -> None:
        r"""Assign ``bucket_params`` to the rank with least size so far."""
        overlap_info = self._overlap_info
        if len(bucket_params) == 0:
            raise ValueError("Empty bucket assignment")
        params_per_rank = overlap_info.params_per_rank
        offsets = overlap_info.offsets

        self._bucket_assignments_per_rank_cache[assigned_rank][bucket_index] = (
            _DDPBucketAssignment(bucket_index, bucket_params, bucket_offset)
        )
        if self.global_rank == assigned_rank:
            offsets[bucket_index] = len(params_per_rank[assigned_rank])
        params_per_rank[assigned_rank].extend(bucket_params)
        assigned_ranks_per_bucket[bucket_index].add(assigned_rank)
        self._overlap_info.num_bucket_assignments += 1

    @property
    def _bucket_assignments_per_rank(self):
        r"""Return DDP bucket parameters assigned per rank."""
        if not self._overlap_with_ddp:
            raise AssertionError(
                "`_bucket_assignments_per_rank` only be used if `overlap_with_ddp=True`"
            )
        if len(self._bucket_assignments_per_rank_cache) > 0:
            return self._bucket_assignments_per_rank_cache

        overlap_info = self._overlap_info
        if overlap_info.status != _OverlapStatus.INITIALIZED:
            raise AssertionError

        self._bucket_assignments_per_rank_cache = [{} for _ in range(self.world_size)]
        params_per_bucket = overlap_info.params_per_bucket

        if overlap_info.shard_buckets:
            # Define the assignment threshold to approximate uniformity
            if overlap_info.total_size is None:
                raise AssertionError("`total_size` was not computed")
            threshold = overlap_info.total_size / self.world_size
            size_per_rank = [0 for _ in range(self.world_size)]

        num_buckets = len(params_per_bucket)
        overlap_info.assigned_ranks_per_bucket = [set() for _ in range(num_buckets)]
        assigned_ranks_per_bucket = overlap_info.assigned_ranks_per_bucket
        if not overlap_info.shard_buckets:
            # Assign each DDP bucket entirely to a single rank
            for bucket_index, bucket_params in enumerate(params_per_bucket):
                if len(bucket_params) <= 0:
                    raise AssertionError("Empty bucket")
                assigned_rank = self._get_assigned_rank(bucket_index)
                self._assign_bucket_subset_to_rank(
                    bucket_index,
                    bucket_params,
                    0,
                    assigned_rank,
                    assigned_ranks_per_bucket,
                )
        else:
            # Assign each DDP bucket to possibly multiple ranks
            params_per_bucket_enum = sorted(
                enumerate(params_per_bucket),
                key=lambda x: sum(p.numel() for p in x[1]),
            )
            for bucket_index, bucket_params in params_per_bucket_enum:
                if len(bucket_params) <= 0:
                    raise AssertionError("Empty bucket")
                bucket_offset = 0
                assignment_size = 0
                for param_index, param in enumerate(bucket_params):
                    param_numel = param.numel()
                    if (
                        assignment_size + param_numel >= threshold
                        and param_index > bucket_offset
                    ):
                        assigned_rank = self._get_min_index(
                            size_per_rank,
                            assigned_ranks_per_bucket[bucket_index],
                        )
                        self._assign_bucket_subset_to_rank(
                            bucket_index,
                            bucket_params[bucket_offset:param_index],
                            bucket_offset,
                            assigned_rank,
                            assigned_ranks_per_bucket,
                        )
                        size_per_rank[assigned_rank] += assignment_size
                        bucket_offset = param_index
                        assignment_size = 0
                    assignment_size += param_numel
                # Assign the remainder of the bucket so that no assignment
                # spans across two buckets
                assigned_rank = self._get_min_index(
                    size_per_rank,
                    assigned_ranks_per_bucket[bucket_index],
                )
                self._assign_bucket_subset_to_rank(
                    bucket_index,
                    bucket_params[bucket_offset:],
                    bucket_offset,
                    assigned_rank,
                    assigned_ranks_per_bucket,
                )
                size_per_rank[assigned_rank] += assignment_size

        return self._bucket_assignments_per_rank_cache

    def _local_step(
        self,
        gradients: list[tp.Tensor | None] | None = None,
        closure=None,
        **kwargs: Any,
    ) -> float | None:
        r"""
        Perform a single optimizer step without syncing parameters across ranks.
        """
        Join.notify_join_context(self)
        # Check if the model trainability has changed
        is_trainable_mask = self._get_is_trainable_mask()
        if is_trainable_mask != self._is_trainable_mask:
            if self._overlap_with_ddp:
                raise RuntimeError(
                    "ZeroRedundancyOptimizer with `overlap_with_ddp=True` "
                    "does not support changing parameter trainability at run "
                    "time"
                )
            logger.warning(
                "ZeroRedundancyOptimizer detected that the trainable "
                "parameters changed; rebuilding the parameter buckets if "
                "enabled"
            )
            self._build_param_buckets()
            self._is_trainable_mask = is_trainable_mask

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only
        if gradients is None:
            loss = (
                self.optim.step(**kwargs)
                if closure is None
                else self.optim.step(closure=closure, **kwargs)
            )
        else:
            if not self._overlap_with_ddp:
                raise AssertionError(
                    "Specifying `gradients` should not "
                    "be used when `overlap_with_ddp=False`"
                )
            if closure is not None:
                raise AssertionError(
                    "`closure` is not supported when using a local functional optimizer"
                )
            loss = self.optim.step(gradients=gradients)

        # Sync any updated attributes in the local optimizer to the exposed
        # `param_groups`
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def step(
        self,
        closure=None,
        **kwargs: Any,
    ) -> float | None:
        r"""
        Perform a single optimizer step and syncs parameters across all ranks.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note:: Any extra parameters are passed to the base optimizer as-is.
        """
        if self._overlap_with_ddp:
            logger.warning(
                "`step()` should not be included in the training loop when "
                "`overlap_with_ddp=True`"
            )
            return None

        # Perform the local optimizer step
        loss = self._local_step(closure=closure, **kwargs)

        # Sync all of the updated parameter shards across the ranks
        self._sync_params()

        return loss

    def join_hook(self, **_kwargs: Any) -> JoinHook:
        r"""
        Return the ZeRO join hook.

        It enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.
        """
        return _ZeROJoinHook(self)

    @property
    def join_device(self):
        r"""Return default device."""
        return self._default_device

    @property
    def join_process_group(self) -> Any:
        r"""Return process group."""
        return self.process_group

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""
        Load the state pertaining to the given rank from the input ``state_dict``, updating the local optimizer as needed.
        """
        self._check_overlap_initialized()

        for index, value in state_dict["state"].items():
            param = self._index_to_param[index]
            if self._param_to_rank[param] != self.rank:
                # Clear any state irrelevant to this rank
                state_dict["state"][index] = None
            else:
                # Load the parameter state to the local optimizer
                self.optim.state[param] = _recursive_copy_to_device(
                    value, non_blocking=True, device=param.device
                )
                # Force zero-dimensional tensors (like Adam "step") on CPU
                for state_name, state_value in self.optim.state[param].items():
                    if tp.is_tensor(state_value) and state_value.ndimension() == 0:
                        self.optim.state[param][state_name] = state_value.cpu()

        Optimizer.load_state_dict(self, state_dict)

        # Sync the input state with the exposed and local optimizer states
        self._sync_param_groups(state_dict["param_groups"], self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

    def state_dict(self) -> dict[str, Any]:
        r"""
        Return the last global optimizer state known to this rank.

        Raises:
            RuntimeError: if this method is called without a preceding call
                to :meth:`consolidate_state_dict`.
        """
        self._check_overlap_initialized()

        if len(self._all_state_dicts) == 0:
            raise RuntimeError(
                "Optimizer state has not been consolidated on this rank. "
                f"Please call `consolidate_state_dict(to={self.rank})` on "
                "all ranks beforehand if you meant to save the global state."
            )

        # Get the possibly-stale global optimizer state that uses global
        # parameter indexing
        state_dict = Optimizer.state_dict(self)

        # Update the global optimizer state with local state information,
        # factoring in the translation from local to global indexing
        for rank, local_state_dict in enumerate(self._all_state_dicts):
            local_param_groups = local_state_dict["param_groups"]
            global_param_groups = self._partition_parameters()[rank]
            if len(local_param_groups) != len(global_param_groups):
                raise AssertionError(
                    "Mismatch between number of local and global parameter groups"
                )

            for local_param_group, global_param_group in zip(
                local_param_groups, global_param_groups
            ):
                # `local_param_group` stores local indices, while
                # `global_param_group` stores the tensors directly
                local_param_indices = local_param_group["params"]
                global_params = global_param_group["params"]

                if len(local_param_indices) != len(global_params):
                    raise AssertionError(
                        "Mismatch between number of local and global "
                        "parameters in parameter group"
                    )
                for local_param_index, global_param in zip(
                    local_param_indices, global_params
                ):
                    # Update the global parameter state, if any
                    if local_param_index in local_state_dict["state"]:
                        global_param_index = self._param_to_index[global_param]
                        state_dict["state"][global_param_index] = local_state_dict[
                            "state"
                        ][local_param_index]

        # Sort the parameters in the state
        state_dict["state"] = dict(sorted(state_dict["state"].items()))
        return state_dict

    @staticmethod
    def _sync_param_groups(
        src_param_groups: list[dict[Any, Any]],
        dst_param_groups: list[dict[Any, Any]],
    ) -> None:
        r"""
        Sync the attributes from the source parameter groups to the destination parameter groups.

        Example attributes include learning rate or scheduler attributes. The
        two parameter groups should have the same length.
        """
        if len(src_param_groups) != len(dst_param_groups):
            raise AssertionError(
                "Mismatch between number of source and destination parameter groups"
            )
        for src_param_group, dst_param_group in zip(src_param_groups, dst_param_groups):
            # Sync all attributes except the parameters
            for attr in filter(lambda x: x != "params", src_param_group.keys()):
                dst_param_group[attr] = src_param_group[attr]

    def _build_param_buckets(self) -> None:
        r"""
        Build parameter buckets if ``parameters_as_bucket_view=True``.

        For each device that stores this rank's parameters, there is a
        bucket containing all of the parameters on that device assigned to a
        given rank in the parameter update partition.
        """
        if not self.parameters_as_bucket_view or self._overlap_with_ddp:
            return

        # `self._buckets[i][j]` are the parameters stored on device i and
        # assigned to rank j
        num_devices = len(self._device_to_params_per_rank)
        self._buckets = [[] for _ in range(num_devices)]  # type: ignore[assignment]

        for dev_i, (device, params_per_rank) in enumerate(
            self._device_to_params_per_rank.items()
        ):
            for params in params_per_rank:
                bucket_size = 0
                dtype = None
                trainable_params = []
                for param in params:
                    if not _is_trainable(param):
                        # Clone in case the parameter was previously part of
                        # a bucket to avoid the data from being destroyed
                        param.data = param.data.detach().clone()
                    else:
                        bucket_size += param.numel()
                        trainable_params.append(param)
                    dtype = param.dtype  # assumes all same dtype

                if bucket_size == 0:
                    # Create a dummy bucket if there are no parameters
                    bucket = tp.zeros(1, device=device)
                else:
                    # Construct the bucket (assuming all dense and same dtype)
                    bucket = tp.empty(bucket_size, dtype=dtype, device=device)
                    offset = 0
                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.reshape(-1))
                        param.data = bucket[offset:offset_next].view(param.shape)
                        offset = offset_next
                self._buckets[dev_i].append(bucket)  # type: ignore[arg-type]

    def _build_ddp_param_buckets(self) -> None:
        r"""
        Build the DDP bucket with parameters assigned to this rank.
        """
        for bucket_assignments in self._bucket_assignments_per_rank:
            for bucket_assignment in bucket_assignments.values():
                params = bucket_assignment.parameters
                bucket_size = 0
                dtype = None
                for param in params:
                    if not _is_trainable(param):
                        raise AssertionError(
                            "Model parameter "
                            "corresponding to a gradient in a DDP bucket should "
                            "require a gradient"
                        )
                    bucket_size += param.numel()
                    dtype = param.dtype  # assumes all same dtype
                if bucket_size <= 0:
                    raise AssertionError("Empty bucket")

                # Construct the bucket tensor (assuming all dense and same dtype)
                tensor = tp.empty(
                    bucket_size, dtype=dtype, device=bucket_assignment.device
                )
                offset = 0
                for param in params:
                    offset_next = offset + param.numel()
                    tensor[offset:offset_next].copy_(param.data.reshape(-1))
                    param.data = tensor[offset:offset_next].view(param.shape)
                    offset = offset_next
                bucket_assignment.tensor = tensor

    def _verify_and_init_params(
        self,
        params: Any,
    ) -> list[tp.Tensor] | list[dict]:
        r"""
        Verify the type of ``params`` and initializes ``self._all_params``.
        """
        if isinstance(params, tp.Tensor):
            raise TypeError(
                "`params` argument should be an iterable of "
                f"Tensors, but got {type(params).__name__}"
            )
        try:
            all_params = list(params)
        except TypeError as e:
            raise TypeError(
                "`params` argument should be an iterable of Tensors"
                f" or dicts, but got {type(params).__name__}"
            ) from e
        if len(all_params) == 0:
            raise ValueError("ZeroRedundancyOptimizer got an empty parameter list")
        all_tensors = True
        all_dicts = True
        for param in all_params:
            all_tensors &= isinstance(param, tp.Tensor)
            all_dicts &= isinstance(param, dict)
        if not all_tensors and not all_dicts:
            raise TypeError(
                "`params` argument should be an iterable of Tensors or dicts"
            )
        # Ensure that `self._all_params` contains a list of all parameters
        if all_tensors:
            self._all_params = all_params
        elif all_dicts:
            self._all_params = []
            # `all_params` contains parameter groups (not parameters)
            for param_group in all_params:
                if "params" not in param_group:
                    raise ValueError(
                        "Each parameter group passed-in via `params` must "
                        "have a 'params' key mapping to the parameters in "
                        "the group"
                    )
                self._all_params.extend(param_group["params"])
        return all_params

    def _verify_same_dense_param_type(self) -> None:
        r"""
        Verify that all parameters are of the same dense type.
        """
        first = self._all_params[0]
        typename = str(first.dtype)
        if first.is_sparse():
            raise ValueError(
                "ZeroRedundancyOptimizer only supports using "
                "the same dense type for all parameters but got "
                f"{typename}"
            )
        for param in self._all_params[1:]:
            other_typename = str(param.dtype)
            if other_typename != typename or param.is_sparse():
                raise ValueError(
                    "ZeroRedundancyOptimizer only supports "
                    "using the same dense type for all "
                    f"parameters but got both {typename} and "
                    f"{other_typename}"
                )

    def _get_is_trainable_mask(self) -> list[bool]:
        r"""Return a boolean mask indicating if each parameter is trainable."""
        return list(map(_is_trainable, self._all_params))

    def _init_local_optimizer(self) -> None:
        r"""
        Initialize this rank's local optimizer, responsible for its subset of the parameters.

        The local optimizer is saved in ``self.optim``.
        """
        if self._optim_constructor is None:
            raise AssertionError("The local optimizer class has not been set")

        param_groups = self._partition_parameters()[self.rank]
        # `overlap_with_ddp=True` requires a local functional optimizer
        if self._overlap_with_ddp:
            # Functional optimizers only support a single parameter group and
            # require passing in the parameters as a list
            if len(param_groups) != 1:
                raise AssertionError(
                    "Initializing the local functional optimizer "
                    "with more than one parameter group"
                )
            params = param_groups[0]["params"]
            # Try to pass `_allow_empty_param_list=True` to avoid erroring
            if (
                "_allow_empty_param_list"
                in inspect.signature(self._optim_constructor).parameters
            ):
                self.optim: Any = self._optim_constructor(
                    params, **self._optim_defaults, _allow_empty_param_list=True
                )
            else:
                logger.warning(
                    "%s does not support the argument "
                    "`_allow_empty_param_list`; ZeroRedundancyOptimizer may "
                    "error due to an empty parameter list",
                    self._optim_constructor,
                )
                self.optim = self._optim_constructor(
                    params, **self._optim_defaults
                )

            # Log information about the DDP and ZeRO bucketing
            if dist.get_world_size() > 1:
                local_numel = sum(p.numel() for p in params)
                logger.info(
                    "rank %s with %s parameters",
                    self.global_rank,
                    local_numel,
                )
        else:
            # NOTE: Passing `param_groups` into the local optimizer constructor
            # bypasses the empty parameter list check
            self.optim = self._optim_constructor(
                param_groups, **self._optim_defaults
            )

        # Manually add `self.param_groups` if using a functional optimizer
        if self._overlap_with_ddp and not hasattr(self.optim, "param_groups"):
            if not hasattr(self.optim, "param_group"):
                raise AssertionError(
                    "The functional optimizer should set at least one of "
                    "the attributes `param_group` or `param_groups`"
                )
            self.optim.param_groups = [self.optim.param_group]  # type: ignore[attr-defined]

        self._sync_param_groups(self.optim.param_groups, self.param_groups)

    def _init_zero_for_overlap(self) -> None:
        r"""Perform a delayed initialization of the local optimizer and the supporting data structures."""
        if not self._overlap_with_ddp:
            raise AssertionError(
                "`_init_zero_for_overlap()` should only be called when "
                "`overlap_with_ddp=True`"
            )
        self._overlap_info.status = _OverlapStatus.INITIALIZED
        self._clear_cache()
        self._partition_parameters(self._overlap_info.params_per_rank)
        self._build_ddp_param_buckets()
        self._init_local_optimizer()

    def _get_assigned_rank(self, bucket_index: int) -> int:
        r"""Return the single rank assigned to a DDP gradient bucket."""
        if self._overlap_info.shard_buckets:
            raise AssertionError(
                "The bucket assignment requires global bucket information "
                "and will be computed later; there should be no need to "
                "use this method"
            )
        return bucket_index % self.world_size

    def _check_overlap_initialized(self):
        r"""
        Check the delayed initialization depending on the value of ``overlap_with_ddp``.
        """
        if (
            self._overlap_with_ddp
            and self._overlap_info.status != _OverlapStatus.INITIALIZED
        ):
            raise RuntimeError(
                "This method should not be called until this "
                "ZeroRedundancyOptimizer instance has been fully "
                "initialized"
            )

    def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
        r"""
        Return the optimizer constructor using validation and transformation depending on ``overlap_with_ddp``.
        """
        functional_optims = functional_optim_map.values()
        if not self._overlap_with_ddp:
            if optimizer_class in functional_optims:
                # Using a functional optimizer is only supported when
                # `overlap_with_ddp=True`
                raise ValueError(
                    f"Passing in a functional optimizer {optimizer_class} "
                    "when `overlap_with_ddp=False`"
                )
            else:
                return optimizer_class
        else:
            if optimizer_class in functional_optims:
                # Already a functional optimizer
                return optimizer_class
            elif optimizer_class in functional_optim_map:
                # Translate the passed-in optimizer class to its functional
                # equivalent if `overlap_with_ddp=True`
                optim_constructor = functional_optim_map[optimizer_class]
                logger.info(
                    "Using the functional optimizer %s "
                    "instead of %s since "
                    "`overlap_with_ddp=True`",
                    optim_constructor,
                    optimizer_class,
                )
                return optim_constructor
            else:
                raise ValueError(
                    "Using `ddp_with_overlap=True` requires using a "
                    "functional optimizer, but there is no supported functional "
                    f"optimizer equivalent for {optimizer_class}"
                )
