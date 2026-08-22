"""DistributedDataParallel — ported from ``torch/nn/parallel/distributed.py``.

The public API and reduction machinery mirror torch's DDP: gradient
bucketing via ``_compute_bucket_assignment_by_size``, post-accumulate grad
hooks (torch's ``register_post_accumulate_grad_hook`` reducer path),
``find_unused_parameters`` via output-graph traversal, communication hooks
returning ``tensorplay.futures.Future``, and coalesced buffer broadcasts.
"""

import warnings
from contextlib import contextmanager
from typing import Any, List, Optional

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed.algorithms.join import Join, JoinHook, Joinable
from tensorplay.nn.modules.module import Module

__all__ = ["DistributedDataParallel", "_DDPJoinHook"]


def _recursive_to(inputs: Any, target_device: str) -> Any:
    """Recursively moves input tensors to target_device (torch _recursive_to)."""
    if isinstance(inputs, tp.Tensor):
        return inputs.to(target_device) if inputs.device != tp.device(target_device) else inputs
    if isinstance(inputs, tuple) and len(inputs) > 0:
        return tuple(_recursive_to(x, target_device) for x in inputs)
    if isinstance(inputs, list) and len(inputs) > 0:
        return [_recursive_to(x, target_device) for x in inputs]
    if isinstance(inputs, dict) and len(inputs) > 0:
        return {k: _recursive_to(v, target_device) for k, v in inputs.items()}
    return inputs


def _find_tensors(obj):
    """Recursively find all tensors contained in the specified object."""
    import itertools

    if isinstance(obj, tp.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return list(itertools.chain.from_iterable(map(_find_tensors, obj)))
    if isinstance(obj, dict):
        return list(itertools.chain.from_iterable(
            map(_find_tensors, obj.values())))
    return []


def _param_key(tensor) -> tuple:
    """Stable identity key for a parameter across wrapper objects."""
    return (tensor.data_ptr(), tuple(tensor.shape))


def _verify_param_shape_across_processes(process_group, tensors) -> bool:
    """All-reduce MIN/MAX over tensor sizes; False on mismatch (torch parity)."""
    if dist.get_world_size(process_group) == 1 or len(tensors) == 0:
        return True
    device = tensors[0].device
    sizes = tp.tensor([t.numel() for t in tensors], dtype=tp.int64,
                      device=device)
    mins = sizes.clone()
    dist.all_reduce(mins, op=dist.ReduceOp.MIN, group=process_group)
    maxs = sizes.clone()
    dist.all_reduce(maxs, op=dist.ReduceOp.MAX, group=process_group)
    return int(mins.min().item()) == int(maxs.max().item())


def _sync_params_and_buffers(process_group, module_states,
                             broadcast_bucket_size, src_global_rank: int) -> None:
    """Synchronize module_states by coalesced broadcast (torch parity)."""
    if len(module_states) == 0:
        return
    src = dist.get_global_rank(process_group, src_global_rank)
    dist._broadcast_coalesced(process_group, module_states,
                              broadcast_bucket_size, src)


def _sync_module_states(module: Module, process_group,
                        broadcast_bucket_size: int, src: int,
                        params_and_buffers_to_ignore,
                        broadcast_buffers: bool = True) -> None:
    """Sync ``module``'s parameters and buffers state (torch parity)."""
    module_states: List[tp.Tensor] = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())
    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())
    _sync_params_and_buffers(process_group, module_states,
                             broadcast_bucket_size, src)


class _DDPJoinHook(JoinHook):
    def __init__(self, ddp, divide_by_initial_world_size):
        """Set config variables for internal usage."""
        if not isinstance(ddp, DistributedDataParallel):
            raise AssertionError(
                "DDP join hook requires passing in a DistributedDataParallel "
                f"instance as the state, got {type(ddp).__name__}"
            )
        self.ddp = ddp
        self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()

    def main_hook(self):
        """Shadow the DDP collective communication operations in the forward and backward passes."""
        ddp = self.ddp
        # Schedule a broadcast if we are syncing module buffers in the
        # forward pass
        ddp._check_and_sync_module_buffers()

        # Check if need to sync in the backward pass
        should_sync_backwards = ddp._check_global_requires_backward_grad_sync(
            is_joined_rank=True
        )
        # Forward parameter sync is disabled in the next iteration if we
        # are skipping gradient sync this iteration, so set
        # `require_forward_param_sync` accordingly
        ddp.require_forward_param_sync = should_sync_backwards
        if not should_sync_backwards:
            return

        # Schedule one allreduce per gradient bucket to match the backward
        # pass allreduce (joined ranks contribute zero gradients)
        ddp._match_all_reduce_for_bwd_pass()

    def post_hook(self, is_last_joiner: bool):
        """Sync the final model to ensure that the model is the same across all processes."""
        self.ddp._sync_final_model(is_last_joiner)


class DistributedDataParallel(Module, Joinable):
    r"""Implements distributed data parallelism (torch parity).

    Arguments mirror ``torch.nn.parallel.DistributedDataParallel``. Gradient
    reduction uses buckets built by
    ``dist._compute_bucket_assignment_by_size``; each parameter's post
    accumulate hook copies its gradient into its bucket buffer and, when the
    bucket completes, one all-reduce averages it and the reduced values are
    copied back into ``param.grad``.

    Example::

        >>> tp.distributed.init_process_group(backend="nccl")
        >>> net = DistributedDataParallel(model, device_ids=[rank])
        >>> out = net(input)
    """

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=None,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    ):
        super().__init__()
        Joinable.__init__(self)
        self.logger = None
        self.process_group = (
            process_group if process_group is not None
            else dist._get_default_group()
        )

        self._delay_all_reduce_params = []
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = set(module._ddp_params_and_buffers_to_ignore)
        else:
            self.parameters_to_ignore = set()

        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]
        if not any(p.requires_grad for p in self._module_parameters):
            self._log_and_throw(
                RuntimeError,
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient.",
            )

        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError,
                "device_ids can only be None or contain a single element.",
            )

        self.is_multi_device_module = (
            len({str(p.device) for p in self._module_parameters}) > 1
        )
        distinct_device_types = {
            p.device.type for p in self._module_parameters if p.device is not None
        }
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataParallel's input module must be on "
                f"the same type of devices, but input module parameters locate in {distinct_device_types}.",
            )

        self.device_type = next(iter(distinct_device_types))

        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                devices = {str(p.device) for p in self._module_parameters}
                self._log_and_throw(
                    ValueError,
                    "DistributedDataParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    f"but got device_ids {device_ids}, output_device {output_device}, "
                    f"and module parameters {devices}.",
                )
            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = [int(x) for x in device_ids]
            if output_device is None:
                output_device = device_ids[0]
            self.output_device = int(output_device)

        self.dim = dim
        self.module = module
        self.device = next(iter(self._module_parameters)).device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it.",
                stacklevel=2,
            )

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 250 * 1024 * 1024
        self.bucket_bytes_cap = (
            bucket_cap_mb * 1024 * 1024 if bucket_cap_mb else 25 * 1024 * 1024
        )
        self._comm_hooks = []
        self._static_expected = None
        self._next_bucket = 0

        # Build parameters for the reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()

        # Verify model equivalence and sync params/buffers so all ranks
        # start off at the same value.
        if not _verify_param_shape_across_processes(
                self.process_group, parameters):
            self._log_and_throw(
                RuntimeError,
                "Parameters were not all to the same shape across ranks.",
            )
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=0,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=broadcast_buffers,
        )

        # Builds reducer state and registers grad hooks.
        self._ddp_init_helper(parameters, expect_sparse_gradient)

        self._has_rebuilt_buckets = False
        self._lazy_init_ran = True

    def _log_and_throw(self, err_type, err_msg):
        if self.logger is not None:
            self.logger.set_error_and_log(f"{str(err_type)}: {err_msg}")
        raise err_type(err_msg)

    def _ddp_init_helper(self, parameters, expect_sparse_gradient):
        """Bucket the parameters and register post-accumulate grad hooks.

        Mirrors torch Reducer construction: buckets are passed reversed so
        their readiness approximates gradient production order.
        """
        bucket_indices, _ = dist._compute_bucket_assignment_by_size(
            parameters,
            [self.bucket_bytes_cap],
            expect_sparse_gradient=expect_sparse_gradient,
        )

        self._buckets = []
        self._param_entries = {}  # param_key -> (bucket_state, offset, length)
        self._node_to_param_key = {}  # AccumulateGrad raw ptr -> param key
        self._grad_hooks = []

        for bucket_index, indices in enumerate(reversed(bucket_indices)):
            params = [parameters[i] for i in indices]
            offsets, lengths, shapes = [], [], []
            acc = 0
            for p in params:
                lengths.append(p.numel())
                offsets.append(acc)
                shapes.append(list(p.shape))
                acc += p.numel()
            buffer = tp.zeros(acc, dtype=params[0].dtype,
                              device=params[0].device)
            bstate = {
                "index": bucket_index,
                "buffer": buffer,
                "offsets": offsets,
                "lengths": lengths,
                "shapes": shapes,
                "params": params,
                "pending": set(),
                "expected_keys": set(),
            }
            self._buckets.append(bstate)
            for off, ln, p in zip(offsets, lengths, params):
                self._param_entries[_param_key(p)] = (bstate, off, ln)

        # Map each parameter's AccumulateGrad node to its key so backward
        # graph traversal can identify participating parameters.
        for p in parameters:
            node = p._accumulate_grad_node
            if node is not None:
                self._node_to_param_key[node._raw_ptr()] = _param_key(p)

        for index, param in enumerate(parameters):
            if not param.requires_grad:
                continue
            self._grad_hooks.append(param.register_post_accumulate_grad_hook(
                self._make_reducer_hook(param)))

        total = len(self._buckets)
        for bstate in self._buckets:
            bstate["grad_bucket"] = dist.GradBucket(
                bstate["index"], bstate["buffer"], bstate["offsets"],
                bstate["lengths"], bstate["shapes"], bstate["params"],
                num_total_buckets=total,
            )

    def _make_reducer_hook(self, param):
        def hook(_param):
            self._mark_param_ready(param)
        return hook

    def _reset_iteration_state(self):
        """Prepare per-iteration pending sets (torch prepare_for_forward).

        Parameters not expected this iteration get their bucket slice
        zero-filled up front, mirroring the reducer marking unused
        parameters ready with zero gradients.
        """
        self._next_bucket = 0
        for bid, bstate in enumerate(self._buckets):
            if self.find_unused_parameters and self.static_graph \
                    and getattr(self, "_static_expected", None) is not None:
                expected = set(self._static_expected[bid])
            elif self.find_unused_parameters:
                # Filled later by _prepare_for_backward after forward.
                expected = None
            else:
                expected = {_param_key(p) for p in bstate["params"]}
            bstate["expected_keys"] = expected
            bstate["pending"] = set(expected) if expected is not None else None
            if expected is not None:
                self._zero_unused_slices(bstate)

    def _zero_unused_slices(self, bstate):
        with tp.no_grad():
            for off, ln, p in zip(bstate["offsets"], bstate["lengths"],
                                  bstate["params"]):
                if _param_key(p) not in bstate["pending"]:
                    bstate["buffer"][off : off + ln].zero_()

    def _try_flush_ready_buckets(self):
        """Reduce completed buckets strictly in index order.

        All ranks traverse their own graph under find_unused_parameters,
        so completion order is forced to 0,1,... to keep the collective
        sequence aligned across ranks.
        """
        while self._next_bucket < len(self._buckets):
            bstate = self._buckets[self._next_bucket]
            if bstate["pending"] is None or bstate["pending"]:
                break
            self._reduce_bucket(bstate)
            self._next_bucket += 1

    def _prepare_for_backward(self, output):
        """Collect parameters reachable from outputs (find_unused path)."""
        if not self.find_unused_parameters:
            return
        keys = set()
        stack = []
        for t in _find_tensors(output):
            gf = t.grad_fn if hasattr(t, "grad_fn") else None
            if gf is not None:
                stack.append(gf)
        visited = set()
        while stack:
            node = stack.pop()
            ptr = node._raw_ptr()
            if ptr in visited:
                continue
            visited.add(ptr)
            var = getattr(node, "variable", None)
            if var is not None:
                keys.add(_param_key(var))
            for nxt, _input_nr in node.next_functions:
                if nxt is not None:
                    stack.append(nxt)
        self._next_bucket = 0
        for bid, bstate in enumerate(self._buckets):
            member_keys = {_param_key(p) for p in bstate["params"]}
            bstate["expected_keys"] = member_keys & keys
            bstate["pending"] = set(bstate["expected_keys"])
            self._zero_unused_slices(bstate)
        if self.static_graph and self._static_expected is None:
            self._static_expected = {
                bid: set(b["expected_keys"])
                for bid, b in enumerate(self._buckets)
            }
        self._try_flush_ready_buckets()

    def _mark_param_ready(self, param):
        """Copy grad into the bucket; reduce once the bucket completes."""
        if not self.require_backward_grad_sync:
            return
        entry = self._param_entries.get(_param_key(param))
        if entry is None:
            return
        bstate, offset, length = entry
        grad = param.grad
        if grad is None:
            return
        key = _param_key(param)
        # Only wait on params that this iteration expects (find_unused).
        if key in bstate["pending"]:
            with tp.no_grad():
                bstate["buffer"][offset : offset + length].copy_(
                    grad.reshape(-1))
            bstate["pending"].discard(key)
            self._try_flush_ready_buckets()

    def _reduce_bucket(self, bstate):
        """Run the comm hook (or default allreduce) and copy grads back."""
        world_size = self.process_group.size()
        if world_size <= 1:
            return
        bucket = bstate["grad_bucket"]
        if self._comm_hooks:
            new_buffer = None
            for hook, state in self._comm_hooks:
                new_buffer = hook(state, bucket).value()
            self._copy_bucket_back(bstate, new_buffer)
        else:
            buffer = bstate["buffer"]
            buffer.div_(world_size)
            work = dist.all_reduce(buffer, group=self.process_group,
                                   async_op=True)
            work.wait()
            self._copy_bucket_back(bstate, buffer)

    def _copy_bucket_back(self, bstate, buffer):
        with tp.no_grad():
            for off, ln, p in zip(bstate["offsets"], bstate["lengths"],
                                  bstate["params"]):
                if p.grad is not None:
                    p.grad.copy_(buffer[off : off + ln].view(p.shape))

    def register_comm_hook(self, state: object, hook) -> None:
        r"""Register communication hook for custom gradient aggregation.

        The hook has signature ``hook(state, bucket) -> Future[Tensor]``
        where the future holds the reduced bucket buffer (torch contract;
        completion is resolved synchronously on ``wait``).
        """
        if not callable(hook):
            raise TypeError("Communication hook must be callable.")
        if self._comm_hooks:
            raise RuntimeError(
                "DDP communication hook can only be registered once and "
                "should be registered before calling backward."
            )
        self._comm_hooks.append((hook, state))

    def _build_params_for_reducer(self):
        """Build deduplicated parameters and sparse-gradient expectations."""
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
            for parameter in [
                param
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore
            ]
        ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        modules_and_parameters = [
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)
        ]
        parameters = [parameter for _, parameter in modules_and_parameters]

        def produces_sparse_gradient(module):
            embedding_cls = getattr(tp.nn, "Embedding", None)
            bag_cls = getattr(tp.nn, "EmbeddingBag", None)
            if embedding_cls is not None and isinstance(module, embedding_cls):
                return module.sparse
            if bag_cls is not None and isinstance(module, bag_cls):
                return module.sparse
            return False

        expect_sparse_gradient = [
            produces_sparse_gradient(module)
            for module, _ in modules_and_parameters
        ]

        self._module_parameters = list(parameters)
        self._assign_modules_buffers()

        return parameters, expect_sparse_gradient

    def _assign_modules_buffers(self):
        """Assign self.module.named_buffers to self.modules_buffers."""
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            if buffer_name not in self.parameters_to_ignore
        ]
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != dist._get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True
        if pickle_not_supported:
            self._log_and_throw(
                RuntimeError,
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor",
            )

    def __getstate__(self):
        self._check_default_group()
        attrs = {k: v for k, v in self.__dict__.items()
                 if k not in ("process_group", "_grad_hooks")}
        attrs["broadcast_buffers"] = self.broadcast_buffers
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = dist._get_default_group()
        self.__dict__.update(state)
        self.__dict__.setdefault("require_forward_param_sync", True)
        self.__dict__.setdefault("require_backward_grad_sync", True)
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        self._ddp_init_helper(parameters, expect_sparse_gradient)

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)

    @contextmanager
    def no_sync(self):
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def _run_ddp_forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _pre_forward(self, *inputs, **kwargs):
        # Notify the join context that this process has not joined, if needed
        work = Join.notify_join_context(self)
        if work:
            # Keep a handle; the allreduce is stream-ordered like torch's.
            self._join_notify_work = work

        if tp.is_grad_enabled() and self.require_backward_grad_sync:
            self._static_expected = getattr(self, "_static_expected", None)
            if self.find_unused_parameters and self.static_graph \
                    and self._static_expected is not None:
                # Reuse the first-iteration traversal (static graph).
                self._next_bucket = 0
                for bid, keys in self._static_expected.items():
                    bstate = self._buckets[bid]
                    bstate["expected_keys"] = set(keys)
                    bstate["pending"] = set(keys)
                    self._zero_unused_slices(bstate)
            else:
                self._reset_iteration_state()

        if self.will_sync_module_buffers() and tp.is_grad_enabled():
            self._sync_buffers()

        if self.device_ids:
            target = f"{self.device_type}:{self.device_ids[0]}"
            inputs = _recursive_to(inputs, target)
            kwargs = _recursive_to(kwargs, target)
        return inputs, kwargs

    def _post_forward(self, output):
        if tp.is_grad_enabled() and self.require_backward_grad_sync:
            if self.find_unused_parameters and not (
                self.static_graph and self._static_expected is not None
            ):
                self._prepare_for_backward(output)
        return output

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self._pre_forward(*inputs, **kwargs)
        output = self._run_ddp_forward(*inputs, **kwargs)
        return self._post_forward(output)

    def train(self, mode=True):
        super().train(mode)
        return self

    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers) > 0
        )

    def _sync_buffers(self):
        with tp.no_grad():
            # module buffer sync
            # If we are running DDP with the join manager, we have to agree
            # upon a rank to sync module buffers from, since rank 0 may
            # already have been joined and have stale module buffers.
            if getattr(self, "_join_config", None) is not None \
                    and self._join_config.enable:
                authoritative_rank = self._find_common_rank(
                    self._distributed_rank, True
                )
            else:
                # The process with rank 0 is considered the authoritative copy.
                authoritative_rank = 0
            # Update self.modules_buffers in case any buffers were reassigned.
            self._assign_modules_buffers()
            self._sync_module_buffers(authoritative_rank)

    def _sync_module_buffers(self, authoritative_rank):
        self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)

    def _default_broadcast_coalesced(self, bufs=None, bucket_size=None,
                                     authoritative_rank=0):
        """
        Broadcasts buffers from rank 0 to rest of workers.
        """
        if bufs is None:
            bufs = self.modules_buffers
        if bucket_size is None:
            bucket_size = self.broadcast_bucket_size

        self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)

    def _distributed_broadcast_coalesced(self, tensors, buffer_size,
                                         authoritative_rank=0):
        dist._broadcast_coalesced(self.process_group, tensors, buffer_size,
                                  authoritative_rank)

    def _passing_sync_batchnorm_handle(self, module):
        # tp ships no SyncBatchNorm yet; nothing to hand off.
        pass

    def _find_common_rank(self, input_rank, rank_cond):
        # -1 indicates that this rank is not under consideration to be the
        # common_rank
        rank_to_use = tp.tensor(
            [input_rank if rank_cond else -1],
            device=self.device,
        )
        dist.all_reduce(rank_to_use, op=dist.ReduceOp.MAX, group=self.process_group)
        if int(rank_to_use.item()) == -1:
            self._log_and_throw(
                ValueError,
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue.",
            )
        return int(rank_to_use.item())

    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = tp.ones(1, device=self.device)
        else:
            requires_sync_tensor = tp.zeros(1, device=self.device)

        work = dist.all_reduce(
            requires_sync_tensor, group=self.process_group, async_op=True
        )

        # On joined ranks, block on the result and report whether active
        # ranks will sync backwards this iteration.
        if is_joined_rank:
            work.wait()
            should_sync_backwards = requires_sync_tensor.item() != 0
            return should_sync_backwards
        else:
            return None  # Return value is not/should not be used.

    def _check_and_sync_module_buffers(self):
        if self.will_sync_module_buffers():
            authoritative_rank = self._find_common_rank(self._distributed_rank, False)
            self._sync_module_buffers(authoritative_rank)

    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.
        self._authoritative_rank = self._find_common_rank(
            self._distributed_rank, is_last_joiner
        )
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=self._authoritative_rank,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=self.broadcast_buffers,
        )

    def _match_all_reduce_for_bwd_pass(self):
        # Joined processes contribute zero gradient: zero every bucket then
        # run the normal reduction so collective counts match active ranks.
        for bstate in self._buckets:
            with tp.no_grad():
                bstate["buffer"].zero_()
            bstate["pending"] = set(bstate["expected_keys"]) \
                if bstate["expected_keys"] is not None else set()
            self._next_bucket = 0
        self._try_flush_ready_buckets()

    def join_hook(self, **kwargs):
        r"""
        DDP join hook enables training on uneven inputs by mirroring communications in forward and backward passes.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        The hook supports the following keyword arguments:
            divide_by_initial_world_size (bool, optional):
                If ``True``, then gradients are divided by the initial world
                size that DDP was launched with. Default is ``True``.
        """
        divide_by_initial_world_size = kwargs.get("divide_by_initial_world_size", True)
        return _DDPJoinHook(
            self, divide_by_initial_world_size=divide_by_initial_world_size
        )

    @property
    def join_device(self):
        return self.device

    @property
    def join_process_group(self):
        return self.process_group

    def join(
        self,
        divide_by_initial_world_size: bool = True,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):
        r"""
        Context manager for training with uneven inputs across processes in DDP.

        This context manager will keep track of already-joined DDP processes,
        and "shadow" the forward and backward passes by inserting collective
        communication operations to match with the ones created by non-joined
        DDP processes. See torch's ``Join`` docs for details.
        """
        return Join(
            [self],
            enable,
            throw_on_early_termination,
            divide_by_initial_world_size=divide_by_initial_world_size,
        )

    def scatter(self, inputs, kwargs, device_ids):
        raise NotImplementedError

    def gather(self, outputs, output_device):
        raise NotImplementedError
