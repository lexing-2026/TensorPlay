"""Grouped parameter state transitions and communication settings."""

from dataclasses import dataclass, field
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.autograd import Function

from ... import distributed_core as dist
from ...utils import _apply_to_tensors
from ._fsdp_api import CPUOffloadPolicy
from ._fsdp_collectives import (
    AllGatherResult,
    DefaultAllGather,
    DefaultReduceScatter,
    ProcessGroupAllocAllGather,
    ProcessGroupAllocReduceScatter,
    SymmMemAllGather,
    SymmMemReduceScatter,
    _record_event,
    _current_stream,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
    _wait_event,
    _wait_stream,
)
from .._common_utils import collect_grad_tensors, replace_grad_tensors
from ._fsdp_common import (
    DataParallelMeshInfo,
    FSDPMeshInfo,
    TrainingState,
    _disable_functorch_if_active,
    _cast_fp_tensor,
    _dynamo_disable,
    is_bw,
)
from ._fsdp_param import FSDPParam, ParamModuleInfo, ShardedState

__all__ = [
    "FSDPCommContext",
    "AllGatherState",
    "ReduceScatterState",
    "AllReduceState",
    "FSDPParamGroup",
    "RegisterPostBackwardFunction",
    "_get_param_module_infos",
]


class FSDPCommContext:
    def __init__(self) -> None:
        self.device = None
        self.initialized = False
        self.device_handle = None
        self.all_gather_copy_in_stream = None
        self.all_gather_stream = None
        self.reduce_scatter_stream = None
        self.all_reduce_stream = None
        self.all_gather_state: AllGatherState | None = None
        self.reduce_scatter_states: list[ReduceScatterState] = []
        self._last_post_reduce_events: dict[Any, Any] = {}
        self.post_forward_order: list[Any] = []
        self.reduce_scatter_max_input_buffers = 1

    def lazy_init(self, device: Any) -> None:
        self.device = device
        device_type = str(getattr(device, "type", device)).split(":", 1)[0].lower()
        cuda = getattr(tp, "cuda", None)
        self.device_handle = (
            cuda
            if device_type == "cuda"
            and cuda is not None
            and callable(getattr(cuda, "is_available", None))
            and cuda.is_available()
            else None
        )
        if self.device_handle is None:
            self.initialized = True
            return
        try:
            self.all_gather_copy_in_stream = self.device_handle.Stream(
                device=device, priority=-1
            )
            self.all_gather_stream = self.device_handle.Stream(
                device=device, priority=-1
            )
            self.reduce_scatter_stream = self.device_handle.Stream(
                device=device, priority=-1
            )
            self.all_reduce_stream = self.device_handle.Stream(device=device)
        except (RuntimeError, TypeError):
            self.all_gather_copy_in_stream = None
            self.all_gather_stream = None
            self.reduce_scatter_stream = None
            self.all_reduce_stream = None
        self.initialized = True

    def get_all_gather_streams(self, async_op: bool, training_state: Any) -> tuple[Any, Any]:
        if self.device_handle is None or self.device is None:
            return None, None
        state_name = getattr(training_state, "name", str(training_state))
        if (
            not async_op
            and state_name in {"FORWARD", "PRE_BACKWARD"}
            and self.all_gather_copy_in_stream is not None
            and self.all_gather_stream is not None
        ):
            return self.all_gather_copy_in_stream, self.all_gather_stream
        current_stream = _current_stream(self.device)
        return current_stream, current_stream


@dataclass
class AllGatherState:
    results: list[Any] = field(default_factory=list)
    event: Any = None


@dataclass
class ReduceScatterState:
    reduce_scatter_input: Any = None
    event: Any = None


@dataclass
class AllReduceState:
    all_reduce_input: Any = None
    event: Any = None


class FSDPParamGroup:
    def __init__(self, params: Iterable[FSDPParam], modules: Iterable[Any], mesh_info: DataParallelMeshInfo, post_forward_mesh_info: DataParallelMeshInfo | None, device: Any, shard_placement_fn: Any, mp_policy: Any, offload_policy: Any) -> None:
        del shard_placement_fn
        self.params = list(params)
        self.fsdp_params = self.params
        self.modules = modules if isinstance(modules, list) else list(modules)
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.comm_ctx = FSDPCommContext()
        self._module_fqn = None
        self._reshard_after_forward_enabled = True
        self._reshard_after_backward_enabled = True
        self.reshard_after_backward = True
        self._requires_gradient_sync = True
        self._requires_all_reduce = True
        self.reduce_scatter_max_input_buffers = 1
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED
        self._backward_finalized = False
        self._post_backward_done = False
        self._all_gather_result: AllGatherResult | list[AllGatherResult] | None = None
        self._all_gather_async_op = False
        self._all_gather_comm = DefaultAllGather()
        self._reduce_scatter_comm = DefaultReduceScatter()
        self.unshard_async_op = False
        self.unshard_in_backward = True
        self._reset_sharded_params = False
        self._training_state = TrainingState.IDLE
        self._post_forward_indices: list[int] = []
        self._param_group_index = 0
        self._num_param_groups = 1
        self.reduce_grads = True
        self.all_reduce_grads = True
        self.gradient_divide_factor = None
        self.reduce_scatter_unused_params = False
        self.force_sum_reduction_for_comms = False
        self._partial_reduce_output = None
        self._post_reduce_event = None
        self._reshard_after_forward_event = None
        self._all_reduce_state = None
        self._all_reduce_hook = None
        self._all_reduce_hook_stream = None
        self._comm_hook = None
        self._comm_hook_state = None
        self._module_to_pre_save_state_dict_hook_handle: dict[Any, Any] = {}
        self._module_to_pre_load_state_dict_hook_handle: dict[Any, Any] = {}
        self._state_dict_hooks_registered = False
        self._post_forward_recorded = False
        self._post_backward_wrapped = False
        self._symm_mem_backend = None
        self._allocate_from_process_group = False
        for param in self.params:
            param.set_gradient_sync_owner(self)

    def _init_mp_dtypes(self) -> None:
        for param in self.params:
            param.init_dtype_attrs(self.mp_policy)
        trainable = [param for param in self.params if param.param_requires_grad]
        candidates = trainable or [
            param
            for param in self.params
            if getattr(param.orig_dtype, "is_floating_point", False)
        ]
        orig_dtypes = {param.orig_dtype for param in candidates}
        reduce_dtypes = {param.reduce_dtype for param in candidates}
        if trainable and len(orig_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        if trainable and len(reduce_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        self._orig_dtype = next(iter(orig_dtypes)) if len(orig_dtypes) == 1 else None
        self._reduce_dtype = (
            next(iter(reduce_dtypes)) if len(reduce_dtypes) == 1 else None
        )

    def lazy_init(self) -> None:
        if not self.comm_ctx.initialized:
            self.comm_ctx.lazy_init(self.device)
        if not self._reset_sharded_params and self.is_sharded():
            for param in self.params:
                param.reset_sharded_param()
                param._init_extensions()
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._validate_reduce_scatter_max_input_buffers()
        self._init_mp_dtypes()
        self._register_state_dict_hooks()

    def set_symm_mem(self, backend: Any) -> None:
        if not isinstance(self._all_gather_comm, (DefaultAllGather, SymmMemAllGather)):
            raise AssertionError(
                "cannot enable symmetric memory with a custom all-gather"
            )
        self._all_gather_comm = SymmMemAllGather(
            self._all_gather_process_group(), backend
        )
        if not isinstance(
            self._reduce_scatter_comm,
            (DefaultReduceScatter, SymmMemReduceScatter),
        ):
            raise AssertionError(
                "cannot enable symmetric memory with a custom reduce-scatter"
            )
        if self.force_sum_reduction_for_comms:
            self._reduce_scatter_comm = SymmMemReduceScatter(
                self._reduce_scatter_process_group(), backend
            )
        self._symm_mem_backend = backend

    def set_allocate_memory_from_process_group(self, enable: bool) -> None:
        if not isinstance(
            self._all_gather_comm,
            (DefaultAllGather, ProcessGroupAllocAllGather),
        ):
            raise AssertionError(
                "cannot enable process-group allocation with a custom all-gather"
            )
        self._all_gather_comm = (
            ProcessGroupAllocAllGather(self._all_gather_process_group())
            if enable
            else DefaultAllGather()
        )
        if not isinstance(
            self._reduce_scatter_comm,
            (DefaultReduceScatter, ProcessGroupAllocReduceScatter),
        ):
            raise AssertionError(
                "cannot enable process-group allocation with a custom reduce-scatter"
            )
        self._reduce_scatter_comm = (
            ProcessGroupAllocReduceScatter(self._reduce_scatter_process_group())
            if enable
            else DefaultReduceScatter()
        )
        self._allocate_from_process_group = bool(enable)

    def _unshard_impl(self) -> None:
        self.unshard(async_op=False)
        self.wait_for_unshard()

    def _all_gather_world_size(self) -> int:
        mesh_dim = self._active_mesh_info().shard_mesh_dim
        if mesh_dim is None:
            return 1
        return int(self._active_mesh_info().mesh.size(mesh_dim))

    def _active_mesh_info(self) -> DataParallelMeshInfo:
        if self.is_sharded_post_forward() and self.post_forward_mesh_info is not None:
            return self.post_forward_mesh_info
        return self.mesh_info

    @_disable_functorch_if_active
    def unshard(self, async_op: bool = False) -> None:
        if self._is_unsharded:
            return
        if self._all_gather_result is not None:
            if not async_op:
                self.wait_for_unshard()
            return
        self.lazy_init()
        if self._reshard_after_forward_event is not None:
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None
        if (
            not self.unshard_in_backward
            and self._training_state == TrainingState.PRE_BACKWARD
        ):
            return
        world_size = self._all_gather_world_size()
        if world_size == 1:
            results: list[AllGatherResult] = []
            for param in self.params:
                inputs = param.all_gather_inputs
                if len(inputs) != 1:
                    raise ValueError("one parameter needs one all-gather input")
                value = inputs[0]
                param.init_all_gather_outputs(
                    [int(value.numel())], [value.dtype], 1, self.device
                )
                param.alloc_all_gather_outputs()
                output = param.all_gather_outputs[0]
                output.copy_(value)
                results.append(
                    AllGatherResult(output, _record_event(_current_stream(self.device)))
                )
            self._all_gather_result = results
        else:
            self._all_gather_result = foreach_all_gather(
                self.params,
                self._all_gather_process_group(),
                async_op,
                *self.comm_ctx.get_all_gather_streams(
                    async_op, self._training_state
                ),
                self.device,
                self._all_gather_comm,
            )
        self._all_gather_async_op = bool(async_op)

    @_disable_functorch_if_active
    def wait_for_unshard(self) -> None:
        if self._training_state == TrainingState.FORWARD:
            previous = self.comm_ctx.all_gather_state
            if previous is not None:
                self._wait_all_gather_streams_on_event(previous.event)
                self.comm_ctx.all_gather_state = None
        results = self._all_gather_result
        if results is None:
            return
        try:
            world_size = self._all_gather_world_size()
            if world_size == 1:
                if not isinstance(results, list):
                    raise RuntimeError("single-rank all-gather result is malformed")
                for result, param in zip(results, self.params):
                    _wait_event(result.event, _current_stream(result.output.device))
                    result.wait()
                    param.init_unsharded_param()
            else:
                if not isinstance(results, AllGatherResult):
                    raise RuntimeError("multi-rank all-gather result is malformed")
                foreach_all_gather_copy_out(
                    results, self.params, self._all_gather_process_group()
                )
                for param in self.params:
                    param.init_unsharded_param()
            self._sharded_state = ShardedState.UNSHARDED
            self._is_unsharded = True
            copy_out_event = _record_event(_current_stream(self.device))
            if (
                not self._all_gather_async_op
                and self._training_state == TrainingState.FORWARD
                and world_size > 1
            ):
                self.comm_ctx.all_gather_state = AllGatherState(
                    results if isinstance(results, list) else [results],
                    copy_out_event,
                )
            else:
                self._wait_all_gather_streams_on_event(copy_out_event)
        finally:
            self._all_gather_result = None
            self._all_gather_async_op = False

    def _wait_all_gather_streams_on_event(self, event: Any) -> None:
        if event is None:
            return
        _wait_event(event, self.comm_ctx.all_gather_copy_in_stream)
        _wait_event(event, self.comm_ctx.all_gather_stream)

    @_disable_functorch_if_active
    def reshard(self) -> None:
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward_enabled:
                return
            if self._use_post_forward_mesh():
                self._to_sharded_post_forward()
                self._reshard_after_forward_event = _record_event(
                    _current_stream(self.device)
                )
                return
        if self._all_gather_result is not None:
            self.wait_for_unshard()
        if not self._is_unsharded:
            return
        for param in self.params:
            old_local = param._gradient_hook_param
            old_grad = getattr(old_local, "grad", None)
            param.to_sharded()
            from tensorplay.nn.parameter import Parameter

            local = Parameter(
                param._sharded_local_tensor(),
                requires_grad=param.param.requires_grad,
            )
            if old_grad is not None and tuple(old_grad.shape) == tuple(local.shape):
                local.grad = old_grad.detach().clone()
            param.bind_local_param(local)
            param._setattr_on_modules(local)
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED
        self._post_backward_wrapped = False
        self._backward_finalized = False

    def _reset_iter_state(self) -> None:
        current_stream = _current_stream(self.device)
        if self.comm_ctx.all_gather_state is not None:
            _wait_event(self.comm_ctx.all_gather_state.event, current_stream)
            self.comm_ctx.all_gather_state = None
        if self._all_gather_result is not None:
            pending = self._all_gather_result
            if isinstance(pending, list):
                for result in pending:
                    _wait_event(result.event, current_stream)
                    result.wait()
            else:
                _wait_event(pending.event, current_stream)
                pending.wait()
            self._all_gather_result = None
        _wait_event(self._post_reduce_event, current_stream)
        _wait_event(
            getattr(self._all_reduce_state, "event", None), current_stream
        )
        for state in self.comm_ctx.reduce_scatter_states:
            _wait_event(state.event, current_stream)
        for event in self.comm_ctx._last_post_reduce_events.values():
            _wait_event(event, current_stream)
        self.comm_ctx._last_post_reduce_events.clear()
        if self._reshard_after_forward_event is not None:
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None
        self._to_sharded()
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED
        self._backward_finalized = False
        self._post_backward_done = False
        self._post_reduce_event = None
        self._all_reduce_state = None
        for param in self.params:
            event = getattr(param, "grad_offload_event", None)
            if event is not None:
                synchronize = getattr(event, "synchronize", None)
                if callable(synchronize):
                    synchronize()
                param.grad_offload_event = None
        self._post_forward_indices.clear()
        self._training_state = TrainingState.IDLE
        self._partial_reduce_output = None
        self.comm_ctx.reduce_scatter_states.clear()

    def pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        del module
        self.lazy_init()
        self._backward_finalized = False
        self._post_backward_done = False
        entering_forward_pass = self._training_state != TrainingState.FORWARD
        self._training_state = TrainingState.FORWARD
        current_stream = _current_stream(self.device)
        if entering_forward_pass:
            _wait_stream(self.comm_ctx.all_gather_copy_in_stream, current_stream)
            _wait_stream(self.comm_ctx.all_gather_stream, current_stream)
        self.unshard(self.unshard_async_op)
        self.wait_for_unshard()
        for param in self.params:
            param._restore_spmd_types(param.unsharded_param())
        policy = self.mp_policy
        if getattr(policy, "cast_forward_inputs", False):
            dtype = getattr(policy, "param_dtype", None)
            if dtype is not None:
                args = _cast_tree(args, dtype)
                kwargs = _cast_tree(kwargs, dtype)
        if entering_forward_pass:
            args, kwargs = self._register_post_backward_hook(args, kwargs)
        return args, kwargs

    def post_forward(self, module: Any, input: Any, output: Any) -> Any:
        del module, input
        dtype = getattr(self.mp_policy, "output_dtype", None)
        result = _cast_tree(output, dtype) if dtype is not None else output
        if not is_bw():
            if self._reshard_after_forward_enabled:
                if self._use_post_forward_mesh():
                    self._to_sharded_post_forward()
                    self._reshard_after_forward_event = _record_event(
                        _current_stream(self.device)
                    )
                else:
                    self.reshard()
            if self._training_state == TrainingState.FORWARD:
                self._record_post_forward()
        self._training_state = TrainingState.IDLE
        return result

    def finalize_forward(self) -> None:
        state = self.comm_ctx.all_gather_state
        if state is None:
            return
        self._wait_all_gather_streams_on_event(state.event)
        self.comm_ctx.all_gather_state = None

    def _record_post_forward(self) -> None:
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)
        self._post_forward_recorded = True

    @_dynamo_disable
    def pre_backward(self, default_prefetch: Any, *unused: Any) -> None:
        if self._training_state == TrainingState.PRE_BACKWARD:
            return
        del unused
        self._training_state = TrainingState.PRE_BACKWARD
        self.unshard(self.unshard_async_op)
        self.wait_for_unshard()
        if default_prefetch:
            self._backward_prefetch()

    @_dynamo_disable
    def post_backward(self, *unused: Any) -> None:
        del unused
        if self._post_backward_done:
            return
        is_partial_group_backward = (
            len(self.modules) > 1
            and self._training_state == TrainingState.FORWARD
        )
        self._training_state = TrainingState.POST_BACKWARD
        for param in self.params:
            param.accumulate_unsharded_grad_if_needed()
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[Any] = []
        for param in self.params:
            accumulated = param.unsharded_accumulated_grad
            if accumulated is not None:
                fsdp_params_with_grad.append(param)
                unsharded_grads.append(param.unsharded_accumulated_grad_data())
                continue
            gradient = param._unsharded_gradient()
            if gradient is not None:
                fsdp_params_with_grad.append(param)
                unsharded_grads.append(param.unsharded_grad_data())
                continue
            if (
                self.reduce_scatter_unused_params
                and param.param_requires_grad
                and param._unsharded_param is not None
            ):
                fsdp_params_with_grad.append(param)
                unsharded_grads.append(param.unsharded_zero_grad_data())
        if not self.reduce_grads:
            for param in fsdp_params_with_grad:
                if param.unsharded_accumulated_grad is not None:
                    continue
                gradient = param._unsharded_gradient()
                if gradient is None:
                    continue
                if param.reduce_dtype is not None and gradient.dtype != param.reduce_dtype:
                    gradient = gradient.to(dtype=param.reduce_dtype)
                param.unsharded_accumulated_grad = gradient
                if param._unsharded_param is not None:
                    param._unsharded_param.grad = None
                param._unsharded_grad = None
            if self._reshard_after_backward_enabled:
                self.reshard()
            self._post_backward_done = True
            return
        if not fsdp_params_with_grad:
            if self._reshard_after_backward_enabled:
                self.reshard()
            self._post_backward_done = True
            return
        limit = int(self.comm_ctx.reduce_scatter_max_input_buffers)
        if self._param_group_index == self._num_param_groups - 1:
            while len(self.comm_ctx.reduce_scatter_states) >= limit:
                state = self.comm_ctx.reduce_scatter_states.pop(0)
                _wait_event(state.event, _current_stream(self.device))
        for param in fsdp_params_with_grad:
            if param.unsharded_accumulated_grad is not None:
                param.unsharded_accumulated_grad = None
            else:
                if param._unsharded_param is not None:
                    param._unsharded_param.grad = None
                param._unsharded_grad = None
        if self._reshard_after_backward_enabled:
            self.reshard()
        all_reduce_group = self._all_reduce_process_group()
        all_reduce_stream = self.comm_ctx.all_reduce_stream
        if all_reduce_group is None and self._all_reduce_hook_stream is not None:
            if self._all_reduce_hook is None:
                raise RuntimeError("all-reduce hook stream requires an all-reduce hook")
            all_reduce_stream = self._all_reduce_hook_stream
        self._wait_for_post_backward()
        reduce_result = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group(),
            self.comm_ctx.reduce_scatter_stream,
            self._reduce_scatter_comm,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.gradient_divide_factor,
            all_reduce_group,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
            self.force_sum_reduction_for_comms,
            self._comm_hook,
            self._comm_hook_state,
        )
        (
            reduce_scatter_input,
            reduce_scatter_event,
            post_reduce_stream,
            post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            partial_reduce_output,
        ) = reduce_result
        self.comm_ctx.reduce_scatter_states.append(
            ReduceScatterState(reduce_scatter_input, reduce_scatter_event)
        )
        self._post_reduce_event = post_reduce_event
        self.comm_ctx._last_post_reduce_events[post_reduce_stream] = post_reduce_event
        self._partial_reduce_output = partial_reduce_output
        if is_partial_group_backward:
            _wait_event(post_reduce_event, _current_stream(self.device))
        if all_reduce_input is not None:
            device_type = str(getattr(self.device, "type", self.device)).split(":", 1)[0]
            if device_type != "cpu" and all_reduce_event is None:
                raise RuntimeError("all-reduce completion event is unavailable")
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
            )
        self._post_backward_done = True

    def finalize_backward(self) -> None:
        if self._backward_finalized:
            return
        self._backward_finalized = True
        if not self._post_backward_done:
            self.post_backward()
        current_stream = _current_stream(self.device)
        for event in self.comm_ctx._last_post_reduce_events.values():
            _wait_event(event, current_stream)
        self.comm_ctx._last_post_reduce_events.clear()
        self._post_reduce_event = None
        for state in self.comm_ctx.reduce_scatter_states:
            _wait_event(state.event, current_stream)
        self.comm_ctx.reduce_scatter_states.clear()
        _wait_event(getattr(self._all_reduce_state, "event", None), current_stream)
        self._all_reduce_state = None
        for param in self.params:
            event = getattr(param, "grad_offload_event", None)
            if event is not None:
                synchronize = getattr(event, "synchronize", None)
                if callable(synchronize):
                    synchronize()
                param.grad_offload_event = None
        if self._all_gather_result is not None:
            pending = self._all_gather_result
            if isinstance(pending, list):
                for result in pending:
                    _wait_event(result.event, current_stream)
                    result.wait()
            else:
                _wait_event(pending.event, current_stream)
                pending.wait()
            self._all_gather_result = None
        self._post_forward_indices.clear()
        self._partial_reduce_output = None

    def _wait_for_post_backward(self) -> None:
        self.wait_for_unshard()
        current_stream = _current_stream(self.device)
        _wait_event(self._post_reduce_event, current_stream)
        self._post_reduce_event = None
        _wait_event(getattr(self._all_reduce_state, "event", None), current_stream)
        self._all_reduce_state = None

    def _backward_prefetch(self) -> None:
        if not self._post_forward_indices:
            return
        current_index = self._post_forward_indices.pop()
        if self._num_param_groups > 1:
            if self._param_group_index != 1:
                return
            current_modules = self.modules
            target_modules = None
            for step in range(1, current_index + 1):
                target = self.comm_ctx.post_forward_order[current_index - step]
                if target.modules is current_modules:
                    continue
                if target_modules is None:
                    target_modules = target.modules
                elif target.modules is not target_modules:
                    break
                self._prefetch_unshard(target, "backward")
        elif current_index > 0:
            self._prefetch_unshard(
                self.comm_ctx.post_forward_order[current_index - 1], "backward"
            )

    @staticmethod
    def _prefetch_unshard(target_fsdp_param_group: Any, pass_type: Any) -> None:
        if pass_type not in {"forward", "backward"}:
            raise ValueError(f"unknown prefetch pass: {pass_type}")
        state = (
            TrainingState.FORWARD
            if pass_type == "forward"
            else TrainingState.PRE_BACKWARD
        )
        with target_fsdp_param_group.use_training_state(state):
            target_fsdp_param_group.unshard(target_fsdp_param_group.unshard_async_op)

    def _to_sharded(self) -> None:
        if self._sharded_state == ShardedState.SHARDED:
            return
        for param in self.params:
            param.to_sharded()
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED
        self._post_backward_wrapped = False
        self._backward_finalized = False

    def _to_sharded_post_forward(self) -> None:
        if self._all_gather_result is not None:
            self.wait_for_unshard()
        if not self._is_unsharded:
            return
        for param in self.params:
            param.to_sharded_post_forward()
            from tensorplay.nn.parameter import Parameter

            local = Parameter(
                param._sharded_local_tensor(),
                requires_grad=param.param.requires_grad,
            )
            param.bind_local_param(local)
            param._setattr_on_modules(local)
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED_POST_FORWARD
        self._post_backward_wrapped = False
        self._backward_finalized = False

    def _to_unsharded(self) -> None:
        if self._sharded_state == ShardedState.UNSHARDED:
            return
        for param in self.params:
            param.to_unsharded()
        self._is_unsharded = True
        self._sharded_state = ShardedState.UNSHARDED

    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    def is_sharded_post_forward(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED_POST_FORWARD

    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    def use_training_state(self, training_state: Any) -> Any:
        old_training_state = self._training_state
        self._training_state = training_state

        class _TrainingStateGuard:
            def __enter__(guard_self: Any) -> Any:
                return guard_self

            def __exit__(guard_self: Any, exc_type: Any, exc_value: Any, traceback: Any) -> None:
                del guard_self, exc_type, exc_value, traceback
                self._training_state = old_training_state

        return _TrainingStateGuard()

    def _register_post_backward_hook(self, args: Any, kwargs: Any) -> tuple[Any, Any]:
        if not getattr(tp, "is_grad_enabled", lambda: True)():
            return args, kwargs
        input_tensors = collect_grad_tensors((args, kwargs))
        if not input_tensors:
            return args, kwargs
        output_tensors = RegisterPostBackwardFunction.apply(self, *input_tensors)
        if not isinstance(output_tensors, tuple):
            output_tensors = (output_tensors,)
        replaced = replace_grad_tensors((args, kwargs), iter(output_tensors))
        return replaced

    def _install_post_backward_wrappers(self) -> None:
        if self._post_backward_wrapped:
            return
        if not getattr(tp, "is_grad_enabled", lambda: True)():
            return
        for param in self.params:
            source = param.unsharded_param()
            if not getattr(source, "requires_grad", False):
                continue
            wrapped = RegisterPostBackwardFunction.apply(self, source)
            if isinstance(wrapped, tuple):
                wrapped = wrapped[0]
            param._setattr_on_modules(wrapped)
        self._post_backward_wrapped = True

    def _register_state_dict_hooks(self) -> None:
        num_pre_save_hooks = len(self._module_to_pre_save_state_dict_hook_handle)
        num_pre_load_hooks = len(self._module_to_pre_load_state_dict_hook_handle)
        if num_pre_save_hooks != num_pre_load_hooks:
            raise AssertionError(
                f"pre-save hooks={num_pre_save_hooks} pre-load hooks={num_pre_load_hooks}"
            )
        if num_pre_save_hooks > 0:
            self._state_dict_hooks_registered = True
            return
        modules_with_params = {param.module_info.module for param in self.params}

        def to_sharded_hook(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            if getattr(
                getattr(self, "_fsdp_state", None),
                "_summoning_full_params",
                False,
            ):
                return
            self._to_sharded()

        for module in modules_with_params:
            self._module_to_pre_save_state_dict_hook_handle[module] = (
                module.register_state_dict_pre_hook(to_sharded_hook)
            )
            self._module_to_pre_load_state_dict_hook_handle[module] = (
                module._register_load_state_dict_pre_hook(to_sharded_hook)
            )
        self._state_dict_hooks_registered = True

    def _reshard_after_forward(self) -> bool:
        return self._reshard_after_forward_enabled

    def _use_post_forward_mesh(self) -> bool:
        return (
            self.post_forward_mesh_info is not None
            and self.post_forward_mesh_info is not self.mesh_info
        )

    def _is_hsdp(self) -> bool:
        return self.mesh_info.replicate_mesh_dim is not None

    def _all_gather_process_group(self) -> Any:
        mesh_info = self.mesh_info
        if self.is_sharded_post_forward() and self.post_forward_mesh_info is not None:
            mesh_info = self.post_forward_mesh_info
        group = getattr(mesh_info, "shard_process_group", None)
        if group is not None:
            return group
        mesh_dim = mesh_info.shard_mesh_dim
        if mesh_dim is None or mesh_info.shard_world_size <= 1:
            return None
        return mesh_info.mesh.get_group(mesh_dim)

    def _reduce_scatter_process_group(self) -> Any:
        group = getattr(self.mesh_info, "reduce_scatter_process_group", None)
        return group or self._all_gather_process_group()

    def _all_reduce_process_group(self) -> Any:
        group = getattr(self.mesh_info, "replicate_process_group", None)
        if group is not None:
            return group
        if self.mesh_info.replicate_mesh_dim is None:
            return None
        if self.mesh_info.replicate_world_size <= 1:
            return None
        return self.mesh_info.mesh.get_group(self.mesh_info.replicate_mesh_dim)

    def _set_separate_reduce_scatter_group(
        self, enable: bool, new_groups: dict[tuple[int, ...], Any] | None = None
    ) -> None:
        if not isinstance(self.mesh_info, FSDPMeshInfo):
            raise AssertionError(
                f"Expected FSDPMeshInfo, got {type(self.mesh_info).__name__}"
            )
        if not enable:
            self.mesh_info.reduce_scatter_process_group = None
            return
        ranks = tuple(
            dist.get_process_group_ranks(self.mesh_info.shard_process_group)
        )
        cache = new_groups if new_groups is not None else {}
        if ranks not in cache:
            existing = self.mesh_info.reduce_scatter_process_group
            if existing is None:
                existing = dist.new_group(
                    list(ranks),
                    group_desc="fsdp_reduce_scatter",
                )
                if existing == dist.GroupMember.NON_GROUP_MEMBER:
                    raise AssertionError(
                        f"Current rank was not included in process group {ranks}"
                    )
            cache[ranks] = existing
        self.mesh_info.reduce_scatter_process_group = cache[ranks]

    def _with_fqn(self, label: str) -> str:
        return f"{label}[{', '.join(p.module_info.fqn for p in self.params)}]"

    def _validate_no_meta_params(self) -> None:
        if any(getattr(param.param.device, "type", None) == "meta" for param in self.params):
            raise RuntimeError("meta parameters must be materialized before sharding")

    def _validate_cpu_offload_params(self) -> None:
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        invalid = [
            param
            for param in self.params
            if str(
                getattr(
                    getattr(param._sharded_local_tensor(), "device", None),
                    "type",
                    getattr(param._sharded_local_tensor(), "device", None),
                )
            )
            != "cpu"
        ]
        if invalid:
            raise RuntimeError("CPU offload requires sharded parameters on CPU")

    def _validate_reduce_scatter_max_input_buffers(self) -> None:
        limit = int(getattr(self, "reduce_scatter_max_input_buffers", 1))
        if limit <= 0:
            raise ValueError("reduce_scatter_max_input_buffers must be positive")
        if limit > 1 and isinstance(self._reduce_scatter_comm, SymmMemReduceScatter):
            raise ValueError(
                "max_input_buffers greater than one is not supported with the "
                "symmetric-memory reduce-scatter communication path"
            )

    def __repr__(self) -> str:
        return f"FSDPParamGroup(num_params={len(self.params)}, sharded={self.is_sharded()})"


def _cast_tree(value: Any, dtype: Any) -> Any:
    return _apply_to_tensors(lambda tensor: _cast_fp_tensor(dtype, tensor), value)


def _get_param_module_infos(
    params: list[Any], modules: Iterable[Any]
) -> list[Any]:
    params_set = set(params)
    param_to_module_info: dict[Any, Any] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in submodule.named_parameters(
                recurse=False, remove_duplicate=False
            ):
                if param not in params_set:
                    continue
                if param not in param_to_module_info:
                    param_to_module_info[param] = ParamModuleInfo(submodule, param_name)
                else:
                    info = param_to_module_info[param]
                    info.shared_modules.append(submodule)
                    info.shared_param_names.append(param_name)
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {modules}")
    return [param_to_module_info[param] for param in params]


class RegisterPostBackwardFunction(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(param_group: FSDPParamGroup, *inputs: Any) -> tuple[Any, ...]:
        return inputs

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.param_group = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grads: Any) -> tuple[None, ...]:
        ctx.param_group.post_backward()
        return (None,) + tuple(grads)

    @staticmethod
    def jvp(ctx: Any, param_group_tangent: Any, *grad_inputs: Any) -> tuple[Any, ...]:
        del ctx, param_group_tangent
        return grad_inputs
