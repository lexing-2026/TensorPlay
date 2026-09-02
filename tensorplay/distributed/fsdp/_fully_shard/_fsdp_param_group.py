"""Grouped parameter state transitions and communication settings."""

from dataclasses import dataclass, field
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.autograd import Function

from ._fsdp_api import CPUOffloadPolicy
from ._fsdp_collectives import (
    AllGatherResult,
    DefaultAllGather,
    DefaultReduceScatter,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from ._fsdp_common import DataParallelMeshInfo, FSDPMeshInfo, TrainingState
from ._fsdp_param import FSDPParam, ShardedState

__all__ = [
    "FSDPCommContext",
    "AllGatherState",
    "ReduceScatterState",
    "AllReduceState",
    "FSDPParamGroup",
    "RegisterPostBackwardFunction",
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
        self.post_forward_order: list[Any] = []
        self.reduce_scatter_max_input_buffers = 1

    def lazy_init(self, device: Any) -> None:
        self.device = device
        self.device_handle = getattr(tp, "cuda", None)
        self.initialized = True

    def get_all_gather_streams(self, async_op: bool, training_state: Any) -> tuple[Any, Any]:
        del training_state
        if not async_op or self.device is None:
            return None, None
        device_type = getattr(self.device, "type", str(self.device).split(":", 1)[0])
        if str(device_type) != "cuda" or not getattr(tp, "cuda", None) or not tp.cuda.is_available():
            return None, None
        try:
            return tp.cuda.Stream(device=self.device), tp.cuda.Stream(device=self.device)
        except (RuntimeError, TypeError):
            return None, None


@dataclass
class AllGatherState:
    results: list[Any] = field(default_factory=list)
    event: Any = None


@dataclass
class ReduceScatterState:
    results: list[Any] = field(default_factory=list)
    event: Any = None


@dataclass
class AllReduceState:
    results: list[Any] = field(default_factory=list)
    event: Any = None


class FSDPParamGroup:
    def __init__(self, params: Iterable[FSDPParam], modules: Iterable[Any], mesh_info: DataParallelMeshInfo, post_forward_mesh_info: DataParallelMeshInfo | None, device: Any, shard_placement_fn: Any, mp_policy: Any, offload_policy: Any) -> None:
        del shard_placement_fn
        self.params = list(params)
        self.fsdp_params = self.params
        self.modules = list(modules)
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.comm_ctx = FSDPCommContext()
        self._reshard_after_forward_enabled = True
        self._reshard_after_backward_enabled = True
        self._requires_gradient_sync = True
        self._requires_all_reduce = True
        self.reduce_scatter_max_input_buffers = 1
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED
        self._backward_finalized = False
        self._all_gather_result: list[AllGatherResult] | None = None
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
        self._all_reduce_state = None
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

    def lazy_init(self) -> None:
        if not self.comm_ctx.initialized:
            self.comm_ctx.lazy_init(self.device)
        if not self._reset_sharded_params and self.is_sharded():
            for param in self.params:
                param.reset_sharded_param()
                param._init_extensions()
            self._reset_sharded_params = True
        self._init_mp_dtypes()
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._validate_reduce_scatter_max_input_buffers()

    def set_symm_mem(self, backend: Any) -> None:
        self._symm_mem_backend = backend

    def set_allocate_memory_from_process_group(self, enable: bool) -> None:
        self._allocate_from_process_group = bool(enable)

    def _unshard_impl(self) -> None:
        self.unshard(async_op=False)
        self.wait_for_unshard()

    def _all_gather_process_group(self) -> Any:
        mesh_info = self.mesh_info
        if self.is_sharded_post_forward() and self.post_forward_mesh_info is not None:
            mesh_info = self.post_forward_mesh_info
        mesh_dim = mesh_info.shard_mesh_dim
        if mesh_dim is None or mesh_info.shard_world_size <= 1:
            return None
        return getattr(mesh_info, "shard_process_group", None) or mesh_info.mesh.get_group(mesh_dim)

    def _all_gather_world_size(self) -> int:
        mesh_dim = self._active_mesh_info().shard_mesh_dim
        if mesh_dim is None:
            return 1
        return int(self._active_mesh_info().mesh.size(mesh_dim))

    def _active_mesh_info(self) -> DataParallelMeshInfo:
        if self.is_sharded_post_forward() and self.post_forward_mesh_info is not None:
            return self.post_forward_mesh_info
        return self.mesh_info

    def unshard(self, async_op: bool = False) -> None:
        if self._is_unsharded:
            return
        if self._all_gather_result is not None:
            if not async_op:
                self.wait_for_unshard()
            return
        self.lazy_init()
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
                output = param.all_gather_outputs[0]
                output.copy_(value)
                results.append(AllGatherResult(output, None))
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
        if not async_op:
            self.wait_for_unshard()

    def wait_for_unshard(self) -> None:
        results = self._all_gather_result
        if results is None:
            return
        try:
            foreach_all_gather_copy_out(
                results, self.params, self._all_gather_process_group()
            )
            self._sharded_state = ShardedState.UNSHARDED
            self._is_unsharded = True
        finally:
            self._all_gather_result = None

    def _wait_all_gather_streams_on_event(self, event: Any) -> None:
        wait = getattr(event, "wait", None)
        if wait is not None:
            wait()

    def reshard(self) -> None:
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
        self._is_unsharded = False
        self._sharded_state = ShardedState.SHARDED
        self._backward_finalized = False

    def pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        del module
        self._backward_finalized = False
        self._training_state = TrainingState.FORWARD
        self.unshard(self.unshard_async_op)
        self.wait_for_unshard()
        self._install_post_backward_wrappers()
        policy = self.mp_policy
        if getattr(policy, "cast_forward_inputs", False):
            dtype = getattr(policy, "param_dtype", None)
            if dtype is not None:
                args = _cast_tree(args, dtype)
                kwargs = _cast_tree(kwargs, dtype)
        return self._register_post_backward_hook(args, kwargs)

    def post_forward(self, module: Any, input: Any, output: Any) -> Any:
        del module, input
        dtype = getattr(self.mp_policy, "output_dtype", None)
        result = _cast_tree(output, dtype) if dtype is not None else output
        if self._reshard_after_forward_enabled:
            if self._use_post_forward_mesh():
                self._to_sharded_post_forward()
            else:
                self.reshard()
        self._training_state = TrainingState.IDLE
        return result

    def _record_post_forward(self) -> None:
        self._post_forward_recorded = True

    def pre_backward(self, default_prefetch: Any, *unused: Any) -> None:
        del default_prefetch, unused
        self._training_state = TrainingState.PRE_BACKWARD
        self.unshard(self.unshard_async_op)
        self.wait_for_unshard()

    def post_backward(self, *unused: Any) -> None:
        del unused
        self._training_state = TrainingState.POST_BACKWARD
        if self._reshard_after_backward_enabled:
            self.reshard()

    def finalize_backward(self) -> None:
        if self._backward_finalized:
            return
        self._backward_finalized = True
        self.post_backward()

    def _wait_for_post_backward(self) -> None:
        self.wait_for_unshard()

    def _backward_prefetch(self) -> None:
        self.wait_for_unshard()

    def _prefetch_unshard(self, target_fsdp_param_group: Any, pass_type: Any) -> None:
        del pass_type
        target_fsdp_param_group.unshard()

    def _to_sharded(self) -> None:
        self.reshard()

    def _to_sharded_post_forward(self) -> None:
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
        self.unshard()

    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    def is_sharded_post_forward(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED_POST_FORWARD

    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    def use_training_state(self, training_state: Any) -> Any:
        self._training_state = training_state
        return self

    def _register_post_backward_hook(self, args: Any, kwargs: Any) -> tuple[Any, Any]:
        values = [
            value
            for value in _iter_tensors((args, kwargs))
            if getattr(value, "requires_grad", False)
        ]
        if not values:
            return args, kwargs
        wrapped = RegisterPostBackwardFunction.apply(self, *values)
        if not isinstance(wrapped, tuple):
            wrapped = (wrapped,)
        iterator = iter(wrapped)

        def replace(value: Any) -> Any:
            if isinstance(value, tp.Tensor):
                if getattr(value, "requires_grad", False):
                    return next(iterator)
                return value
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        return replace(args), replace(kwargs)

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

    def _set_separate_reduce_scatter_group(self, enable: bool, new_groups: Any = None) -> None:
        self._separate_reduce_scatter = bool(enable)
        self._new_reduce_scatter_groups = new_groups

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
            if str(getattr(param.param.device, "type", param.param.device)) != "cpu"
        ]
        if invalid:
            raise RuntimeError("CPU offload requires sharded parameters on CPU")

    def _validate_reduce_scatter_max_input_buffers(self) -> None:
        limit = int(getattr(self, "reduce_scatter_max_input_buffers", 1))
        if limit <= 0:
            raise ValueError("reduce_scatter_max_input_buffers must be positive")

    def __repr__(self) -> str:
        return f"FSDPParamGroup(num_params={len(self.params)}, sharded={self.is_sharded()})"


def _cast_tree(value: Any, dtype: Any) -> Any:
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_cast_tree(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_tree(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_tree(item, dtype) for key, item in value.items()}
    return value


def _iter_tensors(value: Any):
    if isinstance(value, tp.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


class RegisterPostBackwardFunction(Function):
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
