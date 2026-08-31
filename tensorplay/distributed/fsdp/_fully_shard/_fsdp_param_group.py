"""Grouped parameter state transitions and communication settings."""

from dataclasses import dataclass, field
from typing import Any, Iterable

from ._fsdp_common import DataParallelMeshInfo
from ._fsdp_param import FSDPParam

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

    def lazy_init(self, device: Any) -> None:
        self.device = device
        self.initialized = True

    def get_all_gather_streams(self, async_op: bool, training_state: Any) -> tuple[Any, Any]:
        del async_op, training_state
        return None, None


@dataclass
class AllGatherState:
    results: list[Any] = field(default_factory=list)


@dataclass
class ReduceScatterState:
    results: list[Any] = field(default_factory=list)


@dataclass
class AllReduceState:
    results: list[Any] = field(default_factory=list)


class FSDPParamGroup:
    def __init__(self, params: Iterable[FSDPParam], modules: Iterable[Any], mesh_info: DataParallelMeshInfo, post_forward_mesh_info: DataParallelMeshInfo | None, device: Any, shard_placement_fn: Any, mp_policy: Any, offload_policy: Any) -> None:
        del shard_placement_fn
        self.params = list(params)
        self.modules = list(modules)
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info or mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.comm_ctx = FSDPCommContext()
        self._reshard_after_forward_enabled = True
        self._requires_gradient_sync = True
        self._requires_all_reduce = True
        self._is_unsharded = False

    def _init_mp_dtypes(self) -> None:
        for param in self.params:
            param.init_dtype_attrs(self.mp_policy)

    def lazy_init(self) -> None:
        if not self.comm_ctx.initialized:
            self.comm_ctx.lazy_init(self.device)
        self._init_mp_dtypes()

    def set_symm_mem(self, backend: Any) -> None:
        self._symm_mem_backend = backend

    def set_allocate_memory_from_process_group(self, enable: bool) -> None:
        self._allocate_from_process_group = bool(enable)

    def unshard(self, async_op: bool = False) -> None:
        del async_op
        self.lazy_init()
        for param in self.params:
            param.to_unsharded()
            param._setattr_on_modules(param.unsharded_param())
        self._is_unsharded = True

    def wait_for_unshard(self) -> None:
        return None

    def _wait_all_gather_streams_on_event(self, event: Any) -> None:
        wait = getattr(event, "wait", None)
        if wait is not None:
            wait()

    def reshard(self) -> None:
        for param in self.params:
            param.to_sharded()
            from tensorplay.nn.parameter import Parameter

            local = Parameter(
                param._sharded_local_tensor(),
                requires_grad=param.param.requires_grad,
            )
            param._setattr_on_modules(local)
        self._is_unsharded = False

    def _reset_iter_state(self) -> None:
        self._is_unsharded = False

    def pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        del module
        self.unshard()
        policy = self.mp_policy
        if getattr(policy, "cast_forward_inputs", False):
            dtype = getattr(policy, "param_dtype", None)
            if dtype is not None:
                args = _cast_tree(args, dtype)
                kwargs = _cast_tree(kwargs, dtype)
        return args, kwargs

    def post_forward(self, module: Any, input: Any, output: Any) -> Any:
        del module, input
        dtype = getattr(self.mp_policy, "output_dtype", None)
        return _cast_tree(output, dtype) if dtype is not None else output

    def _record_post_forward(self) -> None:
        return None

    def pre_backward(self, default_prefetch: Any, *unused: Any) -> None:
        del default_prefetch, unused
        self.unshard()

    def post_backward(self, *unused: Any) -> None:
        del unused
        if self._reshard_after_forward:
            self.reshard()

    def finalize_backward(self) -> None:
        return None

    def _wait_for_post_backward(self) -> None:
        return None

    def _backward_prefetch(self) -> None:
        return None

    def _prefetch_unshard(self, target_fsdp_param_group: Any, pass_type: Any) -> None:
        del pass_type
        target_fsdp_param_group.unshard()

    def _to_sharded(self) -> None:
        self.reshard()

    _to_sharded_post_forward = _to_sharded

    def _to_unsharded(self) -> None:
        self.unshard()

    def is_sharded(self) -> bool:
        return not self._is_unsharded

    def is_sharded_post_forward(self) -> bool:
        return self.is_sharded()

    def is_unsharded(self) -> bool:
        return self._is_unsharded

    def use_training_state(self, training_state: Any) -> Any:
        self.training_state = training_state
        return training_state

    def _register_post_backward_hook(self, args: Any, kwargs: Any) -> None:
        del args, kwargs

    def _register_state_dict_hooks(self) -> None:
        return None

    def _reshard_after_forward(self) -> bool:
        return self._reshard_after_forward_enabled

    def _use_post_forward_mesh(self) -> bool:
        return self.post_forward_mesh_info is not self.mesh_info

    def _is_hsdp(self) -> bool:
        return self.mesh_info.replicate_mesh_dim is not None

    def _all_gather_process_group(self) -> Any:
        return self.mesh_info.mesh.get_group(self.mesh_info.shard_mesh_dim)

    _reduce_scatter_process_group = _all_gather_process_group
    _all_reduce_process_group = _all_gather_process_group

    def _set_separate_reduce_scatter_group(self, enable: bool, new_groups: Any = None) -> None:
        self._separate_reduce_scatter = bool(enable)
        self._new_reduce_scatter_groups = new_groups

    def _with_fqn(self, label: str) -> str:
        return f"{label}[{', '.join(p.module_info.fqn for p in self.params)}]"

    def _validate_no_meta_params(self) -> None:
        if any(getattr(param.param.device, "type", None) == "meta" for param in self.params):
            raise RuntimeError("meta parameters must be materialized before sharding")

    def _validate_cpu_offload_params(self) -> None:
        return None

    def _validate_reduce_scatter_max_input_buffers(self) -> None:
        return None

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


class RegisterPostBackwardFunction:
    @staticmethod
    def forward(param_group: FSDPParamGroup, *inputs: Any) -> tuple[Any, ...]:
        return inputs

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.param_group = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grads: Any) -> tuple[None, ...]:
        ctx.param_group.post_backward()
        return (None,) * (len(grads) + 1)

    @staticmethod
    def jvp(ctx: Any, param_group_tangent: Any, *grad_inputs: Any) -> tuple[Any, ...]:
        del ctx, param_group_tangent
        return grad_inputs
