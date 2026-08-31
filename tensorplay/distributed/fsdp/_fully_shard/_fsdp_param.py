"""Parameter state transitions for composable sharding."""

from dataclasses import dataclass, field
from typing import Any, Callable

import tensorplay as tp
from tensorplay.nn.parameter import Parameter

from ...tensor import DTensor, Replicate, Shard, distribute_tensor
from ._fsdp_common import DataParallelMeshInfo, resolve_shard_placement

__all__ = [
    "ShardedState",
    "ParamModuleInfo",
    "ExtensionsData",
    "FSDPParam",
    "copy_",
    "copy__functionalize",
    "alloc_storage",
    "free_storage",
    "unsafe_setattr_param",
    "set_requires_grad_if_needed",
]


@dataclass
class ShardedState:
    SHARDED = "sharded"
    UNSHARDED = "unsharded"
    SHARDED_POST_FORWARD = "sharded_post_forward"


@dataclass
class ParamModuleInfo:
    module: Any
    fqn: str
    name: str


@dataclass
class ExtensionsData:
    value: Any = None
    metadata: Any = None

    def clear(self) -> None:
        self.value = None
        self.metadata = None


def _get_orig_param_uid(param: Any) -> int:
    return id(param)


def _spans_same_mesh(lhs_axes: Any, rhs_axes: Any) -> bool:
    return tuple(lhs_axes) == tuple(rhs_axes)


def copy_(tensor: Any, data: Any) -> Any:
    tensor.copy_(data)
    return tensor


def copy__functionalize(tensor: Any, data: Any) -> Any:
    return copy_(tensor, data)


class FSDPParam:
    """Owns one logical parameter and its local sharded representation."""

    def __init__(
        self,
        param: Any,
        module_info: ParamModuleInfo,
        mesh_info: DataParallelMeshInfo,
        post_forward_mesh_info: DataParallelMeshInfo | None = None,
        device: Any = None,
        shard_placement_fn: Callable[[Any], Any] | None = None,
        mp_policy: Any = None,
        offload_policy: Any = None,
    ) -> None:
        self.param = param
        self.module_info = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info or mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self._state = ShardedState.UNSHARDED
        self._full_tensor = param
        self._sharded_tensor: DTensor | None = None
        self._sharded_grad: Any = None
        self._extensions = ExtensionsData()
        self._placement, self._mesh = resolve_shard_placement(shard_placement_fn(param) if shard_placement_fn else None, mesh_info)
        self._init_sharded_param(param, device, shard_placement_fn, mesh_info)

    def _init_sharded_param(self, param: Any, device: Any, shard_placement_fn: Any, mesh_info: DataParallelMeshInfo) -> None:
        del device, shard_placement_fn
        value = param.detach()
        self._sharded_tensor = distribute_tensor(value, mesh_info.mesh, [self._placement])
        self._state = ShardedState.SHARDED

    def _resolve_spmd_types_for_storage(self, param: Any, partition_spec: Any, init_local_type: Any, mesh_info: Any) -> Any:
        del param, partition_spec, mesh_info
        return init_local_type

    def _restore_spmd_types(self, tensor: Any) -> Any:
        return tensor

    def _init_sharding_spec(self, param: Any, fsdp_placement: Any, shard_dim: int) -> Any:
        del param, shard_dim
        return fsdp_placement

    _init_sharding_spec_spmd = _init_sharding_spec
    _init_sharding_spec_tp = _init_sharding_spec
    _init_sharding_spec_plain = _init_sharding_spec

    def _build_spmd_sharding_spec(self, dp_dim_names: Any, dp_shard_indices: Any, fsdp_placement: Any) -> tuple[Any, ...]:
        del dp_dim_names, dp_shard_indices
        return (fsdp_placement,)

    def _init_sharded_post_forward_param_metadata(self, param: Any) -> None:
        self._post_forward_shape = tuple(param.shape)

    def init_dtype_attrs(self, mp_policy: Any) -> None:
        self.orig_dtype = self.param.dtype
        self._compute_dtype = getattr(mp_policy, "param_dtype", None) or self.orig_dtype
        self._reduce_dtype = getattr(mp_policy, "reduce_dtype", None) or self._compute_dtype

    def _init_extensions(self) -> None:
        self._extensions = ExtensionsData()

    def init_all_gather_outputs(self, all_gather_input_numels: Any, all_gather_input_dtypes: Any, world_size: int, device: Any) -> None:
        del all_gather_input_dtypes
        self._all_gather_output = tp.empty(sum(all_gather_input_numels) * world_size, device=device, dtype=self.param.dtype)

    def init_unsharded_param(self) -> Any:
        self.to_unsharded()
        return self._full_tensor

    def _release_all_gather_outputs_if_needed(self) -> None:
        if hasattr(self, "_all_gather_output"):
            del self._all_gather_output

    def _get_unsharded_dtensor_spec(self, unsharded_param: Any) -> Any:
        del unsharded_param
        return None

    def _unflatten_all_gather_outputs(self) -> Any:
        return self._full_tensor

    def to_sharded(self) -> None:
        if self._state == ShardedState.SHARDED:
            return
        self._sharded_tensor = distribute_tensor(self._full_tensor.detach(), self.mesh_info.mesh, [self._placement])
        self._state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        self.to_sharded()
        self._state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        if self._sharded_tensor is None:
            self._sharded_tensor = distribute_tensor(self._full_tensor.detach(), self.mesh_info.mesh, [self._placement])
        self._full_tensor = self._sharded_tensor.full_tensor()
        if self._compute_dtype != getattr(self._full_tensor, "dtype", None):
            self._full_tensor = self._full_tensor.to(dtype=self._compute_dtype)
        self._state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: Any) -> None:
        unsafe_setattr_param(self.module_info.module, self.module_info.name, param)

    def to_sharded_dtensor(self, tensor: Any) -> DTensor:
        return tensor if isinstance(tensor, DTensor) else distribute_tensor(tensor, self.mesh_info.mesh, [self._placement])

    def to_sharded_post_forward_dtensor(self, tensor: Any) -> DTensor:
        return self.to_sharded_dtensor(tensor)

    def to_accumulated_grad_if_needed(self) -> Any:
        return self._sharded_grad

    def accumulate_unsharded_grad_if_needed(self) -> Any:
        return self._sharded_grad

    def alloc_all_gather_outputs(self) -> None:
        self._all_gather_output = self._sharded_tensor.to_local().clone() if self._sharded_tensor is not None else None

    def free_all_gather_outputs(self) -> None:
        self._release_all_gather_outputs_if_needed()

    def free_unsharded_param(self) -> None:
        self.to_sharded()

    def all_gather_inputs(self) -> tuple[Any, ...]:
        return (self._sharded_local_tensor(),)

    def unsharded_param(self) -> Any:
        if self._state != ShardedState.UNSHARDED:
            self.to_unsharded()
        return self._full_tensor

    def unsharded_grad_data(self) -> Any:
        return getattr(self._full_tensor, "grad", None)

    def unsharded_accumulated_grad_data(self) -> Any:
        return self._sharded_grad

    def unsharded_zero_grad_data(self) -> Any:
        return tp.zeros_like(self._full_tensor)

    def _get_grad_inner_tensor(self, grad: Any) -> Any:
        return grad.to_local() if isinstance(grad, DTensor) else grad

    def _sharded_local_tensor(self) -> Any:
        if self._sharded_tensor is None:
            self.to_sharded()
        return self._sharded_tensor.to_local()

    def _init_shard_mesh(self) -> None:
        self._shard_mesh = self.mesh_info.mesh

    def shard_mesh(self) -> Any:
        return getattr(self, "_shard_mesh", self.mesh_info.mesh)

    def shard_mesh_from_root(self, root_mesh: Any) -> Any:
        return root_mesh

    def _assert_in_states(self, *states: Any) -> None:
        if self._state not in states:
            raise RuntimeError(f"parameter state {self._state!r} is not one of {states!r}")

    def reset_sharded_param(self) -> None:
        self._sharded_tensor = None
        self._state = ShardedState.UNSHARDED

    def _use_unsharded_tensor(self, tensor: Any) -> None:
        self._full_tensor = tensor.reshape(self.param.shape)
        self._state = ShardedState.UNSHARDED

    def _set_sharded_grad(self, grad: Any) -> None:
        self._sharded_grad = grad

    def __repr__(self) -> str:
        return f"FSDPParam(fqn={self.module_info.fqn!r}, state={self._state!r})"


def alloc_storage(tensor: Any) -> Any:
    return tensor


def free_storage(tensor: Any) -> None:
    del tensor


def unsafe_setattr_param(module: Any, param_name: str, param: Any) -> None:
    module._parameters[param_name] = param


def set_requires_grad_if_needed(src_tensor: Any, dst_tensor: Any) -> None:
    dst_tensor.requires_grad = getattr(src_tensor, "requires_grad", False)
