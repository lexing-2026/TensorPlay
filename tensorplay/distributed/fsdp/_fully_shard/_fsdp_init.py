"""Initialization helpers for composable sharding."""

from typing import Any, Iterable

from ...device_mesh import DeviceMesh, init_device_mesh
from ... import distributed_core as dist
from ._fsdp_common import DataParallelMeshInfo, FSDPMeshInfo
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPParamGroup

__all__ = [
    "_validate_module",
    "_validate_mesh",
    "_get_mesh_info",
    "_get_mesh_info_from_named_dims",
    "_get_post_forward_mesh_info",
    "_init_default_mesh",
    "_init_default_fully_shard_mesh",
    "_get_device_from_mesh",
    "_ignore_module",
    "_adjust_managed_modules",
    "_get_managed_modules",
    "_verify_managed_param",
    "_get_managed_states",
    "_move_states_to_device",
    "_apply_to_module",
    "_init_param_group",
    "_get_modules_and_states",
]


def _validate_module(module: Any) -> None:
    if not hasattr(module, "named_parameters") or not hasattr(module, "forward"):
        raise TypeError("fully_shard expects a module with parameters and forward")


def _validate_mesh(mesh: Any) -> None:
    if mesh is None or not hasattr(mesh, "ndim") or not hasattr(mesh, "size"):
        raise TypeError("mesh must provide ndim() and size()")


def _get_mesh_info(mesh: Any, dp_mesh_dims: Any = None) -> DataParallelMeshInfo:
    _validate_mesh(mesh)
    if dp_mesh_dims is None:
        return FSDPMeshInfo(mesh, 0)
    shard_dim = dp_mesh_dims.shard_names[0] if dp_mesh_dims.shard_names else 0
    replicate_dim = dp_mesh_dims.replicate_names[0] if dp_mesh_dims.replicate_names else None
    return FSDPMeshInfo(mesh, shard_dim, replicate_dim)


def _get_mesh_info_from_named_dims(mesh: Any, shard_names: Any, replicate_names: Any = None) -> DataParallelMeshInfo:
    class Dims:
        shard = tuple(shard_names) if not isinstance(shard_names, str) else shard_names
        replicate = tuple(replicate_names) if replicate_names else None
        @property
        def shard_names(self):
            return (self.shard,) if isinstance(self.shard, str) else self.shard
        @property
        def replicate_names(self):
            if self.replicate is None:
                return ()
            return (self.replicate,) if isinstance(self.replicate, str) else self.replicate
    return _get_mesh_info(mesh, Dims())


def _get_post_forward_mesh_info(mesh_info: DataParallelMeshInfo, reshard_after_forward: Any) -> DataParallelMeshInfo:
    del reshard_after_forward
    return mesh_info


def _init_default_mesh(device_type: str = "cpu") -> Any:
    if dist.is_initialized():
        return init_device_mesh(device_type, (dist.get_world_size(),))
    return DeviceMesh(device_type, [0])


def _init_default_fully_shard_mesh(device_type: str = "cpu") -> Any:
    return _init_default_mesh(device_type)


def _get_device_from_mesh(mesh: Any) -> Any:
    return getattr(mesh, "device_type", "cpu")


def _ignore_module(module: Any, ignored_modules: set[Any] | None = None) -> bool:
    return ignored_modules is not None and module in ignored_modules


def _adjust_managed_modules(modules: Iterable[Any]) -> list[Any]:
    return list(modules)


def _get_managed_modules(root: Any, ignored_modules: set[Any] | None = None) -> list[Any]:
    return [module for module in root.modules() if not _ignore_module(module, ignored_modules)]


def _verify_managed_param(param: Any, ignored_params: set[Any] | None = None) -> bool:
    return param not in (ignored_params or set())


def _get_managed_states(modules: Iterable[Any], ignored_params: set[Any] | None = None) -> list[tuple[Any, str, Any]]:
    result = []
    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if _verify_managed_param(param, ignored_params):
                result.append((module, name, param))
    return result


def _move_states_to_device(modules: Iterable[Any], device: Any) -> None:
    for module in modules:
        module.to(device)


def _apply_to_module(module: Any, state: Any) -> Any:
    module._fsdp_state = state
    return module


def _init_param_group(modules: Iterable[Any], mesh_info: DataParallelMeshInfo, device: Any, mp_policy: Any, offload_policy: Any, shard_placement_fn: Any = None) -> FSDPParamGroup:
    params = [FSDPParam(param, type("Info", (), {"module": module, "fqn": name, "name": name})(), mesh_info, device=device, shard_placement_fn=shard_placement_fn, mp_policy=mp_policy, offload_policy=offload_policy) for module in modules for name, param in module.named_parameters(recurse=False)]
    return FSDPParamGroup(params, modules, mesh_info, None, device, shard_placement_fn, mp_policy, offload_policy)


def _get_modules_and_states(module: Any) -> tuple[list[Any], list[Any]]:
    modules = list(module.modules())
    return modules, [getattr(item, "_fsdp_state", None) for item in modules]
