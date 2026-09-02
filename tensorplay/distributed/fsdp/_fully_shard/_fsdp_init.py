"""Initialization helpers for composable sharding."""

from typing import Any, Iterable

from ...device_mesh import DeviceMesh, init_device_mesh
from ... import distributed_core as dist
from ._fsdp_common import (
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
)
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


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    value = value() if callable(value) else value
    return int(value)


def _mesh_dim_index(mesh: Any, mesh_dim: int | str) -> int:
    if isinstance(mesh_dim, bool):
        raise TypeError("mesh dimension must be an integer or string")
    if isinstance(mesh_dim, str):
        names = getattr(mesh, "mesh_dim_names", None)
        if names is None or mesh_dim not in names:
            raise ValueError(
                f"mesh dimension {mesh_dim!r} is not present in mesh_dim_names"
            )
        return int(names.index(mesh_dim))
    dim = int(mesh_dim)
    ndim = _mesh_ndim(mesh)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("mesh dimension is outside the mesh")
    return dim


def _normalize_dp_dims(mesh: Any, dp_mesh_dims: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shard_names = tuple(getattr(dp_mesh_dims, "shard_names", ()) or ())
    replicate_names = tuple(getattr(dp_mesh_dims, "replicate_names", ()) or ())
    if not shard_names and not replicate_names:
        raise ValueError("at least one data-parallel mesh dimension is required")
    shard_dims = tuple(_mesh_dim_index(mesh, value) for value in shard_names)
    replicate_dims = tuple(_mesh_dim_index(mesh, value) for value in replicate_names)
    if set(shard_dims).intersection(replicate_dims):
        raise ValueError("shard and replicate dimensions must be different")
    if len(set(shard_dims)) != len(shard_dims) or len(set(replicate_dims)) != len(replicate_dims):
        raise ValueError("data-parallel mesh dimensions must be unique")
    return shard_dims, replicate_dims


def _get_mesh_info(mesh: Any, dp_mesh_dims: Any = None) -> DataParallelMeshInfo:
    _validate_mesh(mesh)
    ndim = _mesh_ndim(mesh)
    if dp_mesh_dims is None:
        if ndim == 1:
            return FSDPMeshInfo(mesh, 0)
        if ndim == 2:
            if getattr(mesh, "mesh_dim_names", None) is None:
                raise ValueError("a two-dimensional mesh requires dimension names")
            return HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
        raise ValueError("fully_shard expects a one- or two-dimensional mesh")
    return _get_mesh_info_from_named_dims(mesh, dp_mesh_dims)


def _get_mesh_info_from_named_dims(
    mesh: Any, shard_names: Any, replicate_names: Any = None
) -> DataParallelMeshInfo:
    if hasattr(shard_names, "shard_names") and replicate_names is None:
        dp_mesh_dims = shard_names
    else:
        shard_values = (shard_names,) if isinstance(shard_names, (str, int)) else tuple(shard_names or ())
        replicate_values = (
            (replicate_names,)
            if isinstance(replicate_names, (str, int))
            else tuple(replicate_names or ())
        )

        class Dims:
            @property
            def shard_names(self):
                return shard_values

            @property
            def replicate_names(self):
                return replicate_values

        dp_mesh_dims = Dims()

    _normalize_dp_dims(mesh, dp_mesh_dims)
    shard_names = tuple(dp_mesh_dims.shard_names)
    replicate_names = tuple(dp_mesh_dims.replicate_names)

    def _get_submesh(names: tuple[str, ...]) -> DeviceMesh:
        if len(names) == 1:
            return mesh[names[0]]
        return mesh[names]._flatten("_".join(names))

    if len(shard_names) == 0:
        dp_mesh = _get_submesh(replicate_names)
        return DDPMeshInfo(
            dp_mesh,
            replicate_mesh_dim=0,
            dp_mesh_dims=dp_mesh_dims,
            spmd_mesh=mesh,
        )
    if len(replicate_names) == 0:
        dp_mesh = _get_submesh(shard_names)
        return FSDPMeshInfo(
            dp_mesh,
            shard_mesh_dim=0,
            dp_mesh_dims=dp_mesh_dims,
            spmd_mesh=mesh,
        )

    shard_mesh = _get_submesh(shard_names)
    replicate_mesh = _get_submesh(replicate_names)
    hsdp_mesh = DeviceMesh._concatenate([replicate_mesh, shard_mesh])
    return HSDPMeshInfo(
        hsdp_mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0,
        dp_mesh_dims=dp_mesh_dims,
        spmd_mesh=mesh,
    )


def _flatten_mesh_value(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            result.extend(_flatten_mesh_value(item))
        return result
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _flatten_mesh_value(tolist())
    return [int(value)]


def _get_post_forward_mesh_info(
    reshard_after_forward: Any, mesh_info: DataParallelMeshInfo
) -> DataParallelMeshInfo | None:
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    if isinstance(reshard_after_forward, bool):
        return mesh_info if reshard_after_forward else None
    shard_size = mesh_info.shard_world_size
    if (
        reshard_after_forward < 1
        or reshard_after_forward > shard_size
        or shard_size % reshard_after_forward != 0
    ):
        raise ValueError(
            "If passing reshard_after_forward as an int, it should be a "
            f"factor of {shard_size}, not {reshard_after_forward}"
        )
    if reshard_after_forward == 1:
        return None
    if reshard_after_forward == shard_size:
        return mesh_info
    shard_ranks = _flatten_mesh_value(mesh_info.mesh.mesh)
    if len(shard_ranks) % reshard_after_forward:
        raise ValueError("post-forward shard layout is not rectangular")
    post_shape = (len(shard_ranks) // reshard_after_forward, reshard_after_forward)
    post_mesh = DeviceMesh(
        getattr(mesh_info.mesh, "device_type", "cpu"),
        [
            shard_ranks[offset : offset + post_shape[1]]
            for offset in range(0, len(shard_ranks), post_shape[1])
        ],
        mesh_dim_names=("replicate", "shard"),
        _root_mesh=getattr(mesh_info.mesh, "_root_mesh", None)
        or mesh_info.mesh,
        _backend_override=getattr(mesh_info.mesh, "_backend_override", None),
    )
    return HSDPMeshInfo(
        post_mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0,
    )


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
