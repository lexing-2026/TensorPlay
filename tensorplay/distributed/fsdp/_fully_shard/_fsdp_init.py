"""Initialization helpers for composable sharding."""

import warnings
import itertools
from typing import Any, Iterable

from tensorplay.nn.modules.container import ModuleDict, ModuleList
from tensorplay.nn.modules.module import Module

from ...device_mesh import DeviceMesh, init_device_mesh
from ... import distributed_core as dist
from ...tensor import Replicate
from ...utils import _get_root_modules
from ._fsdp_common import (
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    _is_composable_with_fsdp,
)
from .._common_utils import _get_module_fsdp_state
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


def _validate_module(module: Any, func_name: str = "fully_shard") -> None:
    if not hasattr(module, "named_parameters") or not hasattr(module, "forward"):
        raise TypeError(f"{func_name} expects a module with parameters and forward")
    if isinstance(module, (ModuleList, ModuleDict)) and type(module).forward is Module.forward:
        raise ValueError(
            f"{func_name} does not support containers that do not implement forward: {module}"
        )


def _validate_mesh(mesh: Any, dp_mesh_dims: Any = None) -> None:
    if mesh is None or not hasattr(mesh, "ndim") or not hasattr(mesh, "size"):
        raise TypeError("mesh must provide ndim() and size()")
    if dp_mesh_dims is not None:
        shard_names = tuple(getattr(dp_mesh_dims, "shard_names", ()) or ())
        replicate_names = tuple(getattr(dp_mesh_dims, "replicate_names", ()) or ())
        if not shard_names and not replicate_names:
            raise ValueError(
                "at least one data-parallel mesh dimension is required"
            )
        mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
        if mesh_dim_names is None:
            raise ValueError(
                "mesh dimension names are required when data-parallel dimensions are specified"
            )
        for name in shard_names + replicate_names:
            if name not in mesh_dim_names:
                raise ValueError(
                    f"mesh dimension {name!r} is not present in mesh_dim_names"
                )
        if set(shard_names).intersection(replicate_names):
            raise ValueError("shard and replicate dimensions must be different")
        return
    ndim = _mesh_ndim(mesh)
    if ndim not in (1, 2):
        raise ValueError("fully_shard expects a one- or two-dimensional mesh")
    if ndim == 2 and getattr(mesh, "mesh_dim_names", None) is None:
        raise ValueError("a two-dimensional mesh requires dimension names")


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
    _validate_mesh(mesh, dp_mesh_dims)
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
        warnings.warn(
            "reshard_after_forward=1 uses a world-size-one layout; use True for full sharding",
            stacklevel=2,
        )
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


def _ignore_module(
    module: Any,
    ignored_params: set[Any] | None = None,
    ignore_decision: dict[Any, bool] | None = None,
) -> bool:
    if ignore_decision is None:
        return ignored_params is not None and module in ignored_params
    if module in ignore_decision:
        return ignore_decision[module]
    if any(module.buffers(recurse=False)):
        ignore_decision[module] = False
        return False
    ignored_params = ignored_params or set()
    if any(param not in ignored_params for _, param in module.named_parameters(recurse=False)):
        ignore_decision[module] = False
        return False
    if any(
        not _ignore_module(child, ignored_params, ignore_decision)
        for child in module.children()
    ):
        ignore_decision[module] = False
        return False
    ignore_decision[module] = True
    return True


def _adjust_managed_modules(
    modules: Iterable[Any], ignored_params: set[Any] | None = None
) -> list[Any]:
    modules = list(modules)
    if ignored_params is None:
        return modules
    decisions: dict[Any, bool] = {}
    return [
        module
        for module in modules
        if not _ignore_module(module, ignored_params, decisions)
    ]


def _get_managed_modules(
    root: Any,
    ignored_params: set[Any] | None = None,
    is_composable_fn: Any = None,
    get_state_fn: Any = None,
) -> list[Any]:
    roots = tuple(root) if isinstance(root, (list, tuple)) else (root,)
    root_set = set(roots)
    is_composable_fn = is_composable_fn or _is_composable_with_fsdp
    get_state_fn = get_state_fn or _get_module_fsdp_state
    visited: set[int] = set()
    managed: list[Any] = []

    def dfs(module: Any) -> None:
        if id(module) in visited:
            return
        if not is_composable_fn(module):
            return
        if module not in root_set and get_state_fn(module) is not None:
            return
        visited.add(id(module))
        for child in module.children():
            dfs(child)
        managed.append(module)

    for root_module in roots:
        dfs(root_module)
    return _adjust_managed_modules(managed, ignored_params)


def _verify_managed_param(param: Any, ignored_params: set[Any] | None = None) -> bool:
    if ignored_params is None:
        if len(getattr(param, "shape", ())) == 0:
            raise ValueError(
                "fully_shard does not support scalar parameters; use a one-dimensional parameter"
            )
        return True
    return param not in ignored_params


def _get_managed_states(
    modules: Iterable[Any], ignored_params: set[Any] | None = None
) -> tuple[list[Any], list[Any]]:
    params: list[Any] = []
    buffers: list[Any] = []
    seen_params: set[int] = set()
    seen_buffers: set[int] = set()
    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if id(param) not in seen_params and _verify_managed_param(param, ignored_params):
                if len(getattr(param, "shape", ())) == 0:
                    raise ValueError(
                        f"fully_shard does not support scalar parameter {name!r}"
                    )
                params.append(param)
                seen_params.add(id(param))
        for buffer in module.buffers(recurse=False):
            if id(buffer) not in seen_buffers:
                buffers.append(buffer)
                seen_buffers.add(id(buffer))
    return params, buffers


def _move_states_to_device(
    params_or_modules: Iterable[Any], buffers_or_device: Any, device: Any = None
) -> None:
    if device is None:
        for module in params_or_modules:
            module.to(buffers_or_device)
        return
    for tensor in itertools.chain(params_or_modules, buffers_or_device):
        tensor_device = getattr(tensor, "device", None)
        if tensor_device == device or str(tensor_device) == "meta":
            continue
        tensor.data = tensor.to(device)


def _apply_to_module(module: Any, state: Any) -> Any:
    module._fsdp_state = state
    return module


def _init_param_group(
    state: Any,
    params: list[FSDPParam],
    modules: Iterable[Any],
    mesh_info: DataParallelMeshInfo,
    post_forward_mesh_info: DataParallelMeshInfo | None,
    device: Any,
    shard_placement_fn: Any,
    mp_policy: Any,
    offload_policy: Any,
    reshard_after_forward: Any = True,
) -> None:
    if not params:
        return
    modules = list(modules)
    if shard_placement_fn is None:
        state._fsdp_param_groups.append(
            FSDPParamGroup(
                params,
                modules,
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
        )
        return
    if not isinstance(mesh_info, FSDPMeshInfo):
        if all(isinstance(fsdp_param._placement, Replicate) for fsdp_param in params):
            state._fsdp_param_groups.append(
                FSDPParamGroup(
                    params,
                    modules,
                    mesh_info,
                    post_forward_mesh_info,
                    device,
                    shard_placement_fn,
                    mp_policy,
                    offload_policy,
                )
            )
            return
        raise ValueError("per-parameter placement requires an FSDP mesh")
    grouped: dict[tuple[int, int], tuple[DataParallelMeshInfo, list[FSDPParam]]] = {}
    for fsdp_param in params:
        param_mesh_info = fsdp_param.mesh_info
        if not isinstance(param_mesh_info, FSDPMeshInfo):
            raise ValueError("per-parameter placement must return an FSDP mesh")
        shard_group = getattr(param_mesh_info, "shard_process_group", None)
        replicate_group = (
            getattr(param_mesh_info, "replicate_process_group", None)
            if isinstance(param_mesh_info, HSDPMeshInfo)
            else None
        )
        key = (id(shard_group), id(replicate_group))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = (param_mesh_info, [fsdp_param])
        else:
            existing_mesh_info, existing_params = existing
            if existing_mesh_info is not param_mesh_info:
                raise ValueError(
                    "parameters sharing a process group must share mesh info"
                )
            existing_params.append(fsdp_param)
    for group_mesh_info, group_params in grouped.values():
        group_post_forward_mesh_info = (
            post_forward_mesh_info
            if group_mesh_info is mesh_info
            else _get_post_forward_mesh_info(
                reshard_after_forward, group_mesh_info
            )
        )
        for fsdp_param in group_params:
            fsdp_param.post_forward_mesh_info = group_post_forward_mesh_info
        state._fsdp_param_groups.append(
            FSDPParamGroup(
                group_params,
                modules,
                group_mesh_info,
                group_post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
        )


def _get_modules_and_states(
    module: Any,
    device: Any,
    ignored_params: set[Any] | None,
    is_composable_fn: Any = None,
    get_state_fn: Any = None,
) -> tuple[Any, tuple[Any, ...], list[Any], list[Any], list[Any]]:
    arg_module = module
    if hasattr(module, "named_modules"):
        root_modules = (module,)
    else:
        candidates = list(module)
        if not candidates:
            raise ValueError("fully_shard expects at least one module")
        root_modules = tuple(_get_root_modules(candidates))
    if not root_modules:
        raise ValueError("fully_shard could not find a root module")
    managed_modules = _get_managed_modules(
        root_modules,
        ignored_params,
        is_composable_fn,
        get_state_fn,
    )
    params, buffers = _get_managed_states(managed_modules, ignored_params)
    _move_states_to_device(params, buffers, device)
    return arg_module, root_modules, managed_modules, params, buffers
