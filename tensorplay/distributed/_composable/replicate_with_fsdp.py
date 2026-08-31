from __future__ import annotations

from typing import Any

import tensorplay.nn as nn

from tensorplay.distributed._composable.contract import _get_registry
from tensorplay.distributed._composable.replicate import replicate as _replicate
from tensorplay.distributed.device_mesh import DeviceMesh, init_device_mesh
import tensorplay.distributed as dist

__all__ = ["replicate", "ReplicateModule", "is_composable_with_replicate"]


class _ReplicateStateContext:
    def __init__(self) -> None:
        self.states: list[Any] = []


class _ReplicateState:
    _state_name = "Replicate"

    def __init__(self) -> None:
        self._state_ctx = _ReplicateStateContext()
        self.modules: tuple[nn.Module, ...] = ()
        self.mesh: DeviceMesh | None = None

    def init(self, modules: tuple[nn.Module, ...], device: Any = None, mp_policy: Any = None, auto_reshard_after_forward: bool = False) -> None:
        del device, mp_policy, auto_reshard_after_forward
        self.modules = modules

    def _get_state_for_module(self, module: nn.Module) -> Any:
        return _get_module_replicate_state(module)


def _get_module_replicate_state(module: nn.Module) -> _ReplicateState | None:
    value = _get_registry(module).get("__replicate_with_fsdp_state__")
    return value if isinstance(value, _ReplicateState) else None


class ReplicateModule:
    _orig_cls_mro_index = 1


def _validate_module(module: nn.Module) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError("replicate expects a module")
    if not is_composable_with_replicate(module):
        raise RuntimeError("replicate cannot be applied after fully_shard")


def _validate_mesh(mesh: DeviceMesh) -> None:
    ndim = mesh.ndim() if callable(getattr(mesh, "ndim", None)) else mesh.ndim
    if ndim != 1:
        raise ValueError("replicate expects a one-dimensional device mesh")


def is_composable_with_replicate(module: nn.Module) -> bool:
    registry = _get_registry(module)
    return "fully_shard" not in registry


def replicate(module: nn.Module | list[nn.Module], *, mesh: DeviceMesh | None = None, mp_policy: Any = None, offload_policy: Any = None, ignored_params: set[Any] | None = None, dp_mesh_dims: Any = None):
    del mp_policy, offload_policy, ignored_params, dp_mesh_dims
    modules = module if isinstance(module, list) else [module]
    for item in modules:
        _validate_module(item)
    if mesh is not None:
        active_mesh = mesh
    elif dist.is_initialized():
        active_mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("replicate",))
    else:
        active_mesh = DeviceMesh("cpu", [0], mesh_dim_names=("replicate",))
    _validate_mesh(active_mesh)
    result = []
    for item in modules:
        state = _ReplicateState()
        state.mesh = active_mesh
        state.init((item,))
        _get_registry(item)["__replicate_with_fsdp_state__"] = state
        result.append(_replicate(item))
    return result if isinstance(module, list) else result[0]
