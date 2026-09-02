from __future__ import annotations

from typing import Any

import tensorplay.nn as nn

from tensorplay.distributed._composable.contract import _get_registry
from tensorplay.distributed._composable.replicate import replicate as _replicate
from tensorplay.distributed.device_mesh import DeviceMesh, init_device_mesh
from tensorplay.distributed.fsdp._fully_shard._fsdp_api import (
    MixedPrecisionPolicy,
    OffloadPolicy,
)
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

    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: Any = None,
        mp_policy: MixedPrecisionPolicy | None = None,
        auto_reshard_after_forward: bool = False,
        offload_policy: OffloadPolicy | None = None,
        ignored_params: set[Any] | None = None,
    ) -> None:
        self.modules = modules
        self.device = device
        self.mp_policy = mp_policy or MixedPrecisionPolicy()
        self.auto_reshard_after_forward = bool(auto_reshard_after_forward)
        self.offload_policy = offload_policy or OffloadPolicy()
        self.ignored_params = set(ignored_params or ())
        self._ddp = None
        self._legacy_state = None
        self._pending_requires_gradient_sync = True

    def set_ddp(self, ddp: Any) -> None:
        self._ddp = ddp

    def set_legacy_state(self, state: Any) -> None:
        self._legacy_state = state

    def set_requires_gradient_sync(self, value: bool) -> None:
        value = bool(value)
        self._pending_requires_gradient_sync = value
        if self._ddp is not None:
            self._ddp.require_backward_grad_sync = value
        if self._legacy_state is not None:
            self._legacy_state._no_sync = not value

    def register_comm_hook(self, *args: Any, **kwargs: Any) -> None:
        if self._ddp is not None:
            self._ddp.register_comm_hook(*args, **kwargs)
        elif self._legacy_state is not None:
            self._legacy_state._comm_hook_args.append((args, kwargs))
        else:
            raise RuntimeError("replicate state has not been initialized")

    def _get_state_for_module(self, module: nn.Module) -> Any:
        return _get_module_replicate_state(module)


def _get_module_replicate_state(module: nn.Module) -> _ReplicateState | None:
    value = _get_registry(module).get("__replicate_with_fsdp_state__")
    return value if isinstance(value, _ReplicateState) else None


class ReplicateModule:
    _orig_cls_mro_index = 2

    def _get_replicate_state(self) -> _ReplicateState:
        state = _get_module_replicate_state(self)
        if state is None:
            raise RuntimeError("module is not managed by replicate")
        return state

    def set_requires_gradient_sync(self, value: bool) -> None:
        self._get_replicate_state().set_requires_gradient_sync(value)

    def set_requires_all_reduce(self, value: bool, recurse: bool = True) -> None:
        del recurse
        self.set_requires_gradient_sync(value)

    def register_comm_hook(self, *args: Any, **kwargs: Any) -> None:
        self._get_replicate_state().register_comm_hook(*args, **kwargs)

    def reshard(self) -> None:
        return None

    def unshard(self, async_op: bool = False) -> Any:
        del async_op
        return _CompletedUnshardHandle()

    def set_reshard_after_forward(self, value: bool, recurse: bool = True) -> None:
        del recurse
        self._get_replicate_state().auto_reshard_after_forward = bool(value)


def _validate_module(module: nn.Module) -> None:
    if not hasattr(module, "named_parameters") or not hasattr(module, "forward"):
        raise TypeError("replicate expects a module with parameters and forward")
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
    return (
        getattr(module, "_fsdp_state", None) is None
        and "fully_shard" not in registry
    )


def replicate(
    module: nn.Module | list[nn.Module],
    *,
    mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    ignored_params: set[Any] | None = None,
    dp_mesh_dims: Any = None,
    **kwargs: Any,
):
    modules = module if isinstance(module, list) else [module]
    if not modules:
        raise ValueError("replicate expects at least one module")
    if any(not isinstance(item, nn.Module) for item in modules):
        raise TypeError("replicate expects modules")
    if len({id(item) for item in modules}) != len(modules):
        raise ValueError("replicate cannot receive the same module twice")
    for item in modules:
        _validate_module(item)
    if ignored_params is not None:
        ignored_params = set(ignored_params)
        all_params = {param for item in modules for param in item.parameters()}
        unknown = ignored_params - all_params
        if unknown:
            raise ValueError("ignored_params contains a parameter outside the module")
    if mp_policy is not None and not isinstance(mp_policy, MixedPrecisionPolicy):
        raise TypeError("mp_policy must be a MixedPrecisionPolicy")
    if offload_policy is not None and not isinstance(offload_policy, OffloadPolicy):
        raise TypeError("offload_policy must be an OffloadPolicy")
    device_id = kwargs.get("device_id")
    if device_id is not None and not isinstance(device_id, (int, str)) and not hasattr(device_id, "type"):
        raise TypeError("device_id must be an integer or device value")
    if mesh is not None:
        active_mesh = mesh
    elif dist.is_initialized():
        active_mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("replicate",))
    else:
        raise RuntimeError("replicate requires an initialized process group")
    _validate_mesh(active_mesh)
    if dp_mesh_dims is not None:
        shard_names = getattr(dp_mesh_dims, "shard_names", ())
        if shard_names:
            raise ValueError("replicate only accepts replicate mesh dimensions")
        replicate_names = getattr(dp_mesh_dims, "replicate_names", ())
        if replicate_names and tuple(replicate_names) != tuple(active_mesh.mesh_dim_names or ()):
            raise ValueError("replicate mesh dimensions must cover the selected mesh")
    process_group = kwargs.pop("process_group", None)
    if process_group is None:
        process_group = active_mesh.get_group(0)
    ignored_names: set[str] = set()
    for item in modules:
        for name, parameter in item.named_parameters():
            if ignored_params is not None and parameter in ignored_params:
                ignored_names.add(name)
        if ignored_names:
            item._ddp_params_and_buffers_to_ignore = set(ignored_names)
    result = []
    for item in modules:
        state = _ReplicateState()
        state.mesh = active_mesh
        state.init(
            (item,),
            device=getattr(active_mesh, "device_type", "cpu"),
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            ignored_params=ignored_params,
        )
        _get_registry(item)["__replicate_with_fsdp_state__"] = state
        ddp_kwargs = dict(kwargs)
        ddp_kwargs["process_group"] = process_group
        if mp_policy is not None:
            state.mp_policy = mp_policy
        replicated = _replicate(item, **ddp_kwargs)
        legacy_state = _replicate_state(replicated)
        state.set_legacy_state(legacy_state)
        ddp_state = getattr(legacy_state, "_ddp", None)
        if ddp_state is not None:
            state.set_ddp(ddp_state)
        state.set_requires_gradient_sync(state._pending_requires_gradient_sync)
        cls = type(replicated)
        wrapped_cls = type(f"Replicate{cls.__name__}", (ReplicateModule, cls), {})
        replicated.__class__ = wrapped_cls
        result.append(replicated)
    return result if isinstance(module, list) else result[0]


class _CompletedUnshardHandle:
    def wait(self) -> None:
        return None


def _replicate_state(module: nn.Module) -> Any:
    registry = _get_registry(module)
    state = registry.get("__replicate_state_key__")
    return state
