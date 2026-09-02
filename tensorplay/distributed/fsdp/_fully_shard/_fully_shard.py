"""Composable fully sharded module operations."""

import contextlib
import functools
from typing import Any, Callable, Iterable

from tensorplay.nn.parameter import Parameter

from ..._composable.contract import _get_registry
from .._common_utils import _get_module_fsdp_state
from ..api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from ._fsdp_api import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from ._fsdp_common import FSDPMeshInfo, resolve_shard_placement
from ._fsdp_init import (
    _get_device_from_mesh,
    _get_mesh_info,
    _get_post_forward_mesh_info,
    _init_default_mesh,
    _validate_module,
)
from ._fsdp_state import FSDPState

__all__ = [
    "fully_shard",
    "FSDPModule",
    "UnshardHandle",
    "register_fsdp_forward_method",
    "get_cls_to_fsdp_cls",
    "disable_fsdp_module_new_init",
    "share_comm_ctx",
]

_cls_to_fsdp_cls: dict[type, type] = {}
_FULLY_SHARD_STATE_KEY = "fully_shard"


def get_cls_to_fsdp_cls() -> dict[type, type]:
    return _cls_to_fsdp_cls


def _as_fsdp_module(module: Any) -> Any:
    if isinstance(module, FSDPModule):
        return module
    original_cls = type(module)
    fsdp_cls = _cls_to_fsdp_cls.get(original_cls)
    if fsdp_cls is None:
        fsdp_cls = type(f"{original_cls.__name__}FSDPModule", (FSDPModule, original_cls), {})
        _cls_to_fsdp_cls[original_cls] = fsdp_cls
    module.__class__ = fsdp_cls
    return module


def fully_shard(
    module: Any,
    *,
    mesh: Any = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[Any], Any] | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    ignored_params: set[Any] | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
) -> Any:
    modules = list(module) if isinstance(module, list) else [module]
    for item in modules:
        _validate_module(item)
        registry = _get_registry(item)
        if _FULLY_SHARD_STATE_KEY in registry:
            raise RuntimeError("fully_shard has already been applied to this module")
        if "__replicate_state_key__" in registry or "__replicate_with_fsdp_state__" in registry:
            raise RuntimeError("fully_shard cannot be applied to a replicated module")
    if mesh is None:
        mesh = _init_default_mesh("cpu")
    mp_policy = mp_policy or MixedPrecisionPolicy()
    offload_policy = offload_policy or OffloadPolicy()
    mesh_info = _get_mesh_info(mesh, dp_mesh_dims)
    auto_reshard_after_forward = reshard_after_forward is None
    if isinstance(mesh_info, FSDPMeshInfo):
        if (
            mesh_info.is_spmd_mesh
            and not auto_reshard_after_forward
            and not isinstance(reshard_after_forward, bool)
            and isinstance(reshard_after_forward, int)
        ):
            raise NotImplementedError(
                "reshard_after_forward as int is not supported with an SPMD mesh"
            )
        post_forward_mesh_info = _get_post_forward_mesh_info(
            True if auto_reshard_after_forward else reshard_after_forward,
            mesh_info,
        )
    else:
        post_forward_mesh_info = None
    converted = [_as_fsdp_module(item) for item in modules]
    states: list[FSDPState] = []
    for item in converted:
        state = FSDPState(item)
        state.mesh = mesh
        state.mesh_info = mesh_info
        state.offload_policy = offload_policy
        state.ignored_params = ignored_params or set()
        managed_modules: list[Any] = []

        def collect_managed(candidate: Any) -> None:
            if candidate is not item and isinstance(candidate, FSDPModule):
                return
            if any(candidate.named_parameters(recurse=False)):
                managed_modules.append(candidate)
            for _, child in candidate.named_children():
                collect_managed(child)

        collect_managed(item)
        state.init(
            managed_modules or [item],
            _get_device_from_mesh(mesh),
            mp_policy,
            auto_reshard_after_forward,
            shard_placement_fn,
            post_forward_mesh_info=post_forward_mesh_info,
        )
        for fsdp_param in state._fsdp_param_group().params:
            if fsdp_param.param in state.ignored_params:
                continue
            fsdp_param.to_sharded()
            local = Parameter(fsdp_param._sharded_local_tensor(), requires_grad=fsdp_param.param.requires_grad)
            fsdp_param.bind_local_param(local)
            fsdp_param._setattr_on_modules(local)
        item._fsdp_state = state
        item._fsdp_state_obj = state
        _get_registry(item)[_FULLY_SHARD_STATE_KEY] = state
        item._state_dict_type = getattr(item, "_state_dict_type", StateDictType.FULL_STATE_DICT)
        item._state_dict_config = getattr(item, "_state_dict_config", FullStateDictConfig())
        item._optim_state_dict_config = getattr(item, "_optim_state_dict_config", FullOptimStateDictConfig())
        for managed_module in managed_modules:
            managed_module._fsdp_state = state
            _get_registry(managed_module)[_FULLY_SHARD_STATE_KEY] = state
        states.append(state)
    if len(converted) == 1:
        return converted[0]
    return converted


class FSDPModule:
    """Methods mixed into modules managed by :func:`fully_shard`."""

    def reshard(self) -> None:
        state = self._get_fsdp_state()
        state._fsdp_param_group().reshard()

    def unshard(self, async_op: bool = False) -> "UnshardHandle":
        state = self._get_fsdp_state()
        state._fsdp_param_group().unshard(async_op)
        return UnshardHandle(state._fsdp_param_group())

    def reset_iter_state(self) -> None:
        self._get_fsdp_state()._reset_iter_state()

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        self._get_fsdp_state()._is_last_backward = bool(is_last_backward)

    def set_requires_gradient_sync(self, requires_gradient_sync: bool, recurse: bool = True) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            state._requires_gradient_sync = bool(requires_gradient_sync)
            if state._param_group is not None:
                state._param_group._requires_gradient_sync = bool(requires_gradient_sync)

    def set_requires_all_reduce(self, requires_all_reduce: bool, recurse: bool = True) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            state._requires_all_reduce = bool(requires_all_reduce)
            if state._param_group is not None:
                state._param_group._requires_all_reduce = bool(requires_all_reduce)

    def set_reshard_after_forward(self, reshard_after_forward: bool, recurse: bool = True) -> None:
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                "reshard_after_forward should be a bool, "
                f"got {type(reshard_after_forward)}"
            )
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            if not isinstance(state.mesh_info, FSDPMeshInfo):
                raise AssertionError("reshard_after_forward requires FSDP mesh info")
            state._auto_reshard_after_forward = False
            post_forward_mesh_info = _get_post_forward_mesh_info(
                reshard_after_forward,
                state.mesh_info,
            )
            state._reshard_after_forward = post_forward_mesh_info is not None
            group = state._fsdp_param_group()
            group.post_forward_mesh_info = post_forward_mesh_info
            group._reshard_after_forward_enabled = post_forward_mesh_info is not None
            for param in group.params:
                param.post_forward_mesh_info = post_forward_mesh_info

    def set_reshard_after_backward(self, reshard_after_backward: bool) -> None:
        state = self._get_fsdp_state()
        state._reshard_after_backward = bool(reshard_after_backward)
        state._fsdp_param_group()._reshard_after_backward_enabled = bool(reshard_after_backward)

    def set_modules_to_forward_prefetch(self, modules: Iterable[Any]) -> None:
        self._get_fsdp_state()._modules_to_forward_prefetch = list(modules)

    def set_modules_to_backward_prefetch(self, modules: Iterable[Any]) -> None:
        self._get_fsdp_state()._modules_to_backward_prefetch = list(modules)

    def set_custom_all_gather(self, comm: Any) -> None:
        state = self._get_fsdp_state()
        state._custom_all_gather = comm
        state._fsdp_param_group()._all_gather_comm = comm

    def set_custom_reduce_scatter(self, comm: Any) -> None:
        state = self._get_fsdp_state()
        state._custom_reduce_scatter = comm
        state._fsdp_param_group()._reduce_scatter_comm = comm

    def set_all_reduce_hook(self, hook: Any, stream: Any = None) -> None:
        state = self._get_fsdp_state()
        state._all_reduce_hook = (hook, stream)
        group = state._fsdp_param_group()
        group._all_reduce_hook = hook
        group._all_reduce_hook_stream = stream

    def set_post_optim_event(self, event: Any) -> None:
        self._get_fsdp_state()._post_optim_event = event

    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        self.set_gradient_divide_factor(factor)

    def set_gradient_divide_factor(self, factor: float) -> None:
        state = self._get_fsdp_state()
        state._gradient_divide_factor = factor
        state._fsdp_param_group().gradient_divide_factor = factor

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        state = self._get_fsdp_state()
        state._force_sum_reduction_for_comms = bool(enable)
        state._fsdp_param_group().force_sum_reduction_for_comms = bool(enable)

    def set_reduce_scatter_unused_params(self, reduce_scatter_unused_params: bool, recurse: bool = True) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            state._reduce_scatter_unused_params = bool(reduce_scatter_unused_params)
            if state._param_group is not None:
                state._param_group.reduce_scatter_unused_params = bool(
                    reduce_scatter_unused_params
                )

    def set_reduce_scatter_max_input_buffers(self, max_input_buffers: int) -> None:
        value = int(max_input_buffers)
        if value <= 0:
            raise ValueError("max_input_buffers must be positive")
        state = self._get_fsdp_state()
        state._reduce_scatter_max_input_buffers = value
        state._param_group.reduce_scatter_max_input_buffers = value

    def set_separate_reduce_scatter_group(self, enable: bool, recurse: bool = True) -> None:
        for module in _selected_modules(self, recurse):
            _get_module_fsdp_state(module)._param_group._set_separate_reduce_scatter_group(enable)

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        state = self._get_fsdp_state()
        state._unshard_in_backward = bool(unshard_in_backward)
        state._fsdp_param_group().unshard_in_backward = bool(unshard_in_backward)

    def set_allocate_memory_from_process_group_for_comm(self, enable: bool) -> None:
        self._get_fsdp_state()._param_group.set_allocate_memory_from_process_group(enable)

    def set_symm_mem_for_comm(self, backend: Any) -> None:
        self._get_fsdp_state()._param_group.set_symm_mem(backend)

    def _set_unshard_async_op(self, async_op: bool) -> None:
        state = self._get_fsdp_state()
        state._unshard_async_op = bool(async_op)
        state._fsdp_param_group().unshard_async_op = bool(async_op)

    def _get_fsdp_state(self) -> FSDPState:
        state = _get_module_fsdp_state(self)
        if state is None:
            raise RuntimeError("module is not managed by fully_shard")
        return state

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        state = self._get_fsdp_state()
        state._fsdp_param_group().reshard()
        result = super()._apply(fn, *args, **kwargs)
        state._fsdp_param_group().reshard()
        return result


class UnshardHandle:
    def __init__(self, param_group: Any) -> None:
        self._param_group = param_group

    def wait(self) -> None:
        self._param_group.wait_for_unshard()


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_groups: Iterable[Any]) -> None:
        self._param_groups = list(fsdp_param_groups)

    def wait(self) -> None:
        for group in self._param_groups:
            group.wait_for_unshard()


def register_fsdp_forward_method(module: Any, method_name: str) -> None:
    method = getattr(module, method_name)
    if getattr(method, "_fsdp_wrapped", False):
        return

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        state = _get_module_fsdp_state(module)
        if state is None:
            return method(*args, **kwargs)
        state._fsdp_param_group().unshard()
        try:
            return method(*args, **kwargs)
        finally:
            state._fsdp_param_group().reshard()

    wrapped._fsdp_wrapped = True
    setattr(module, method_name, wrapped)


def share_comm_ctx(modules: list[FSDPModule]) -> None:
    contexts = [module._get_fsdp_state()._param_group.comm_ctx for module in modules]
    if not contexts:
        return
    for module in modules[1:]:
        module._get_fsdp_state()._param_group.comm_ctx = contexts[0]


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise TypeError("all modules must be managed by fully_shard")


def _selected_modules(module: Any, recurse: bool) -> list[Any]:
    if not recurse:
        return [module]
    return [item for item in module.modules() if isinstance(item, FSDPModule)]


@contextlib.contextmanager
def disable_fsdp_module_new_init():
    yield
