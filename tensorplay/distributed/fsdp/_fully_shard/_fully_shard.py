"""Composable fully sharded module operations."""

import contextlib
import functools
from typing import Any, Callable, Iterable

import tensorplay as tp
from tensorplay.nn.parameter import Parameter

from ..._composable.contract import _get_registry
from .._common_utils import _FSDPDeviceHandle, _get_module_fsdp_state
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
    _get_modules_and_states,
    _get_post_forward_mesh_info,
    _init_default_mesh,
    _validate_mesh,
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
_enable_fsdp_module_new_init = True


def get_cls_to_fsdp_cls() -> dict[type, type]:
    return _cls_to_fsdp_cls


def _as_fsdp_module(module: Any) -> Any:
    if isinstance(module, FSDPModule):
        return module
    original_cls = type(module)
    fsdp_cls = _cls_to_fsdp_cls.get(original_cls)
    if fsdp_cls is None:
        fsdp_cls = type(
            f"{original_cls.__name__}FSDPModule",
            (FSDPModule, original_cls),
            {"__deepcopy__": _unimplemented_deepcopy},
        )
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
    input_is_list = isinstance(module, list)
    input_modules = tuple(module) if input_is_list else (module,)
    if not input_modules:
        raise ValueError("fully_shard expects at least one module")
    if len({id(item) for item in input_modules}) != len(input_modules):
        raise ValueError("fully_shard cannot receive the same module twice")
    for item in input_modules:
        _validate_module(item)
        registry = _get_registry(item)
        if _FULLY_SHARD_STATE_KEY in registry:
            raise RuntimeError("fully_shard has already been applied to this module")
        if "__replicate_state_key__" in registry or "__replicate_with_fsdp_state__" in registry:
            raise RuntimeError("fully_shard cannot be applied to a replicated module")
    if mesh is None:
        mesh = _init_default_mesh("cpu")
    _validate_mesh(mesh, dp_mesh_dims)
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
    converted = tuple(_as_fsdp_module(item) for item in input_modules)
    arg_module, root_modules, managed_modules, _, _ = _get_modules_and_states(
        converted if input_is_list else converted[0],
        _get_device_from_mesh(mesh),
        ignored_params,
    )
    state = FSDPState(root_modules[0])
    state._root_modules = root_modules
    state.mesh = mesh
    state.mesh_info = mesh_info
    state._device = _get_device_from_mesh(mesh)
    state._device_handle = _FSDPDeviceHandle.from_device(state._device)
    state.compute_device = state._device
    state._device_mesh = mesh
    state.rank = int(
        getattr(
            mesh_info,
            "shard_mesh_rank",
            getattr(mesh_info, "replicate_mesh_rank", 0),
        )
    )
    state.world_size = int(
        getattr(
            mesh_info,
            "shard_world_size",
            getattr(mesh_info, "replicate_world_size", 1),
        )
    )
    state.process_group = getattr(mesh_info, "shard_process_group", None)
    state.offload_policy = offload_policy
    state.ignored_params = set(ignored_params or ())
    state.init(
        root_modules,
        _get_device_from_mesh(mesh),
        mp_policy,
        auto_reshard_after_forward,
        shard_placement_fn,
        post_forward_mesh_info=post_forward_mesh_info,
        reshard_after_forward=(
            True if auto_reshard_after_forward else reshard_after_forward
        ),
        managed_modules=managed_modules,
    )
    for group in state._all_param_groups():
        for fsdp_param in group.params:
            if fsdp_param.param in state.ignored_params:
                continue
            fsdp_param.to_sharded()
            local = Parameter(
                fsdp_param._sharded_local_tensor(),
                requires_grad=fsdp_param.param.requires_grad,
            )
            fsdp_param.bind_local_param(local)
            fsdp_param._setattr_on_modules(local)
    for item in root_modules:
        item._fsdp_state = state
        item._fsdp_state_obj = state
        _get_registry(item)[_FULLY_SHARD_STATE_KEY] = state
        item._state_dict_type = getattr(
            item, "_state_dict_type", StateDictType.FULL_STATE_DICT
        )
        item._state_dict_config = getattr(
            item, "_state_dict_config", FullStateDictConfig()
        )
        item._optim_state_dict_config = getattr(
            item, "_optim_state_dict_config", FullOptimStateDictConfig()
        )
    state._state_dict_type = getattr(
        root_modules[0], "_state_dict_type", StateDictType.FULL_STATE_DICT
    )
    state._state_dict_config = getattr(
        root_modules[0], "_state_dict_config", FullStateDictConfig()
    )
    state._optim_state_dict_config = getattr(
        root_modules[0], "_optim_state_dict_config", FullOptimStateDictConfig()
    )
    state._buffer_names = {
        name
        for root_module in root_modules
        for name, _ in root_module.named_buffers()
    }
    state._ignored_buffer_names = set()
    for managed_module in managed_modules:
        managed_module._fsdp_state = state
        _get_registry(managed_module)[_FULLY_SHARD_STATE_KEY] = state
    if input_is_list:
        return list(converted)
    return converted[0]


def _fully_shard_state(module: Any) -> FSDPState:
    state = _get_module_fsdp_state(module)
    if state is None:
        raise RuntimeError("module is not managed by fully_shard")
    return state


fully_shard.state = _fully_shard_state


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(
        "FSDP modules do not support deepcopy; use state dict serialization"
    )


class FSDPModule:
    """Methods mixed into modules managed by :func:`fully_shard`."""

    _orig_cls_mro_index = 2

    @property
    def module(self) -> Any:
        return self

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        orig_cls = cls.__mro__[cls._orig_cls_mro_index]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        if _enable_fsdp_module_new_init:
            self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        state = self._get_fsdp_state()
        for group in state._all_param_groups():
            group.reshard()

    def unshard(self, async_op: bool = False) -> "UnshardHandle | None":
        state = self._get_fsdp_state()
        groups = state._all_param_groups()
        for group in groups:
            group.lazy_init()
            group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(groups or None)
        if async_op:
            return handle
        handle.wait()
        return None

    def reset_iter_state(self) -> None:
        self._get_fsdp_state()._reset_iter_state()

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        self._get_fsdp_state()._state_ctx.is_last_backward = bool(is_last_backward)

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            if state is None:
                continue
            state._requires_gradient_sync = bool(requires_gradient_sync)
            for group in state._all_param_groups():
                group.reduce_grads = bool(requires_gradient_sync)
                group.all_reduce_grads = bool(requires_gradient_sync)
                group._requires_gradient_sync = bool(requires_gradient_sync)

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            if state is None:
                continue
            state._requires_all_reduce = bool(requires_all_reduce)
            for group in state._all_param_groups():
                group.all_reduce_grads = bool(requires_all_reduce)
                group._requires_all_reduce = bool(requires_all_reduce)

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
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
            state._reshard_after_forward = False
            for group in state._all_param_groups():
                post_forward_mesh_info = _get_post_forward_mesh_info(
                    reshard_after_forward,
                    group.mesh_info,
                )
                group.post_forward_mesh_info = post_forward_mesh_info
                group._reshard_after_forward_enabled = (
                    post_forward_mesh_info is not None
                )
                state._reshard_after_forward |= post_forward_mesh_info is not None
                for param in group.params:
                    param.post_forward_mesh_info = post_forward_mesh_info

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            if state is None:
                continue
            state._reshard_after_backward = bool(reshard_after_backward)
            for group in state._all_param_groups():
                group.reshard_after_backward = bool(reshard_after_backward)
                group._reshard_after_backward_enabled = bool(reshard_after_backward)

    def set_modules_to_forward_prefetch(self, modules: Iterable[Any]) -> None:
        states = []
        for module in modules:
            if not isinstance(module, FSDPModule):
                raise TypeError("prefetch targets must be managed modules")
            states.append(module._get_fsdp_state())
        self._get_fsdp_state()._states_to_forward_prefetch = states

    def set_modules_to_backward_prefetch(self, modules: Iterable[Any]) -> None:
        states = []
        for module in modules:
            if not isinstance(module, FSDPModule):
                raise TypeError("prefetch targets must be managed modules")
            states.append(module._get_fsdp_state())
        self._get_fsdp_state()._states_to_backward_prefetch = states

    def set_custom_all_gather(self, comm: Any) -> None:
        state = self._get_fsdp_state()
        if len(state._all_param_groups()) > 1:
            raise ValueError(
                "custom all-gather is unavailable with multiple parameter groups"
            )
        state._custom_all_gather = comm
        for group in state._all_param_groups():
            group._all_gather_comm = comm

    def set_custom_reduce_scatter(self, comm: Any) -> None:
        state = self._get_fsdp_state()
        if len(state._all_param_groups()) > 1:
            raise ValueError(
                "custom reduce-scatter is unavailable with multiple parameter groups"
            )
        state._custom_reduce_scatter = comm
        for group in state._all_param_groups():
            group._reduce_scatter_comm = comm

    def set_all_reduce_hook(self, hook: Any, *, stream: Any = None) -> None:
        state = self._get_fsdp_state()
        if len(state._all_param_groups()) > 1:
            raise ValueError(
                "all-reduce hooks are unavailable with multiple parameter groups"
            )
        state._all_reduce_hook = (hook, stream)
        for group in state._all_param_groups():
            group._all_reduce_hook = hook
            if stream is not None:
                if group._is_hsdp():
                    raise ValueError("stream cannot be set when using native HSDP")
                group._all_reduce_hook_stream = stream

    def set_post_optim_event(self, event: Any) -> None:
        state = self._get_fsdp_state()
        state._state_ctx.post_optim_event = event
        state._post_optim_event = event

    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        self.set_gradient_divide_factor(factor)

    def set_gradient_divide_factor(self, factor: float) -> None:
        state = self._get_fsdp_state()
        state._gradient_divide_factor = factor
        for group in state._all_param_groups():
            group.gradient_divide_factor = factor

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        state = self._get_fsdp_state()
        state._force_sum_reduction_for_comms = bool(enable)
        for group in state._all_param_groups():
            group.force_sum_reduction_for_comms = bool(enable)

    def set_reduce_scatter_unused_params(
        self, reduce_scatter_unused_params: bool, *, recurse: bool = True
    ) -> None:
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            state._reduce_scatter_unused_params = bool(reduce_scatter_unused_params)
            for group in state._all_param_groups():
                group.reduce_scatter_unused_params = bool(
                    reduce_scatter_unused_params
                )

    def set_reduce_scatter_max_input_buffers(
        self, max_input_buffers: int, *, recurse: bool = True
    ) -> None:
        if isinstance(max_input_buffers, bool) or not isinstance(max_input_buffers, int):
            raise TypeError(
                "max_input_buffers must be an int, got "
                f"{type(max_input_buffers).__name__}"
            )
        if max_input_buffers < 1:
            raise ValueError(
                f"max_input_buffers must be a positive int, got {max_input_buffers}"
            )
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            if state is None:
                continue
            state._reduce_scatter_max_input_buffers = max_input_buffers
            for group in state._all_param_groups():
                group.reduce_scatter_max_input_buffers = max_input_buffers

    def set_separate_reduce_scatter_group(
        self, enable: bool = True, *, recurse: bool = True
    ) -> None:
        new_groups: dict[tuple[int, ...], Any] = {}
        for module in _selected_modules(self, recurse):
            state = _get_module_fsdp_state(module)
            for group in state._all_param_groups():
                group._set_separate_reduce_scatter_group(enable, new_groups)

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        state = self._get_fsdp_state()
        state._unshard_in_backward = bool(unshard_in_backward)
        for group in state._all_param_groups():
            group.unshard_in_backward = bool(unshard_in_backward)

    def set_allocate_memory_from_process_group_for_comm(self, enable: bool) -> None:
        state = self._get_fsdp_state()
        for group in state._all_param_groups():
            group.set_allocate_memory_from_process_group(enable)

    def set_symm_mem_for_comm(self, backend: Any = "NCCL") -> None:
        state = self._get_fsdp_state()
        for group in state._all_param_groups():
            group.set_symm_mem(backend)

    def _set_unshard_async_op(self, async_op: bool) -> None:
        for module in _selected_modules(self, True):
            state = _get_module_fsdp_state(module)
            if state is None:
                continue
            state._unshard_async_op = bool(async_op)
            for group in state._all_param_groups():
                group.unshard_async_op = bool(async_op)

    def _get_fsdp_state(self) -> FSDPState:
        state = _get_module_fsdp_state(self)
        if state is None:
            raise RuntimeError("module is not managed by fully_shard")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        self.reshard()
        result = super()._apply(*args, **kwargs)
        state = self._get_fsdp_state()
        groups = state._all_param_groups()
        if not groups:
            return result
        with tp.no_grad():
            for group in groups:
                for fsdp_param in group.fsdp_params:
                    fsdp_param.reset_sharded_param()
        return result


class UnshardHandle:
    def wait(self) -> None:
        return None


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_groups: list[Any] | None) -> None:
        self._param_groups = fsdp_param_groups

    def wait(self) -> None:
        if self._param_groups is not None:
            for group in self._param_groups:
                group.wait_for_unshard()
            self._param_groups = None


def register_fsdp_forward_method(module: Any, method_name: str) -> None:
    if not isinstance(module, FSDPModule):
        return
    if not hasattr(module, method_name):
        raise ValueError(f"{type(module)} does not have a method {method_name}")
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self: Any, *args: Any, **kwargs: Any) -> Any:
        state = self._get_fsdp_state()
        args, kwargs = state._pre_forward(self, args, kwargs)
        output = orig_method(*args, **kwargs)
        return state._post_forward(self, args, output)

    setattr(module, method_name, wrapped_method.__get__(module, type(module)))


def share_comm_ctx(modules: list[FSDPModule]) -> None:
    if not modules:
        return
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError(f"expected managed module, got {module}")
    states = [module._get_fsdp_state() for module in modules]
    comm_ctx = states[0]._comm_ctx
    for state in states[1:]:
        state._comm_ctx = comm_ctx
        for group in state._all_param_groups():
            group.comm_ctx = comm_ctx


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
    global _enable_fsdp_module_new_init
    previous = _enable_fsdp_module_new_init
    _enable_fsdp_module_new_init = False
    try:
        yield
    finally:
        _enable_fsdp_module_new_init = previous
