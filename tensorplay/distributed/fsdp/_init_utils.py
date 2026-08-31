"""Initialization helpers for the wrapper implementation."""

from typing import Any, Iterable

from ._common_utils import _FSDPDeviceHandle
from ._fsdp_extensions import _set_fsdp_extensions
from ._fully_shard._fsdp_common import DataParallelMeshInfo
from ._fully_shard._fsdp_param import ParamModuleInfo
from ._fully_shard._fsdp_param_group import FSDPParamGroup

__all__ = []


def _init_process_group_state(state: Any, process_group: Any, sharding_strategy: Any, policy: Any, device_mesh: Any) -> None:
    state.process_group = process_group
    state.sharding_strategy = sharding_strategy
    state.device_mesh = device_mesh


def _init_process_group_state_for_hybrid_shard(state: Any, process_group: Any, device_mesh: Any) -> None:
    _init_process_group_state(state, process_group, None, None, device_mesh)


def _is_valid_hybrid_shard_pg_type(process_group: Any) -> bool:
    return process_group is None or hasattr(process_group, "size")


def _is_valid_hybrid_shard_device_mesh(device_mesh: Any) -> bool:
    return device_mesh is None or hasattr(device_mesh, "ndim")


def _init_intra_node_process_group(num_devices_per_node: int) -> Any:
    del num_devices_per_node
    return None


def _init_inter_node_process_group(global_process_group: Any, num_devices_per_node: int) -> Any:
    del num_devices_per_node
    return global_process_group


def _init_intra_and_inter_node_groups(global_process_group: Any, num_devices_per_node: int) -> tuple[Any, Any]:
    return _init_intra_node_process_group(num_devices_per_node), _init_inter_node_process_group(global_process_group, num_devices_per_node)


def _init_ignored_module_states(state: Any, module: Any, ignored_modules: Iterable[Any] | None, ignored_states: Iterable[Any] | None) -> None:
    state._ignored_modules = set(ignored_modules or ())
    state._ignored_params = {value for value in (ignored_states or ()) if hasattr(value, "numel")}


def _check_ignored_states(ignored_states: Any, passed_as_ignored_states: Any) -> None:
    if ignored_states is not None and passed_as_ignored_states is not None and set(ignored_states) != set(passed_as_ignored_states):
        raise ValueError("ignored state collections disagree")


def _init_device_handle(state: Any, module: Any, ignored_params: Any, device_id: Any) -> None:
    del module, ignored_params, device_id
    state._device_handle = _FSDPDeviceHandle("cpu")


def _init_buffer_state(state: Any, module: Any) -> None:
    state._buffer_names = [name for name, _ in module.named_buffers()]


def _init_core_state(state: Any, sharding_strategy: Any, mixed_precision: Any, cpu_offload: Any, limit_all_gathers: bool, use_orig_params: bool, backward_prefetch_limit: int, forward_prefetch_limit: int) -> None:
    state.sharding_strategy = sharding_strategy
    state.mixed_precision = mixed_precision
    state.cpu_offload = cpu_offload
    state.limit_all_gathers = limit_all_gathers
    state.use_orig_params = use_orig_params
    state.backward_prefetch_limit = backward_prefetch_limit
    state.forward_prefetch_limit = forward_prefetch_limit


def _init_runtime_state(state: Any) -> None:
    state._training_state = None


def _init_prefetching_state(state: Any, backward_prefetch: Any, forward_prefetch: bool) -> None:
    state.backward_prefetch = backward_prefetch
    state.forward_prefetch = forward_prefetch


def _init_extension(state: Any, device_mesh: Any) -> None:
    del state, device_mesh
    _set_fsdp_extensions(None)


def _init_state_dict_state(state: Any) -> None:
    state._state_dict_type = None


def _verify_managed_params(module: Any, params: Iterable[Any]) -> None:
    available = set(module.parameters())
    if any(param not in available for param in params):
        raise ValueError("managed parameters must belong to the module")


def _init_param_handle_from_module(state: Any, fully_sharded_module: Any, device_id: Any, param_init_fn: Any, sync_module_states: bool) -> None:
    del device_id, sync_module_states
    if param_init_fn is not None:
        param_init_fn(fully_sharded_module)
    state._handles = []


def _init_param_handle_from_params(state: Any, params: Iterable[Any], fully_sharded_module: Any) -> None:
    del params, fully_sharded_module
    state._handles = []


def _get_ignored_modules(root_module: Any, ignored_modules: set[Any] | None) -> set[Any]:
    return set(ignored_modules or ())


def _get_ignored_params(root_module: Any, ignored_modules: set[Any] | None, ignored_parameters: set[Any] | None) -> set[Any]:
    del root_module
    return set(ignored_parameters or ()) | {param for module in ignored_modules or () for param in module.parameters()}


def _get_ignored_buffer_names(root_module: Any, ignored_modules: set[Any] | None) -> set[str]:
    del root_module
    return {name for module in ignored_modules or () for name, _ in module.named_buffers()}


def _get_buffer_names(root_module: Any) -> set[str]:
    return {name for name, _ in root_module.named_buffers()}


def _check_single_device_module(module: Any, ignored_params: Any, device_id: Any) -> Any:
    del ignored_params, device_id
    devices = {str(param.device) for param in module.parameters()}
    if len(devices) > 1:
        raise ValueError("all managed parameters must use one device")
    return next(iter(devices), "cpu")


def _get_device_from_device_id(device_id: Any, rank: int, device_handle: Any) -> Any:
    del rank, device_handle
    return device_id or "cpu"


def _need_to_materialize_module(module: Any, ignored_params: Any, ignored_modules: Any) -> bool:
    del ignored_params, ignored_modules
    return any(getattr(param.device, "type", None) == "meta" for param in module.parameters())


def _materialize_with_param_init_fn(root_module: Any, param_init_fn: Any, ignored_modules: Any) -> None:
    del ignored_modules
    if param_init_fn is not None:
        param_init_fn(root_module)


def _materialize_meta_module(root_module: Any, device_from_device_id: Any, ignored_modules: Any, device_handle: Any) -> None:
    del root_module, device_from_device_id, ignored_modules, device_handle


def _get_modules_to_materialize(root_module: Any, ignored_modules: Any) -> list[Any]:
    return [module for module in root_module.modules() if module not in (ignored_modules or set())]


def _move_module_to_device(module: Any, ignored_params: Any, ignored_buffers: Any, device_from_device_id: Any) -> None:
    del ignored_params, ignored_buffers
    module.to(device_from_device_id)


def _move_states_to_device(params: Iterable[Any], buffers: Iterable[Any], device_from_device_id: Any) -> None:
    for value in list(params) + list(buffers):
        value.data = value.to(device_from_device_id)


def _warn_cpu_init() -> None:
    return None


def _get_compute_device(module: Any, ignored_params: Any, device_from_device_id: Any, rank: int, device_handle: Any) -> Any:
    del ignored_params, rank, device_handle
    return device_from_device_id or next((param.device for param in module.parameters()), "cpu")


def _sync_module_params_and_buffers(module: Any, params: Any, process_group: Any) -> None:
    del module, params, process_group


def _check_module_states_for_sync_module_states(module_states: Any) -> None:
    if not isinstance(module_states, (list, tuple)):
        raise TypeError("module states must be a sequence")


def _get_orig_params(module: Any, ignored_params: Any) -> list[Any]:
    return [param for param in module.parameters() if param not in (ignored_params or set())]


def _check_orig_params_flattened(fsdp_module: Any, ignored_params: Any) -> None:
    del fsdp_module, ignored_params


def _get_default_comm_hook(sharding_strategy: Any) -> Any:
    return sharding_strategy


def _get_default_comm_hook_state(process_group: Any) -> dict[str, Any]:
    return {"process_group": process_group}


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
