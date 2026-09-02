"""Initialization helpers for sharded module state."""

import warnings
from typing import Any, Iterable

import tensorplay as tp

from .. import distributed_core as dist
from ._common_utils import _FSDPDeviceHandle
from ._flat_param import FlatParamHandle, HandleShardingStrategy
from ._fsdp_extensions import _set_fsdp_extensions
from ._fully_shard._fsdp_common import DataParallelMeshInfo
from ._fully_shard._fsdp_param import ParamModuleInfo
from ._fully_shard._fsdp_param_group import FSDPParamGroup

__all__ = []


def _device_type(value: Any) -> str | None:
    if value is None:
        return None
    kind = getattr(value, "type", None)
    if kind is not None:
        return str(kind)
    if isinstance(value, int):
        return "cuda"
    text = str(value)
    return text.split(":", 1)[0] if text else None


def _device_name(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    return str(value)


def _init_process_group_state(state: Any, process_group: Any, sharding_strategy: Any, policy: Any, device_mesh: Any) -> None:
    state.process_group = process_group
    state.sharding_strategy = sharding_strategy
    state.device_mesh = device_mesh


def _init_process_group_state_for_hybrid_shard(state: Any, process_group: Any, device_mesh: Any) -> None:
    _init_process_group_state(state, process_group, None, None, device_mesh)


def _is_valid_hybrid_shard_pg_type(process_group: Any) -> bool:
    if process_group is None:
        return True
    return (
        isinstance(process_group, tuple)
        and len(process_group) == 2
        and all(hasattr(group, "size") for group in process_group)
    )


def _is_valid_hybrid_shard_device_mesh(device_mesh: Any) -> bool:
    return device_mesh is None or hasattr(device_mesh, "ndim")


def _init_intra_node_process_group(num_devices_per_node: int) -> Any:
    num_devices_per_node = int(num_devices_per_node)
    if num_devices_per_node <= 0:
        raise ValueError("num_devices_per_node must be positive")
    if not dist.is_initialized():
        return None
    world_size = int(dist.get_world_size())
    if world_size % num_devices_per_node:
        raise ValueError("world size must be divisible by devices per node")
    current, _ = dist.new_subgroups(num_devices_per_node)
    return current


def _init_inter_node_process_group(global_process_group: Any, num_devices_per_node: int) -> Any:
    num_devices_per_node = int(num_devices_per_node)
    if num_devices_per_node <= 0:
        raise ValueError("num_devices_per_node must be positive")
    if not dist.is_initialized():
        return global_process_group
    world_size = int(dist.get_world_size(global_process_group))
    if world_size % num_devices_per_node:
        raise ValueError("world size must be divisible by devices per node")
    world_ranks = dist.get_process_group_ranks(global_process_group)
    local_rank = int(dist.get_rank(global_process_group)) % num_devices_per_node
    num_nodes = world_size // num_devices_per_node
    selected = None
    for node_rank in range(num_devices_per_node):
        ranks = [
            world_ranks[node_rank + node_index * num_devices_per_node]
            for node_index in range(num_nodes)
        ]
        group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(global_process_group),
        )
        if node_rank == local_rank:
            selected = group
    if selected is None:
        raise RuntimeError("failed to select the inter-node process group")
    return selected


def _init_intra_and_inter_node_groups(global_process_group: Any, num_devices_per_node: int) -> tuple[Any, Any]:
    return _init_intra_node_process_group(num_devices_per_node), _init_inter_node_process_group(global_process_group, num_devices_per_node)


def _init_ignored_module_states(state: Any, module: Any, ignored_modules: Iterable[Any] | None, ignored_states: Iterable[Any] | None) -> None:
    if ignored_modules is not None and ignored_states is not None:
        raise ValueError("ignored_modules and ignored_states cannot both be supplied")
    states = tuple(ignored_states or ())
    if ignored_modules is None and states and all(hasattr(value, "parameters") for value in states):
        ignored_modules = states
        states = ()
    elif ignored_modules is not None and states:
        raise ValueError("ignored_modules and ignored_states cannot both be supplied")
    state._ignored_modules = set(ignored_modules or ())
    state._ignored_params = {
        value for value in states if hasattr(value, "numel")
    }
    if states and len(state._ignored_params) != len(states):
        raise TypeError("ignored_states must contain only parameters or modules")
    if module in state._ignored_modules:
        warnings.warn("all parameters of the root module are ignored", stacklevel=2)


def _check_ignored_states(ignored_states: Any, passed_as_ignored_states: Any) -> None:
    if ignored_states is not None and passed_as_ignored_states is not None and set(ignored_states) != set(passed_as_ignored_states):
        raise ValueError("ignored state collections disagree")


def _init_device_handle(state: Any, module: Any, ignored_params: Any, device_id: Any) -> None:
    ignored = set(ignored_params or ())
    requested = _device_type(device_id)
    observed: set[str] = set()
    for param in module.parameters():
        if param in ignored:
            continue
        kind = _device_type(getattr(param, "device", None))
        if kind not in {"cpu", "meta", ""}:
            observed.add(kind)
    if len(observed) > 1:
        raise ValueError("managed parameters must use one device type")
    if requested is not None and observed and requested != next(iter(observed)):
        raise ValueError("device_id does not match the managed parameter device")
    device_type = requested or next(iter(observed), "cpu")
    state._device_handle = _FSDPDeviceHandle(device_type)


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
    if param_init_fn is not None:
        if not callable(param_init_fn):
            raise TypeError("param_init_fn must be callable")
        for candidate in _get_modules_to_materialize(
            fully_sharded_module, getattr(state, "_ignored_modules", set())
        ):
            param_init_fn(candidate)
    ignored = set(getattr(state, "_ignored_params", set()))
    params = [param for param in fully_sharded_module.parameters() if param not in ignored]
    if sync_module_states:
        _sync_module_params_and_buffers(
            fully_sharded_module,
            params,
            getattr(state, "process_group", None),
        )
    _init_param_handle_from_params(state, params, fully_sharded_module)
    if device_id is not None:
        _move_states_to_device(params, fully_sharded_module.buffers(), device_id)


def _init_param_handle_from_params(state: Any, params: Iterable[Any], fully_sharded_module: Any) -> None:
    values = [param for param in params if hasattr(param, "numel")]
    if not values:
        state._handles = []
        state._flat_param = None
        return
    strategy_name = getattr(getattr(state, "sharding_strategy", None), "name", "FULL_SHARD")
    strategy = getattr(HandleShardingStrategy, strategy_name, HandleShardingStrategy.FULL_SHARD)
    handle = FlatParamHandle(
        values,
        fully_sharded_module,
        device=getattr(state, "compute_device", None),
        sharding_strategy=strategy,
        offload_params=bool(getattr(getattr(state, "cpu_offload", None), "offload_params", False)),
        process_group=getattr(state, "process_group", None),
        use_orig_params=bool(getattr(state, "use_orig_params", False)),
    )
    state._handles = [handle]
    state._flat_param = handle.flat_param


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
    ignored = set(ignored_params or ())
    devices = {
        str(param.device)
        for param in module.parameters()
        if param not in ignored
    }
    if len(devices) > 1:
        raise ValueError("all managed parameters must use one device")
    return _device_name(device_id) if device_id is not None else next(iter(devices), "cpu")


def _get_device_from_device_id(device_id: Any, rank: int, device_handle: Any) -> Any:
    del rank
    if device_id is None:
        return None
    if isinstance(device_id, int):
        return f"{getattr(device_handle, 'device_type', 'cuda')}:{device_id}"
    return device_id


def _need_to_materialize_module(module: Any, ignored_params: Any, ignored_modules: Any) -> bool:
    ignored_params = set(ignored_params or ())
    ignored_modules = set(ignored_modules or ())
    for candidate in module.modules():
        if candidate in ignored_modules:
            continue
        if any(
            getattr(param.device, "type", None) == "meta"
            for param in candidate.parameters(recurse=False)
            if param not in ignored_params
        ):
            return True
        if any(getattr(buffer.device, "type", None) == "meta" for buffer in candidate.buffers(recurse=False)):
            return True
    return False


def _materialize_with_param_init_fn(root_module: Any, param_init_fn: Any, ignored_modules: Any) -> None:
    del ignored_modules
    if param_init_fn is not None:
        param_init_fn(root_module)


def _materialize_meta_module(root_module: Any, device_from_device_id: Any, ignored_modules: Any, device_handle: Any) -> None:
    device = device_from_device_id
    if device is None:
        device_type = getattr(device_handle, "device_type", "cpu")
        if device_type == "cuda" and hasattr(getattr(tp, "cuda", None), "current_device"):
            device = f"cuda:{tp.cuda.current_device()}"
        else:
            device = device_type
    for module in _get_modules_to_materialize(root_module, set(ignored_modules or ())):
        direct_states = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
        if not any(getattr(value.device, "type", None) == "meta" for value in direct_states):
            continue
        to_empty = getattr(module, "to_empty", None)
        if not callable(to_empty):
            raise RuntimeError("meta module does not provide to_empty")
        to_empty(device=device, recurse=False)
        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            raise RuntimeError("meta module does not provide reset_parameters")
        reset()


def _get_modules_to_materialize(root_module: Any, ignored_modules: Any) -> list[Any]:
    return [module for module in root_module.modules() if module not in (ignored_modules or set())]


def _move_module_to_device(module: Any, ignored_params: Any, ignored_buffers: Any, device_from_device_id: Any) -> None:
    if device_from_device_id is None:
        return
    ignored_params = set(ignored_params or ())
    ignored_buffers = set(ignored_buffers or ())
    _move_states_to_device(
        [param for param in module.parameters() if param not in ignored_params],
        [buffer for buffer in module.buffers() if buffer not in ignored_buffers],
        device_from_device_id,
    )


def _move_states_to_device(params: Iterable[Any], buffers: Iterable[Any], device_from_device_id: Any) -> None:
    for value in list(params) + list(buffers):
        if value is None:
            continue
        value.data = value.to(device_from_device_id)
        grad = getattr(value, "grad", None)
        if grad is not None:
            value.grad = grad.to(device_from_device_id)


def _warn_cpu_init() -> None:
    warnings.warn(
        "sharding initialization is running on CPU; pass device_id to select an accelerator",
        stacklevel=2,
    )


def _get_compute_device(module: Any, ignored_params: Any, device_from_device_id: Any, rank: int, device_handle: Any) -> Any:
    del rank
    ignored = set(ignored_params or ())
    observed = next(
        (param.device for param in module.parameters() if param not in ignored),
        None,
    )
    result = device_from_device_id if device_from_device_id is not None else observed
    if result is None:
        result = getattr(device_handle, "device_type", "cpu")
    return result


def _sync_module_params_and_buffers(module: Any, params: Any, process_group: Any) -> None:
    if not dist.is_initialized():
        return
    values: list[Any] = []
    seen: set[int] = set()
    for value in list(params or ()) + list(module.buffers()):
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        values.append(value)
    for value in values:
        dist.broadcast(value, src=0, group=process_group)


def _check_module_states_for_sync_module_states(module_states: Any) -> None:
    if not isinstance(module_states, (list, tuple)):
        raise TypeError("module states must be a sequence")


def _get_orig_params(module: Any, ignored_params: Any) -> list[Any]:
    return [param for param in module.parameters() if param not in (ignored_params or set())]


def _check_orig_params_flattened(fsdp_module: Any, ignored_params: Any) -> None:
    from ._common_utils import _is_fsdp_flattened

    ignored = set(ignored_params or ())
    unflattened = [
        name
        for name, param in fsdp_module.named_parameters()
        if param not in ignored and not _is_fsdp_flattened(param)
    ]
    if unflattened:
        raise ValueError(f"parameters are not flattened: {unflattened}")


def _get_default_comm_hook(sharding_strategy: Any) -> Any:
    return sharding_strategy


def _get_default_comm_hook_state(process_group: Any) -> dict[str, Any]:
    return {"process_group": process_group}


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
