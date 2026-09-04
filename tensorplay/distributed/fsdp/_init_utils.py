"""Initialization helpers for sharded module state."""

import collections
import warnings
from typing import Any, Iterable, Iterator

import tensorplay as tp

from .. import distributed_core as dist
from ._common_utils import (
    _FSDPDeviceHandle,
    _get_module_fsdp_state,
    _is_fsdp_flattened,
    _set_fsdp_flattened,
    clean_tensor_name,
    TrainingState,
)
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
    if process_group is not None and device_mesh is not None:
        raise ValueError("process_group and device_mesh cannot both be specified")
    name = getattr(sharding_strategy, "name", str(sharding_strategy))
    is_hybrid = name in {"HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"}
    if is_hybrid:
        if device_mesh is not None:
            if not _is_valid_hybrid_shard_device_mesh(device_mesh):
                raise ValueError("hybrid sharding requires a two-dimensional mesh")
            process_group = (
                device_mesh.get_group(mesh_dim=1),
                device_mesh.get_group(mesh_dim=0),
            )
        elif process_group is not None and not _is_valid_hybrid_shard_pg_type(process_group):
            raise ValueError("hybrid sharding requires two process groups")
        _init_process_group_state_for_hybrid_shard(state, process_group, device_mesh)
        process_group = getattr(state, "process_group", process_group)
    elif device_mesh is not None:
        process_group = device_mesh.get_group(mesh_dim=0)
    elif process_group is None and dist.is_initialized():
        process_group = getattr(getattr(dist, "group", None), "WORLD", None)
    state.process_group = process_group
    state.sharding_strategy = sharding_strategy
    state.device_mesh = device_mesh
    try:
        state.rank = int(dist.get_rank(process_group))
        state.world_size = int(dist.get_world_size(process_group))
    except (RuntimeError, ValueError, AttributeError):
        state.rank = 0
        state.world_size = 1
    if is_hybrid:
        inter_group = getattr(state, "_inter_node_pg", None)
        state._gradient_predivide_factor = 1.0
        state._gradient_postdivide_factor = float(
            state.world_size * (int(dist.get_world_size(inter_group)) if inter_group is not None else 1)
        )
    else:
        state._gradient_predivide_factor = 1.0
        state._gradient_postdivide_factor = float(state.world_size)
    state._device_mesh = device_mesh
    state._process_group = process_group
    state._policy = policy


def _init_process_group_state_for_hybrid_shard(state: Any, process_group: Any, device_mesh: Any) -> None:
    groups = list(process_group) if isinstance(process_group, tuple) else []
    if groups:
        state.process_group = groups[0]
        state._intra_node_pg = groups[0]
        state._inter_node_pg = groups[1]
    else:
        state.process_group = None
        state._inter_node_pg = None
        state._intra_node_pg = None
    state.device_mesh = device_mesh


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
    state._buffer_names = {name for name, _ in module.named_buffers()}
    state._buffer_name_to_orig_dtype = {
        clean_tensor_name(name): getattr(buffer, "dtype", None)
        for name, buffer in module.named_buffers()
    }
    state._ignored_buffer_names = set()


def _init_core_state(state: Any, sharding_strategy: Any, mixed_precision: Any, cpu_offload: Any, limit_all_gathers: bool, use_orig_params: bool, backward_prefetch_limit: int, forward_prefetch_limit: int) -> None:
    state.sharding_strategy = sharding_strategy
    state.mixed_precision = mixed_precision
    state.cpu_offload = cpu_offload
    state.limit_all_gathers = limit_all_gathers
    state.use_orig_params = use_orig_params
    state.backward_prefetch_limit = backward_prefetch_limit
    state.forward_prefetch_limit = forward_prefetch_limit
    state._use_orig_params = bool(use_orig_params)
    state._use_full_prec_in_eval = False
    state._reshard_after_forward = True
    state._reshard_after_backward = True
    state._sync_gradients = True
    state._ignored_buffer_names = set(getattr(state, "_ignored_buffer_names", ()))


def _init_runtime_state(state: Any) -> None:
    state._training_state = TrainingState.IDLE
    state._is_root = None
    state._handle = None
    state._fully_sharded_module_to_handle = {}
    state.compute_device = None
    state._gradient_predivide_factor = 1.0
    state._gradient_postdivide_factor = 1.0
    state._comm_hook = None
    state._comm_hook_state = None
    state._unshard_event = None
    state._post_backward_callback_queued = False
    state._root_pre_forward_handles = []
    state._pre_forward_handles = []
    state._post_forward_handles = []
    state._streams_initialized = False
    state._default_stream = None
    state._unshard_stream = None
    state._pre_unshard_stream = None
    state._post_backward_stream = None
    state._all_reduce_stream = None
    state._all_fsdp_states = []
    state._all_handles = []
    state._sync_gradients = True
    state._reshard_after_backward = True
    state._reshard_after_forward = True
    state._needs_buffer_dtype_restore_check = False
    state._free_event_queue = None
    state._post_backward_callback_queued = False


def _init_prefetching_state(state: Any, backward_prefetch: Any, forward_prefetch: bool) -> None:
    state.backward_prefetch = backward_prefetch
    state.forward_prefetch = forward_prefetch
    state._states_to_backward_prefetch = []
    state._states_to_forward_prefetch = []
    state._modules_to_forward_prefetch = []
    state._modules_to_backward_prefetch = []
    from ._exec_order_utils import _ExecOrderData

    state._exec_order_data = _ExecOrderData(
        backward_prefetch_limit=1,
        forward_prefetch_limit=1,
    )


def _init_extension(state: Any, device_mesh: Any = None) -> Any:
    extension = None
    root_mesh = None
    if device_mesh is not None:
        get_root_mesh = getattr(device_mesh, "_get_root_mesh", None)
        root_mesh = get_root_mesh() if callable(get_root_mesh) else device_mesh
    state_mesh = getattr(state, "_device_mesh", None)
    if device_mesh is not None and root_mesh != state_mesh:
        from ..tensor.parallel.fsdp import DTensorExtensions

        extension = DTensorExtensions(getattr(state, "_device_handle", None))
    state._fsdp_extension = extension
    _set_fsdp_extensions(extension)
    return state


def _init_state_dict_state(state: Any) -> None:
    from .api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

    state._state_dict_type = StateDictType.FULL_STATE_DICT
    state._state_dict_config = FullStateDictConfig()
    state._optim_state_dict_config = FullOptimStateDictConfig()


def _verify_managed_params(module: Any, params: Iterable[Any]) -> None:
    available = set(module.parameters())
    if any(param not in available for param in params):
        raise ValueError("managed parameters must belong to the module")


def _init_param_handle_from_module(state: Any, fully_sharded_module: Any, device_id: Any, param_init_fn: Any, sync_module_states: bool) -> None:
    _check_single_device_module(
        fully_sharded_module,
        getattr(state, "_ignored_params", set()),
        device_id,
    )
    device_from_device_id = _get_device_from_device_id(
        device_id,
        int(getattr(state, "rank", 0)),
        getattr(state, "_device_handle", _FSDPDeviceHandle("cpu")),
    )
    needs_materialization = _need_to_materialize_module(
        fully_sharded_module,
        getattr(state, "_ignored_params", set()),
        getattr(state, "_ignored_modules", set()),
    )
    if needs_materialization and param_init_fn is not None:
        if not callable(param_init_fn):
            raise TypeError("param_init_fn must be callable")
        _materialize_with_param_init_fn(
            fully_sharded_module,
            param_init_fn,
            getattr(state, "_ignored_modules", set()),
        )
    elif needs_materialization:
        _materialize_meta_module(
            fully_sharded_module,
            device_from_device_id,
            getattr(state, "_ignored_modules", set()),
            getattr(state, "_device_handle", _FSDPDeviceHandle("cpu")),
        )
    ignored = set(getattr(state, "_ignored_params", set()))
    ignored_buffers = {
        buffer
        for ignored_module in getattr(state, "_ignored_modules", set())
        for buffer in ignored_module.buffers()
    }
    _move_module_to_device(
        fully_sharded_module,
        ignored,
        ignored_buffers,
        device_from_device_id,
    )
    params = [param for param in fully_sharded_module.parameters() if param not in ignored]
    state.compute_device = _get_compute_device(
        fully_sharded_module,
        ignored,
        device_from_device_id,
        int(getattr(state, "rank", 0)),
        getattr(state, "_device_handle", _FSDPDeviceHandle("cpu")),
    )
    if sync_module_states:
        _sync_module_params_and_buffers(
            fully_sharded_module,
            params,
            getattr(state, "process_group", None),
        )
    _init_param_handle_from_params(state, params, fully_sharded_module)


def _init_param_handle_from_params(state: Any, params: Iterable[Any], fully_sharded_module: Any) -> None:
    values = [
        param
        for param in params
        if hasattr(param, "numel") and not _is_fsdp_flattened(param)
    ]
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
    handle.shard()
    state._handles = [handle]
    state._flat_param = handle.flat_param
    state._handle = handle
    state._fully_sharded_module_to_handle = {
        fully_sharded_module: handle,
    }
    _set_fsdp_flattened(handle.flat_param, True)


def _get_ignored_modules(root_module: Any, ignored_modules: set[Any] | None) -> set[Any]:
    try:
        roots = set(ignored_modules or ())
    except TypeError as exc:
        raise TypeError("ignored_modules must be an iterable of modules") from exc
    for module in roots:
        if not hasattr(module, "modules"):
            raise TypeError("ignored_modules must contain modules")
        if _get_module_fsdp_state(module) is not None:
            raise ValueError("ignored_modules cannot contain a sharded module")
    from . import _traversal_utils

    for module in root_module.modules():
        if not _traversal_utils._composable(module):
            roots.add(module)
    result = {
        child
        for module in roots
        for child in module.modules()
        if _get_module_fsdp_state(child) is None
    }
    for module in root_module.modules():
        nested = _get_module_fsdp_state(module)
        if nested is not None:
            result.update(getattr(nested, "_ignored_modules", ()))
    return result


def _get_ignored_params(root_module: Any, ignored_modules: set[Any] | None, ignored_parameters: set[Any] | None) -> set[Any]:
    ignored = {
        param
        for param in set(ignored_parameters or ())
        | {param for module in ignored_modules or () for param in module.parameters()}
        if not _is_fsdp_flattened(param)
    }
    for module in root_module.modules():
        nested = _get_module_fsdp_state(module)
        if nested is not None:
            ignored.update(getattr(nested, "_ignored_params", ()))
    return ignored


def _get_ignored_buffer_names(root_module: Any, ignored_modules: set[Any] | None) -> set[str]:
    buffers = {
        buffer
        for module in ignored_modules or ()
        for buffer in module.buffers()
    }
    ignored = {
        clean_tensor_name(name)
        for name, buffer in root_module.named_buffers()
        if buffer in buffers
    }
    for module in root_module.modules():
        nested = _get_module_fsdp_state(module)
        if nested is not None:
            ignored.update(getattr(nested, "_ignored_buffer_names", ()))
    return ignored


def _get_buffer_names(root_module: Any) -> set[str]:
    return {clean_tensor_name(name) for name, _ in root_module.named_buffers()}


def _check_single_device_module(module: Any, ignored_params: Any, device_id: Any) -> Any:
    ignored = set(ignored_params or ())
    devices = {
        _device_type(getattr(param, "device", None))
        for param in _get_orig_params(module, ignored)
        if _device_type(getattr(param, "device", None)) not in {None, "cpu", "meta"}
    }
    if len(devices) > 1:
        raise RuntimeError("managed parameters use different device types")
    requested = _device_type(device_id)
    if requested is not None and devices and requested not in devices:
        raise RuntimeError("device_id does not match the managed parameter device")
    return _device_name(device_id) if device_id is not None else next(iter(devices), "cpu")


def _get_device_from_device_id(device_id: Any, rank: int, device_handle: Any) -> Any:
    del rank
    if device_id is None:
        return None
    if isinstance(device_id, int):
        device_type = getattr(device_handle, "device_type", "cuda")
        device_factory = getattr(tp, "device", None)
        return device_factory(device_type, device_id) if callable(device_factory) else f"{device_type}:{device_id}"
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
    if param_init_fn is None:
        return
    if not callable(param_init_fn):
        raise ValueError(f"parameter initialization callback is not callable: {param_init_fn}")
    for module in _get_modules_to_materialize(root_module, set(ignored_modules or ())):
        param_init_fn(module)


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
    ignored = set(ignored_modules or ())
    result: list[Any] = []
    queue: collections.deque[Any] = collections.deque([root_module])
    visited = {id(root_module)}
    while queue:
        module = queue.popleft()
        if module in ignored:
            continue
        result.append(module)
        for child in module.children():
            if id(child) in visited or child in ignored:
                continue
            if _get_module_fsdp_state(child) is not None:
                continue
            visited.add(id(child))
            queue.append(child)
    return result


def _move_module_to_device(module: Any, ignored_params: Any, ignored_buffers: Any, device_from_device_id: Any) -> None:
    if device_from_device_id is None:
        param = next(iter(_get_orig_params(module, set(ignored_params or ()))), None)
        if param is not None and _device_type(getattr(param, "device", None)) == "cpu":
            _warn_cpu_init()
        return
    ignored_params = set(ignored_params or ())
    ignored_buffers = set(ignored_buffers or ())
    queue: collections.deque[Any] = collections.deque([module])
    visited: set[int] = set()
    params: list[Any] = []
    buffers: list[Any] = []
    while queue:
        current = queue.popleft()
        if id(current) in visited:
            continue
        visited.add(id(current))
        params.extend(
            param
            for param in current.parameters(recurse=False)
            if param not in ignored_params
            and _device_type(getattr(param, "device", None)) == "cpu"
        )
        buffers.extend(
            buffer
            for buffer in current.buffers(recurse=False)
            if buffer not in ignored_buffers
            and _device_type(getattr(buffer, "device", None)) == "cpu"
        )
        for child in current.children():
            if _get_module_fsdp_state(child) is None:
                queue.append(child)
    _move_states_to_device(
        params,
        buffers,
        device_from_device_id,
    )


def _move_states_to_device(params: Iterable[Any], buffers: Iterable[Any], device_from_device_id: Any) -> None:
    for value in list(params) + list(buffers):
        if value is None:
            continue
        value.data = value.to(device_from_device_id)
        grad = getattr(value, "grad", None)
        if grad is not None:
            grad.data = grad.to(device_from_device_id)


def _warn_cpu_init() -> None:
    warnings.warn(
        "sharding initialization is running on CPU; pass device_id to select an accelerator",
        stacklevel=2,
    )


def _get_compute_device(module: Any, ignored_params: Any, device_from_device_id: Any, rank: int, device_handle: Any) -> Any:
    ignored = set(ignored_params or ())
    param = next(iter(_get_orig_params(module, ignored)), None)
    observed = getattr(param, "device", None) if param is not None else None
    result = observed if observed is not None else device_from_device_id
    if result is None:
        current_device = getattr(device_handle, "current_device", None)
        result = current_device() if callable(current_device) else getattr(device_handle, "device_type", "cpu")
    if device_from_device_id is not None and str(result) != str(device_from_device_id):
        raise ValueError(
            f"inconsistent compute device and device_id on rank {rank}: {result} vs {device_from_device_id}"
        )
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
        setattr(value, "_fsdp_synced", True)


def _check_module_states_for_sync_module_states(module_states: Any) -> None:
    if not isinstance(module_states, (list, tuple)):
        raise TypeError("module states must be a sequence")


def _get_orig_params(module: Any, ignored_params: Any) -> Iterator[Any]:
    ignored = set(ignored_params or ())
    return (
        param
        for param in module.parameters()
        if param not in ignored and not _is_fsdp_flattened(param)
    )


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
    name = getattr(sharding_strategy, "name", sharding_strategy)
    if name == "NO_SHARD":
        return getattr(dist, "all_reduce", None)
    return getattr(dist, "reduce_scatter", None)


def _get_default_comm_hook_state(process_group: Any) -> dict[str, Any]:
    return {"process_group": process_group}


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
