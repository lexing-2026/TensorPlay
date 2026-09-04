"""State-dict hook helpers for sharded modules."""

import contextlib
import math
import warnings
from typing import Any, Callable, Iterator

import tensorplay as tp

from .. import distributed_core as dist
from ..tensor import DTensor, Replicate, Shard, distribute_tensor

from ._common_utils import (
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
    _get_module_fsdp_state,
    _has_fsdp_params,
    clean_tensor_name,
)
from ._runtime_utils import (
    _cast_buffers_to_dtype_and_device,
    _get_orig_buffer_dtypes,
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from ._fsdp_extensions import (
    _ext_all_gather_dtensor,
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
    _ext_post_unflatten_transform,
    _ext_pre_load_state_dict_transform,
)
from .api import (
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

__all__ = ["_register_all_state_dict_hooks"]


def _param_groups(fsdp_state: Any) -> list[Any]:
    getter = getattr(fsdp_state, "_all_param_groups", None)
    if callable(getter):
        return list(getter())
    group = getattr(fsdp_state, "_param_group", None)
    return [group] if group is not None else []


def _should_unshard_params(fsdp_state: Any) -> bool:
    strategy = getattr(fsdp_state, "sharding_strategy", None)
    use_orig_params = bool(getattr(fsdp_state, "use_orig_params", False))
    composable = hasattr(fsdp_state, "_fsdp_param_groups")
    return not (
        strategy == ShardingStrategy.NO_SHARD and (composable or use_orig_params)
    )


def _convert_to_wrapped_module_name(module_name: str) -> str:
    module_name = module_name.replace(FSDP_PREFIX, "")
    module_name = module_name.replace(FSDP_WRAPPED_MODULE, "")
    return f"{module_name}." if module_name else ""


def _param_name_infos(
    module: Any, fsdp_state: Any
) -> Iterator[tuple[str, str, str]]:
    del module
    for group in _param_groups(fsdp_state):
        for fsdp_param in getattr(group, "params", ()):
            info = fsdp_param.module_info
            param_name = info.name
            module_name = info.fqn[: -len(param_name)] if param_name else info.fqn
            module_name = _convert_to_wrapped_module_name(module_name.rstrip("."))
            yield f"{module_name}{param_name}", param_name, module_name


def _shared_param_name_infos(
    module: Any, fsdp_state: Any
) -> Iterator[tuple[str, str, str]]:
    del module
    seen: set[tuple[int, str]] = set()
    for group in _param_groups(fsdp_state):
        for fsdp_param in getattr(group, "params", ()):
            info = fsdp_param.module_info
            for shared_module, shared_name in zip(
                info.shared_modules, info.shared_param_names
            ):
                key = (id(shared_module), shared_name)
                if key in seen:
                    continue
                seen.add(key)
                prefix = next(
                    (
                        name
                        for name, candidate in info.module.named_modules()
                        if candidate is shared_module
                    ),
                    "",
                )
                prefix = _convert_to_wrapped_module_name(prefix)
                yield f"{prefix}{shared_name}", shared_name, prefix


def _context_map(fsdp_state: Any) -> dict[Any, Any]:
    contexts = getattr(fsdp_state, "_unshard_params_ctx", None)
    if contexts is None:
        contexts = {}
        fsdp_state._unshard_params_ctx = contexts
    return contexts


def _enter_unshard_params_ctx(
    module: Any,
    fsdp_state: Any,
    writeback: bool = False,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
) -> None:
    contexts = _context_map(fsdp_state)
    if module in contexts:
        raise AssertionError("unshard context is already active for this module")
    from ._unshard_param_utils import _unshard_fsdp_state_params

    context_manager = _unshard_fsdp_state_params(
        module,
        fsdp_state,
        writeback,
        rank0_only,
        offload_to_cpu,
        with_grads,
    )
    context_manager.__enter__()
    contexts[module] = {
        "writeback": bool(writeback),
        "rank0_only": bool(rank0_only),
        "offload_to_cpu": bool(offload_to_cpu),
        "with_grads": bool(with_grads),
        "context_manager": context_manager,
    }


def _exit_unshard_params_ctx(module: Any, fsdp_state: Any) -> None:
    contexts = _context_map(fsdp_state)
    if module not in contexts:
        return
    context = contexts.pop(module)
    context_manager = context.get("context_manager")
    if context_manager is not None:
        context_manager.__exit__(None, None, None)
        return
    for group in reversed(_param_groups(fsdp_state)):
        group.reshard()


def _common_pre_state_dict_hook(module: Any, fsdp_state: Any) -> None:
    handle = getattr(fsdp_state, "_device_handle", None)
    synchronize = getattr(handle, "synchronize", None)
    if callable(synchronize):
        synchronize()
    _lazy_init(fsdp_state, module)
    if getattr(fsdp_state, "_is_root", False):
        handles = getattr(fsdp_state, "_all_handles", None)
        _reset_flat_param_grad_info_if_needed(
            handles if handles is not None else _param_groups(fsdp_state)
        )


def _common_unshard_pre_state_dict_hook(
    module: Any,
    fsdp_state: Any,
    offload_to_cpu: bool,
    rank0_only: bool,
) -> None:
    if _should_unshard_params(fsdp_state):
        _enter_unshard_params_ctx(
            module,
            fsdp_state,
            writeback=False,
            offload_to_cpu=offload_to_cpu,
            rank0_only=rank0_only,
        )


def _clone_state_value(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _common_unshard_post_state_dict_hook(
    module: Any,
    fsdp_state: Any,
    state_dict: dict[str, Any],
    prefix: str,
    param_hook: Callable[[dict[str, Any], str, str], None],
) -> dict[str, Any]:
    wrapped_prefix = prefix + getattr(
        fsdp_state, "_state_dict_wrapped_prefix", FSDP_PREFIX
    )
    for key in tuple(state_dict):
        if key.startswith(wrapped_prefix):
            state_dict[prefix + key[len(wrapped_prefix) :]] = state_dict.pop(key)
    if not state_dict or not _has_fsdp_params(module):
        if _should_unshard_params(fsdp_state):
            _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict
    config = getattr(fsdp_state, "_state_dict_config", None)
    rank0_only = isinstance(config, FullStateDictConfig) and bool(config.rank0_only)
    no_return = rank0_only and int(getattr(fsdp_state, "rank", 0)) != 0
    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        key = prefix + fqn
        if no_return:
            state_dict.pop(key, None)
        elif key in state_dict:
            param_hook(state_dict, prefix, key)
    if _should_unshard_params(fsdp_state):
        _exit_unshard_params_ctx(module, fsdp_state)
    ignored = set(getattr(fsdp_state, "_ignored_buffer_names", ()))
    clean_names: list[str] = []
    buffers: list[Any] = []
    for name in getattr(fsdp_state, "_buffer_names", ()):
        clean_name = clean_tensor_name(name)
        key = prefix + clean_name
        if key not in state_dict:
            continue
        if no_return:
            state_dict.pop(key, None)
            continue
        value = state_dict[key]
        if getattr(config, "offload_to_cpu", False):
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                value = cpu()
                state_dict[key] = value
        if clean_name not in ignored:
            clean_names.append(clean_name)
            buffers.append(value)
    buffer_dtype = getattr(getattr(fsdp_state, "mixed_precision", None), "buffer_dtype", None)
    if buffers and buffer_dtype is not None:
        dtypes = _get_orig_buffer_dtypes(fsdp_state, clean_names)
        _cast_buffers_to_dtype_and_device(
            buffers, dtypes, getattr(fsdp_state, "compute_device", None)
        )
        for name, value in zip(clean_names, buffers):
            state_dict[prefix + name] = _clone_state_value(value)
    return state_dict


def _full_pre_state_dict_hook(fsdp_state: Any, module: Any, *args: Any, **kwargs: Any) -> None:
    del args, kwargs
    _common_pre_state_dict_hook(module, fsdp_state)
    config = getattr(fsdp_state, "_state_dict_config", FullStateDictConfig())
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        bool(getattr(config, "offload_to_cpu", False)),
        bool(getattr(config, "rank0_only", False)),
    )


def _full_post_state_dict_hook(
    module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str
) -> dict[str, Any]:
    def param_hook(destination: dict[str, Any], unused_prefix: str, key: str) -> None:
        del unused_prefix
        destination[key] = _clone_state_value(destination[key])

    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


def _full_pre_load_state_dict_hook(
    module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str
) -> None:
    _lazy_init(fsdp_state, module)
    if _should_unshard_params(fsdp_state):
        _enter_unshard_params_ctx(module, fsdp_state, writeback=True)
    wrapped_prefix = prefix + getattr(
        fsdp_state, "_state_dict_wrapped_prefix", FSDP_PREFIX
    )
    for key in tuple(state_dict):
        if key.startswith(prefix) and not key.startswith(wrapped_prefix):
            state_dict[wrapped_prefix + key[len(prefix) :]] = state_dict.pop(key)


def _full_post_load_state_dict_hook(module: Any, fsdp_state: Any, *args: Any, **kwargs: Any) -> None:
    del args, kwargs
    if _should_unshard_params(fsdp_state):
        _exit_unshard_params_ctx(module, fsdp_state)


def _local_pre_state_dict_hook(fsdp_state: Any, module: Any, *args: Any, **kwargs: Any) -> None:
    del args, kwargs
    _common_pre_state_dict_hook(module, fsdp_state)


def _local_post_state_dict_hook(
    module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str
) -> dict[str, Any]:
    wrapped_prefix = prefix + getattr(
        fsdp_state, "_state_dict_wrapped_prefix", FSDP_PREFIX
    )
    for key in tuple(state_dict):
        if key.startswith(wrapped_prefix):
            state_dict[prefix + key[len(wrapped_prefix) :]] = state_dict.pop(key)
    for group in _param_groups(fsdp_state):
        for param in getattr(group, "params", ()):
            key = prefix + param.module_info.fqn
            if key in state_dict:
                state_dict[key] = _clone_state_value(param._sharded_local_tensor())
    return state_dict


def _local_pre_load_state_dict_hook(
    module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str
) -> None:
    _lazy_init(fsdp_state, module)
    wrapped_prefix = prefix + getattr(
        fsdp_state, "_state_dict_wrapped_prefix", FSDP_PREFIX
    )
    for key in tuple(state_dict):
        if key.startswith(prefix) and not key.startswith(wrapped_prefix):
            state_dict[wrapped_prefix + key[len(prefix) :]] = state_dict.pop(key)


def _local_post_load_state_dict_hook(module: Any, fsdp_state: Any, *args: Any, **kwargs: Any) -> None:
    del module, args, kwargs
    for group in _param_groups(fsdp_state):
        group.reshard()


def _sharded_pre_state_dict_hook(
    fsdp_state: Any, module: Any, *args: Any, **kwargs: Any
) -> None:
    del args, kwargs
    _common_pre_state_dict_hook(module, fsdp_state)
    _common_unshard_pre_state_dict_hook(module, fsdp_state, False, False)


def _sharded_post_state_dict_hook(
    module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str
) -> dict[str, Any]:
    def param_hook(destination: dict[str, Any], unused_prefix: str, key: str) -> None:
        del unused_prefix
        value = destination[key]
        target = None
        for group in _param_groups(fsdp_state):
            target = next(
                (
                    param
                    for param in getattr(group, "params", ())
                    if prefix + param.module_info.fqn == key
                ),
                None,
            )
            if target is not None:
                break
        config = getattr(fsdp_state, "_state_dict_config", None)
        use_dtensor = bool(
            getattr(config, "_use_dtensor", False)
            or getattr(fsdp_state, "_use_dtensor", False)
        )
        mesh_info = getattr(target, "mesh_info", None) if target is not None else None
        mesh = getattr(mesh_info, "mesh", None)
        if target is None or mesh is None:
            destination[key] = _clone_state_value(value)
            return
        if use_dtensor:
            if isinstance(value, DTensor):
                destination[key] = _ext_chunk_dtensor(
                    value,
                    int(getattr(fsdp_state, "rank", 0)),
                    mesh,
                    getattr(fsdp_state, "_fsdp_extension", None),
                )
            else:
                ndim = getattr(mesh, "ndim", 1)
                ndim = int(ndim() if callable(ndim) else ndim)
                placements = [Replicate() for _ in range(ndim)]
                shard_dim = getattr(mesh_info, "shard_mesh_dim", None)
                if shard_dim is not None:
                    placements[int(shard_dim)] = Shard(0)
                destination[key] = distribute_tensor(value, mesh, placements)
        else:
            device_handle = getattr(fsdp_state, "_device_handle", None)
            device_count = getattr(device_handle, "device_count", None)
            device_count = int(device_count() if callable(device_count) else 1)
            destination[key] = _ext_chunk_tensor(
                value,
                int(getattr(fsdp_state, "rank", 0)),
                int(getattr(fsdp_state, "world_size", 1)),
                device_count,
                getattr(fsdp_state, "process_group", None),
                getattr(fsdp_state, "_fsdp_extension", None),
                getattr(fsdp_state, "compute_device", None),
            )
        if bool(getattr(config, "offload_to_cpu", False)):
            cpu = getattr(destination[key], "cpu", None)
            if callable(cpu):
                destination[key] = cpu()

    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


def _sharded_pre_load_state_dict_hook(
    module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str
) -> None:
    _lazy_init(fsdp_state, module)
    composable = hasattr(fsdp_state, "_fsdp_param_groups")
    if not composable:
        _full_pre_load_state_dict_hook(module, fsdp_state, state_dict, prefix)
    if not _has_fsdp_params(module):
        return
    config = getattr(fsdp_state, "_state_dict_config", None)
    use_dtensor = bool(
        getattr(config, "_use_dtensor", False)
        or getattr(fsdp_state, "_use_dtensor", False)
    )
    extension = getattr(fsdp_state, "_fsdp_extension", None)
    param_extensions: dict[str, Any] = {}
    handle = getattr(fsdp_state, "_handle", None)
    flat_param = getattr(handle, "flat_param", None)
    if flat_param is not None:
        for name, value in zip(
            getattr(flat_param, "_fqns", ()),
            getattr(flat_param, "_param_extensions", ()),
        ):
            param_extensions[name] = value

    wrapped_prefix = getattr(
        fsdp_state, "_state_dict_wrapped_prefix", FSDP_PREFIX
    )
    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        if composable:
            source_key = prefix + fqn
            state_dict_key = prefix + wrapped_prefix + fqn
        else:
            source_key = prefix + wrapped_prefix + fqn
            state_dict_key = source_key
        if source_key not in state_dict:
            warnings.warn(
                f"missing sharded parameter {source_key!r}; leaving its current value",
                stacklevel=2,
            )
            continue
        value = state_dict.pop(source_key)
        if use_dtensor and isinstance(value, DTensor):
            root_mesh = getattr(fsdp_state, "_device_mesh", None)
            if root_mesh is None:
                root_mesh = getattr(fsdp_state, "mesh", None)
            local_value = _ext_all_gather_dtensor(value, root_mesh, extension)
            param_extension = param_extensions.get(fqn)
            if param_extension is not None:
                local_value = _ext_post_unflatten_transform(
                    local_value, param_extension, extension
                )
            state_dict[state_dict_key] = local_value
            continue
        local_shards_fn = getattr(value, "local_shards", None)
        if not callable(local_shards_fn):
            state_dict[state_dict_key] = _clone_state_value(value)
            continue
        transformed, shards = _ext_pre_load_state_dict_transform(value, extension)
        if len(shards) >= 2:
            raise AssertionError(
                f"expected at most one local shard, got {len(shards)}"
            )
        shape = tuple(getattr(transformed, "shape", ()))
        if not shape:
            state_dict[state_dict_key] = transformed
            continue
        total_numel = math.prod(int(size) for size in shape)
        world_size = int(getattr(fsdp_state, "world_size", 1))
        chunk_size = math.ceil(int(shape[0]) / max(1, world_size))
        chunk_size = chunk_size * (total_numel // int(shape[0]))
        device = getattr(fsdp_state, "compute_device", None)
        if device is None:
            device = getattr(transformed, "device", None)
        if shards:
            local_tensor = shards[0].tensor.reshape(-1)
            local_tensor = local_tensor.to(device) if device is not None else local_tensor
            padding = chunk_size - int(local_tensor.numel())
            if padding > 0:
                local_tensor = tp.cat(
                    (local_tensor, local_tensor.new_zeros(padding)), dim=0
                )
        else:
            local_tensor = tp.zeros(
                chunk_size,
                dtype=getattr(transformed, "dtype", None),
                device=device,
            )
        gathered = tp.empty(
            chunk_size * world_size,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
        )
        if world_size > 1:
            dist.all_gather_single(
                gathered,
                local_tensor,
                group=getattr(fsdp_state, "process_group", None),
            )
        else:
            gathered.copy_(local_tensor)
        state_dict[state_dict_key] = gathered.narrow(
            0, 0, total_numel
        ).reshape(shape)
    _enter_unshard_params_ctx(module, fsdp_state, writeback=True)


def _sharded_post_load_state_dict_hook(
    module: Any, fsdp_state: Any, *args: Any, **kwargs: Any
) -> None:
    _full_post_load_state_dict_hook(module, fsdp_state, *args, **kwargs)


def _replace_with_full_state_dict_type(fsdp_state: Any) -> Any:
    fsdp_state._state_dict_type = StateDictType.FULL_STATE_DICT
    fsdp_state._state_dict_config = FullStateDictConfig()
    return fsdp_state


def _post_state_dict_hook(
    module: Any,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: Any = None,
) -> dict[str, Any]:
    del local_metadata
    fsdp_state = _get_module_fsdp_state(module)
    if fsdp_state is None:
        return state_dict
    state_type = getattr(fsdp_state, "_state_dict_type", StateDictType.FULL_STATE_DICT)
    hook = {
        StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
        StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
        StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
    }[state_type]
    return hook(module, fsdp_state, state_dict, prefix)


def _pre_state_dict_hook(module: Any, prefix: str, keep_vars: bool) -> None:
    del prefix, keep_vars
    fsdp_state = _get_module_fsdp_state(module)
    if fsdp_state is None:
        return
    state_type = getattr(fsdp_state, "_state_dict_type", StateDictType.FULL_STATE_DICT)
    hook = {
        StateDictType.FULL_STATE_DICT: _full_pre_state_dict_hook,
        StateDictType.LOCAL_STATE_DICT: _local_pre_state_dict_hook,
        StateDictType.SHARDED_STATE_DICT: _sharded_pre_state_dict_hook,
    }[state_type]
    hook(fsdp_state, module)


def _set_use_dtensor(fsdp_state: Any, use_dtensor: bool) -> None:
    config = getattr(fsdp_state, "_state_dict_config", None)
    if config is not None and hasattr(config, "_use_dtensor"):
        config._use_dtensor = bool(use_dtensor)
    fsdp_state._use_dtensor = bool(use_dtensor)


def _pre_load_state_dict_hook(
    module: Any,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: Any,
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    fsdp_state = _get_module_fsdp_state(module)
    if fsdp_state is None:
        return
    state_type = getattr(fsdp_state, "_state_dict_type", StateDictType.FULL_STATE_DICT)
    hook = {
        StateDictType.FULL_STATE_DICT: _full_pre_load_state_dict_hook,
        StateDictType.LOCAL_STATE_DICT: _local_pre_load_state_dict_hook,
        StateDictType.SHARDED_STATE_DICT: _sharded_pre_load_state_dict_hook,
    }[state_type]
    hook(module, fsdp_state, state_dict, prefix)


def _post_load_state_dict_hook(module: Any, incompatible_keys: Any) -> None:
    fsdp_state = _get_module_fsdp_state(module)
    if fsdp_state is None:
        return
    state_type = getattr(fsdp_state, "_state_dict_type", StateDictType.FULL_STATE_DICT)
    hook = {
        StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook,
        StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook,
        StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook,
    }[state_type]
    hook(module, fsdp_state)
    if isinstance(incompatible_keys, tuple):
        key_lists = incompatible_keys
    else:
        key_lists = (
            incompatible_keys.missing_keys,
            incompatible_keys.unexpected_keys,
        )
    for key_list in key_lists:
        for index, key in enumerate(key_list):
            key_list[index] = clean_tensor_name(key)


def _register_state_dict_hooks_base(
    state: Any,
    hook_registration_fn_name: str,
    hook: Any,
    hook_registration_fn_kwargs: dict[str, Any] | None = None,
    module: Any = None,
) -> Any:
    module = module if module is not None else getattr(state, "module", None)
    register = getattr(module, hook_registration_fn_name, None)
    if not callable(register):
        return None
    return register(hook, **(hook_registration_fn_kwargs or {}))


def _register_all_state_dict_hooks(state: Any, module: Any = None) -> None:
    if getattr(state, "_state_dict_hooks_registered", False):
        return
    for name, hook, kwargs in (
        ("register_state_dict_pre_hook", _pre_state_dict_hook, {}),
        ("_register_state_dict_hook", _post_state_dict_hook, {}),
        (
            "_register_load_state_dict_pre_hook",
            _pre_load_state_dict_hook,
            {"with_module": True},
        ),
        ("register_load_state_dict_post_hook", _post_load_state_dict_hook, {}),
    ):
        _register_state_dict_hooks_base(state, name, hook, kwargs, module)
    state._state_dict_hooks_registered = True
