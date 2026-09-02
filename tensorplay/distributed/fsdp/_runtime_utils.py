"""Runtime hooks and transitions for sharded parameters."""

from enum import Enum, auto
from typing import Any, Iterable

from ._common_utils import TrainingState

__all__ = ["_PrefetchMode"]


class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()


def _get_fsdp_root_states_with_modules(module: Any) -> list[tuple[Any, Any]]:
    result: list[tuple[Any, Any]] = []
    seen: set[int] = set()
    for item in module.modules():
        state = getattr(item, "_fsdp_state", None)
        if state is None or id(state) in seen:
            continue
        seen.add(id(state))
        if _is_fsdp_root(state, item):
            result.append((item, state))
    return result


def _get_fsdp_root_states(module: Any) -> list[Any]:
    return [state for _, state in _get_fsdp_root_states_with_modules(module)]


def _is_fsdp_root(state: Any, module: Any) -> bool:
    if getattr(state, "_is_root", None) is None:
        state._is_root = True
    if getattr(state, "_is_root", True) and getattr(state, "module", module) is not module:
        return bool(getattr(state, "module", module) is module)
    return bool(getattr(state, "_is_root", True))


def _lazy_init(state: Any, root_module: Any) -> None:
    initializer = getattr(state, "_lazy_init", None)
    if callable(initializer):
        initializer()
    elif getattr(state, "_param_group", None) is not None:
        state._lazy_init()
    else:
        state._is_root = True
    if getattr(state, "module", None) is None:
        state.module = root_module


def _check_flat_params_on_expected_device(state: Any, module: Any) -> None:
    expected = getattr(state, "compute_device", None)
    if expected is None:
        expected = getattr(getattr(state, "_device_handle", None), "device_type", None)
    if expected is None:
        return
    handles = list(getattr(state, "_handles", ()))
    group = getattr(state, "_param_group", None)
    if group is not None:
        handles.extend(getattr(group, "params", ()))
    for handle in handles:
        value = getattr(handle, "flat_param", getattr(handle, "param", None))
        if value is None:
            continue
        actual = getattr(value, "device", None)
        if actual is not None and str(actual) != str(expected):
            raise RuntimeError(f"managed parameter is on {actual}, expected {expected}")


def _share_state_and_init_handle_attrs(root_state: Any, root_module: Any) -> None:
    states = _get_fsdp_root_states(root_module)
    for state in states:
        state._shared_state = root_state
        if state is root_state:
            continue
        for name in ("_streams_initialized", "_default_stream", "_unshard_stream", "_post_backward_stream"):
            if hasattr(root_state, name):
                setattr(state, name, getattr(root_state, name))
        if hasattr(state, "_handles") and hasattr(root_state, "_handles"):
            state._all_handles = root_state._handles
        for handle in getattr(state, "_handles", ()):
            initializer = getattr(handle, "init_flat_param_attributes", None)
            if callable(initializer):
                initializer()


def _init_streams(state: Any) -> None:
    if getattr(state, "_streams_initialized", False):
        return
    handle = getattr(state, "_device_handle", None)
    state._default_stream = handle.current_stream() if handle is not None and hasattr(handle, "current_stream") else None
    if handle is not None and hasattr(handle, "Stream") and getattr(handle, "is_available", lambda: False)():
        state._unshard_stream = handle.Stream()
        state._post_backward_stream = handle.Stream()
        state._pre_unshard_stream = handle.Stream()
    else:
        state._unshard_stream = None
        state._post_backward_stream = None
        state._pre_unshard_stream = None
    state._streams_initialized = True


def _unshard(state: Any, handle: Any, unshard_stream: Any = None, pre_unshard_stream: Any = None) -> None:
    del unshard_stream, pre_unshard_stream
    if handle is None:
        return
    unshard = getattr(handle, "unshard", None)
    if not callable(unshard):
        return
    try:
        unshard(getattr(state, "_unshard_async_op", False))
    except TypeError:
        unshard()


def _reshard(state: Any, handle: Any, free_unsharded_flat_param: bool = True) -> None:
    del state
    if handle is not None:
        handle.reshard(free_unsharded_flat_param)


def _unshard_grads(handle: Any) -> None:
    if handle is None:
        return
    for param in getattr(handle, "params", ()):
        grad = getattr(param, "_sharded_grad", None)
        if grad is not None:
            param._full_tensor.grad = param._gather_with_local_gradient(grad.detach())


def _reshard_grads(handle: Any) -> None:
    if handle is None:
        return
    for param in getattr(handle, "params", ()):
        full_grad = getattr(param._full_tensor, "grad", None)
        if full_grad is not None:
            param._capture_full_gradient(full_grad)


def _pre_forward(state: Any, handle: Any, unshard_fn: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    del module
    if unshard_fn is not None:
        unshard_fn()
    elif handle is not None:
        _unshard(state, handle)
    caster = getattr(state, "_cast_forward_inputs", None)
    if callable(caster):
        args, kwargs = caster(args, kwargs)
    return args, kwargs


def _pre_forward_unshard(state: Any, handle: Any) -> None:
    _unshard(state, handle)


def _post_forward(state: Any, handle: Any, reshard_fn: Any, module: Any, input: Any, output: Any) -> Any:
    del module, input
    if reshard_fn is not None:
        reshard_fn()
    elif handle is not None and getattr(state, "_reshard_after_forward", True):
        _reshard(state, handle)
    state._training_state = TrainingState.IDLE
    return output


def _post_forward_reshard(state: Any, handle: Any) -> None:
    _reshard(state, handle)


def _root_pre_forward(state: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    del module
    state._training_state = TrainingState.FORWARD
    return args, kwargs


def _root_cast_forward_input(state: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    del module
    return state._cast_forward_inputs(args, kwargs)


def _pre_backward_hook(state: Any, module: Any, handle: Any, grad: Any) -> Any:
    del module, handle
    callback = getattr(state, "_pre_backward", None)
    return callback(grad) if callable(callback) else grad


def _post_backward_hook(state: Any, handle: Any, flat_param: Any) -> None:
    if flat_param is not None and getattr(flat_param, "grad", None) is not None:
        _accumulate_sharded_grad(state, handle, flat_param.grad)
    if handle is not None:
        _reshard_grads(handle)


def _post_backward_reshard_only_hook(state: Any, handle: Any) -> None:
    _reshard(state, handle)


def _post_backward_reshard(state: Any, handle: Any) -> None:
    _reshard(state, handle)


def _should_free_in_backward(state: Any, handle: Any) -> bool:
    del handle
    return bool(getattr(state, "_reshard_after_backward", True))


def _reduce_grad(state: Any, handle: Any) -> Any:
    del state
    return handle.sharded_grad()


def _get_reduce_scatter_tensors(state: Any, unsharded_grad: Any) -> tuple[Any, ...]:
    handle = getattr(state, "_param_group", None)
    mesh_info = getattr(handle, "mesh_info", None)
    count = int(mesh_info.shard_world_size) if mesh_info is not None else int(getattr(state, "world_size", 1))
    if count <= 1:
        return (unsharded_grad,)
    return tuple(unsharded_grad.chunk(count, dim=0))


def _accumulate_sharded_grad(state: Any, handle: Any, sharded_grad: Any) -> None:
    del state
    if hasattr(handle, "flat_param"):
        handle.flat_param.grad = sharded_grad
    elif hasattr(handle, "_set_sharded_grad"):
        handle._set_sharded_grad(sharded_grad)
    else:
        for param in getattr(handle, "params", ()):
            param._set_sharded_grad(sharded_grad)


def _reduce_grad_no_shard(state: Any, handle: Any) -> Any:
    return _reduce_grad(state, handle)


def _post_reduce_grad_callback(state: Any, handle: Any, grad_to_offload: Any) -> None:
    if grad_to_offload is None:
        return
    if getattr(getattr(state, "cpu_offload", None), "offload_params", False):
        grad_to_offload = grad_to_offload.cpu()
    _accumulate_sharded_grad(state, handle, grad_to_offload)


def _offload_grad(state: Any, handle: Any, grad_to_offload: Any) -> Any:
    del handle
    if grad_to_offload is None:
        return None
    enabled = getattr(getattr(state, "cpu_offload", None), "offload_params", False)
    return grad_to_offload.cpu() if enabled and hasattr(grad_to_offload, "cpu") else grad_to_offload


def _post_backward_use_sharded_grad_views(handle: Any) -> Any:
    return handle.sharded_grad()


def _div_if_needed(tensor: Any, div_factor: float | None) -> Any:
    return tensor if div_factor in (None, 1) else tensor / div_factor


def _cast_grad_to_param_dtype(state: Any, sharded_grad: Any, param: Any) -> Any:
    del state
    return sharded_grad.to(dtype=param.dtype) if getattr(sharded_grad, "dtype", None) != getattr(param, "dtype", None) else sharded_grad


def _check_grad_to_accumulate(new_sharded_grad: Any, accumulated_grad: Any) -> Any:
    if accumulated_grad is None:
        return new_sharded_grad
    return accumulated_grad + new_sharded_grad


def _low_precision_hook_enabled(state: Any) -> bool:
    return getattr(state, "_comm_hook", None) is not None


def _post_backward_final_callback(state: Any, module: Any) -> None:
    del module
    state._root_post_backward_final_callback()


def _catch_all_reshard(state: Any) -> None:
    state._fsdp_param_group().reshard()


def _finalize_params(state: Any) -> None:
    state._fsdp_param_group().finalize_backward()


def _prefetch_handle(state: Any, current_handle: Any, prefetch_mode: _PrefetchMode) -> None:
    target = _get_handle_to_prefetch(state, current_handle)
    if target is not None:
        _unshard(state, target)


def _get_handle_to_prefetch(state: Any, current_handle: Any) -> Any:
    handles = list(getattr(state, "_handles", ()))
    if not handles:
        group = getattr(state, "_param_group", None)
        handles = [group] if group is not None else []
    if current_handle is None and handles:
        return handles[0]
    try:
        index = handles.index(current_handle)
        return handles[index + 1] if index + 1 < len(handles) else None
    except (ValueError, IndexError):
        return None


def _get_training_state(handle: Any) -> Any:
    state = getattr(handle, "_training_state", None)
    return state if state is not None else getattr(handle, "training_state", TrainingState.IDLE)


def _register_pre_forward_hook(state: Any, module: Any) -> Any:
    return module.register_forward_pre_hook(state._pre_forward_hook, with_kwargs=True)


def _register_post_forward_hook(state: Any, module: Any) -> Any:
    return module.register_forward_hook(state._post_forward_hook, with_kwargs=True)


def _register_root_pre_forward_hook(state: Any, module: Any) -> Any:
    return _register_pre_forward_hook(state, module)


def _register_pre_backward_hooks(state: Any, module: Any, outputs: Any, handle: Any) -> Any:
    del module, handle
    register = getattr(state, "_register_pre_backward_hook", None)
    return register(outputs) if callable(register) else outputs


def _register_post_backward_hook(state: Any, handle: Any) -> None:
    group_register = getattr(handle, "_register_post_backward_hook", None)
    if callable(group_register):
        group_register((), {})
        return
    for param in getattr(handle, "params", ()):
        register = getattr(param, "register_hook", None)
        if register is not None:
            register(lambda grad: _post_backward_hook(state, handle, param) or grad)


def _register_post_backward_reshard_only_hook(state: Any, handle: Any, args: Any, kwargs: Any) -> None:
    del args, kwargs
    _post_backward_reshard_only_hook(state, handle)


def _register_post_backward_final_callback(state: Any, module: Any) -> None:
    del module
    state._register_root_post_backward_final_callback()


def _wait_for_computation_stream(computation_stream: Any, unshard_stream: Any, pre_unshard_stream: Any) -> None:
    if computation_stream is not None and unshard_stream is not None:
        wait_stream = getattr(computation_stream, "wait_stream", None)
        if callable(wait_stream):
            wait_stream(unshard_stream)
    if computation_stream is not None and pre_unshard_stream is not None:
        wait_stream = getattr(computation_stream, "wait_stream", None)
        if callable(wait_stream):
            wait_stream(pre_unshard_stream)


def _reset_flat_param_grad_info_if_needed(handles: Iterable[Any]) -> None:
    for handle in handles:
        flat_param = getattr(handle, "flat_param", None)
        if flat_param is not None and hasattr(flat_param, "grad"):
            flat_param.grad = None
        for param in getattr(handle, "params", ()):
            if hasattr(param, "_sharded_grad"):
                param._sharded_grad = None


def _get_buffers_and_dtypes_for_computation(state: Any, root_module: Any) -> tuple[list[Any], list[Any]]:
    del state
    buffers = list(root_module.buffers())
    return buffers, [getattr(value, "dtype", None) for value in buffers]


def _get_orig_buffer_dtypes(state: Any, buffer_names: Iterable[str]) -> dict[str, Any]:
    module = getattr(state, "module", None)
    buffers = dict(module.named_buffers()) if module is not None else {}
    return {name: getattr(buffers.get(name), "dtype", None) for name in buffer_names}


def _cast_buffers_to_dtype_and_device(buffers: Iterable[Any], buffer_dtypes: Iterable[Any], device: Any) -> None:
    for buffer, dtype in zip(buffers, buffer_dtypes):
        if dtype is not None:
            buffer.data = buffer.to(dtype=dtype, device=device)
