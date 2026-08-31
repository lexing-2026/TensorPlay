"""Runtime hooks and transitions for sharded parameters."""

from enum import Enum, auto
from typing import Any, Iterable

from ._common_utils import TrainingState

__all__ = ["_PrefetchMode"]


class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()


def _get_fsdp_root_states_with_modules(module: Any) -> list[tuple[Any, Any]]:
    return [(item, getattr(item, "_fsdp_state")) for item in module.modules() if getattr(item, "_fsdp_state", None) is not None]


def _get_fsdp_root_states(module: Any) -> list[Any]:
    return [state for _, state in _get_fsdp_root_states_with_modules(module)]


def _is_fsdp_root(state: Any, module: Any) -> bool:
    del module
    return getattr(state, "_is_root", True)


def _lazy_init(state: Any, root_module: Any) -> None:
    del root_module
    state._lazy_init()


def _check_flat_params_on_expected_device(state: Any, module: Any) -> None:
    del state, module


def _share_state_and_init_handle_attrs(root_state: Any, root_module: Any) -> None:
    del root_state, root_module


def _init_streams(state: Any) -> None:
    state._streams_initialized = True


def _unshard(state: Any, handle: Any, unshard_stream: Any = None, pre_unshard_stream: Any = None) -> None:
    del unshard_stream, pre_unshard_stream
    handle.unshard()


def _reshard(state: Any, handle: Any, free_unsharded_flat_param: bool = True) -> None:
    del state
    handle.reshard(free_unsharded_flat_param)


def _unshard_grads(handle: Any) -> None:
    del handle


def _reshard_grads(handle: Any) -> None:
    del handle


def _pre_forward(state: Any, handle: Any, unshard_fn: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    del state, handle, module
    if unshard_fn is not None:
        unshard_fn()
    return args, kwargs


def _pre_forward_unshard(state: Any, handle: Any) -> None:
    del state
    handle.unshard()


def _post_forward(state: Any, handle: Any, reshard_fn: Any, module: Any, input: Any, output: Any) -> Any:
    del state, handle, module, input
    if reshard_fn is not None:
        reshard_fn()
    return output


def _post_forward_reshard(state: Any, handle: Any) -> None:
    del state
    handle.reshard()


def _root_pre_forward(state: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    del module
    return args, kwargs


def _root_cast_forward_input(state: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    del module
    return state._cast_forward_inputs(args, kwargs)


def _pre_backward_hook(state: Any, module: Any, handle: Any, grad: Any) -> Any:
    del module, handle, grad
    state._pre_backward(None)
    return None


def _post_backward_hook(state: Any, handle: Any, flat_param: Any) -> None:
    del state, handle, flat_param


def _post_backward_reshard_only_hook(state: Any, handle: Any) -> None:
    del state
    handle.reshard()


def _post_backward_reshard(state: Any, handle: Any) -> None:
    del state
    handle.reshard()


def _should_free_in_backward(state: Any, handle: Any) -> bool:
    del state, handle
    return True


def _reduce_grad(state: Any, handle: Any) -> Any:
    del state
    return handle.sharded_grad()


def _get_reduce_scatter_tensors(state: Any, unsharded_grad: Any) -> tuple[Any, ...]:
    del state
    return tuple(unsharded_grad.chunk(1))


def _accumulate_sharded_grad(state: Any, handle: Any, sharded_grad: Any) -> None:
    del state
    handle.flat_param.grad = sharded_grad


def _reduce_grad_no_shard(state: Any, handle: Any) -> Any:
    return _reduce_grad(state, handle)


def _post_reduce_grad_callback(state: Any, handle: Any, grad_to_offload: Any) -> None:
    del state, handle, grad_to_offload


def _offload_grad(state: Any, handle: Any, grad_to_offload: Any) -> Any:
    del state, handle
    return grad_to_offload.cpu() if hasattr(grad_to_offload, "cpu") else grad_to_offload


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
        target.unshard()


def _get_handle_to_prefetch(state: Any, current_handle: Any) -> Any:
    handles = list(getattr(state, "_handles", ()))
    try:
        return handles[handles.index(current_handle) + 1]
    except (ValueError, IndexError):
        return None


def _get_training_state(handle: Any) -> Any:
    return getattr(handle, "training_state", TrainingState.IDLE)


def _register_pre_forward_hook(state: Any, module: Any) -> Any:
    return module.register_forward_pre_hook(state._pre_forward_hook, with_kwargs=True)


def _register_post_forward_hook(state: Any, module: Any) -> Any:
    return module.register_forward_hook(state._post_forward_hook, with_kwargs=True)


def _register_root_pre_forward_hook(state: Any, module: Any) -> Any:
    return _register_pre_forward_hook(state, module)


def _register_pre_backward_hooks(state: Any, module: Any, outputs: Any, handle: Any) -> Any:
    del module, handle
    return state._register_pre_backward_hook(outputs)


def _register_post_backward_hook(state: Any, handle: Any) -> None:
    del state, handle


def _register_post_backward_reshard_only_hook(state: Any, handle: Any, args: Any, kwargs: Any) -> None:
    del args, kwargs
    _post_backward_reshard_only_hook(state, handle)


def _register_post_backward_final_callback(state: Any, module: Any) -> None:
    del module
    state._register_root_post_backward_final_callback()


def _wait_for_computation_stream(computation_stream: Any, unshard_stream: Any, pre_unshard_stream: Any) -> None:
    del computation_stream, unshard_stream, pre_unshard_stream


def _reset_flat_param_grad_info_if_needed(handles: Iterable[Any]) -> None:
    for handle in handles:
        if hasattr(handle.flat_param, "grad"):
            handle.flat_param.grad = None


def _get_buffers_and_dtypes_for_computation(state: Any, root_module: Any) -> tuple[list[Any], list[Any]]:
    del state
    buffers = list(root_module.buffers())
    return buffers, [getattr(value, "dtype", None) for value in buffers]


def _get_orig_buffer_dtypes(state: Any, buffer_names: Iterable[str]) -> dict[str, Any]:
    del state
    return {name: None for name in buffer_names}


def _cast_buffers_to_dtype_and_device(buffers: Iterable[Any], buffer_dtypes: Iterable[Any], device: Any) -> None:
    for buffer, dtype in zip(buffers, buffer_dtypes):
        if dtype is not None:
            buffer.data = buffer.to(dtype=dtype, device=device)
