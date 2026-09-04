"""Runtime hooks and transitions for sharded parameters."""

import functools
from enum import Enum, auto
from typing import Any, Iterable

import tensorplay as tp

from .. import distributed_core as dist
from ..utils import _apply_to_tensors, _to_kwargs
from ._common_utils import (
    HandleTrainingState,
    TrainingState,
    _assert_in_training_states,
    collect_grad_tensors,
    clean_tensor_name,
)

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
    _lazy_init(state, module)
    if getattr(state, "_is_root", None) is None:
        raise AssertionError("expected FSDP root state to be initialized")
    return bool(state._is_root)


def _lazy_init(state: Any, root_module: Any) -> Any:
    if getattr(state, "_is_root", None) is not None:
        return state
    initializer = getattr(state, "_lazy_init", None)
    if callable(initializer):
        initializer()
        return state
    state._is_root = True
    _assert_in_training_states(state, [TrainingState.IDLE])
    _check_flat_params_on_expected_device(state, root_module)
    from . import _traversal_utils as traversal_utils

    state._all_fsdp_states = traversal_utils._get_fsdp_states(root_module)
    _init_streams(state)
    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_computation(
        state, root_module
    )
    if getattr(state, "compute_device", None) is not None:
        _cast_buffers_to_dtype_and_device(
            buffers, buffer_dtypes, state.compute_device
        )
    execution = getattr(state, "_exec_order_data", None)
    init = getattr(execution, "init", None)
    if callable(init):
        init(state, root_module, getattr(state, "process_group", None))
    _share_state_and_init_handle_attrs(state, root_module)
    return state


def _check_flat_params_on_expected_device(state: Any, module: Any) -> None:
    expected = getattr(state, "compute_device", None)
    if expected is None:
        expected = getattr(getattr(state, "_device_handle", None), "device_type", None)
    if expected is None:
        return
    handles = list(getattr(state, "_handles", ()))
    groups_getter = getattr(state, "_all_param_groups", None)
    if callable(groups_getter):
        handles.extend(
            param
            for group in groups_getter()
            for param in getattr(group, "params", ())
        )
    else:
        group = getattr(state, "_param_group", None)
        if group is not None:
            handles.extend(getattr(group, "params", ()))
    for handle in handles:
        value = getattr(handle, "flat_param", getattr(handle, "param", handle))
        if value is None:
            continue
        actual = getattr(value, "device", None)
        if actual is not None and str(actual) != str(expected):
            raise RuntimeError(f"managed parameter is on {actual}, expected {expected}")


def _share_state_and_init_handle_attrs(root_state: Any, root_module: Any) -> None:
    states = _get_fsdp_root_states(root_module)
    root_groups_getter = getattr(root_state, "_all_param_groups", None)
    root_groups = (
        list(root_groups_getter())
        if callable(root_groups_getter)
        else [getattr(root_state, "_param_group", None)]
    )
    for state in states:
        state._shared_state = root_state
        if state is root_state:
            continue
        for name in (
            "_streams_initialized",
            "_default_stream",
            "_unshard_stream",
            "_post_backward_stream",
            "_pre_unshard_stream",
            "_all_reduce_stream",
        ):
            if hasattr(root_state, name):
                setattr(state, name, getattr(root_state, name))
        if hasattr(state, "_handles") and hasattr(root_state, "_handles"):
            state._all_handles = root_state._handles
        groups_getter = getattr(state, "_all_param_groups", None)
        groups = (
            list(groups_getter())
            if callable(groups_getter)
            else [getattr(state, "_param_group", None)]
        )
        for handle in (*getattr(state, "_handles", ()), *groups):
            if handle is None:
                continue
            initializer = getattr(handle, "init_flat_param_attributes", None)
            if callable(initializer):
                initializer()
    if root_groups:
        for handle in root_groups:
            if handle is None:
                continue
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
    if handle is None:
        return
    pre_unshard = getattr(handle, "pre_unshard", None)
    if callable(pre_unshard):
        try:
            ran_pre_unshard = bool(pre_unshard())
        except TypeError:
            ran_pre_unshard = True
        if not ran_pre_unshard:
            return
    if (
        unshard_stream is not None
        and pre_unshard_stream is not None
        and hasattr(unshard_stream, "wait_stream")
    ):
        unshard_stream.wait_stream(pre_unshard_stream)
    unshard = getattr(handle, "unshard", None)
    if not callable(unshard):
        return
    try:
        unshard()
    except TypeError:
        unshard(getattr(state, "_unshard_async_op", False))
    post_unshard = getattr(handle, "post_unshard", None)
    if callable(post_unshard):
        post_unshard()


def _reshard(state: Any, handle: Any, free_unsharded_flat_param: bool = True) -> None:
    del state
    if handle is None:
        return
    reshard = getattr(handle, "reshard", None)
    if callable(reshard):
        try:
            reshard(free_unsharded_flat_param)
        except TypeError:
            reshard()
    post_reshard = getattr(handle, "post_reshard", None)
    if callable(post_reshard):
        post_reshard()
    handle._prefetched = False


def _unshard_grads(handle: Any) -> None:
    if handle is not None:
        unshard_grad = getattr(handle, "unshard_grad", None)
        if callable(unshard_grad):
            unshard_grad()


def _reshard_grads(handle: Any) -> None:
    if handle is not None:
        reshard_grad = getattr(handle, "reshard_grad", None)
        if callable(reshard_grad):
            reshard_grad()


def _pre_forward(state: Any, handle: Any, unshard_fn: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    if handle is not None and getattr(handle, "_training_state", None) == HandleTrainingState.BACKWARD_PRE:
        return args, kwargs
    state._training_state = TrainingState.FORWARD_BACKWARD
    execution = getattr(state, "_exec_order_data", None)
    record = getattr(execution, "record_pre_forward", None)
    if callable(record):
        try:
            record(handle, bool(getattr(module, "training", True)))
        except TypeError:
            record(handle)
    if handle is not None:
        handle._training_state = HandleTrainingState.FORWARD
    if unshard_fn is not None:
        unshard_fn(state, handle)
    elif handle is not None:
        _unshard(state, handle)
    _register_post_backward_hook(state, handle)
    if (
        handle is not None
        and getattr(handle, "_offload_params", False)
        and getattr(getattr(handle, "flat_param", None), "_cpu_grad", None) is None
    ):
        flat_param = handle.flat_param
        local_shard = getattr(flat_param, "_local_shard", flat_param)
        try:
            flat_param._cpu_grad = tp.zeros_like(local_shard, device="cpu")
            if getattr(handle, "pin_memory", False):
                flat_param._cpu_grad = flat_param._cpu_grad.pin_memory()
        except (AttributeError, RuntimeError):
            flat_param._cpu_grad = tp.zeros_like(local_shard)
    caster = getattr(state, "_cast_forward_inputs", None)
    mixed_precision = getattr(state, "mixed_precision", None)
    force_full_precision = getattr(handle, "_force_full_precision", False)
    if callable(force_full_precision):
        force_full_precision = force_full_precision()
    should_cast = handle is not None and not bool(force_full_precision)
    if callable(caster) and should_cast and (
        mixed_precision is None
        or bool(getattr(mixed_precision, "cast_forward_inputs", True))
    ):
        args, kwargs = caster(args, kwargs)
    _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
    return args, kwargs


def _pre_forward_unshard(state: Any, handle: Any) -> None:
    if handle is None:
        return
    if not getattr(handle, "_prefetched", False):
        _unshard(state, handle)
    handle._needs_pre_forward_unshard = False
    if getattr(handle, "_prefetched", False):
        handle._prefetched = False
    current_stream = getattr(getattr(state, "_device_handle", None), "current_stream", None)
    if callable(current_stream):
        current_stream = current_stream()
    if current_stream is not None:
        event = getattr(state, "_unshard_event", None)
        if event is not None and callable(getattr(current_stream, "wait_event", None)):
            current_stream.wait_event(event)
            state._unshard_event = None
        elif callable(getattr(current_stream, "wait_stream", None)):
            stream = getattr(state, "_unshard_stream", None)
            if stream is not None:
                current_stream.wait_stream(stream)
    _prefetch_handle(state, handle, _PrefetchMode.FORWARD)


def _post_forward(state: Any, handle: Any, reshard_fn: Any, module: Any, input: Any, output: Any) -> Any:
    del input
    if handle is not None and getattr(handle, "_training_state", None) == HandleTrainingState.BACKWARD_PRE:
        return output
    execution = getattr(state, "_exec_order_data", None)
    record = getattr(execution, "record_post_forward", None)
    if callable(record):
        record(handle)
    if reshard_fn is not None:
        reshard_fn(state, handle)
    elif handle is not None and getattr(state, "_reshard_after_forward", True):
        _reshard(state, handle)
    output = _register_pre_backward_hooks(state, module, output, handle)
    state._training_state = TrainingState.IDLE
    if handle is not None:
        handle._training_state = HandleTrainingState.IDLE
    return output


def _post_forward_reshard(state: Any, handle: Any) -> None:
    if handle is None:
        return
    strategy = getattr(handle, "_sharding_strategy", None)
    strategy_type = type(strategy)
    reshard_strategies = {
        getattr(strategy_type, "FULL_SHARD", object()),
        getattr(strategy_type, "HYBRID_SHARD", object()),
        getattr(strategy_type, "_HYBRID_SHARD_ZERO2", object()),
    }
    free_unsharded_flat_param = not bool(getattr(state, "_is_root", False)) and (
        strategy in reshard_strategies
    )
    _reshard(state, handle, free_unsharded_flat_param)


def _root_pre_forward(state: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    _lazy_init(state, module)
    if getattr(state, "_is_root", True) is not True:
        return _root_cast_forward_input(state, module, args, kwargs)
    handle = getattr(state, "_handle", None)
    force_full_precision = bool(getattr(handle, "_force_full_precision", False))
    if handle is None:
        force_full_precision = any(
            bool(getattr(candidate, "_force_full_precision", False))
            for candidate in getattr(state, "_all_handles", ())
        )
    if force_full_precision:
        buffers = list(dict(module.named_buffers()).values())
        original_dtypes = getattr(state, "_buffer_name_to_orig_dtype", {})
        dtypes = [
            original_dtypes.get(clean_tensor_name(name))
            for name, _ in module.named_buffers()
        ]
        if buffers:
            _cast_buffers_to_dtype_and_device(
                buffers, dtypes, getattr(state, "compute_device", None)
            )
        state._needs_buffer_dtype_restore_check = True
    elif getattr(state, "_needs_buffer_dtype_restore_check", False):
        buffers, dtypes = _get_buffers_and_dtypes_for_computation(state, module)
        if any(
            getattr(buffer, "dtype", None) != dtype
            for buffer, dtype in zip(buffers, dtypes)
            if dtype is not None
        ):
            _cast_buffers_to_dtype_and_device(
                buffers, dtypes, getattr(state, "compute_device", None)
            )
        state._needs_buffer_dtype_restore_check = False
    state._training_state = TrainingState.FORWARD
    if getattr(state, "forward_prefetch", False):
        handles = list(getattr(state, "_all_handles", ()))
        if not handles:
            handles = list(getattr(state, "_handles", ()))
        for candidate in handles:
            candidate._needs_pre_forward_unshard = True
            candidate._prefetched = False
    _wait_for_computation_stream(
        getattr(getattr(state, "_device_handle", None), "current_stream", lambda: None)(),
        getattr(state, "_unshard_stream", None),
        getattr(state, "_pre_unshard_stream", None),
    )
    _reset_flat_param_grad_info_if_needed(getattr(state, "_all_handles", ()))
    device = getattr(state, "compute_device", None)
    device_type = str(getattr(device, "type", device)).split(":", 1)[0].lower()
    if device is not None and device_type in {"cuda", "hpu", "xpu", "mtia"} and (
        args or kwargs
    ):
        args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, device, False)
        args = args_tuple[0] if args_tuple else ()
        kwargs = kwargs_tuple[0] if kwargs_tuple else {}
    return _root_cast_forward_input(state, module, args, kwargs)


def _root_cast_forward_input(state: Any, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
    policy = getattr(state, "mixed_precision", None)
    if policy is None:
        policy = getattr(state, "_mp_policy", None)
    handle = getattr(state, "_handle", None)
    force_full_precision = not bool(getattr(handle, "_force_full_precision", False))
    should_cast = (
        (bool(getattr(module, "training", True)) or not bool(getattr(state, "_use_full_prec_in_eval", False)))
        and force_full_precision
        and bool(getattr(policy, "cast_root_forward_inputs", True))
    )
    if not should_cast:
        return args, kwargs
    dtype = getattr(policy, "param_dtype", None)
    caster = getattr(state, "_cast_forward_inputs", None)
    if callable(caster):
        return caster(args, kwargs)
    if dtype is None:
        return args, kwargs
    return (
        _apply_to_tensors(lambda value: value.to(dtype=dtype) if getattr(value, "is_floating_point", lambda: False)() else value, args),
        _apply_to_tensors(lambda value: value.to(dtype=dtype) if getattr(value, "is_floating_point", lambda: False)() else value, kwargs),
    )


def _pre_backward_hook(state: Any, module: Any, handle: Any, grad: Any, *unused: Any) -> Any:
    if handle is not None and getattr(handle, "_ran_pre_backward_hook", False):
        return grad
    if getattr(state, "_is_root", False) and not getattr(
        state, "_post_backward_callback_queued", False
    ):
        _register_post_backward_final_callback(state, module)
        _reset_flat_param_grad_info_if_needed(getattr(state, "_all_handles", ()))
    elif handle is not None:
        allowed = [TrainingState.IDLE]
        if type(getattr(state, "_training_state", None)) is TrainingState:
            allowed.append(TrainingState.FORWARD_BACKWARD)
        _assert_in_training_states(state, allowed)
    state._training_state = TrainingState.FORWARD_BACKWARD
    if handle is None:
        callback = getattr(state, "_pre_backward", None)
        return callback(grad) if callable(callback) else grad
    handle._training_state = HandleTrainingState.BACKWARD_PRE
    if getattr(handle, "_needs_pre_backward_unshard", False):
        if not getattr(handle, "_prefetched", False):
            _unshard(
                state,
                handle,
                getattr(state, "_unshard_stream", None),
                getattr(state, "_pre_unshard_stream", None),
            )
        current_stream = getattr(getattr(state, "_device_handle", None), "current_stream", None)
        if callable(current_stream):
            current_stream = current_stream()
        unshard_stream = getattr(state, "_unshard_stream", None)
        if current_stream is not None and unshard_stream is not None and callable(
            getattr(current_stream, "wait_stream", None)
        ):
            current_stream.wait_stream(unshard_stream)
    handle._needs_pre_backward_unshard = False
    _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
    prepare = getattr(handle, "prepare_gradient_for_backward", None)
    if callable(prepare):
        prepare()
    handle._ran_pre_backward_hook = True
    callback = getattr(state, "_pre_backward", None)
    return callback(grad) if callable(callback) else grad


@tp.no_grad()
def _post_backward_hook(state: Any, handle: Any, flat_param: Any, *unused: Any) -> None:
    del unused
    if handle is None:
        return
    flat_param = getattr(handle, "flat_param", flat_param)
    if flat_param is None:
        return
    flat_param._post_backward_called = True
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    if getattr(handle, "_training_state", None) not in (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
    ):
        raise AssertionError("invalid handle state during backward")
    handle._training_state = HandleTrainingState.BACKWARD_POST
    grad = getattr(flat_param, "grad", None)
    if grad is None:
        return
    if getattr(grad, "requires_grad", False):
        raise RuntimeError("gradients of gradients are not supported")
    _post_backward_reshard(state, handle)
    sync_gradients = bool(
        getattr(state, "_sync_gradients", getattr(state, "_requires_gradient_sync", True))
    )
    if not sync_gradients:
        _post_backward_use_sharded_grad_views(handle)
        return
    post_stream = getattr(state, "_post_backward_stream", None)
    current_stream = getattr(getattr(state, "_device_handle", None), "current_stream", None)
    if callable(current_stream):
        current_stream = current_stream()
    if post_stream is not None and current_stream is not None and callable(
        getattr(post_stream, "wait_stream", None)
    ):
        post_stream.wait_stream(current_stream)
    if not _low_precision_hook_enabled(state) and getattr(grad, "dtype", None) != getattr(
        handle, "_reduce_dtype", getattr(flat_param, "dtype", None)
    ) and not bool(getattr(handle, "_force_full_precision", False)):
        flat_param.grad = grad.to(dtype=getattr(handle, "_reduce_dtype", grad.dtype))
    if handle.uses_sharded_strategy:
        _reduce_grad(state, handle)
    else:
        _reduce_grad_no_shard(state, handle)


def _post_backward_reshard_only_hook(state: Any, handle: Any) -> None:
    state._training_state = TrainingState.FORWARD_BACKWARD
    if handle is not None:
        handle._training_state = HandleTrainingState.BACKWARD_POST
    _post_backward_reshard(state, handle)


def _post_backward_reshard(state: Any, handle: Any) -> None:
    _reshard(state, handle, _should_free_in_backward(state, handle))
    _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)


def _should_free_in_backward(state: Any, handle: Any) -> bool:
    if handle is None or not handle.uses_sharded_strategy:
        return False
    sync_gradients = bool(
        getattr(state, "_sync_gradients", getattr(state, "_requires_gradient_sync", True))
    )
    strategy = getattr(handle, "_sharding_strategy", None)
    return sync_gradients or strategy in {
        getattr(type(strategy), "FULL_SHARD", object()),
        getattr(type(strategy), "HYBRID_SHARD", object()),
        getattr(type(strategy), "_HYBRID_SHARD_ZERO2", object()),
    }


def _reduce_grad(state: Any, handle: Any) -> Any:
    flat_param = handle.flat_param
    unsharded_grad = getattr(flat_param, "grad", None)
    if unsharded_grad is None:
        return None
    flat_param.grad = None
    padded_grad, sharded_grad = _get_reduce_scatter_tensors(state, unsharded_grad)
    predivide = getattr(state, "_gradient_predivide_factor", 1.0)
    postdivide = getattr(state, "_gradient_postdivide_factor", 1.0)
    comm_hook = getattr(state, "_comm_hook", None)
    group = getattr(handle, "process_group", None) or getattr(state, "process_group", None)
    if comm_hook is None:
        _div_if_needed(padded_grad, predivide)
        if int(getattr(handle, "world_size", 1)) > 1 and dist.is_initialized():
            dist.reduce_scatter_single(sharded_grad, padded_grad, group=group)
        else:
            sharded_grad.copy_(padded_grad.reshape(-1)[: int(sharded_grad.numel())])
    else:
        comm_hook(getattr(state, "_comm_hook_state", None), padded_grad, sharded_grad)
    _div_if_needed(sharded_grad, postdivide)
    if getattr(handle, "_sharding_strategy", None) in (
        getattr(type(handle._sharding_strategy), "HYBRID_SHARD", None),
        getattr(type(handle._sharding_strategy), "_HYBRID_SHARD_ZERO2", None),
    ):
        all_reduce_group = getattr(state, "_inter_node_pg", None)
        if all_reduce_group is not None:
            dist.all_reduce(sharded_grad, group=all_reduce_group)
    grad_to_offload = _accumulate_sharded_grad(state, handle, sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)
    return sharded_grad


def _get_reduce_scatter_tensors(state: Any, unsharded_grad: Any) -> tuple[Any, Any]:
    count = int(getattr(state, "world_size", 1) or 1)
    handle = getattr(state, "_handle", None)
    if handle is not None:
        count = int(getattr(handle, "world_size", count) or count)
    chunks = list(unsharded_grad.reshape(-1).chunk(count))
    if not chunks:
        chunks = [unsharded_grad.reshape(-1)]
    chunk_size = int(chunks[0].numel())
    padded_numel = count * chunk_size
    if int(unsharded_grad.numel()) != padded_numel:
        padded = unsharded_grad.new_zeros((padded_numel,))
        padded[: int(unsharded_grad.numel())].copy_(unsharded_grad.reshape(-1))
    else:
        padded = unsharded_grad.reshape(-1)
    return padded, unsharded_grad.new_empty((chunk_size,))


def _accumulate_sharded_grad(state: Any, handle: Any, sharded_grad: Any) -> Any:
    del state
    flat_param = handle.flat_param
    _cast_grad_to_param_dtype(None, sharded_grad, flat_param)
    if hasattr(flat_param, "_saved_grad_shard"):
        _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
        flat_param._saved_grad_shard += sharded_grad
    else:
        flat_param._saved_grad_shard = sharded_grad
    return flat_param._saved_grad_shard


def _reduce_grad_no_shard(state: Any, handle: Any) -> Any:
    flat_param = handle.flat_param
    grad = getattr(flat_param, "grad", None)
    if grad is None:
        return None
    comm_hook = getattr(state, "_comm_hook", None)
    group = getattr(handle, "process_group", None) or getattr(state, "process_group", None)
    if comm_hook is None:
        _div_if_needed(grad, getattr(state, "_gradient_predivide_factor", 1.0))
        if dist.is_initialized() and int(getattr(handle, "world_size", 1)) > 1:
            dist.all_reduce(grad, group=group)
        _div_if_needed(grad, getattr(state, "_gradient_postdivide_factor", 1.0))
    else:
        comm_hook(getattr(state, "_comm_hook_state", None), grad)
    if not getattr(handle, "_keep_low_precision_grads", False):
        _cast_grad_to_param_dtype(state, grad, flat_param)
    _post_reduce_grad_callback(state, handle, grad)
    return grad


def _post_reduce_grad_callback(state: Any, handle: Any, grad_to_offload: Any) -> None:
    if grad_to_offload is None:
        return
    _offload_grad(state, handle, grad_to_offload)
    _post_backward_use_sharded_grad_views(handle)


def _offload_grad(state: Any, handle: Any, grad_to_offload: Any) -> Any:
    if grad_to_offload is None or not getattr(handle, "_offload_params", False):
        return grad_to_offload
    flat_param = handle.flat_param
    cpu_grad = getattr(flat_param, "_cpu_grad", None)
    if cpu_grad is None:
        cpu_grad = tp.zeros_like(grad_to_offload, device="cpu")
        flat_param._cpu_grad = cpu_grad
    cpu_grad.copy_(grad_to_offload.detach())
    return cpu_grad


def _post_backward_use_sharded_grad_views(handle: Any) -> Any:
    if handle is None or not getattr(handle, "_use_orig_params", False):
        return None
    reset = getattr(handle, "_reset_is_grad_none", None)
    if callable(reset):
        reset()
    use_views = getattr(handle, "_use_sharded_grad_views", None)
    if callable(use_views):
        use_views()
    return getattr(handle, "sharded_grad", None)


def _div_if_needed(tensor: Any, div_factor: float | None) -> Any:
    if tensor is not None and div_factor not in (None, 1):
        tensor.div_(div_factor)
    return tensor


def _cast_grad_to_param_dtype(state: Any, sharded_grad: Any, param: Any) -> Any:
    if (
        sharded_grad is not None
        and getattr(sharded_grad, "dtype", None) != getattr(param, "dtype", None)
        and not _low_precision_hook_enabled(state)
    ):
        sharded_grad.data = sharded_grad.to(dtype=param.dtype)
    return sharded_grad


def _check_grad_to_accumulate(new_sharded_grad: Any, accumulated_grad: Any) -> Any:
    if accumulated_grad is None:
        return new_sharded_grad
    if tuple(new_sharded_grad.shape) != tuple(accumulated_grad.shape):
        raise ValueError("gradient accumulation shape mismatch")
    if getattr(new_sharded_grad, "device", None) != getattr(accumulated_grad, "device", None):
        raise ValueError("gradient accumulation device mismatch")
    return accumulated_grad + new_sharded_grad


def _low_precision_hook_enabled(state: Any) -> bool:
    return state is not None and getattr(state, "_comm_hook", None) is not None


@tp.no_grad()
def _post_backward_final_callback(state: Any, module: Any) -> None:
    del module
    callback = getattr(state, "_root_post_backward_final_callback", None)
    if callable(callback):
        callback()
        return
    current_stream = getattr(getattr(state, "_device_handle", None), "current_stream", None)
    if callable(current_stream):
        current_stream = current_stream()
    post_stream = getattr(state, "_post_backward_stream", None)
    if current_stream is not None and post_stream is not None and callable(
        getattr(current_stream, "wait_stream", None)
    ):
        current_stream.wait_stream(post_stream)
    execution = getattr(state, "_exec_order_data", None)
    next_iter = getattr(execution, "next_iter", None)
    if callable(next_iter):
        next_iter()
    states = getattr(state, "_all_fsdp_states", ()) or (state,)
    for fsdp_state in states:
        _catch_all_reshard(fsdp_state)
        _finalize_params(fsdp_state)
        fsdp_state._training_state = TrainingState.IDLE
        fsdp_state._post_backward_callback_queued = False


def _catch_all_reshard(state: Any) -> None:
    groups_getter = getattr(state, "_all_param_groups", None)
    if callable(groups_getter):
        for group in groups_getter():
            group.reshard()
        return
    group_getter = getattr(state, "_fsdp_param_group", None)
    if callable(group_getter):
        group_getter().reshard()
        return
    handle = getattr(state, "_handle", None)
    if handle is not None:
        _reshard(state, handle, _should_free_in_backward(state, handle))


def _finalize_params(state: Any) -> None:
    groups_getter = getattr(state, "_all_param_groups", None)
    if callable(groups_getter):
        for group in groups_getter():
            group.finalize_backward()
        return
    group_getter = getattr(state, "_fsdp_param_group", None)
    if callable(group_getter):
        group_getter().finalize_backward()
        return
    handle = getattr(state, "_handle", None)
    if handle is None:
        return
    flat_param = getattr(handle, "flat_param", None)
    hook_state = getattr(flat_param, "_post_backward_hook_state", None)
    if hook_state:
        for hook in hook_state:
            remove = getattr(hook, "remove", None)
            if callable(remove):
                remove()
        delattr(flat_param, "_post_backward_hook_state")
    prepare = getattr(handle, "prepare_gradient_for_optim", None)
    if callable(prepare) and bool(
        getattr(state, "_sync_gradients", getattr(state, "_requires_gradient_sync", True))
    ):
        prepare()
    if flat_param is not None:
        flat_param._post_backward_called = False


def _prefetch_handle(state: Any, current_handle: Any, prefetch_mode: _PrefetchMode) -> None:
    if current_handle is None:
        return
    target = _get_handle_to_prefetch(state, current_handle)
    if target is None:
        return
    previous = getattr(target, "_training_state", HandleTrainingState.IDLE)
    if prefetch_mode is _PrefetchMode.BACKWARD:
        target._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode is _PrefetchMode.FORWARD:
        target._training_state = HandleTrainingState.FORWARD
    else:
        raise ValueError("invalid prefetch mode")
    _unshard(
        state,
        target,
        getattr(state, "_unshard_stream", None),
        getattr(state, "_pre_unshard_stream", None),
    )
    target._training_state = previous
    target._prefetched = True


def _get_handle_to_prefetch(state: Any, current_handle: Any) -> Any:
    if current_handle is None:
        return None
    training_state = _get_training_state(current_handle)
    execution = getattr(state, "_exec_order_data", None)
    target = None
    backward_prefetch = getattr(getattr(state, "backward_prefetch", None), "name", None)
    if training_state is HandleTrainingState.BACKWARD_PRE and backward_prefetch == "BACKWARD_PRE":
        getter = getattr(execution, "get_handle_to_backward_prefetch", None)
        if callable(getter):
            target = getter(current_handle)
    elif training_state is HandleTrainingState.BACKWARD_POST and backward_prefetch == "BACKWARD_POST":
        getter = getattr(execution, "get_handle_to_backward_prefetch", None)
        if callable(getter):
            target = getter(current_handle)
    elif training_state is HandleTrainingState.FORWARD and getattr(state, "forward_prefetch", False):
        getter = getattr(execution, "get_handle_to_forward_prefetch", None)
        if callable(getter):
            target = getter(current_handle)
    if target is not None and (
        not getattr(target, "_needs_pre_backward_unshard", True)
        and not getattr(target, "_needs_pre_forward_unshard", True)
    ):
        return None
    if target is not None and getattr(target, "_prefetched", False):
        return None
    return target


def _get_training_state(handle: Any) -> Any:
    state = getattr(handle, "_training_state", None)
    return state if state is not None else getattr(handle, "training_state", TrainingState.IDLE)


def _register_pre_forward_hook(state: Any, module: Any) -> Any:
    handles = getattr(state, "_pre_forward_handles", None)
    if handles is None:
        handles = []
        state._pre_forward_handles = handles
    for handle in handles:
        handle.remove()
    handles.clear()
    mapping = getattr(state, "_fully_sharded_module_to_handle", {})
    handle = mapping.get(module, getattr(state, "_handle", None))
    hook = functools.partial(_pre_forward, state, handle, _pre_forward_unshard)
    handle = module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    handles.append(handle)
    return handle


def _register_post_forward_hook(state: Any, module: Any) -> Any:
    handles = getattr(state, "_post_forward_handles", None)
    if handles is None:
        handles = []
        state._post_forward_handles = handles
    for handle in handles:
        handle.remove()
    handles.clear()
    mapping = getattr(state, "_fully_sharded_module_to_handle", {})
    handle = mapping.get(module, getattr(state, "_handle", None))
    hook = functools.partial(_post_forward, state, handle, _post_forward_reshard)
    handle = module.register_forward_hook(hook)
    handles.append(handle)
    return handle


def _register_root_pre_forward_hook(state: Any, module: Any) -> Any:
    handles = getattr(state, "_root_pre_forward_handles", None)
    if handles is None:
        handles = []
        state._root_pre_forward_handles = handles
    for handle in handles:
        handle.remove()
    handles.clear()
    hook = functools.partial(_root_pre_forward, state)
    handle = module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    handles.append(handle)
    return handle


def _register_pre_backward_hooks(state: Any, module: Any, outputs: Any, handle: Any) -> Any:
    if not getattr(tp, "is_grad_enabled", lambda: True)():
        return outputs
    if getattr(state, "_is_root", False):
        state._post_backward_callback_queued = False
    if handle is not None:
        handle._needs_pre_backward_unshard = False
        handle._ran_pre_backward_hook = False

    def _register_hook(tensor: Any) -> Any:
        if getattr(tensor, "requires_grad", False):
            tensor.register_hook(
                functools.partial(_pre_backward_hook, state, module, handle)
            )
            if handle is not None:
                handle._needs_pre_backward_unshard = True
        return tensor

    return _apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hook(state: Any, handle: Any) -> None:
    if handle is None or not getattr(tp, "is_grad_enabled", lambda: True)():
        return
    install = getattr(handle, "_install_post_backward_wrappers", None)
    if callable(install):
        install()
        return
    flat_param = getattr(handle, "flat_param", None)
    if flat_param is None or not getattr(flat_param, "requires_grad", False):
        return
    if getattr(flat_param, "_post_backward_hook_state", None) is not None:
        return
    register = getattr(flat_param, "register_post_accumulate_grad_hook", None)
    if callable(register):
        hook_handle = register(functools.partial(_post_backward_hook, state, handle))
        flat_param._post_backward_hook_state = (hook_handle,)


def _register_post_backward_reshard_only_hook(state: Any, handle: Any, args: Any, kwargs: Any) -> None:
    if handle is None or not getattr(tp, "is_grad_enabled", lambda: True)():
        return
    flat_param = getattr(handle, "flat_param", None)
    if flat_param is not None and (
        getattr(flat_param, "requires_grad", False)
        or getattr(flat_param, "_post_backward_hook_state", None) is not None
    ):
        return
    tensors = list(collect_grad_tensors((args, kwargs)))
    if not tensors:
        return
    remaining = [len(tensors)]
    hook_handles: list[Any] = []

    def _post_input_hook(grad: Any) -> Any:
        remaining[0] -= 1
        if remaining[0] == 0:
            _post_backward_reshard_only_hook(state, handle)
        return grad

    for tensor in tensors:
        hook_handles.append(tensor.register_hook(_post_input_hook))
    if flat_param is not None:
        flat_param._post_backward_hook_state = tuple(hook_handles)


def _register_post_backward_final_callback(state: Any, module: Any) -> None:
    del module
    if getattr(state, "_post_backward_callback_queued", False):
        return
    register = getattr(state, "_register_root_post_backward_final_callback", None)
    if callable(register):
        state._post_backward_callback_queued = True
        register()


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
        reset = getattr(handle, "_reset_flat_param_grad_info_if_needed", None)
        if callable(reset):
            reset()
            continue
        flat_param = getattr(handle, "flat_param", None)
        if flat_param is not None and hasattr(flat_param, "grad"):
            flat_param.grad = None
        for param in getattr(handle, "params", ()):
            if hasattr(param, "_sharded_grad"):
                param._sharded_grad = None


def _get_buffers_and_dtypes_for_computation(state: Any, root_module: Any) -> tuple[list[Any], list[Any]]:
    buffers: list[Any] = []
    dtypes: list[Any] = []
    seen: set[int] = set()
    from . import _traversal_utils as traversal_utils

    fsdp_states, fsdp_modules = traversal_utils._get_fsdp_states_with_modules(
        root_module
    )
    if not fsdp_modules:
        fsdp_modules = [root_module]
        fsdp_states = [state]
    for owner, module in zip(reversed(fsdp_states), reversed(fsdp_modules)):
        ignored = set(getattr(owner, "_ignored_buffer_names", ()))
        policy = getattr(owner, "mixed_precision", None)
        if policy is None:
            policy = getattr(owner, "_mp_policy", None)
        dtype = getattr(policy, "buffer_dtype", None)
        for name, buffer in module.named_buffers():
            clean_name = clean_tensor_name(name)
            if id(buffer) in seen or clean_name in ignored:
                continue
            seen.add(id(buffer))
            buffers.append(buffer)
            dtypes.append(dtype)
    if len(buffers) != len(dtypes):
        raise AssertionError("buffer and computation dtype lists must have equal length")
    return buffers, dtypes


def _get_orig_buffer_dtypes(state: Any, buffer_names: Iterable[str]) -> list[Any]:
    original_dtypes = getattr(state, "_buffer_name_to_orig_dtype", {})
    result: list[Any] = []
    for name in buffer_names:
        clean_name = clean_tensor_name(name)
        if clean_name not in original_dtypes:
            raise ValueError(f"buffer {name!r} is not registered")
        result.append(original_dtypes[clean_name])
    return result


def _cast_buffers_to_dtype_and_device(buffers: Iterable[Any], buffer_dtypes: Iterable[Any], device: Any) -> None:
    for buffer, dtype in zip(buffers, buffer_dtypes):
        is_floating = getattr(buffer, "is_floating_point", None)
        if not callable(is_floating) or not is_floating() or dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(dtype=dtype, device=device)
