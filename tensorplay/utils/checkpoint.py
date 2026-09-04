"""Activation checkpointing helpers."""

from __future__ import annotations

import functools
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Any, Callable, Iterable

from tensorplay._C import _autograd as _native_autograd
from tensorplay.overrides import TensorPlayFunctionMode


__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointError",
    "CheckpointFunction",
    "check_backward_validity",
    "detach_variable",
    "get_device_states",
    "set_device_states",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
    "set_checkpoint_debug_enabled",
    "CheckpointPolicy",
    "SelectiveCheckpointContext",
    "create_selective_checkpoint_contexts",
    "GraphExecGroup",
]


CheckpointError = RuntimeError
_checkpoint_early_stop = True
_checkpoint_debug_enabled: bool | None = None


class CheckpointFunction:
    @staticmethod
    def apply(function: Callable[..., Any], preserve_rng_state: bool,
              *args: Any) -> Any:
        return checkpoint(
            function,
            *args,
            use_reentrant=True,
            preserve_rng_state=bool(preserve_rng_state),
        )


def _tensor_type() -> type:
    import tensorplay

    return tensorplay.Tensor


def _walk_tensors(value: Any) -> Iterable[Any]:
    tensor_type = _tensor_type()
    if isinstance(value, tensor_type):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)


class DefaultDeviceType:
    _default_device_type: str | None = "cuda"

    @staticmethod
    def set_device_type(device: str = "cuda") -> None:
        if not isinstance(device, str) or not device:
            raise TypeError("device type must be a non-empty string")
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        if not DefaultDeviceType._default_device_type:
            DefaultDeviceType._default_device_type = "cuda"
        return DefaultDeviceType._default_device_type


def _infer_device_type(*args: Any) -> str:
    types = {str(t.device.type) for t in _walk_tensors(args)
             if str(t.device.type) not in {"cpu", "meta"}}
    if not types:
        return DefaultDeviceType.get_device_type()
    if "cuda" in types:
        return "cuda"
    return next(iter(types))


def get_device_states(*args: Any) -> tuple[list[int], list[Any]]:
    devices: list[int] = []
    states: list[Any] = []
    for tensor in _walk_tensors(args):
        device = tensor.device
        device_type = str(device.type)
        if device_type in {"cpu", "meta"}:
            continue
        index = device.index
        if index is None or index < 0:
            continue
        devices.append(int(index))

    if not devices:
        return devices, states

    if _infer_device_type(*args) == "cuda":
        from tensorplay import cuda

        for index in devices:
            with cuda.device(index):
                states.append(cuda.get_rng_state(index))
    return devices, states


def set_device_states(
    devices: Iterable[int],
    states: Iterable[Any],
    *,
    device_type: str | None = None,
) -> None:
    selected_type = device_type or DefaultDeviceType.get_device_type()
    if selected_type == "meta":
        return
    if selected_type != "cuda":
        return
    from tensorplay import cuda

    for device, state in zip(devices, states):
        with cuda.device(int(device)):
            cuda.set_rng_state(state, int(device))


def detach_variable(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    if not isinstance(inputs, tuple):
        raise RuntimeError("detach_variable expects a tuple of values")
    result = []
    for value in inputs:
        if isinstance(value, _tensor_type()):
            detached = value.detach()
            if value.requires_grad:
                detached.requires_grad_(True)
            result.append(detached)
        else:
            result.append(value)
    return tuple(result)


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(
        isinstance(value, _tensor_type()) and value.requires_grad
        for value in inputs
    ):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None",
            stacklevel=2,
        )


def noop_context_fn():
    return nullcontext(), nullcontext()


@contextmanager
def set_checkpoint_early_stop(enabled: bool):
    global _checkpoint_early_stop
    previous = _checkpoint_early_stop
    _checkpoint_early_stop = bool(enabled)
    try:
        yield
    finally:
        _checkpoint_early_stop = previous


@contextmanager
def set_checkpoint_debug_enabled(enabled: bool | None):
    global _checkpoint_debug_enabled
    previous = _checkpoint_debug_enabled
    _checkpoint_debug_enabled = enabled
    try:
        yield
    finally:
        _checkpoint_debug_enabled = previous


class _CheckpointedFunction:
    def __init__(self, function: Callable[..., Any], **options: Any) -> None:
        self.function = function
        self.options = dict(options)
        functools.update_wrapper(self, function, updated=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return checkpoint(self.function, *args, **self.options, **kwargs)

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        getter = getattr(self.function, "__get__", None)
        if getter is None:
            return self
        return type(self)(getter(instance, owner), **self.options)


def checkpoint(
    function: Callable[..., Any] | None = None,
    *args: Any,
    use_reentrant: bool | None = None,
    preserve_rng_state: bool = True,
    context_fn: Callable[[], Any] | None = None,
    determinism_check: str = "default",
    debug: bool = False,
    early_stop: bool | None = None,
    respect_saved_tensors_hooks: bool | None = None,
    **kwargs: Any,
) -> Any:
    if function is None:
        if args or kwargs:
            raise ValueError("a callable is required before runtime arguments")
        return lambda wrapped: _CheckpointedFunction(
            wrapped,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            early_stop=early_stop,
            respect_saved_tensors_hooks=respect_saved_tensors_hooks,
        )

    if use_reentrant is None:
        warnings.warn(
            "use_reentrant should be passed explicitly; the non-reentrant "
            "implementation is recommended for new code",
            stacklevel=2,
        )
        use_reentrant = True
    if respect_saved_tensors_hooks is not None and use_reentrant:
        raise ValueError(
            "respect_saved_tensors_hooks is only supported for non-reentrant checkpointing")
    if early_stop is None:
        early_stop = _checkpoint_early_stop
    if _checkpoint_debug_enabled is not None:
        debug = _checkpoint_debug_enabled

    native_entry = getattr(_native_autograd, "_activation_checkpoint", None)
    if native_entry is None:
        raise RuntimeError("native activation checkpoint entry is unavailable")
    return native_entry(
        function,
        tuple(args),
        dict(kwargs),
        bool(use_reentrant),
        bool(preserve_rng_state),
        context_fn,
        determinism_check,
        bool(debug),
        bool(early_stop),
    )


def checkpoint_sequential(
    function: Callable[..., Any],
    chunks: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if not isinstance(chunks, int) or chunks <= 0:
        raise ValueError("chunks must be a positive integer")
    if hasattr(function, "children"):
        modules = tuple(function.children())
    elif isinstance(function, (tuple, list)):
        modules = tuple(function)
    else:
        return checkpoint(function, *args, **kwargs)
    if len(args) != 1:
        raise ValueError("checkpoint_sequential expects one input value")
    if not modules:
        return args[0]

    segment_count = min(chunks, len(modules))
    quotient, remainder = divmod(len(modules), segment_count)
    segments: list[tuple[Any, ...]] = []
    start = 0
    for index in range(segment_count):
        width = quotient + (1 if index < remainder else 0)
        segments.append(modules[start:start + width])
        start += width

    value = args[0]
    for index, segment in enumerate(segments):
        def run_segment(value: Any, segment: tuple[Any, ...] = segment) -> Any:
            for module in segment:
                value = module(value)
            return value

        def parameters(segment: tuple[Any, ...] = segment):
            for module in segment:
                if hasattr(module, "parameters"):
                    yield from module.parameters()

        run_segment.parameters = parameters
        if index + 1 != len(segments):
            value = checkpoint(run_segment, value, **kwargs)
        else:
            value = run_segment(value)
    return value


class CheckpointPolicy(Enum):
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    MUST_CPU_OFFLOAD = 4
    PREFER_CPU_OFFLOAD = 5


class SelectiveCheckpointContext:
    def __init__(self, *, is_recompute: bool, op_output: Any = None) -> None:
        self.is_recompute = is_recompute
        self.op_output = op_output


class _CachedTensor:
    def __init__(self, value: Any, device: Any = None,
                 requires_grad: bool = False, version: int | None = None) -> None:
        self.value = value
        self.device = device
        self.requires_grad = requires_grad
        self.version = (getattr(value, "_version", None)
                         if version is None else version)

    def unpack(self, allow_mutation: bool) -> Any:
        if not allow_mutation and self.version is not None:
            current = getattr(self.value, "_version", self.version)
            if current != self.version:
                raise RuntimeError(
                    "tensor cached by selective checkpointing was modified")
        result = (self.value.to(self.device, copy=True)
                  if self.device is not None else self.value)
        if self.requires_grad:
            result.requires_grad_(True)
        return result


def _cache_tree(value: Any, *, offload: bool) -> Any:
    tensor_type = _tensor_type()
    if isinstance(value, tensor_type):
        requires_grad = bool(value.requires_grad)
        version = getattr(value, "_version", None)
        if offload and str(value.device.type) != "cpu":
            import tensorplay

            host = value.detach().to(
                tensorplay.Device(tensorplay.DeviceType.CPU), copy=True)
            return _CachedTensor(
                host, value.device, requires_grad=requires_grad,
                version=version)
        return _CachedTensor(
            value.detach(), requires_grad=requires_grad, version=version)
    if isinstance(value, tuple):
        return tuple(_cache_tree(item, offload=offload) for item in value)
    if isinstance(value, list):
        return [_cache_tree(item, offload=offload) for item in value]
    if isinstance(value, dict):
        return {key: _cache_tree(item, offload=offload)
                for key, item in value.items()}
    return value


def _uncache_tree(value: Any, *, allow_mutation: bool) -> Any:
    if isinstance(value, _CachedTensor):
        return value.unpack(allow_mutation)
    if isinstance(value, tuple):
        return tuple(_uncache_tree(item, allow_mutation=allow_mutation)
                     for item in value)
    if isinstance(value, list):
        return [_uncache_tree(item, allow_mutation=allow_mutation)
                for item in value]
    if isinstance(value, dict):
        return {key: _uncache_tree(item, allow_mutation=allow_mutation)
                for key, item in value.items()}
    return value


def _operation_key(function: Any) -> Any:
    try:
        hash(function)
        return function
    except TypeError:
        return id(function)


def _invoke_function(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "self" in kwargs:
        import tensorplay

        name = getattr(function, "__name__", None)
        native = getattr(getattr(tensorplay, "_C", None), name, None)
        if callable(native):
            return native(*args, **kwargs)
    return function(*args, **kwargs)


def _call_policy(policy_fn: Callable[..., Any], output: Any, function: Any,
                 args: tuple[Any, ...], kwargs: dict[str, Any]) -> CheckpointPolicy:
    from tensorplay.overrides import _disable_tensorplay_function

    with _disable_tensorplay_function():
        policy = policy_fn(
            SelectiveCheckpointContext(is_recompute=False, op_output=output),
            function,
            *args,
            **kwargs,
        )
    if isinstance(policy, bool):
        return CheckpointPolicy.MUST_SAVE if policy else CheckpointPolicy.PREFER_RECOMPUTE
    if not isinstance(policy, CheckpointPolicy):
        raise TypeError("selective checkpoint policy must return CheckpointPolicy")
    return policy


class _SelectiveCachingMode(TensorPlayFunctionMode):
    def __init__(self, policy_fn: Callable[..., Any], storage: dict) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.func_counter: defaultdict[Any, int] = defaultdict(int)

    def __enter__(self):
        self.func_counter.clear()
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __tensorplay_function__(self, function, types_, args=(), kwargs=None):
        del types_
        kwargs = {} if kwargs is None else kwargs
        operation = _native_autograd._checkpoint_operation_begin()
        try:
            output = _invoke_function(function, args, kwargs)
        finally:
            _native_autograd._checkpoint_operation_end(operation)

        key = _operation_key(function)
        index = self.func_counter[key]
        self.func_counter[key] += 1
        policy = _call_policy(self.policy_fn, output, function, args, kwargs)
        should_cache = policy in {
            CheckpointPolicy.MUST_SAVE,
            CheckpointPolicy.PREFER_SAVE,
            CheckpointPolicy.MUST_CPU_OFFLOAD,
            CheckpointPolicy.PREFER_CPU_OFFLOAD,
        }
        _native_autograd._checkpoint_operation_cache(operation, should_cache)
        if should_cache:
            self.storage[key][index] = _cache_tree(
                output,
                offload=policy in {
                    CheckpointPolicy.MUST_CPU_OFFLOAD,
                    CheckpointPolicy.PREFER_CPU_OFFLOAD,
                },
            )
        else:
            self.storage[key][index] = _RECOMPUTE
        return output


class _SelectiveReplayMode(TensorPlayFunctionMode):
    def __init__(self, storage: dict, allow_cache_entry_mutation: bool) -> None:
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self.func_counter: defaultdict[Any, int] = defaultdict(int)

    def __enter__(self):
        self.func_counter.clear()
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __tensorplay_function__(self, function, types_, args=(), kwargs=None):
        del types_
        kwargs = {} if kwargs is None else kwargs
        key = _operation_key(function)
        index = self.func_counter[key]
        self.func_counter[key] += 1
        entry = self.storage.get(key, {}).get(index)
        if entry is None:
            raise RuntimeError(
                "selective checkpoint replay encountered an unknown operation")
        if entry is _RECOMPUTE:
            return _invoke_function(function, args, kwargs)
        operation = _native_autograd._checkpoint_operation_begin()
        try:
            _native_autograd._checkpoint_operation_reuse(operation)
            return _uncache_tree(
                entry,
                allow_mutation=self.allow_cache_entry_mutation,
            )
        finally:
            _native_autograd._checkpoint_operation_end(operation)


_RECOMPUTE = object()


def create_selective_checkpoint_contexts(
    policy_fn_or_list: Callable[..., Any] | list[Any],
    allow_cache_entry_mutation: bool = False,
):
    if isinstance(policy_fn_or_list, list):
        selected = set(policy_fn_or_list)

        def policy_fn(context, function, *args, **kwargs):
            del context, args, kwargs
            return (CheckpointPolicy.MUST_SAVE
                    if function in selected
                    else CheckpointPolicy.PREFER_RECOMPUTE)
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be a function or a list")

    storage: defaultdict[Any, dict[int, Any]] = defaultdict(dict)
    return (
        _SelectiveCachingMode(policy_fn, storage),
        _SelectiveReplayMode(storage, allow_cache_entry_mutation),
    )


class GraphExecGroup:
    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    @classmethod
    def _get_current_group(cls):
        return None
