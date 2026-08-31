"""Shared state and traversal helpers for sharded modules."""

from contextlib import nullcontext
from enum import Enum, auto
from typing import Any, Callable, Iterable

import tensorplay as tp

__all__ = [
    "FSDP_PREFIX",
    "FSDP_WRAPPED_MODULE",
    "TrainingState",
    "HandleTrainingState",
    "collect_grad_tensors",
    "replace_grad_tensors",
    "_FSDPState",
    "_FSDPDeviceHandle",
    "_get_module_fsdp_state",
]

FSDP_PREFIX = "_fsdp_wrapped_module"
FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"


class TrainingState(Enum):
    IDLE = auto()
    FORWARD_BACKWARD = auto()
    FORWARD = auto()
    PRE_BACKWARD = auto()
    POST_BACKWARD = auto()


class HandleTrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


class _FSDPDeviceHandle:
    def __init__(self, device_type: str = "cpu") -> None:
        self.device_type = device_type

    def current_stream(self, device: Any = None) -> Any:
        del device
        cuda = getattr(tp, "cuda", None)
        return cuda.current_stream() if cuda is not None and hasattr(cuda, "current_stream") else None

    def synchronize(self, device: Any = None) -> None:
        del device
        cuda = getattr(tp, "cuda", None)
        if cuda is not None and hasattr(cuda, "synchronize"):
            cuda.synchronize()

    def __getattr__(self, name: str) -> Any:
        cuda = getattr(tp, "cuda", None)
        if cuda is None:
            raise AttributeError(name)
        return getattr(cuda, name)


class _UninitializedDeviceHandle(_FSDPDeviceHandle):
    def __init__(self) -> None:
        super().__init__("cpu")


class _FSDPState:
    def __init__(self) -> None:
        self._fsdp_state = self
        self._training_state = TrainingState.IDLE
        self._handles: list[Any] = []
        self._state_dict_type = None
        self._state_dict_config = None
        self._optim_state_dict_config = None


def _get_module_fsdp_state(module: Any) -> Any:
    return getattr(module, "_fsdp_state", getattr(module, "_fsdp_state_obj", None))


def _get_module_fsdp_state_if_fully_sharded_module(module: Any) -> Any:
    return _get_module_fsdp_state(module)


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(type(value), "_fields")


def _collect_grad_tensors(value: Any, result: list[Any]) -> None:
    if isinstance(value, tp.Tensor):
        grad = getattr(value, "grad", None)
        if grad is not None:
            result.append(grad)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_grad_tensors(item, result)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_grad_tensors(item, result)


def collect_grad_tensors(value: Any) -> list[Any]:
    result: list[Any] = []
    _collect_grad_tensors(value, result)
    return result


def _replace_grad_tensors(value: Any, grads: Iterable[Any], index: list[int]) -> Any:
    if isinstance(value, tp.Tensor):
        if getattr(value, "grad", None) is not None and index[0] < len(grads):
            value.grad = list(grads)[index[0]]
            index[0] += 1
        return value
    if isinstance(value, dict):
        return {key: _replace_grad_tensors(item, grads, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return type(value)(_replace_grad_tensors(item, grads, index) for item in value)
    if isinstance(value, list):
        return [_replace_grad_tensors(item, grads, index) for item in value]
    return value


def replace_grad_tensors(value: Any, grads: Iterable[Any]) -> Any:
    materialized = list(grads)
    return _replace_grad_tensors(value, materialized, [0])


def _collect_grad_tensors(value: Any, result: list[Any]) -> None:
    if isinstance(value, tp.Tensor):
        grad = getattr(value, "grad", None)
        if grad is not None:
            result.append(grad)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_grad_tensors(item, result)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_grad_tensors(item, result)


def _module_handle(module: Any) -> Any:
    state = _get_module_fsdp_state(module)
    return getattr(state, "_handle", None) if state is not None else None


def _has_fsdp_params(module: Any) -> bool:
    return bool(_module_handle(module))


def _get_sharding_strategy(state: Any) -> Any:
    return getattr(state, "sharding_strategy", None)


def clean_tensor_name(name: str) -> str:
    return name.replace("._fsdp_wrapped_module", "")


def _set_fsdp_flattened(param: Any, value: bool) -> None:
    setattr(param, "_fsdp_flattened", bool(value))


def _is_fsdp_flattened(param: Any) -> bool:
    return bool(getattr(param, "_fsdp_flattened", False))


def _named_parameters_with_duplicates(module: Any, prefix: str = ""):
    yield from module.named_parameters(prefix=prefix, remove_duplicate=False)


def _get_param_to_fqns(module: Any) -> dict[Any, list[str]]:
    result: dict[Any, list[str]] = {}
    for name, param in _named_parameters_with_duplicates(module):
        result.setdefault(param, []).append(name)
    return result


def _log_post_backward_hook(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _get_handle_fqns_from_root(root: Any, handle: Any) -> list[str]:
    names = []
    params = set(getattr(handle, "params", ()))
    for name, param in root.named_parameters():
        if param in params:
            names.append(name)
    return names


def _apply_to_modules(root: Any, fn: Callable[[Any], Any]) -> Any:
    for module in root.modules():
        fn(module)
    return root


def _assert_in_training_states(state: Any, states: Iterable[Any]) -> None:
    if getattr(state, "_training_state", None) not in set(states):
        raise RuntimeError("module is not in an active training state")


def _get_root_modules(root: Any) -> list[Any]:
    return [root]


def _override_module_mixed_precision(module: Any, mixed_precision: Any) -> Any:
    setattr(module, "_mixed_precision", mixed_precision)
    return module


def _no_dispatch_record_stream(tensor: Any, stream: Any) -> None:
    record = getattr(tensor, "record_stream", None)
    if record is not None:
        record(stream)
