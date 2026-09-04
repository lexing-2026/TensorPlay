"""Shared state and traversal helpers for sharded modules."""

import dataclasses
import functools
import traceback
from contextlib import nullcontext
from enum import Enum, auto
from typing import Any, Callable, Iterable, Iterator

import tensorplay as tp

from ..utils import _apply_to_tensors

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
    SUMMON_FULL_PARAMS = auto()
    FORWARD = auto()
    PRE_BACKWARD = auto()
    POST_BACKWARD = auto()


class HandleTrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class _FSDPDeviceHandle:
    def __init__(self, device_type: str = "cpu") -> None:
        self.device_type = device_type

    @classmethod
    def from_device(cls, device: Any) -> "_FSDPDeviceHandle":
        device_type = str(getattr(device, "type", device)).split(":", 1)[0]
        backend = getattr(tp, device_type, None)
        if backend is not None:
            return backend
        return cls(device_type)

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

    def __getattribute__(self, name: str) -> Any:
        if name not in {"__class__", "device_type", "__dict__", "__getattribute__"}:
            raise RuntimeError("device handle is not initialized")
        return super().__getattribute__(name)


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
    state = _get_module_fsdp_state(module)
    if state is None:
        return None
    if state is module:
        return state
    mapping = getattr(state, "_fully_sharded_module_to_handle", None)
    if isinstance(mapping, dict) and module in mapping:
        return state
    get_groups = getattr(state, "_all_param_groups", None)
    try:
        groups = get_groups() if callable(get_groups) else ()
    except RuntimeError:
        groups = ()
    for group in groups:
        if module in getattr(group, "modules", ()):
            return state
    registered = getattr(module, "_modules", {}).get("module")
    if registered is not None:
        if registered is getattr(state, "module", None):
            return state
        for group in groups:
            if registered in getattr(group, "modules", ()):
                return state
    return None


def _is_namedtuple(value: Any) -> bool:
    fields = getattr(type(value), "_fields", None)
    return (
        isinstance(value, tuple)
        and hasattr(value, "_asdict")
        and isinstance(fields, tuple)
        and all(isinstance(field, str) for field in fields)
    )


_MAX_TRAVERSE_DEPTH = 128


def _collect_grad_tensors(
    value: Any, result: list[Any], depth: int = 0
) -> None:
    if depth >= _MAX_TRAVERSE_DEPTH:
        raise RuntimeError(
            f"collect_grad_tensors exceeded max depth ({_MAX_TRAVERSE_DEPTH})"
        )
    if isinstance(value, tp.Tensor):
        if bool(getattr(value, "requires_grad", False)):
            result.append(value)
    elif _is_namedtuple(value):
        for item in value:
            _collect_grad_tensors(item, result, depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _collect_grad_tensors(getattr(value, field.name), result, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_grad_tensors(item, result, depth + 1)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_grad_tensors(item, result, depth + 1)


def collect_grad_tensors(value: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    _collect_grad_tensors(value, result)
    return tuple(result)


def _replace_grad_tensors(
    value: Any, grads: Iterator[Any], depth: int = 0
) -> Any:
    if depth >= _MAX_TRAVERSE_DEPTH:
        raise RuntimeError(
            f"replace_grad_tensors exceeded max depth ({_MAX_TRAVERSE_DEPTH})"
        )
    if isinstance(value, tp.Tensor):
        return next(grads) if bool(getattr(value, "requires_grad", False)) else value
    if _is_namedtuple(value):
        items = []
        changed = False
        for item in value:
            replaced = _replace_grad_tensors(item, grads, depth + 1)
            items.append(replaced)
            changed = changed or replaced is not item
        return type(value)(*items) if changed else value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for field in dataclasses.fields(value):
            old_value = getattr(value, field.name)
            new_value = _replace_grad_tensors(old_value, grads, depth + 1)
            if new_value is not old_value:
                changes[field.name] = new_value
        return dataclasses.replace(value, **changes) if changes else value
    if isinstance(value, dict):
        replaced = {
            key: _replace_grad_tensors(item, grads, depth + 1)
            for key, item in value.items()
        }
        if type(value) is dict:
            return replaced
        try:
            return type(value)(replaced)
        except (TypeError, ValueError):
            return replaced
    if isinstance(value, (tuple, list)):
        items = [_replace_grad_tensors(item, grads, depth + 1) for item in value]
        if type(value) is tuple:
            return tuple(items)
        if type(value) is list:
            return items
        try:
            return type(value)(items)
        except TypeError:
            return list(items) if isinstance(value, list) else tuple(items)
    return value


def replace_grad_tensors(value: Any, grads: Iterable[Any]) -> Any:
    iterator = iter(grads)
    result = _replace_grad_tensors(value, iterator)
    sentinel = object()
    if next(iterator, sentinel) is not sentinel:
        raise RuntimeError(
            f"replacement tensors were not consumed while processing {type(value).__qualname__}"
        )
    return result


def _is_composable(state: Any) -> bool:
    return not hasattr(state, "_fully_sharded_module_to_handle")


def _module_handle(state_or_module: Any, module: Any = None) -> Any:
    explicit_module = module is not None
    if not explicit_module:
        module = state_or_module
        state = _get_module_fsdp_state(module)
    else:
        state = state_or_module
    if state is None:
        return None
    if not explicit_module and state is module:
        return getattr(state, "_handle", None)
    mapping = getattr(state, "_fully_sharded_module_to_handle", None)
    if isinstance(mapping, dict):
        return mapping.get(module)
    if not _is_composable(state):
        return getattr(module, "_handle", getattr(state, "_handle", None))
    get_groups = getattr(state, "_all_param_groups", None)
    try:
        groups = get_groups() if callable(get_groups) else ()
    except RuntimeError:
        groups = ()
    for group in groups:
        if module in getattr(group, "modules", ()):
            return group
    registered = getattr(module, "_modules", {}).get("module")
    if registered is getattr(state, "module", None):
        return getattr(state, "_handle", None)
    for group in groups:
        if registered in getattr(group, "modules", ()):
            return group
    return None


def _has_fsdp_params(state_or_module: Any, module: Any = None) -> bool:
    return bool(_module_handle(state_or_module, module))


def _get_sharding_strategy(state: Any) -> Any:
    return getattr(state, "sharding_strategy", None)


def clean_tensor_name(name: str) -> str:
    return name.replace("._fsdp_wrapped_module", "")


def _set_fsdp_flattened(param: Any, value: bool) -> None:
    setattr(param, "_fsdp_flattened", bool(value))


def _is_fsdp_flattened(param: Any) -> bool:
    return bool(getattr(param, "_fsdp_flattened", False))


def _named_parameters_with_duplicates(module: Any, **kwargs: Any) -> list[tuple[str, Any]]:
    if "remove_duplicate" in kwargs:
        raise AssertionError("remove_duplicate is managed by this helper")
    kwargs["remove_duplicate"] = False
    try:
        return list(module.named_parameters(**kwargs))
    except (TypeError, AssertionError):
        kwargs.pop("remove_duplicate", None)
        return list(module.named_parameters(**kwargs))


def _get_param_to_fqns(
    module: Any, dedup_shared_params: bool = True
) -> dict[Any, list[str]]:
    result: dict[Any, list[str]] = {}

    def module_fn(
        current: Any, prefix: str, tree_level: int, param_to_fqns: dict[Any, list[str]]
    ) -> None:
        del tree_level
        for param_name, param in _named_parameters_with_duplicates(
            current, recurse=False
        ):
            try:
                from ._flat_param import FlatParameter
            except ImportError:
                FlatParameter = ()
            local_fqns = getattr(param, "_fqns", [param_name]) if isinstance(param, FlatParameter) else [param_name]
            global_fqns = [clean_tensor_name(prefix + name) for name in local_fqns]
            if param not in param_to_fqns:
                param_to_fqns[param] = global_fqns
            elif isinstance(param, FlatParameter):
                param_to_fqns[param] = global_fqns
            elif not dedup_shared_params:
                param_to_fqns[param].extend(global_fqns)

    def return_fn(param_to_fqns: dict[Any, list[str]]) -> dict[Any, list[str]]:
        return param_to_fqns

    names = [name for name, _ in _named_parameters_with_duplicates(module)]
    return _apply_to_modules(module, module_fn, return_fn, names, result)


def _log_post_backward_hook(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _get_handle_fqns_from_root(root: Any, handle: Any) -> list[str]:
    names = []
    params = set(getattr(handle, "params", ()))
    for name, param in root.named_parameters():
        if param in params:
            names.append(name)
    return names


def _apply_to_modules(
    root: Any,
    module_fn: Callable[..., Any],
    return_fn: Callable[..., Any] | None = None,
    filter_fqns: list[str] | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if return_fn is None:
        for module in root.modules():
            module_fn(module)
        return root
    filter_prefixes: set[str] | None = None
    if filter_fqns is not None:
        filter_prefixes = set()
        for fqn in filter_fqns:
            index = fqn.find(".")
            while index != -1:
                filter_prefixes.add(fqn[: index + 1])
                index = fqn.find(".", index + 1)

    def f(current: Any, prefix: str, tree_level: int) -> None:
        module_fn(current, prefix, tree_level, *args, **kwargs)
        for child_name, child in current.named_children():
            if child is None:
                continue
            child_prefix = prefix + child_name + "."
            if filter_prefixes is not None and child_prefix not in filter_prefixes:
                if child_name in {"_fsdp_wrapped_module", "module"}:
                    child_prefix = prefix
            f(child, child_prefix, tree_level + 1)

    f(root, "", 0)
    return return_fn(*args, **kwargs)


def _assert_in_training_states(state: Any, states: Iterable[Any]) -> None:
    current = getattr(state, "training_state", None)
    if current is None:
        current = getattr(state, "_training_state", None)
    expected = set(states)
    if current not in expected:
        message = f"expected to be in states {expected} but current state is {current}"
        if getattr(state, "rank", None) == 0:
            traceback.print_stack()
        raise ValueError(message)


def _get_root_modules(root: Any) -> Any:
    if not isinstance(root, (set, list, tuple)):
        return [root]
    modules = set(root)
    roots = set(modules)
    descendants = {module for module in modules for module in module.modules()}
    for candidate in tuple(modules):
        if any(candidate is not parent and candidate in parent.modules() for parent in modules):
            roots.discard(candidate)
    return roots if isinstance(root, set) else list(roots & descendants)


_MODULE_TO_INP_DTYPE: dict[Any, Any] = {}


def _override_module_mixed_precision(
    root: Any,
    module_classes_to_override: Iterable[type[Any]],
    wrap_override_dict: dict[str, Any] | None = None,
) -> set[type[Any]]:
    module_classes = tuple(set(module_classes_to_override))
    overrides = wrap_override_dict or {"mixed_precision": None}
    overridden: set[type[Any]] = set()
    for module in root.modules():
        if not isinstance(module, module_classes):
            continue
        overridden.add(type(module))
        module._wrap_overrides = overrides

        def cast_fn(dtype: Any, target: Any, tensor: Any) -> Any:
            is_floating = getattr(tensor, "is_floating_point", None)
            if callable(is_floating) and not is_floating():
                return tensor
            if not callable(is_floating) and not getattr(tensor, "dtype", None):
                return tensor
            old_dtype = getattr(tensor, "dtype", None)
            if old_dtype == dtype:
                return tensor
            _MODULE_TO_INP_DTYPE[target] = old_dtype
            return tensor.to(dtype)

        def forward_pre_hook(target: Any, args: Any) -> Any:
            return _apply_to_tensors(functools.partial(cast_fn, tp.float32, target), args)

        def forward_post_hook(target: Any, args: Any, output: Any) -> Any:
            del args
            old_dtype = _MODULE_TO_INP_DTYPE.get(target)
            if old_dtype is None:
                return output
            return _apply_to_tensors(
                functools.partial(cast_fn, old_dtype, target), output
            )

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_post_hook)
    return overridden


def _no_dispatch_record_stream(tensor: Any, stream: Any) -> None:
    record = getattr(tensor, "record_stream", None)
    if record is not None:
        record(stream)
