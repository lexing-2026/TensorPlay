"""Automatic wrapping policies for module trees."""

import contextlib
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Sequence

from .fully_sharded_data_parallel import FullyShardedDataParallel

__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]


def _post_order_apply(root_module: Any, fn: Callable[[Any], Any]) -> Any:
    visited: set[int] = set()

    def visit(module: Any, parent: Any = None, name: str = "") -> None:
        if id(module) in visited:
            return
        visited.add(id(module))
        for child_name, child in list(module.named_children()):
            visit(child, module, child_name)
        replacement = fn(module)
        if replacement is not None and parent is not None:
            setattr(parent, name, replacement)

    visit(root_module)
    return root_module


def _construct_wrap_fn(root_module: Any, target_module_to_kwargs: dict[Any, dict[str, Any]], fsdp_fn: Callable) -> Callable[[Any], Any]:
    def apply(module: Any) -> Any:
        if module is root_module:
            return None
        if module in target_module_to_kwargs:
            return fsdp_fn(module, **target_module_to_kwargs[module])
        return None
    return apply


def _run_mixed_precision_override_policy(root_module: Any, module_classes: Iterable[type], ignored_modules: set[Any], root_kwargs: dict[str, Any], target_module_to_kwargs: dict[Any, dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    for module in root_module.modules():
        if module in ignored_modules or not isinstance(module, tuple(module_classes)):
            continue
        kwargs = target_module_to_kwargs.setdefault(module, copy.copy(root_kwargs))
        kwargs["mixed_precision"] = None
    return target_module_to_kwargs


def always_wrap_policy(*args: Any, **kwargs: Any) -> bool:
    del args, kwargs
    return True


class _Policy(ABC):
    @abstractmethod
    def _run_policy(self, root_module: Any, ignored_modules: set[Any], root_kwargs: dict[str, Any]) -> dict[Any, dict[str, Any]]:
        raise NotImplementedError


def _module_wrap_policy(module: Any, recurse: bool, nonwrapped_numel: int, module_classes: set[type]) -> bool:
    del nonwrapped_numel
    return True if recurse else isinstance(module, tuple(module_classes))


class ModuleWrapPolicy(_Policy):
    def __init__(self, module_classes: Iterable[type]) -> None:
        self._module_classes = set(module_classes)

    def _run_policy(self, root_module: Any, ignored_modules: set[Any], root_kwargs: dict[str, Any]) -> dict[Any, dict[str, Any]]:
        return {module: copy.copy(root_kwargs) for module in root_module.modules() if module not in ignored_modules and isinstance(module, tuple(self._module_classes)) and module is not root_module}

    def __call__(self, module: Any, recurse: bool, nonwrapped_numel: int = 0) -> bool:
        return _module_wrap_policy(module, recurse, nonwrapped_numel, self._module_classes)

    def __repr__(self) -> str:
        return f"ModuleWrapPolicy(module_classes={self._module_classes!r})"


class CustomPolicy(_Policy):
    def __init__(self, lambda_fn: Callable[[Any], bool | dict[str, Any]]) -> None:
        self._lambda_fn = lambda_fn

    def _run_policy(self, root_module: Any, ignored_modules: set[Any], root_kwargs: dict[str, Any]) -> dict[Any, dict[str, Any]]:
        result: dict[Any, dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            decision = self._lambda_fn(module)
            if isinstance(decision, bool):
                if decision:
                    result[module] = copy.copy(root_kwargs)
            elif isinstance(decision, dict):
                values = copy.copy(root_kwargs)
                values.update(decision)
                result[module] = values
            else:
                raise TypeError("custom policy must return bool or a mapping")
        return result


def lambda_auto_wrap_policy(module: Any, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable[[Any], bool]) -> bool:
    del nonwrapped_numel
    return True if recurse else bool(lambda_fn(module))


def transformer_auto_wrap_policy(module: Any, recurse: bool, nonwrapped_numel: int, transformer_layer_cls: set[type]) -> bool:
    return _module_wrap_policy(module, recurse, nonwrapped_numel, transformer_layer_cls)


def _wrap_module_cls_individually(module: Any, module_classes: Sequence[type], recurse: bool, *args: Any, **kwargs: Any) -> bool:
    del args, kwargs
    return True if recurse else isinstance(module, tuple(module_classes))


def _or_policy(module: Any, recurse: bool, nonwrapped_numel: int, policies: Iterable[Callable[..., bool]]) -> bool:
    return any(policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel) for policy in policies)


def size_based_auto_wrap_policy(module: Any, recurse: bool, nonwrapped_numel: int, min_num_params: int = int(1e8), force_leaf_modules: set[type] | None = None, exclude_wrap_modules: set[type] | None = None) -> bool:
    force_leaf_modules = force_leaf_modules or set()
    exclude_wrap_modules = exclude_wrap_modules or set()
    if isinstance(module, tuple(force_leaf_modules)):
        return False
    if recurse:
        return not isinstance(module, tuple(force_leaf_modules))
    return nonwrapped_numel >= min_num_params and not isinstance(module, tuple(exclude_wrap_modules))


class _ConfigAutoWrap:
    def __init__(self) -> None:
        self.wrapper_cls: Any = FullyShardedDataParallel
        self.kwargs: dict[str, Any] = {}

    def enable_autowrap_context(self, kwargs: dict[str, Any]) -> None:
        self.wrapper_cls = kwargs.pop("wrapper_cls", FullyShardedDataParallel)
        self.kwargs = kwargs

    def disable_autowrap_context(self) -> None:
        self.wrapper_cls = FullyShardedDataParallel
        self.kwargs = {}

    def __enter__(self) -> "_ConfigAutoWrap":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        self.disable_autowrap_context()


_config = _ConfigAutoWrap()


@contextlib.contextmanager
def enable_wrap(wrapper_cls: Any = FullyShardedDataParallel, **kwargs: Any):
    previous_cls, previous_kwargs = _config.wrapper_cls, _config.kwargs
    _config.wrapper_cls = wrapper_cls
    _config.kwargs = dict(kwargs)
    try:
        yield
    finally:
        _config.wrapper_cls, _config.kwargs = previous_cls, previous_kwargs


def wrap(module: Any, **kwargs: Any) -> Any:
    options = dict(_config.kwargs)
    options.update(kwargs)
    return _config.wrapper_cls(module, **options)


def _wrap(module: Any, wrapper_cls: Any = FullyShardedDataParallel, **kwargs: Any) -> Any:
    return wrapper_cls(module, **kwargs)


def _recursive_wrap(module: Any, auto_wrap_policy: Callable[..., bool], wrapper_cls: Any, ignored_modules: set[Any], ignored_params: set[Any], only_wrap_children: bool = False, **kwargs: Any) -> tuple[Any, int]:
    del ignored_params
    total = 0
    for name, child in list(module.named_children()):
        if child in ignored_modules:
            continue
        child, child_count = _recursive_wrap(child, auto_wrap_policy, wrapper_cls, ignored_modules, set(), False, **kwargs)
        total += child_count
        setattr(module, name, child)
    own = sum(int(param.numel()) for param in module.parameters(recurse=False))
    total += own
    if (not only_wrap_children and auto_wrap_policy(module, False, total)):
        return wrapper_cls(module, **kwargs), 0
    return module, total
