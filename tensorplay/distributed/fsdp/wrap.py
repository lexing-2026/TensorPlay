"""Automatic wrapping policies for module trees."""

import contextlib
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Sequence

from tensorplay.nn.modules.container import ModuleDict, ModuleList
from tensorplay.nn.modules.multihead_attention import MultiheadAttention
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

    def _post_order_apply_inner(module: Any, parent: Any = None, name: str = "") -> None:
        if id(module) in visited:
            return
        visited.add(id(module))
        for child_name, child in list(module.named_children()):
            _post_order_apply_inner(child, module, child_name)
        replacement = fn(module)
        if replacement is not None and parent is not None:
            setattr(parent, name, replacement)

    _post_order_apply_inner(root_module)
    return root_module


def _construct_wrap_fn(root_module: Any, target_module_to_kwargs: dict[Any, dict[str, Any]], fsdp_fn: Callable) -> Callable[[Any], Any]:
    def fn(module: Any) -> Any:
        if module is root_module:
            return None
        if module in target_module_to_kwargs:
            return fsdp_fn(module, **target_module_to_kwargs[module])
        return None
    return fn


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
                if not decision:
                    continue
                values = copy.copy(root_kwargs)
                values.update(decision)
                result[module] = values
            else:
                raise ValueError(
                    "custom policy must return False, True, or a configuration mapping"
                )
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
    if force_leaf_modules is None:
        force_leaf_modules = size_based_auto_wrap_policy.FORCE_LEAF_MODULES
    if exclude_wrap_modules is None:
        exclude_wrap_modules = size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES
    is_large = nonwrapped_numel >= min_num_params
    if recurse:
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    return is_large and not isinstance(module, tuple(exclude_wrap_modules))


size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {ModuleList, ModuleDict}
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {MultiheadAttention}


class _ConfigAutoWrap:
    in_autowrap_context = False
    wrapper_cls: Any = None
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: dict[str, Any]) -> None:
        if _ConfigAutoWrap.in_autowrap_context:
            raise RuntimeError("nested automatic wrapping contexts are not supported")
        if "wrapper_cls" not in kwargs:
            raise AssertionError("wrapper_cls is required")
        _ConfigAutoWrap.in_autowrap_context = True
        _ConfigAutoWrap.wrapper_cls = kwargs["wrapper_cls"]
        _ConfigAutoWrap.kwargs = {
            key: value for key, value in kwargs.items() if key != "wrapper_cls"
        }

    @staticmethod
    def disable_autowrap_context() -> None:
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.wrapper_cls = None
        _ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> "_ConfigAutoWrap":
        self.enable_autowrap_context(self.kwargs)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        self.disable_autowrap_context()


_config = _ConfigAutoWrap()


@contextlib.contextmanager
def enable_wrap(wrapper_cls: Any = FullyShardedDataParallel, **kwargs: Any):
    with _ConfigAutoWrap(wrapper_cls=wrapper_cls, **kwargs):
        yield


def wrap(module: Any, **kwargs: Any) -> Any:
    if not _ConfigAutoWrap.in_autowrap_context:
        return module
    if _ConfigAutoWrap.wrapper_cls is None:
        raise AssertionError("wrapper_cls is required")
    options = {**_ConfigAutoWrap.kwargs, **kwargs}
    return _wrap(module, _ConfigAutoWrap.wrapper_cls, **options)


def _wrap(module: Any, wrapper_cls: Any = FullyShardedDataParallel, **kwargs: Any) -> Any:
    if wrapper_cls is None:
        raise AssertionError("wrapper_cls is required")
    if hasattr(module, "_wrap_overrides"):
        kwargs = {**kwargs, **module._wrap_overrides}
    return wrapper_cls(module, **kwargs)


def _recursive_wrap(module: Any, auto_wrap_policy: Callable[..., bool], wrapper_cls: Any, ignored_modules: set[Any], ignored_params: set[Any], only_wrap_children: bool = False, **kwargs: Any) -> tuple[Any, int]:
    if auto_wrap_policy is None:
        raise AssertionError("auto_wrap_policy is required")
    if wrapper_cls is None:
        raise AssertionError("wrapper_cls is required")
    try:
        wrapper_type = wrapper_cls
        for _, child in module.named_modules():
            if child in ignored_modules:
                continue
            if isinstance(child, wrapper_type):
                raise AssertionError(f"child module {child} is already wrapped")
    except TypeError:
        pass
    nonwrapped_numel = sum(
        int(param.numel())
        for param in module.parameters()
        if param not in ignored_params
    )
    if not auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        return module, 0
    total_wrapped_numel = 0
    for name, child in module.named_children():
        if child in ignored_modules:
            continue
        wrapped_child, child_numel = _recursive_wrap(
            child,
            auto_wrap_policy,
            wrapper_cls,
            ignored_modules,
            ignored_params,
            **kwargs,
        )
        setattr(module, name, wrapped_child)
        total_wrapped_numel += child_numel
    remainder = nonwrapped_numel - total_wrapped_numel
    if not only_wrap_children and auto_wrap_policy(
        module=module, recurse=False, nonwrapped_numel=remainder
    ):
        return _wrap(module, wrapper_cls, **kwargs), nonwrapped_numel
    return module, total_wrapped_numel
