"""Module selection helpers used by automatic wrapping."""

import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Iterable

from ._common_utils import _get_module_fsdp_state, _override_module_mixed_precision

__all__ = ["_auto_wrap", "_check_nested_wrapping", "_warn_on_overridden_mixed_precision", "_validate_frozen_params", "_get_post_order_named_modules", "_get_managed_param_to_fqn"]


def _auto_wrap(
    root_module: Any,
    auto_wrap_policy: Any,
    ignored_modules: set[Any],
    ignored_params: set[Any],
    root_kwargs: dict[str, Any],
    wrapper_cls: Any,
) -> Any:
    from .wrap import (
        _Policy,
        _construct_wrap_fn,
        _or_policy,
        _post_order_apply,
        _recursive_wrap,
        _run_mixed_precision_override_policy,
        _wrap_module_cls_individually,
    )

    mixed_precision = root_kwargs.get("mixed_precision")
    is_wrapper = inspect.isclass(wrapper_cls)
    _check_nested_wrapping(root_module)
    if isinstance(auto_wrap_policy, _Policy):
        root_kwargs["auto_wrap_policy" if is_wrapper else "policy"] = None
        target_module_to_kwargs = auto_wrap_policy._run_policy(
            root_module, ignored_modules, root_kwargs
        )
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(
                root_module,
                mixed_precision._module_classes_to_ignore,
                ignored_modules,
                root_kwargs,
                target_module_to_kwargs,
            )
            overridden = _override_module_mixed_precision(
                root_module, mixed_precision._module_classes_to_ignore
            )
            _warn_on_overridden_mixed_precision(overridden)
        _validate_frozen_params(
            root_module,
            set(target_module_to_kwargs),
            ignored_params,
            bool(root_kwargs.get("use_orig_params", False)),
        )
        _post_order_apply(
            root_module,
            _construct_wrap_fn(root_module, target_module_to_kwargs, wrapper_cls),
        )
        return root_module

    recursive_kwargs = {
        "module": root_module,
        "auto_wrap_policy": auto_wrap_policy,
        "wrapper_cls": wrapper_cls,
        "ignored_modules": ignored_modules,
        "ignored_params": ignored_params,
        "only_wrap_children": True,
    }
    if mixed_precision is not None:
        overridden = _override_module_mixed_precision(
            root_module, mixed_precision._module_classes_to_ignore
        )
        recursive_kwargs["auto_wrap_policy"] = functools.partial(
            _or_policy,
            policies=[
                auto_wrap_policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
        )
        _warn_on_overridden_mixed_precision(overridden)
    _recursive_wrap(**recursive_kwargs, **root_kwargs)
    return root_module


def _parents(root: Any):
    for parent in root.modules():
        yield "", parent


def _check_nested_wrapping(root_module: Any) -> None:
    for module_name, module in root_module.named_modules():
        if _get_module_fsdp_state(module) is not None:
            raise ValueError(
                "automatic wrapping requires modules without existing sharding "
                f"but found {module_name} in {root_module}"
            )


def _warn_on_overridden_mixed_precision(
    overridden_module_classes: set[type[Any]],
) -> None:
    if not overridden_module_classes:
        return
    warnings.warn(
        "Mixed precision was disabled for separately wrapped module classes: "
        f"{overridden_module_classes}",
        stacklevel=2,
    )


def _validate_frozen_params(
    root_module: Any,
    modules_to_wrap: set[Any] | None = None,
    ignored_params: set[Any] | None = None,
    use_orig_params: bool = False,
) -> None:
    modules_to_wrap = modules_to_wrap or set()
    ignored_params = ignored_params or set()
    visited_modules: set[Any] = set()
    for module_name, module in _get_post_order_named_modules(root_module):
        if module not in modules_to_wrap:
            continue
        param_to_fqn = _get_managed_param_to_fqn(
            module, ignored_params, visited_modules, module_name
        )
        frozen = [fqn for param, fqn in param_to_fqn.items() if not param.requires_grad]
        trainable = [fqn for param, fqn in param_to_fqn.items() if param.requires_grad]
        if not frozen or not trainable:
            continue
        message = (
            f"{module_name} has parameters with mixed requires_grad values: "
            f"trainable={trainable}, frozen={frozen}"
        )
        if use_orig_params:
            warnings.warn(message, stacklevel=2)
        else:
            raise ValueError(message)


def _get_post_order_named_modules(root_module: Any) -> list[tuple[str, Any]]:
    visited_modules = {root_module}
    stack = [("", root_module)]
    reverse_order: list[tuple[str, Any]] = []
    while stack:
        module_name, module = stack.pop()
        reverse_order.append((module_name, module))
        for child_name, child in module.named_children():
            if child is None or child in visited_modules:
                continue
            visited_modules.add(child)
            full_name = child_name if not module_name else f"{module_name}.{child_name}"
            stack.append((full_name, child))
    return list(reversed(reverse_order))


def _get_managed_param_to_fqn(
    root_module: Any,
    ignored_params: set[Any] | None = None,
    visited_modules: set[Any] | None = None,
    root_prefix: str = "",
) -> dict[Any, str]:
    ignored = ignored_params or set()
    visited = visited_modules if visited_modules is not None else set()
    queue = collections.deque([(root_module, root_prefix)])
    visited.add(root_module)
    result: dict[Any, str] = {}
    while queue:
        module, prefix = queue.popleft()
        for name, param in module.named_parameters(recurse=False):
            if param in ignored:
                continue
            result[param] = name if not prefix else f"{prefix}.{name}"
        for child_name, child in module.named_children():
            if child is None or child in visited:
                continue
            visited.add(child)
            child_prefix = child_name if not prefix else f"{prefix}.{child_name}"
            queue.append((child, child_prefix))
    return result
