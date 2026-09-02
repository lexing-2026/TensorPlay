"""Module selection helpers used by automatic wrapping."""

from typing import Any, Iterable

__all__ = ["_auto_wrap", "_check_nested_wrapping", "_warn_on_overridden_mixed_precision", "_validate_frozen_params", "_get_post_order_named_modules", "_get_managed_param_to_fqn"]


def _auto_wrap(root_module: Any, auto_wrap_policy: Any, ignored_modules: set[Any], ignored_params: set[Any], root_kwargs: dict[str, Any], wrapper_cls: Any) -> Any:
    if hasattr(auto_wrap_policy, "_run_policy"):
        targets = auto_wrap_policy._run_policy(root_module, ignored_modules, root_kwargs)
    else:
        targets = {}
        for module in root_module.modules():
            count = sum(int(param.numel()) for param in module.parameters())
            if module not in ignored_modules and auto_wrap_policy(module=module, recurse=False, nonwrapped_numel=count):
                targets[module] = dict(root_kwargs)
    for module in _get_post_order_named_modules(root_module):
        if module is root_module or module not in targets:
            continue
        kwargs = targets[module]
        replacement = wrapper_cls(module, **kwargs)
        replaced = False
        for parent in list(root_module.modules()):
            for child_name, child in list(parent.named_children()):
                if child is module:
                    setattr(parent, child_name, replacement)
                    replaced = True
                    break
            if replaced:
                break
    return root_module


def _parents(root: Any):
    for parent in root.modules():
        yield "", parent


def _check_nested_wrapping(root_module: Any) -> None:
    del root_module


def _warn_on_overridden_mixed_precision(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _validate_frozen_params(root_module: Any) -> None:
    del root_module


def _get_post_order_named_modules(root_module: Any) -> list[Any]:
    result: list[Any] = []
    visited: set[int] = set()

    def visit(module: Any) -> None:
        if id(module) in visited:
            return
        visited.add(id(module))
        for _, child in module.named_children():
            visit(child)
        result.append(module)

    visit(root_module)
    return result


def _get_managed_param_to_fqn(root_module: Any, ignored_params: set[Any] | None = None) -> dict[Any, str]:
    ignored = ignored_params or set()
    return {param: name for name, param in root_module.named_parameters() if param not in ignored}
