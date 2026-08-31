"""Helpers for invoking user-defined operations during graph execution."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "_access_subclass_inner_tensor",
    "_call_custom_autograd_function_in_pre_dispatch",
]


def _access_subclass_inner_tensor(source: Any, attribute: str) -> Any:
    value = getattr(source, attribute, None)
    if value is None or not hasattr(value, "shape"):
        raise AttributeError(f"{attribute!r} is not a tensor-valued attribute")
    return value


def _call_custom_autograd_function_in_pre_dispatch(
    function_class_name: str, *args: Any, **kwargs: Any
) -> Any:
    if not isinstance(function_class_name, str) or "." not in function_class_name:
        raise ValueError("function class name must include a module path")
    module_name, class_name = function_class_name.rsplit(".", 1)
    function_class = getattr(importlib.import_module(module_name), class_name)
    apply = getattr(function_class, "apply", None)
    if not callable(apply):
        raise TypeError(f"{function_class_name!r} does not expose apply")
    return apply(*args, **kwargs)
