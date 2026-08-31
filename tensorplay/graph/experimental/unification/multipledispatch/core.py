from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .dispatcher import Dispatcher, MethodDispatcher

__all__ = ["dispatch", "ismethod"]

global_namespace: dict[str, Dispatcher] = {}


def dispatch(*types: type, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    namespace: dict[str, Dispatcher] = kwargs.get("namespace", global_namespace)

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        name = function.__name__
        if ismethod(function):
            frame = inspect.currentframe()
            previous = frame.f_back.f_locals.get(name) if frame and frame.f_back else None
            dispatcher = previous if isinstance(previous, MethodDispatcher) else MethodDispatcher(name)
        else:
            dispatcher = namespace.setdefault(name, Dispatcher(name))
        dispatcher.add(tuple(types), function)
        return dispatcher

    return decorate


def ismethod(function: Callable[..., Any]) -> bool:
    try:
        return "self" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
