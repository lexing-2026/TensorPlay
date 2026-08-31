from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["dispatch"]

_namespace: dict[str, "_Dispatch"] = {}


class _Dispatch:
    def __init__(self, name: str, signatures: tuple[type, ...], function: Callable[..., Any]) -> None:
        self.name = name
        self._entries: list[tuple[tuple[type, ...], Callable[..., Any]]] = [(signatures, function)]

    def add(self, signatures: tuple[type, ...], function: Callable[..., Any]) -> "_Dispatch":
        self._entries.append((signatures, function))
        self._entries.sort(key=lambda item: sum(cls is object for cls in item[0]))
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for signatures, function in self._entries:
            if len(signatures) == len(args) and all(
                signature is object or isinstance(value, signature)
                for signature, value in zip(signatures, args)
            ):
                return function(*args, **kwargs)
        raise TypeError(f"no dispatch match for {self.name}{tuple(type(arg).__name__ for arg in args)}")

    def register(self, *signatures: type) -> Callable[[Callable[..., Any]], "_Dispatch"]:
        def decorate(function: Callable[..., Any]) -> "_Dispatch":
            return self.add(signatures, function)

        return decorate


def dispatch(*signatures: type, namespace: dict[str, "_Dispatch"] | None = None) -> Callable[[Callable[..., Any]], _Dispatch]:
    registry = _namespace if namespace is None else namespace

    def decorate(function: Callable[..., Any]) -> _Dispatch:
        dispatcher = registry.get(function.__name__)
        if dispatcher is None:
            dispatcher = _Dispatch(function.__name__, signatures, function)
            registry[function.__name__] = dispatcher
        else:
            dispatcher.add(signatures, function)
        return dispatcher

    return decorate
