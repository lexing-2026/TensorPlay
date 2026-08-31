from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .core import reify, unify
from .unification_tools import first, groupby
from .utils import _toposort, freeze
from .variable import Var, isvar

__all__ = [
    "Dispatcher",
    "VarDispatcher",
    "edge",
    "global_namespace",
    "match",
    "ordering",
    "supercedes",
]


class Dispatcher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.funcs: dict[Any, Callable[..., Any]] = {}
        self.ordering: list[Any] = []

    def add(self, signature: tuple[Any, ...], function: Callable[..., Any]) -> None:
        frozen = freeze(signature)
        self.funcs[frozen] = function
        self.ordering = ordering(self.funcs)

    def resolve(self, args: tuple[Any, ...]) -> tuple[Callable[..., Any], dict[Var, Any]]:
        for signature in self.ordering:
            if len(signature) != len(args):
                continue
            substitutions = unify(freeze(args), signature)
            if substitutions is not False:
                return self.funcs[signature], substitutions
        raise NotImplementedError(
            f"no match found for {self.name}: known={self.ordering!r}, input={args!r}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function, _ = self.resolve(args)
        return function(*args, **kwargs)

    def register(self, *signature: Any) -> Callable[[Callable[..., Any]], "Dispatcher"]:
        def decorate(function: Callable[..., Any]) -> "Dispatcher":
            self.add(signature, function)
            return self

        return decorate


class VarDispatcher(Dispatcher):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function, substitutions = self.resolve(args)
        values = {variable.token: value for variable, value in substitutions.items()}
        return function(**values, **kwargs)


global_namespace: dict[str, Dispatcher] = {}


def match(*signature: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Dispatcher]:
    namespace = kwargs.get("namespace", global_namespace)
    dispatcher_type = kwargs.get("Dispatcher", Dispatcher)

    def decorate(function: Callable[..., Any]) -> Dispatcher:
        dispatcher = namespace.setdefault(function.__name__, dispatcher_type(function.__name__))
        dispatcher.add(signature, function)
        return dispatcher

    return decorate


def supercedes(first_signature: Any, second_signature: Any) -> bool:
    if isvar(second_signature) and not isvar(first_signature):
        return True
    substitutions = unify(first_signature, second_signature)
    if substitutions is False:
        return False
    substitutions = {
        key: value
        for key, value in substitutions.items()
        if not isvar(key) or not isvar(value)
    }
    if reify(first_signature, substitutions) == first_signature:
        return True
    if reify(second_signature, substitutions) == second_signature:
        return False
    return False


def edge(first_signature: Any, second_signature: Any, tie_breaker: Callable[[Any], int] = hash) -> bool:
    if supercedes(first_signature, second_signature):
        if supercedes(second_signature, first_signature):
            return tie_breaker(first_signature) > tie_breaker(second_signature)
        return True
    return False


def ordering(signatures: Iterable[Any]) -> list[Any]:
    signatures = [tuple(signature) for signature in signatures]
    edges = [(left, right) for left in signatures for right in signatures if edge(left, right)]
    grouped = groupby(first, edges)
    for signature in signatures:
        grouped.setdefault(signature, [])
    return _toposort({key: [right for _, right in values] for key, values in grouped.items()})
