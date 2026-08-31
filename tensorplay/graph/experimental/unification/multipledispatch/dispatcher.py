from __future__ import annotations

import inspect
from collections.abc import Callable, Generator, Iterable
from typing import Any
import warnings

from .conflict import AmbiguityWarning, ambiguities, ordering
from .utils import expand_tuples, typename
from .variadic import Variadic, isvariadic

__all__ = [
    "Dispatcher",
    "MDNotImplementedError",
    "MethodDispatcher",
    "ambiguity_warn",
    "halt_ordering",
    "restart_ordering",
    "source",
    "str_signature",
    "variadic_signature_matches",
    "variadic_signature_matches_iter",
    "warning_text",
]


class MDNotImplementedError(NotImplementedError):
    pass


def str_signature(signature: Iterable[type]) -> str:
    return ", ".join(getattr(item, "__name__", str(item)) for item in signature)


def warning_text(name: str, values: set[tuple[tuple[type, ...], tuple[type, ...]]]) -> str:
    rendered = "\n".join(
        "\t" + ", ".join(f"[{str_signature(signature)}]" for signature in pair)
        for pair in values
    )
    return f"ambiguous dispatch for {name}:\n{rendered}"


def ambiguity_warn(dispatcher: "Dispatcher", values: set[tuple[tuple[type, ...], tuple[type, ...]]]) -> None:
    warnings.warn(warning_text(dispatcher.name, values), AmbiguityWarning, stacklevel=2)


def halt_ordering() -> None:
    return None


def restart_ordering(on_ambiguity: Callable[..., None] = ambiguity_warn) -> None:
    del on_ambiguity


def variadic_signature_matches_iter(
    types: tuple[type, ...], signature: tuple[type, ...]
) -> Generator[bool, None, None]:
    if not signature:
        return
    iterator = iter(signature)
    current = next(iterator)
    for value in types:
        yield issubclass(value, current)
        if not isvariadic(current):
            try:
                current = next(iterator)
            except StopIteration:
                return
    try:
        next(iterator)
    except StopIteration:
        yield isvariadic(current)
    else:
        yield False


def variadic_signature_matches(types: tuple[type, ...], signature: tuple[type, ...]) -> bool:
    return bool(signature) and all(variadic_signature_matches_iter(types, signature))


class Dispatcher:
    def __init__(self, name: str, doc: str | None = None) -> None:
        self.name = self.__name__ = name
        self.funcs: dict[tuple[type, ...], Callable[..., Any]] = {}
        self.doc = doc
        self._ordering: list[tuple[type, ...]] | None = None
        self._cache: dict[tuple[type, ...], Callable[..., Any]] = {}

    def register(self, *types: type, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            self.add(types, function, **kwargs)
            return function

        return decorate

    @classmethod
    def get_func_params(cls, function: Callable[..., Any]) -> Iterable[inspect.Parameter] | None:
        try:
            return inspect.signature(function).parameters.values()
        except (TypeError, ValueError):
            return None

    @classmethod
    def get_func_annotations(cls, function: Callable[..., Any]) -> tuple[type, ...] | None:
        params = cls.get_func_params(function)
        if params is None:
            return None
        positional = [
            parameter
            for parameter in params
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        annotations = tuple(parameter.annotation for parameter in positional)
        return annotations if annotations and all(item is not inspect.Parameter.empty for item in annotations) else None

    def add(self, signature: tuple[type, ...], function: Callable[..., Any], **_: Any) -> None:
        if not signature:
            signature = self.get_func_annotations(function) or signature
        if any(isinstance(item, tuple) for item in signature):
            for expanded in expand_tuples(signature):
                self.add(expanded, function)
            return
        normalized: list[type] = []
        for index, item in enumerate(signature):
            if isinstance(item, list):
                if index != len(signature) - 1 or len(item) != 1:
                    raise TypeError("a variadic signature must be the final single-item list")
                normalized.append(Variadic[item[0]])
            elif isinstance(item, type):
                normalized.append(item)
            else:
                raise TypeError(f"dispatch signatures must contain types, got {item!r}")
        self.funcs[tuple(normalized)] = function
        self._ordering = None
        self._cache.clear()

    @property
    def ordering(self) -> list[tuple[type, ...]]:
        if self._ordering is None:
            self._ordering = ordering(self.funcs)
        return self._ordering

    def reorder(self, on_ambiguity: Callable[..., None] = ambiguity_warn) -> list[tuple[type, ...]]:
        self._ordering = ordering(self.funcs)
        conflicts = ambiguities(self.funcs)
        if conflicts:
            on_ambiguity(self, conflicts)
        return self._ordering

    def dispatch(self, *types: type) -> Callable[..., Any] | None:
        if types in self.funcs:
            return self.funcs[types]
        return next(self.dispatch_iter(*types), None)

    def dispatch_iter(self, *types: type) -> Generator[Callable[..., Any], None, None]:
        for signature in self.ordering:
            if len(signature) == len(types) and all(issubclass(value, expected) for value, expected in zip(types, signature)):
                yield self.funcs[signature]
            elif signature and isvariadic(signature[-1]) and variadic_signature_matches(types, signature):
                yield self.funcs[signature]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        types = tuple(type(value) for value in args)
        function = self._cache.get(types)
        if function is None:
            function = self.dispatch(*types)
            if function is None:
                raise NotImplementedError(f"no dispatch signature for {self.name}: <{str_signature(types)}>")
            self._cache[types] = function
        try:
            return function(*args, **kwargs)
        except MDNotImplementedError:
            for fallback in self.dispatch_iter(*types):
                if fallback is function:
                    continue
                try:
                    return fallback(*args, **kwargs)
                except MDNotImplementedError:
                    continue
            raise

    def __str__(self) -> str:
        return f"<dispatched {self.name}>"

    __repr__ = __str__

    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "funcs": self.funcs, "doc": self.doc}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.name = self.__name__ = state["name"]
        self.funcs = state["funcs"]
        self.doc = state.get("doc")
        self._ordering = None
        self._cache = {}


def source(function: Callable[..., Any]) -> str:
    return f"File: {inspect.getsourcefile(function)}\n\n{inspect.getsource(function)}"


class MethodDispatcher(Dispatcher):
    def __get__(self, instance: Any, owner: type) -> "MethodDispatcher":
        self.obj = instance
        self.cls = owner
        return self

    @classmethod
    def get_func_params(cls, function: Callable[..., Any]) -> Iterable[inspect.Parameter] | None:
        params = super().get_func_params(function)
        return list(params)[1:] if params is not None else None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function = self.dispatch(*(type(value) for value in args))
        if function is None:
            raise NotImplementedError(f"no dispatch signature for {self.name}: <{str_signature(type(value) for value in args)}>")
        return function(self.obj, *args, **kwargs)
