"""Function rewrite tables used by the export pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["PRESERVED_METHODS", "CustomDecompTable"]


class CustomDecompTable(dict[Any, Callable[..., Any]]):
    """A validated mutable mapping from graph targets to replacement callables.

    Entries are keyed by graph target (callable or method name).  Removing a
    key preserves the op from rewriting; ``materialize`` returns a plain dict
    for consumers that require one.
    """

    def __init__(
        self,
        entries: Mapping[Any, Callable[..., Any]] | Iterable[tuple[Any, Callable[..., Any]]] | None = None,
        *,
        defaults: bool = True,
    ) -> None:
        super().__init__()
        self._removed: set[Any] = set()
        self._defaults = dict(_builtin_decompositions()) if defaults else {}
        if entries is not None:
            self.update(entries)
        for key in list(self._removed):
            self.pop(key, None)

    def __setitem__(self, key: Any, value: Callable[..., Any]) -> None:
        if not callable(value):
            raise TypeError(f"decomposition for {key!r} must be callable")
        self._removed.discard(key)
        super().__setitem__(key, value)

    def update(self, other: Mapping[Any, Callable[..., Any]] | Iterable[tuple[Any, Callable[..., Any]]] = (), **kwargs: Any) -> None:
        items = other.items() if hasattr(other, "items") else other
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def register(self, target: Any, replacement: Callable[..., Any]) -> Callable[..., Any]:
        self[target] = replacement
        return replacement

    def remove(self, target: Any) -> Callable[..., Any]:
        """Preserve ``target`` by deleting its decomposition entry."""

        self._removed.add(target)
        try:
            return super().pop(target)
        except KeyError as exc:
            raise KeyError(f"no decomposition registered for {target!r}") from exc

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        self._removed.add(args[0])
        return super().pop(*args, **kwargs)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._removed:
            return default
        entry = dict.get(self, key)
        if entry is not None:
            return entry
        return self._defaults.get(key, default)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or (
            key not in self._removed and key in self._defaults
        )

    def __iter__(self):
        keys = list(dict.keys(self))
        for key in list(self._defaults):
            if key not in self._removed and key not in keys:
                keys.append(key)
        return iter(keys)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def keys(self):
        return {key: None for key in iter(self)}.keys()

    def items(self):
        return [(key, self[key]) for key in iter(self)]

    def values(self):
        return [self[key] for key in iter(self)]

    def materialize(self) -> dict[Any, Callable[..., Any]]:
        """Return a plain dict of effective entries, resolving defaults."""

        merged = dict(self._defaults)
        merged.update(
            {key: dict.__getitem__(self, key) for key in dict.keys(self)}
        )
        for key in self._removed:
            merged.pop(key, None)
        return merged

    def copy(self) -> "CustomDecompTable":
        clone = type(self)(defaults=False)
        clone._defaults = dict(self._defaults)
        clone._removed = set(self._removed)
        for key, value in dict.items(self):
            dict.__setitem__(clone, key, value)
        return clone


def _builtin_decompositions() -> dict[str, Callable[..., Any]]:
    """Name-keyed builders from the shared decomposition registry."""

    try:
        from ..graph.passes.decompose import _DECOMP_METHODS
    except Exception:
        return {}
    return dict(_DECOMP_METHODS)


PRESERVED_METHODS: set[str] = set()
"""Rewrites excluded from the default table; add names here to keep an op
undecomposed unless a caller supplies an explicit table."""


def default_decompositions() -> "CustomDecompTable":
    """The default table: every registered rewrite except preserved ones."""

    table = CustomDecompTable(defaults=True)
    for name in PRESERVED_METHODS:
        if name in table:
            table.remove(name)
    return table
