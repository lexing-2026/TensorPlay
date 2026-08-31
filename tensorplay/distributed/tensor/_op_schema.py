"""Structured operation records used by layout rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["OpSchema"]


@dataclass(frozen=True)
class OpSchema:
    func: Any
    args_schema: tuple[Any, ...] = ()
    kwargs_schema: tuple[tuple[str, Any], ...] = ()

    @property
    def args(self) -> tuple[Any, ...]:
        return self.args_schema

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs_schema)

    def __init__(self, func: Any, args_schema: Any = (), kwargs_schema: Any = ()) -> None:
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "args_schema", tuple(args_schema))
        object.__setattr__(self, "kwargs_schema", tuple(kwargs_schema.items()) if isinstance(kwargs_schema, dict) else tuple(kwargs_schema))
