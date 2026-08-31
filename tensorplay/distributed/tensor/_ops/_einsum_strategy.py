"""Dimension bookkeeping for Einstein-style contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

__all__ = ["EinsumDims", "gen_einsum_strategies"]


@dataclass(frozen=True)
class EinsumDims:
    input_dims: tuple[tuple[str, ...], ...]
    output_dims: tuple[str, ...]
    contracted_dims: tuple[str, ...]

    @classmethod
    def parse(cls, equation: str) -> "EinsumDims":
        left, right = equation.replace(" ", "").split("->", 1)
        inputs = tuple(tuple(part) for part in left.split(","))
        output = tuple(right)
        contracted = tuple(sorted(set(char for dims in inputs for char in dims) - set(output)))
        return cls(inputs, output, contracted)


def gen_einsum_strategies(equation: str, *args: Any, **kwargs: Any) -> list[Any]:
    del args, kwargs
    dims = EinsumDims.parse(equation)
    return [dims]
