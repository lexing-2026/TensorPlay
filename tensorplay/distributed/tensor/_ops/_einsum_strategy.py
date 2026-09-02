"""Dimension bookkeeping for Einstein-style contractions."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from .._dtensor_spec import DTensorSpec
from .._op_schema import OpStrategy, PlacementStrategy
from ..placement_types import Partial, Replicate, Shard

__all__ = ["EinsumDims", "gen_einsum_strategies"]


@dataclass(frozen=True)
class EinsumDims:
    input_dims: tuple[tuple[str, ...], ...]
    output_dims: tuple[str, ...]
    contracted_dims: tuple[str, ...]

    @property
    def contracting_dims(self) -> tuple[str, ...]:
        return self.contracted_dims

    @property
    def batch_dims(self) -> tuple[str, ...]:
        return tuple(
            dim
            for dim in self.output_dims
            if all(dim in input_dims for input_dims in self.input_dims)
        )

    @property
    def lhs_out_only_dims(self) -> tuple[str, ...]:
        if len(self.input_dims) < 2:
            return ()
        return tuple(
            dim
            for dim in self.output_dims
            if dim in self.input_dims[0] and dim not in self.input_dims[1]
        )

    @property
    def rhs_out_only_dims(self) -> tuple[str, ...]:
        if len(self.input_dims) < 2:
            return ()
        return tuple(
            dim
            for dim in self.output_dims
            if dim in self.input_dims[1] and dim not in self.input_dims[0]
        )

    @classmethod
    def parse_equation(cls, equation: str) -> tuple[list[str], str]:
        compact = equation.replace(" ", "")
        if "->" not in compact:
            raise ValueError("einsum equation must contain an output expression")
        inputs, output = compact.split("->", 1)
        input_dims = inputs.split(",")
        if len(input_dims) > 2:
            raise ValueError("einsum strategy supports one or two inputs")
        return input_dims, output

    @classmethod
    def parse_dims(cls, input_dims: list[str], output_dim: str) -> "EinsumDims":
        all_dims = sorted(set("".join(input_dims)))
        contracted = tuple(dim for dim in all_dims if dim not in output_dim)
        return cls(
            tuple(tuple(dim for dim in value) for value in input_dims),
            tuple(output_dim),
            contracted,
        )

    @classmethod
    def parse(cls, equation: str) -> "EinsumDims":
        inputs, output = cls.parse_equation(equation)
        return cls.parse_dims(inputs, output)


def gen_einsum_strategies(
    equation: str,
    mesh: Any | None = None,
    *,
    linearity: bool = False,
) -> OpStrategy | list[EinsumDims]:
    """Generate complete mesh strategies for an Einstein-style contraction."""
    if mesh is None:
        return [EinsumDims.parse(equation)]
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    dims = EinsumDims.parse_dims(input_dims, output_dim)
    per_mesh_dim: list[list[Any]] = [[Replicate()] * (len(input_dims) + 1)]
    for batch_dim in dims.batch_dims:
        per_mesh_dim.append(
            [Shard(output_dim.index(batch_dim))]
            + [Shard(item.index(batch_dim)) for item in input_dims]
        )
    for contracted_dim in dims.contracting_dims:
        per_mesh_dim.append(
            [Partial("sum")]
            + [Shard(item.index(contracted_dim)) for item in input_dims]
        )
    for free_dim in dims.lhs_out_only_dims:
        per_mesh_dim.append(
            [Shard(output_dim.index(free_dim)), Shard(input_dims[0].index(free_dim)), Replicate()]
        )
    for free_dim in dims.rhs_out_only_dims:
        per_mesh_dim.append(
            [Shard(output_dim.index(free_dim)), Replicate(), Shard(input_dims[1].index(free_dim))]
        )
    if linearity:
        per_mesh_dim.append([Partial("sum")] * (len(input_dims) + 1))
    mesh_ndim = int(mesh.ndim() if callable(getattr(mesh, "ndim", None)) else mesh.ndim)
    strategies = []
    for combination in itertools.product(per_mesh_dim, repeat=mesh_ndim):
        placements = tuple(tuple(values[index] for values in combination) for index in range(len(input_dims) + 1))
        specs = tuple(DTensorSpec(mesh, value) for value in placements)
        strategies.append(
            PlacementStrategy(
                output_specs=specs[0],
                input_specs=specs[1:],
                redistribute_cost=[[0.0] for _ in input_dims],
            )
        )
    return OpStrategy(strategies)
