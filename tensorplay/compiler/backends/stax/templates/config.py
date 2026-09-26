"""Tile shapes and launch geometries, before and after they are fitted.

A tile shape here is what the tile would like; the heuristics decide what it gets
once the problem's size is known.  Splitting the two is what lets one table serve
every call of an operation and still yield a different set of candidates per shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class TritonConfig:
    """A launch geometry: block extents plus how many warps and stages."""

    kwargs: dict[str, int]
    num_stages: int = 3
    num_warps: int = 4

    def as_kwargs(self) -> dict[str, Any]:
        return {**self.kwargs, "num_warps": self.num_warps, "num_stages": self.num_stages}
@dataclass
class BaseConfig:
    """A tile shape, before it has been fitted to a particular problem.

    The extents are what the tile would like; the heuristics decide what it
    gets once the problem's size is known.
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    hint_override: int | None = field(default=None, kw_only=True)

    def tile(self) -> dict[str, int]:
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }
@dataclass
class GemmConfig(BaseConfig):
    """A product's tile shape."""
@dataclass
class ConvConfig(BaseConfig):
    """A convolution's tile shape: the same product, with a spatial extent."""
@dataclass
class DepthwiseConvConfig:
    """A depthwise convolution's tiling, which is not a product at all.

    Each input channel is reduced on its own, so the tile is over the output
    positions, the positions along the axis, and the channels -- not over a
    contraction.
    """

    block_n: int
    block_l: int
    block_c: int
    num_stages: int
    num_warps: int

    def tile(self) -> dict[str, int]:
        return {"BLOCK_N": self.block_n, "BLOCK_L": self.block_l, "BLOCK_C": self.block_c}
