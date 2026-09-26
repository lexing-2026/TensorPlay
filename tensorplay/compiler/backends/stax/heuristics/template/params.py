"""Configuration parameters for a kernel template.

A configuration is data before it is anything else: it can be compared, cached and
written down, and a template handed a raw dict cannot tell a caller which parts of
it are a choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

class KernelTemplateParams(ABC):
    """One configuration, in a form that can be written down and read back."""

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        """The configuration as keyword arguments for the template."""

    @abstractmethod
    def to_serializeable_dict(self) -> dict[str, Any]:
        """The configuration as data, for storage and for cache keys."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelTemplateParams":
        """The configuration this data describes."""
class DictKernelTemplateParams(KernelTemplateParams):
    """A configuration held as a dict.

    This is the compatibility layer: it lets a template be wired up before it
    has decided what its parameters mean.  A template whose parameters have
    defaults worth having spells them out in its own class instead, so that a
    caller can leave out what it does not care about.
    """

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = dict(kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def to_serializeable_dict(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DictKernelTemplateParams":
        return cls(data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictKernelTemplateParams):
            return NotImplemented
        return self.kwargs == other.kwargs

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, repr(v)) for k, v in self.kwargs.items())))

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(self.kwargs.items()))
        return f"DictKernelTemplateParams({inner})"
@dataclass(frozen=True)
class GemmTemplateParams(KernelTemplateParams):
    """A product's tile shape, and the launch geometry that goes with it."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_warps: int
    num_stages: int
    choice: str = "triton"

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "BLOCK_K": self.BLOCK_K,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def to_serializeable_dict(self) -> dict[str, Any]:
        return self.to_kwargs()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GemmTemplateParams":
        return cls(**data)
