"""Stax backend: native graph lowering with fused pointwise/reduction
codegen for CPU and Triton for CUDA.

Import-light by design: the lowering module (and through it Triton and the
native extension probes) loads on first attribute access, so importing
``tensorplay`` never pays for this package.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import stax

__all__ = ["stax"]


def __getattr__(name: str) -> Any:
    if name == "stax":
        from .backend import stax

        return stax
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
