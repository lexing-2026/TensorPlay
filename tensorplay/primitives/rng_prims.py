# mypy: allow-untyped-defs
"""Randomness primitives.

Philox-style randomness is modeled as higher-order operators that thread
explicit RNG states through a graph.  This framework has no higher-order
operator machinery and no controllable Philox state object; the registration
point below keeps the namespace complete, and its kernels explain the
limitation when invoked.  For eager randomness, use :func:`tensorplay.rand`,
:func:`tensorplay.normal`, and the in-place fill methods directly.
"""

from typing import cast

import tensorplay
from tensorplay import primitives
from tensorplay.primitives.common import CUDARngStateHelper

__all__ = [
    "PhiloxState",
    "CUDARngStateHelper",
    "philox_rand",
    "philox_rand_like",
    "philox_seed",
    "register_rng_prims",
]


# A minimal, documented stand-in for the explicit RNG state record.
class PhiloxState:
    """Opaque RNG-state handle.

    Explicit-RNG randomness operators pass this record through a lowered
    graph so the same offsets are reproducible.  This framework's eager
    generators are stateful and not graph-threadable, so the handle only
    exists to keep the namespace complete.
    """

    __slots__ = ["seed_", "offset_"]

    def __init__(self, seed: int = 0, offset: int = 0) -> None:
        self.seed_ = seed
        self.offset_ = offset

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PhiloxState)
            and self.seed_ == other.seed_
            and self.offset_ == other.offset_
        )

    def __repr__(self) -> str:
        return f"PhiloxState(seed={self.seed_}, offset={self.offset_})"


def _philox_rand_not_supported(*args, **kwargs):
    raise NotImplementedError(
        "philox_rand is not supported: this framework has no higher-order "
        "randomness operators; use tensorplay.rand / tensorplay.normal instead"
    )


def register_rng_prims() -> None:
    """Register the randomness primitives into the operator registry.

    Kernels are registered but raise on invocation with a message pointing at
    the eager random APIs.
    """
    _register_rng("philox_rand")
    _register_rng("philox_rand_like")
    _register_rng("philox_seed")


def _register_rng(name: str) -> None:
    try:
        qualified = f"prims::{name}"
        if tensorplay.library.has_op(qualified):
            return
        schema = {
            "philox_rand": "(Tensor seed, Tensor offset, SymInt[] size) -> Tensor",
            "philox_rand_like": "(Tensor input, Tensor seed, Tensor offset) -> Tensor",
            "philox_seed": "(SymInt seed, SymInt offset) -> Tensor",
        }[name]
        prim_def = tensorplay.library.custom_op(
            qualified, _philox_rand_not_supported, schema=qualified + schema
        )
        prim_def.register_fake(lambda *a, **k: tensorplay.empty((), dtype=tensorplay.float32))
    except Exception:
        pass
