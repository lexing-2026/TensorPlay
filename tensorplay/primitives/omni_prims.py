# mypy: allow-untyped-defs
"""Operations the attention kernels are built from, which nothing else provides.

These are the operations a kernel body is written in terms of when it is
captured as a graph rather than written out: a body that adds a value into a
position of a larger value is a graph node, and what the node means to a kernel
that contains it is not the same as what it means on its own -- it is a
gradient, accumulated into a value that other positions may also be adding to.

Declared here rather than computed here, for the same reason the rest of the
primitives are: a graph records which operation was called, and the operation
is the thing that has to survive into the kernel.
"""

from typing import Any

import tensorplay

__all__ = [
    "zeros_and_scatter",
    "register_omni_prims",
]


def _zeros_and_scatter_impl(
    shape: Any, indices: Any, values: Any
) -> "tensorplay.Tensor":
    """A value of zeros with a value added into it at a position.

    On its own this is addition into a value, which is an ordinary operation.
    Inside a captured graph it is the gradient a captured value contributes: a
    sum over every position that was read, where several positions may be the
    same one.  Which of them arrives first is not something that can be known
    beforehand, so the eager form cannot be the one the kernel uses -- the
    kernel has to be told, by having this be an operation of its own, that the
    additions are to be made as they happen.
    """

    if not isinstance(indices, (list, tuple)):
        indices = [indices]
    out = tensorplay.zeros(shape, dtype=values.dtype, device=values.device)
    return out.index_put(tuple(indices), values, accumulate=True)


def register_omni_prims() -> None:
    """Register the operations the attention kernels are built from."""

    _register("zeros_and_scatter", _zeros_and_scatter_impl)


def _register(name: str, impl: Any) -> None:
    try:
        qualified = f"omni::{name}"
        if tensorplay.library.has_op(qualified):
            return
        schema = {
            "zeros_and_scatter": (
                "(SymInt[] shape, Tensor[] indices, Tensor values) -> Tensor"
            ),
        }[name]
        prim_def = tensorplay.library.custom_op(
            qualified, impl, schema=qualified + schema
        )
        try:
            prim_def.register_fake(
                lambda shape, indices, values: tensorplay.zeros(
                    shape, dtype=values.dtype, device=values.device
                )
            )
        except Exception:
            # A shape that is not known while the graph is being built cannot
            # be summarised as a value; the registration of the operation
            # itself still proceeds, and the shape is worked out where it is
            # used.
            pass
    except Exception:
        pass
