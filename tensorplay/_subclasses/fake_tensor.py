"""A tensor that carries no memory: its shape is known, its values are not.

A compiler asks questions of shapes -- how big is this, how is it laid out,
what type is it -- long before anything has been computed, and on a machine
where the answer may not be the answer.  A value traced on one machine, or
before an input is known, cannot be handed those questions directly: running
it would compute something, and running it here would compute the wrong
something.  So a value in that state is described rather than carried: this
module holds that description, the mode that produces it, and the way a
description is told apart from a real value.

The description answers shape questions from a symbolic shape environment, so
two traces of the same computation agree about what is fixed and what is not,
and a guard can be recorded about the part that is not.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import tensorplay as tp

from tensorplay.primitives.common import suggest_memory_format

log = logging.getLogger(__name__)


#: The layouts a tensor can have, as the runtime numbers them.  A value in this
#: module is only ever asked which of these it has, and the number is what the
#: runtime answers in, so the names are here for the code that asks.
class Layout:
    """How a tensor's values sit relative to its shape."""

    Strided = 5
    SparseCOO = 0
    SparseCSR = 1
    SparseCSC = 2
    SparseBSR = 3
    SparseBSC = 4

    @staticmethod
    def is_sparse(layout: int) -> bool:
        return layout != Layout.Strided


#: The layouts that store their values in blocks rather than one per position.
_SPARSE_LAYOUTS = frozenset(
    {Layout.SparseCOO, Layout.SparseCSR, Layout.SparseCSC,
     Layout.SparseBSR, Layout.SparseBSC}
)


def is_sparse_any(t) -> bool:
    """Whether this value is stored sparsely, in any of the sparse layouts."""

    return Layout.is_sparse(t.layout)


def is_sparse_coo(t) -> bool:
    return t.layout == Layout.SparseCOO


def is_sparse_compressed(t) -> bool:
    """Whether the values are behind one or two compressed indices."""

    return t.layout in (
        Layout.SparseCSR,
        Layout.SparseCSC,
        Layout.SparseBSR,
        Layout.SparseBSC,
    )


@dataclasses.dataclass
class TensorMetadata:
    """Everything about a tensor that decides whether two of them are the same.

    Not the values -- a description of a value is not a value, and the whole
    point of this is to name a computation rather than to hold its result.  So
    this is the geometry, the type, the device, and the flags that change what
    reading the values means: whether they are conjugate, negated, quantized,
    or laid out so that reading them in order is not reading them all.
    """

    dtype: Any
    shape: tuple
    stride: tuple
    device: Any
    layout: Any
    memory_format: Any
    storage_offset: Any
    storage_bytes: Any | None
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool
    is_coalesced: bool | None
    dense_dim: int | None
    sparse_dim: int | None

    def _flatten_into(self, result: list, mode, state) -> None:
        """Write this out field by field, turning shapes into hashable text.

        A shape can hold a value that is not yet a number, and hashing that
        directly would hash the symbol rather than what it will be; so the
        fields are handed to the mode, which knows how to write down a value
        that is not settled yet.
        """

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (tuple, list, tp.Size)):
                mode._prep_args_for_hash(result, value, state, [])
            elif isinstance(value, tp.SymInt):
                state.convert_sym_int(result, value)
            else:
                result.append(value)


def extract_tensor_metadata(t) -> TensorMetadata:
    """The metadata of a tensor, with what does not belong to it removed.

    A description meant for hashing should not carry a field that says where
    the storage happens to start: two tensors with the same values in different
    places are the same computation, and a name that distinguishes them is a
    name that misses.
    """

    layout = t.layout
    sparse = is_sparse_any(t)
    # Whether a tensor is contiguous has to be asked with numbers in hand.  A
    # tensor whose shape is not yet a number has no answer to that question,
    # and asking anyway would work it out by arithmetic on symbols -- producing
    # a value that is then thrown away, and producing guards as it went.
    if getattr(t, "_has_symbolic_sizes_strides", False) or sparse:
        memory_format = None
    else:
        memory_format = suggest_memory_format(t)
        if not t.is_contiguous(memory_format=memory_format):
            memory_format = None

    return TensorMetadata(
        t.dtype,
        t.shape,
        t.stride() if layout == Layout.Strided else (),
        t.device,
        layout,
        memory_format,
        t.storage_offset(),
        # A sparse tensor has no storage of its own to measure.
        t.untyped_storage().nbytes() if not sparse else None,
        t.requires_grad,
        t.is_quantized,
        t.is_conj(),
        t.is_neg(),
        t.is_inference,
        t.is_sparse,
        t.is_coalesced() if t.is_sparse else None,
        t.dense_dim() if sparse else None,
        t.sparse_dim() if sparse else None,
    )
