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


class FakeTensorMode:
    """A region in which a value is described by a stand-in that holds no data.

    Propagation exists to answer what a computation is shaped like, and
    answering by computing it on real values costs the memory those values
    take -- one region's worth of activations, kept alive for as long as the
    graph that recorded them is read.  Inside this region a value is replaced,
    as soon as it has been computed, by one allocated on the meta device: the
    stand-in carries the geometry, the type and the layout of the value it
    stands for, and no element of it can be read.  A graph propagated here
    therefore describes a computation without ever having held one of its
    results.

    Standing in for what is computed rather than for what is handed in is
    what reaches every value a graph reads: what it reads off its module,
    what it allocates for itself, and what it makes out of a call.  A value
    that is already a stand-in stands for itself.

    Nothing here is allocated where the computation would run, so a caller
    that needs to say where a result would live has to be told: ``device``
    names the device the stand-ins stand for.
    """

    def __init__(self, device: Any = None) -> None:
        self.device = device

    def stand_in(self, value: Any) -> Any:
        """A stand-in for a value the region computed, or for one it was given.

        A value computed out of stand-ins is already on the meta device; it is
        still a description of a value rather than one, and saying so is what
        lets a reader tell it apart from a tensor that really has no memory to
        be read out of.  Anything else is described the way an input is.
        """

        if isinstance(value, tp.Tensor) and value.device.type == "meta":
            from tensorplay._higher_order_ops._hop_base import mark_fake_tensor

            return mark_fake_tensor(value, self.device)
        return self.from_tensor(value)

    def from_tensor(self, value: Any) -> Any:
        """A stand-in for ``value``, or ``value`` itself if it is not a tensor.

        The layout is carried over as well as the geometry: whether a value
        is contiguous decides whether reshaping it needs a copy, and which
        strides it has decides what reads it satisfy without one.
        """

        if not isinstance(value, tp.Tensor):
            return value
        if value.device.type == "meta":
            return value
        from tensorplay._higher_order_ops._hop_base import mark_fake_tensor

        shape = [int(size) for size in value.shape]
        stand_in = tp.empty(list(shape), dtype=value.dtype, device="meta")
        strides = [int(stride) for stride in value.stride()]
        if strides != list(stand_in.stride()) or value.storage_offset() != 0:
            stand_in = stand_in.as_strided(
                list(shape), list(strides), int(value.storage_offset())
            )
        # Recorded as a stand-in, so that what reads it can tell it apart from a
        # value that is really on the meta device and holds nothing to describe.
        return mark_fake_tensor(stand_in, self.device)
