"""What shape each operation produces, derived from the shapes it was given.

A backend is handed values rather than tensors, so it cannot ask one what
shape it has.  This is the layer that answers instead: an operation's result
has the shape its operands imply, and for the operations whose result is not a
tensor at all -- a store, an assertion, a name for a scalar -- the answer is
that there is no shape, which is different from a shape of no extents.

The answer is a sequence of extents, each of which is a number or the name of
a tile, because a backend that works on a whole tile at once names its tiles
rather than numbering them and has to be able to say so here.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Protocol

import sympy

import tensorplay as tp

from .loops import V

#: The shape of a value: one entry per extent, each a number or a tile's name,
#: or nothing at all when the value is not something with a shape.
BlockShapeType = Sequence | None


class ShapeVar(Protocol):
    """A value that knows its own shape."""

    @property
    def shape(self) -> BlockShapeType: ...


ShapeArg = Any


def get_broadcasted_shape(a: BlockShapeType, b: BlockShapeType) -> BlockShapeType:
    """The shape two values brought together take when they are read together.

    An extent of one is not an extent, so it takes whatever the other side has;
    two extents that are neither one nor equal have no shape together, which is
    a mistake in the region rather than a shape.
    """

    if not isinstance(a, Sequence):
        raise AssertionError(f"expected a to be a Sequence, got {type(a)}")
    if not isinstance(b, Sequence):
        raise AssertionError(f"expected b to be a Sequence, got {type(b)}")
    return _get_broadcasted_shape(tuple(a), tuple(b))


@functools.lru_cache(None)
def _get_broadcasted_shape(
    a: tuple, b: tuple
) -> BlockShapeType:
    """The broadcast of two shapes, shorter one padded with extents of one."""

    if len(a) > len(b):
        return _get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    elif len(a) < len(b):
        b, a = a, b
        return _get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    else:

        def _get_broadcasted_dim(d1, d2):
            if str(d1) == "1":
                return d2
            elif str(d2) == "1":
                return d1
            if str(d1) != str(d2):
                raise AssertionError(f"expected str(d1) == str(d2), got {d1} != {d2}")
            return d1

        return tuple(_get_broadcasted_dim(d1, d2) for d1, d2 in zip(a, b))


def broadcast_shapes_for_args(args: Sequence) -> BlockShapeType:
    """The shape a whole list of operands reads as, in the order they are read.

    A scalar among them contributes no extents, so a value read from a scalar
    has the shape of the tensors around it.  An operand whose shape is not
    known makes the whole answer unknown, because one unknown extent would make
    the shape a guess.
    """

    result_shape: BlockShapeType = None

    for arg in args:
        if hasattr(arg, "shape"):
            shape = arg.shape
            if shape is None:
                return None
            elif result_shape is None:
                result_shape = tuple(shape)
            else:
                result_shape = get_broadcasted_shape(result_shape, tuple(shape))
        elif isinstance(arg, (int, float)):
            if result_shape is None:
                result_shape = ()
        elif isinstance(arg, tp.dtype):
            continue
        else:
            from .loops import Loops

            if isinstance(arg, Loops):
                # A region of the body rather than a value of it has no shape
                # of its own that could be read here.
                return None
            raise TypeError(f"Unknown type: {type(arg)}")

    return result_shape


class ShapePropagationOpsHandler:
    """The shape each operation produces.

    The operations that produce a value whose shape is the shape of something
    they were given say so; the ones that produce no value at all answer
    nothing, because there is nothing whose shape could be asked for.
    """

    @staticmethod
    def constant(value, dtype) -> BlockShapeType:
        # A constant is a value rather than a tensor, so it has no extents; a
        # backend that keeps a fixed number of tile extents for everything
        # spells it as a tensor of that many extents of one.
        return ()

    @staticmethod
    def store_reduction(name: str, index, value) -> None:
        return None

    @staticmethod
    def reduction(dtype, src_dtype, reduction_type, value):
        raise NotImplementedError

    @staticmethod
    def store(name: str, index, value, mode=None) -> None:
        return None

    @staticmethod
    def to_dtype(
        value,
        dtype,
        src_dtype=None,
        use_compute_types: bool = True,
    ) -> BlockShapeType:
        """A value read as another type is the same shape read differently."""

        return value.shape

    @staticmethod
    def dot(a, b) -> BlockShapeType:
        """A product's result is two tiles, whatever the operands' extents were.

        A backend that works on tiles names the two the result is written as,
        so the answer is those two names rather than anything derived from the
        operands, which is why this is not a broadcast.
        """

        return ("YBLOCK", "XBLOCK")

    @staticmethod
    def index_expr(expr, dtype) -> BlockShapeType:
        # The shape is carried by the expression itself: an index names the
        # extents it is over, so there is nothing to add to it.
        return None

    @staticmethod
    def value_expr(expr, dtype) -> BlockShapeType:
        return None

    @staticmethod
    def load_seed(name: str, offset) -> BlockShapeType:
        return ()

    @staticmethod
    def indirect_indexing(var, size, check: bool = True, wrap_neg: bool = True) -> None:
        return None

    def __getattr__(self, name: str):
        """Any other operation produces the broadcast of what it was given."""

        return lambda *args, **kwargs: broadcast_shapes_for_args(args)

    @staticmethod
    def device_assert_async(cond, msg: str) -> None:
        return None
