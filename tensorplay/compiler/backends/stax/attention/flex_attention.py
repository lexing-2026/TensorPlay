"""Carrying a structured position through arithmetic that thinks in offsets.

A position in a tensor is usually one number: the distance from the start of
the buffer, which is what indexing arithmetic wants because a load is at an
offset.  The emitter used for these kernels wants the opposite -- the position
as one number per dimension, so it can write ``tensor[i, j]`` and be answerable
for strides itself.

Both are wanted at once, because the position is built by arithmetic that only
understands offsets and is finally handed to an emitter that only understands
dimensions.  So the position stays one expression, and the dimensions ride
along inside it: a value that is an expression to everything that touches it,
and a tuple of coordinates to the one place that asks.
"""

from __future__ import annotations

import sympy


class HierarchicalIndex(sympy.Function):
    """One position in a tensor, held as one number per dimension.

    Nothing is done to the value it holds.  It is not simplified, not flattened
    and not reordered, because the dimensions are not a number that could be:
    they are a position, and a position that was rearranged would be a
    different one.  So evaluating it produces nothing, which is what tells the
    expression machinery to carry the node as it stands.

    A value like this is meant to be short-lived -- built where a position is
    produced and taken apart where it is consumed -- and only the emitter for
    these kernels reads it, by taking the node's arguments as the coordinates.
    """

    @classmethod
    def eval(cls, *args):
        return None
