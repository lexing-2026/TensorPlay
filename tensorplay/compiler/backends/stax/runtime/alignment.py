"""Reading a value whose memory is not aligned the way the kernel needs it.

A vectorized kernel reads several elements at once, and several-at-once has to
start at an address the hardware can address that way.  A tensor's data does not
necessarily start there: it is a window onto somebody else's allocation, and the
window can begin one element in.

Reading it anyway is what produces a wrong answer rather than a crash, because
the hardware reads a whole vector from whatever address it is given and the
elements it lands on are the wrong ones.  So the value is copied once, into an
allocation that does start where it should, and the copy is what the kernel
reads.  A value that is already aligned is handed back as it is, so a tensor
that happens to be aligned does not pay for a copy it does not need.
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp

#: What "aligned" means for a vector load: the largest vector the runtime will
#: use, so a value copied for one kernel is aligned for every kernel.
DEFAULT_ALIGNMENT = 64


def is_aligned(tensor: Any, alignment: int = DEFAULT_ALIGNMENT) -> bool:
    """Whether this value's memory starts where a wide load may be issued.

    The address is asked of the storage rather than computed from the shape,
    because the value may be a window and the window's start is what decides.
    """

    if not tensor.numel():
        # An empty value reads nothing, so where it would have started does not
        # matter and asking would only be a question about an empty allocation.
        return True
    return int(tensor.data_ptr()) % alignment == 0


def copy_if_misaligned(
    tensor: Any,
    alignment: int = DEFAULT_ALIGNMENT,
) -> Any:
    """``tensor`` when its memory is aligned, and a copy of it when it is not.

    The copy is contiguous, so the result is aligned as well as readable, and
    the two are otherwise the same value: the copy changes where the value
    lives and nothing about what it holds.
    """

    if is_aligned(tensor, alignment):
        return tensor
    # A contiguous copy starts at the start of a fresh allocation, which is
    # aligned; going through the element type keeps the copy the same type
    # rather than a reinterpretation of one.
    return tensor.clone().contiguous()


__all__ = ["DEFAULT_ALIGNMENT", "copy_if_misaligned", "is_aligned"]
