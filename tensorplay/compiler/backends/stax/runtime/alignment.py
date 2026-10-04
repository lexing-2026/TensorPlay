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


_NATIVE_CHECK: list = []


def _native_metadata_check():
    """The native shape-and-stride check, looked up once; None without one.

    A region checks every input it is handed on every call, so the lookup
    is kept out of that loop.
    """

    if not _NATIVE_CHECK:
        try:
            from tensorplay._C import _assert_tensor_metadata as check
        except ImportError:
            check = None
        _NATIVE_CHECK.append(check)
    return _NATIVE_CHECK[0]


def assert_size_stride_grouped(
    items: Any,
    sizes: Any,
    strides: Any,
    op_name: Any = None,
) -> None:
    """Check that each value still has the shape and distances it was given.

    A compiled kernel is handed buffers by position, and the shapes it was
    written against are numbers that were true when it was written.  If one of
    them is no longer true then the kernel is reading a shape it was not written
    for, and what comes out is not wrong in a way that shows -- it is a number
    from somewhere else.  So the check is here, at the boundary, where the
    answer can still be about the program rather than about the result.

    Checked together rather than one at a time because that is how they were
    written: the sizes and the distances of a whole group are one set of numbers
    the caller promised together, and a partial check would pass a group that was
    promised one way and delivered another.

    Named by ``op_name`` because "a tensor was not the shape it was called with"
    says nothing about which of several calls was wrong, and the caller of a
    generated kernel cannot see which of its values became which.
    """

    c_check = _native_metadata_check()
    for item, size, stride in zip(items, sizes, strides):
        if c_check is not None:
            try:
                c_check(item, size, stride)
                continue
            except (TypeError, ValueError, RuntimeError):
                pass
        if item.shape != size:
            where = f" in {op_name}" if op_name else ""
            raise AssertionError(
                f"Tensor shape mismatch{where}: expected {size!r}, "
                f"got {item.shape!r}"
            )
        if item.stride() != stride:
            where = f" in {op_name}" if op_name else ""
            raise AssertionError(
                f"Tensor stride mismatch{where}: expected {stride!r}, "
                f"got {item.stride()!r}"
            )


__all__ = [
    "DEFAULT_ALIGNMENT",
    "assert_size_stride_grouped",
    "copy_if_misaligned",
    "is_aligned",
]
