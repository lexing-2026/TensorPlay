"""Checks the generated wrapper makes about the values it was handed.

A wrapper compiled for one shape, one stride and one element type is only
correct for those.  Handed anything else it computes on a buffer laid out
differently from the one it was written for, and the result is a wrong answer
with nothing to point at.

So the generated source checks before it computes.  The check is in the
generated code rather than around it because the thing that knows what the
kernel was compiled for is the code that was generated for it, and a check
written anywhere else would have to be told that.
"""

from __future__ import annotations
from ..compile_log import timed_block

import contextlib
import logging
import time

log = logging.getLogger(__name__)

import functools
import operator
from typing import Any, Hashable

import sympy

import tensorplay as tp

# Re-exported rather than used here: other modules reach for these through this
# one, so that a caller asking where a cache lives does not have to know which
# of these it is.
from .cache_dir_utils import (  # noqa: F401
    cache_dir,
    default_cache_dir,
    triton_cache_dir,
)


#: Whether output may carry colour.  Off when the module that provides it is
#: absent, and off when output is not a terminal, so that a message piped
#: somewhere carries no escape sequences.
HAS_COLORAMA = False

try:  # noqa: SIM105
    import colorama

    HAS_COLORAMA = True
except ImportError:
    pass


if HAS_COLORAMA:

    def _color_text(msg: str, color: str) -> str:
        return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET

else:

    def _color_text(msg: str, color: str) -> str:
        return msg


def green_text(msg: str) -> str:
    """``msg`` in green, where colour is available."""

    return _color_text(msg, "green")


def yellow_text(msg: str) -> str:
    """``msg`` in yellow, where colour is available."""

    return _color_text(msg, "yellow")


def red_text(msg: str) -> str:
    """``msg`` in red, where colour is available."""

    return _color_text(msg, "red")


def get_num_bytes(*args, num_in_out_args: int = 0) -> int:
    """How many bytes a kernel's arguments add up to.

    A value that is both read and written is counted twice, because it is both
    read and written -- which is what makes a kernel that updates in place look
    like the traffic it actually causes.  The first `num_in_out_args` arguments
    are the ones that are both.
    """

    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, tp.Tensor)
    )


def create_bandwidth_info_str(
    ms: float,
    num_gb: float,
    gb_per_s: float,
    prefix: str = "",
    suffix: str = "",
    color: bool = True,
) -> str:
    """One measured kernel as a line, with the slow ones marked.

    A kernel that moves little and takes long is usually not limited by the
    memory rather than by the arithmetic, and that is worth seeing at a glance
    while reading a list of measurements.
    """

    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def validate_triton_config(cfg) -> None:
    """Refuse a configuration that would not survive being written down.

    A configuration may carry a hook to run before the launch.  A hook is a
    function, and a function is not written down with a configuration, so a
    configuration carrying one would come back from a cache without it -- and
    the launch would then differ from the one that was measured.  Rather than
    discover that later, it is refused here.
    """

    if getattr(cfg, "pre_hook", None) is not None:
        raise AssertionError("a configuration carrying a pre-launch hook cannot be kept")


def triton_config_to_hashable(cfg) -> "Hashable":
    """A configuration reduced to something that can key a dictionary.

    What identifies a configuration is every number in it, so the reduction
    keeps them all and orders them, which makes two configurations that
    differ in any of them unequal and two that do not equal.
    """

    items = sorted(cfg.kwargs.items())
    items.append(("num_warps", cfg.num_warps))
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def conditional_product(*args: int) -> int:
    """The product of the arguments that are not zero.

    A zero would make the whole product zero, so an argument that is zero is
    left out rather than multiplied in: what is wanted is the size of what is
    actually there.
    """

    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(number: int, denom: int) -> int:
    """The number of steps of ``denom`` needed to reach ``number``."""

    return -(number // -denom)


def is_power_of_2(n: int) -> bool:
    """Whether ``n`` is a power of two."""

    return n > 0 and n & n - 1 == 0


def get_first_attr(obj, *attrs):
    """The first of several names that this object answers to.

    A runtime renames what it calls things between versions, and a caller that
    has to know which version it is talking about in order to read one value is
    a caller that breaks on a version bump.  So the names are offered in order
    and the first one present is taken; a value that is under none of them is
    refused, because a default would be a value nobody chose.
    """

    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


def triton_hash_to_path_key(key: str) -> str:
    """A kernel's identity, spelled the way a file may hold it.

    What a hash looks like inside the runtime and what may appear in a path name
    have not always been the same: the hash was used directly, then encoded one
    way, then another.  So the encoding is asked of the runtime that is present
    and a key no encoding is offered for is used as it is, which keeps a kernel
    findable under a runtime that offers no help rather than under a name this
    code invented.
    """

    try:
        from triton.runtime.cache import _base64

        return _base64(key)
    except Exception:
        try:
            from triton.runtime.cache import _base32

            return _base32(key)
        except Exception:
            return key


@contextlib.contextmanager
def timed_block(name: str, log_pt2_compile_event: bool = False):
    """How long a named piece of work took, and where it was spent.

    A pass that runs for a long time is worth knowing the name of, and a pass
    that runs for a short time is worth knowing was not the one that ran for a
    long time.  The name is therefore always recorded, and the elapsed time is
    left to whoever reads the log rather than printed from here, so that a
    caller can decide what a duration means for the work it asked about.

    ``log_pt2_compile_event`` says the work is a compilation step whose length
    is worth keeping even when nothing failed; without it the duration is
    recorded at the level that a failure would show, so that a normal run is
    not full of numbers nobody asked for.
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.log(
            logging.DEBUG if not log_pt2_compile_event else logging.INFO,
            "%s took %.3fs",
            name,
            elapsed,
        )


def get_max_y_grid() -> int:
    """How many blocks the second grid axis will hold.

    A launch whose batch is spread over the second axis alone fails once the
    batch outgrows what that axis accepts, so the axis is given a stated limit
    and the batch is split across two axes to stay under it.

    The limit is the one the driver documents rather than one read back from the
    device, because it is a property of the launch interface every device
    implements rather than of any one device, and a device that reported a
    smaller number would be a reason to be more careful rather than a different
    answer to give.
    """

    return 65535


def next_power_of_2(n: int) -> int:
    """The smallest power of two that is at least ``n``."""

    if isinstance(n, sympy.Integer):
        n = int(n)
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def last_power_of_2(n: int) -> int:
    """The largest power of two that is at most ``n``."""

    if isinstance(n, sympy.Integer):
        n = int(n)
    if n <= 0:
        return 1
    return 1 << ((n - 1).bit_length() - 1)




def assert_size_stride(
    name: str,
    size: Any,
    stride: Any,
    op_name: str = "",
) -> None:
    """Raise unless a buffer has the shape and stride it was compiled for.

    The name is the buffer's own, so the message says which argument of which
    call was wrong rather than that something was.
    """

    tensor = _current_buffer(name)
    if tensor is None:
        return
    expected_size = tuple(int(s) for s in size)
    expected_stride = tuple(int(s) for s in stride)
    actual_size = tuple(int(s) for s in tensor.shape)
    actual_stride = tuple(int(s) for s in tensor.stride())
    if actual_size != expected_size or actual_stride != expected_stride:
        where = f" for {op_name}" if op_name else ""
        raise AssertionError(
            f"buffer {name!r}{where} was compiled for shape {expected_size} and "
            f"stride {expected_stride} but arrived with shape {actual_size} and "
            f"stride {actual_stride}"
        )


def assert_tensor_metadata(
    name: str,
    size: Any,
    stride: Any,
    dtype: Any,
    op_name: str = "",
) -> None:
    """Raise unless a value has the shape, stride and element type it was
    compiled for.

    The element type is checked as well because two values of the same shape and
    stride can still be read differently, and a kernel written for one element
    size reading the other produces a number rather than an error.
    """

    tensor = _current_buffer(name)
    if tensor is None:
        return
    assert_size_stride(name, size, stride, op_name)
    if tensor.dtype != dtype:
        where = f" for {op_name}" if op_name else ""
        raise AssertionError(
            f"value {name!r}{where} was compiled for element type {dtype} but "
            f"arrived with element type {tensor.dtype}"
        )


def assert_alignment(name: str, alignment: int) -> None:
    """Raise unless a buffer's memory is aligned as the kernel needs it to be.

    Separate from the shape check because the two fail for different reasons:
    a wrong shape means the caller built something else, while a misaligned
    buffer of the right shape means the caller built the right thing somewhere
    the hardware cannot read it from.
    """

    tensor = _current_buffer(name)
    if tensor is None:
        return
    if tensor.numel() and int(tensor.data_ptr()) % alignment:
        raise AssertionError(
            f"buffer {name!r} is at address {tensor.data_ptr()}, which is not "
            f"a multiple of {alignment}, so a wide load from it would read the "
            "wrong elements"
        )


def _current_buffer(name: str) -> Any:
    """The buffer the generated wrapper is currently working on.

    A miss is not a failure: a name the wrapper never allocated is a name the
    call did not use, and there is nothing to check.
    """

    import sys

    frame = sys._getframe(1)
    while frame is not None:
        if name in frame.f_locals:
            value = frame.f_locals[name]
            if isinstance(value, tp.Tensor):
                return value
        frame = frame.f_back
    return None


__all__ = [
    "assert_alignment",
    "assert_size_stride",
    "assert_tensor_metadata",
]
