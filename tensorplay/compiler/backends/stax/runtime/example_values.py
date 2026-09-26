"""Example values to measure a candidate with, and the state measuring them changed.

Measuring a candidate means running it, and running it needs operands.  A
candidate is not measured on the values the region will actually be given --
those are the program's private data, and a measurement that read them would
report on the data rather than on the candidate -- so it is measured on values
made for the purpose, at the shape and stride and element type the candidate
was compiled for.

Making those values draws from the random number generator, which is the other
half of this module: a measurement that consumed numbers would leave the
generator somewhere else, and a program whose results then depended on whether
it had been autotuned would be a program that gives two answers to the same
question.  So the state is saved before and put back after, on every device
that has a generator, and a measurement that raises still puts it back.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import tensorplay as tp
from tensorplay.primitives.common import is_complex_dtype, is_float_dtype

#: Devices that carry a random number generator of their own.  A device whose
#: generator is not restored changes the program's results, so the list is
#: consulted rather than assumed.
_GENERATOR_DEVICES = ("cuda", "xpu")


def _elements_needed(size: Sequence[int], stride: Sequence[int], extra: int) -> int:
    """How many elements a window of this shape and stride reaches.

    The window does not occupy ``prod(size)`` elements: it steps by ``stride``,
    so the last element it reaches can be far past the last one a dense layout
    would put there.  Allocating the dense count would read past the end of the
    buffer for any window whose stride exceeds its extent.
    """

    needed = extra
    if all(extent > 0 for extent in size):
        needed += sum(
            (extent - 1) * step for extent, step in zip(size, stride)
        ) + 1
    return needed


def rand_strided(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: Any = None,
    device: Any = "cpu",
    extra_size: int = 0,
) -> Any:
    """A random tensor of this shape at this stride.

    Allocated as one run and then addressed, rather than allocated at the shape
    and copied into place, because a copy would cost more than the value is
    worth and the caller only wants something to measure with.
    """

    dtype = tp.float32 if dtype is None else dtype
    needed = _elements_needed(size, stride, extra_size)
    if is_float_dtype(dtype):
        # Drawn around zero rather than from a uniform range, so that a
        # candidate's arithmetic is exercised on values of both signs and on
        # values near zero, which is where a wrong reciprocal or a wrong
        # normalisation shows up.
        values = tp.randn(needed, dtype=tp.float32, device=device)
    elif is_complex_dtype(dtype):
        real = tp.randn(needed, dtype=tp.float32, device=device)
        values = tp.complex(real, tp.randn(needed, dtype=tp.float32, device=device))
    elif dtype is tp.bool:
        # A byte per element, then read as booleans: a count has no meaning in
        # a range a boolean can take, so the draw is made in the byte type and
        # narrowed afterwards.
        raw = tp.randint(0, 2, (needed,), dtype=tp.uint8, device=device)
        values = raw.to(tp.bool)
    else:
        values = tp.randint(0, 128, (needed,), dtype=dtype, device=device)
    flat = values.to(dtype) if values.dtype != dtype else values
    return tp.as_strided(flat, tuple(int(extent) for extent in size),
                         tuple(int(step) for step in stride), 0)


@contextlib.contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Put every generator back where it was, however the block ends.

    Taken before the block and put back in a ``finally``, so a measurement that
    raised does not leave the program's own randomness moved.
    """

    cpu_state = tp.random.get_rng_state()
    device_states: dict[str, Any] = {}
    for name in _GENERATOR_DEVICES:
        device = getattr(tp, name, None)
        if device is None:
            continue
        try:
            if device.is_available():
                device_states[name] = device.get_rng_state()
        except Exception:  # noqa: BLE001 - a device without a usable generator
            continue
    try:
        yield
    finally:
        tp.random.set_rng_state(cpu_state)
        for name, state in device_states.items():
            getattr(tp, name).set_rng_state(state)


__all__ = ["preserve_rng_state", "rand_strided"]
