# mypy: allow-untyped-defs
"""Randomness primitives.

Randomness here is a function of a position rather than of how many values have
been read: a (seed, offset) pair names one point in a stream, and reading there
gives the same values however many came before.  That is what lets a graph of
random operations be reproduced, and what lets two of them be compared -- an
ordinary generator's answer depends on what it was asked for earlier, so the
same graph asked for twice gives two different answers.

The state a graph threads through is a pair of tensors rather than a handle to
something opaque, because a graph records values and not the things that made
them: the pair is what the recording keeps.
"""

from typing import cast

import tensorplay
from tensorplay import primitives
from tensorplay.primitives.common import CUDARngStateHelper

__all__ = [
    "PhiloxState",
    "PhiloxStateTracker",
    "CUDARngStateHelper",
    "philox_rand",
    "philox_rand_like",
    "philox_seed",
    "register_rng_prims",
]


class PhiloxState:
    """A position in a stream of random values, and how much has been read past it.

    The position itself is two values -- which stream, and how far along it --
    and a third number says how much this state has read since it was handed
    that position.  Keeping the two apart is what lets a saved position be
    restored and then read again from where it was rather than from where the
    reading has got to: the relative count is what has been read, not where
    the stream is.
    """

    __slots__ = ["seed", "base_offset", "relative_offset", "offset_advanced_at_least_once"]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seed = tensorplay.tensor(())
        self.base_offset = tensorplay.tensor(())
        self.relative_offset = 0
        self.offset_advanced_at_least_once = False

    def validate_state(self) -> None:
        if self.seed.numel() == 0 or self.base_offset.numel() == 0:
            raise AssertionError(
                f"seed and base_offset must not be empty, got "
                f"seed.numel()={self.seed.numel()}, "
                f"base_offset.numel()={self.base_offset.numel()}"
            )

    def advance_offset(self, consumed_offset) -> None:
        # Added as it stands rather than rounded to a group of four: what is
        # counted here is how much the graph has read, and rounding it as it
        # goes would make two graphs that read the same amounts disagree about
        # where the next read starts.  Rounding happens once, at the end, where
        # the count becomes a position.
        self.offset_advanced_at_least_once = True
        self.relative_offset = self.relative_offset + int(consumed_offset)

    def set_state(self, seed, base_offset, relative_offset: int = 0) -> None:
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        self.validate_state()
        return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        # Asked for as one value because the device's state is read and written
        # as one value, and a pair handed back from a call that saved something
        # would not survive being saved.
        self.validate_state()
        return tensorplay.stack([self.seed, self.base_offset + self.relative_offset])

    def set_state_from_tensor(self, state) -> None:
        self.seed, self.base_offset = tensorplay.unbind(state)
        self.relative_offset = 0


class PhiloxStateTracker:
    """Where each pass over a graph starts reading, and how far it has got.

    One state is kept for the forward pass and one for the backward, because
    the two read the same stream and must not read over each other: the
    backward pass reads after everything the forward pass read, so its position
    starts where the forward one finished rather than where the stream began.
    Which of the two is current is said by marking the beginning of a pass,
    rather than by passing it in, because the decompositions that read are not
    told which pass they are in.
    """

    running_state: PhiloxState
    fwd_state: PhiloxState
    bwd_state: PhiloxState

    def __enter__(self):
        PhiloxStateTracker.reset()
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb):
        PhiloxStateTracker.reset()

    @classmethod
    def reset(cls):
        cls.running_state = PhiloxState()
        cls.fwd_state = PhiloxState()
        cls.bwd_state = PhiloxState()

    @classmethod
    def mark_beginning_of_forward(cls):
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        if mode == "forward":
            cls.fwd_state.set_state(seed, offset)
            cls.mark_beginning_of_forward()
        else:
            if mode != "backward":
                raise AssertionError(f"mode must be 'backward', got {mode}")
            cls.bwd_state.set_state(seed, offset)

    @classmethod
    def get_state_as_tensor(cls):
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def set_state_from_tensor(cls, x):
        cls.running_state.set_state_from_tensor(x)

    @classmethod
    def advance_offset(cls, consumed_offset):
        cls.running_state.advance_offset(consumed_offset)

    @classmethod
    def get_current_relative_offset(cls):
        return cls.running_state.relative_offset

    @classmethod
    def get_updated_fwd_offset(cls):
        # Where the stream is once the forward pass has read what it read.  A
        # pass that read nothing leaves the stream where it was, and one that
        # read something leaves it rounded to a group of four -- because a
        # device reads groups of four, and a position that named a group
        # part-read would have the rest of that group read by whatever came
        # next.
        if not cls.fwd_state.offset_advanced_at_least_once:
            return cls.fwd_state.base_offset
        return PhiloxStateTracker.multiple_of_4(
            cls.fwd_state.base_offset + cls.fwd_state.relative_offset
        )

    @classmethod
    def get_updated_bwd_offset(cls):
        # The backward pass reads after the forward one, so its position starts
        # where the forward one finished rather than where the stream began --
        # otherwise the two passes would read over each other.
        return cls.bwd_state.base_offset + cls.bwd_state.relative_offset

    @staticmethod
    def multiple_of_4(offset):
        # A device reads the stream four values at a time, so a count that is
        # not a multiple of four would leave the next read starting part-way
        # through a group.
        return _philox_multiple_of_4(offset)


def _philox_multiple_of_4(offset) -> int:
    """The next count a device will accept.

    A device reads the stream four values at a time, so a count that is not a
    multiple of four would leave the next read starting part-way through a
    group -- and the values in that group would be read by whichever read
    happened to overlap it.

    What comes back from a read is a position rather than a count, so which of
    its two numbers is the count is asked for here rather than by the caller.
    """

    value = offset
    if hasattr(value, "numel") and value.numel() == 2:
        value = value[1]
    if hasattr(value, "item"):
        value = value.item()
    return (int(value) + 3) // 4 * 4


def _philox_seed_impl(
    seed: int, offset: int, *, device=None, dtype=None
) -> "tensorplay.Tensor":
    """The (seed, offset) pair as the two values a device reads.

    A position in the stream is a number too large for one value, so it is two:
    which stream, and how far along it.  Both are handed over as values rather
    than as numbers so that a graph records them and can hand them to a device
    that never saw the numbers themselves.
    """

    seed_t = tensorplay.tensor(int(seed), dtype=cast(object, tensorplay.uint64), device=device)
    offset_t = tensorplay.tensor(
        int(offset), dtype=cast(object, tensorplay.uint64), device=device
    )
    return tensorplay.stack([seed_t, offset_t])


def _philox_rand_impl(
    size,
    seed: "tensorplay.Tensor",
    offset: "tensorplay.Tensor",
    stride=None,
    device=None,
    dtype=None,
):
    """Values read at a position, and where the stream is afterwards.

    Two answers rather than one because the second is what the next operation
    needs: a graph of random operations threads the position through, so each
    one has to say where it left off.  The count is the number of values read,
    and a shape of no positions reads none.

    A stride would say the stream is shared across devices, which is a
    distributed concern and not one this walks: a stream with a stride has
    positions on it that are not this position.
    """

    if stride is not None:
        raise AssertionError(f"stride must be None, got {stride}")
    if device is None:
        device = seed.device
    elif not isinstance(device, tensorplay.device):
        # Named as a string or an index, which is how a caller usually says it;
        # a device object is what the question below needs to be asked of.
        device = tensorplay.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            f"philox_rand is only written for a graphics device, got {device.type}"
        )

    with tensorplay.random.fork_rng(devices=[]):
        CUDARngStateHelper.set_torch_state_tensor(seed, offset)
        values = tensorplay.rand(size, device=device, dtype=dtype)
    # What comes back is how many values were read, not where the stream now
    # is: the count is what a caller adds to what it has already read, and a
    # position is not something two different readers could add to the same
    # number and mean the same thing by.
    return values, tensorplay.tensor(values.numel(), dtype=tensorplay.int64)


def _philox_rand_like_impl(input: "tensorplay.Tensor", seed, offset):
    """The same values, read into a shape and type taken from another value.

    Asking for the same shape and type as something else is a question about
    the destination rather than about the randomness, so the destination is
    described by what it already is.
    """

    return _philox_rand_impl(
        input.shape, seed, offset, device=input.device, dtype=input.dtype
    )


def register_rng_prims() -> None:
    """Register the randomness primitives into the operator registry."""

    _register_rng("philox_rand", _philox_rand_impl)
    _register_rng("philox_rand_like", _philox_rand_like_impl)
    _register_rng("philox_seed", _philox_seed_impl)


def _register_rng(name: str, impl) -> None:
    try:
        qualified = f"prims::{name}"
        if tensorplay.library.has_op(qualified):
            return
        schema = {
            "philox_rand": (
                "(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, "
                "Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"
            ),
            "philox_rand_like": (
                "(Tensor input, Tensor seed, Tensor offset) -> (Tensor, Tensor)"
            ),
            "philox_seed": "(SymInt seed, SymInt offset, Device? device=None) -> Tensor",
        }[name]
        prim_def = tensorplay.library.custom_op(
            qualified, impl, schema=qualified + schema
        )
    except Exception:
        pass
