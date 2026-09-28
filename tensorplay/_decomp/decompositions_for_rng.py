"""Randomness as something a position decides rather than something a count does.

A generator's answer depends on how much it has been asked for already, so a
graph of random operations asked for twice gives two different answers and the
two cannot be compared.  What is here instead reads at a position: a (seed,
offset) pair names one point in a stream, reading there gives the same values
however many came before, and how far the stream was read is handed back so
that the next read can start after it.

The decompositions exist so that a random operation is written as that read
plus an offset the graph carries, rather than as a call that would consult a
running total.  The offset is carried in the graph, which is what makes a graph
of random operations reproducible: the graph is the record of where each read
happened.
"""

from typing import cast

import tensorplay
from tensorplay._decomp import get_decompositions, register_decomposition
from tensorplay.primitives.rng_prims import PhiloxState, PhiloxStateTracker
from tensorplay.primitives.rng_prims import register_rng_prims

__all__ = [
    "PhiloxState",
    "PhiloxStateTracker",
    "register_rng_decompositions",
    "rng_decompositions",
]

aten = tensorplay.ops.tp
rngprims = tensorplay.ops.prims

# The reads go into a table of their own rather than into the table every other
# decomposition shares.  The reason is the read at a position: it is written by
# asking the framework for random values, and the framework's own answer to
# that question is another read at a position, so the two would ask each other
# for as long as a graph is traced.  Keeping them apart is what stops that --
# a read expands in this table, and the operation it asks is not in it.
rng_decompositions: dict[object, object] = {}


def register_rng_decomposition(op: object):
    """Register a decomposition in the random table rather than the shared one."""

    return register_decomposition(op, registry=rng_decompositions)  # type: ignore[arg-type]

# A device reads the stream four values at a time, so a count that is not a
# multiple of four would leave the next read starting part-way through a group.
extra_random_decomps: dict = {}


def _throw_on_non_cuda(device) -> None:
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but "
        f"{device.type} does not use Philox/counter-based RNG. Therefore, "
        f"functionalizing a {device.type} RNG operator is not supported. We are "
        f"discussing the possibility of a Philox-based RNG implementation for CPU."
    )


def _device_of(x) -> object:
    return x.device


def _philox_rand(size, seed, offset, device, dtype):
    return rngprims.philox_rand(size, seed, offset, None, device, dtype)


def rand(shape, dtype=None, layout=None, device=None, pin_memory=False):
    """Values at the current position, and the position after them.

    The count of what was read comes back and is folded into the position, so
    two of these in a row read two different places rather than the same one
    twice -- and the fold is into the graph, so the second one reads after the
    first whether or not the first has run yet.
    """

    if device is not None and _device_kind(device) != "cuda":
        _throw_on_non_cuda(device)
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    dtype = dtype or tensorplay.float32
    out, offset_jump = _philox_rand(shape, seed, offset, device, dtype)
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


def rand_like(x, dtype=None, layout=None, device=None, pin_memory=False):
    """The same values, into a shape and type taken from another value.

    Asked for as the shape and type of something else rather than as its own,
    because that is the whole difference between this and reading at a shape:
    where the values go and what they are is a question about the destination.
    """

    device = device or x.device
    if _device_kind(device) != "cuda":
        _throw_on_non_cuda(device)
    dtype = dtype or x.dtype
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    out, offset_jump = _philox_rand(x.shape, seed, offset, device, dtype)
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


def bernoulli_(self, p=0.5):
    """A value written in place, each element independently, with chance p.

    Written as a comparison against a value in range rather than as a draw from
    a distribution: a draw from a distribution is a function of a position,
    and a comparison against one is the same function written where the answer
    is a truth value.  The values are read as single precision because that is
    what a draw is a single-precision number, and a narrower type would round
    the threshold rather than the draw.
    """

    if _device_kind(self.device) == "cpu":
        return NotImplemented
    return self.copy_(rand_like(self, dtype=tensorplay.float32) < p)


def bernoulli_p(self, p=0.5, *, generator=None):
    """Which of these are true, each independently, with chance p.

    A generator cannot be threaded through a graph: which stream it names is a
    fact about the call rather than about the positions the graph reads, so
    naming one and reading at a position are two different things and only the
    second is a function of the graph.
    """

    if _device_kind(self.device) == "cpu":
        return NotImplemented
    if generator is not None:
        raise AssertionError(f"generator must be None, got {generator}")
    return rand_like(self, dtype=tensorplay.float32) < p


def _device_kind(device) -> str:
    kind = getattr(device, "type", device)
    return str(kind)


#: Operations whose own decomposition reaches for a random value, and which
#: therefore have to be written in terms of one before they can be written at
#: all.  They are collected first and handed on, so that a decomposition which
#: uses one of them finds a read at a position rather than a call that consults
#: a running total.  Their answers are not the answers of the eager
#: implementations -- a distribution drawn from four values at a position is a
#: different distribution from one drawn from a generator -- which is why this
#: is a separate table rather than a change to theirs.
_EXTRA_RANDOM_NAMES = (
    "cauchy",
    "cauchy_",
    "exponential",
    "exponential_",
    "geometric",
    "geometric_",
    "native_dropout",
    "normal",
    "normal_",
    "normal_functional",
    "log_normal",
    "log_normal_",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "uniform",
    "uniform_",
)


_registered = False


def register_rng_decompositions() -> None:
    """Write the random operations as reads at a position.

    Asked for each time the table of decompositions is read rather than once
    at import, because a program that never captures should not pay for
    reading them.  What it writes goes into a table that outlives the call, so
    writing it twice is not writing it twice: only the first ask does.
    """

    # The prims are what the reads below go through, so they are set up every
    # time: what a read means depends on which pass is running, and that is
    # per pass rather than per registration.
    register_rng_prims()

    global _registered
    if _registered:
        return
    _registered = True

    register_rng_decomposition(aten.rand)(rand)
    register_rng_decomposition(aten.rand_like)(rand_like)
    register_rng_decomposition(aten.bernoulli_)(bernoulli_)
    register_rng_decomposition(aten.bernoulli.p)(bernoulli_p)

    # Collected after the four above, and handed on as their own table, so
    # that an operation reaching for a random value finds the read rather than
    # the call it would otherwise make.
    extra = [
        getattr(aten, name)
        for name in _EXTRA_RANDOM_NAMES
        if hasattr(aten, name)
    ]
    if extra:
        inherited = get_decompositions(extra)
        for op, fn in list(inherited.items()):
            register_decomposition(op, registry=extra_random_decomps)(fn)
