#!/usr/bin/env python3
"""Holding a map that writes two tensors to the tensor it hands back.

A schema may write more than one tensor: the noise a randomised map draws is
laid down beside the answer it computes, and the schema says which of the two
it returns through the write alias the return carries.  The generated
dispatcher has to hand back that one -- the answer -- and not the other.

Every assertion reads the tensor the caller handed in, since that is what the
op contractually fills; the plain call has no destination to read, so its
answer comes from the tensor it returns.
"""
import numpy as np
import pytest
import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE

ops = getattr(tp.ops, NATIVE_NAMESPACE)

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

ROWS, COLS = 5, 7
N = ROWS * COLS
LOWER, UPPER = 0.125, 0.333
SEED = 2026


def read(t):
    """The logical elements: a width narrower than single precision is only
    faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def source_matrix(dtype, device="cpu"):
    rng = np.random.default_rng(11)
    data = rng.standard_normal(N) * 2.0
    return tp.from_numpy(np.array(data, dtype=np.float32)).to(dtype).to(device).reshape(ROWS, COLS)


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_out_hands_back_the_answer_it_wrote(device):
    # The answer and the factors are laid into two destinations of their own,
    # and the call hands back the answer: the tensor holding the factors is
    # not what the caller receives.
    x = source_matrix(tp.float32, device)
    noise = tp.ones([ROWS, COLS], dtype=tp.float32, device=device)
    out = tp.zeros([ROWS, COLS], dtype=tp.float32, device=device)
    address = out.data_ptr()

    returned = ops.rrelu_with_noise.out(x, noise, LOWER, UPPER, True, out=out)

    assert returned.data_ptr() == address, "the answer was not handed back"
    assert returned.data_ptr() != noise.data_ptr(), "the factors were handed back"
    # What was written into the destination is what was handed back.
    assert np.array_equal(read(returned), read(out))
    # The factors live in their own tensor, and they are what the draw gave.
    factors = read(noise).ravel()
    taken = read(x).ravel() <= 0
    assert np.all((factors[taken] >= LOWER) & (factors[taken] <= UPPER))
    assert np.all(factors[~taken] == 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_inplace_hands_back_the_input_it_wrote(device):
    # The in-place spelling writes the answer into the tensor it was handed
    # and hands that same tensor back, not the noise beside it.
    x = source_matrix(tp.float32, device)
    address = x.data_ptr()
    noise = tp.ones([ROWS, COLS], dtype=tp.float32, device=device)

    returned = ops.rrelu_with_noise_.default(x, noise, LOWER, UPPER, True)

    assert returned.data_ptr() == address
    assert returned.data_ptr() != noise.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_out_without_a_destination_hands_back_the_answer(device):
    # Without a destination the call allocates one, and what it hands back is
    # the answer rather than the factors it drew.
    x = source_matrix(tp.float32, device)
    noise = tp.ones([ROWS, COLS], dtype=tp.float32, device=device)

    returned = ops.rrelu_with_noise.default(x, noise, LOWER, UPPER, True)

    assert returned.data_ptr() != noise.data_ptr()
    assert tuple(returned.shape) == (ROWS, COLS)
    factors = read(noise).ravel()
    taken = read(x).ravel() <= 0
    assert np.all((factors[taken] >= LOWER) & (factors[taken] <= UPPER))
