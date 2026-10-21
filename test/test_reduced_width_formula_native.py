#!/usr/bin/env python3
"""The reduced width answers a formula at its own width.

A formula at the reduced width is computed in single precision and stored at
the reduced width.  Every step of the formula therefore rounds at the
reduced width, and the answer of one step is the one the next step reads.

The hardware crossing between the two widths and the software crossing
answer the same, step for step.  The formula is carried through single
precision and narrowed once at the end, which is what both backends answer
and what the reference runtime answers as well.

This states the formulas that way and compares them, element for element,
with what the walks answer.

The pairwise maps are stated the same way, but the answer is computed by
the vector math rather than by the library, so the two may sit one unit in
the last place apart after the narrowing.  They are compared within the
bound the reduced width carries.
"""
import numpy as np
import pytest
import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

COUNT = 4096

# The bound the reduced width carries: one unit in its last place.
BOUND = 2e-3


def read_bits(t):
    """The elements as the reduced width holds them."""
    return np.array(t.tolist(), dtype=np.float16).view(np.uint16)


def to_bf16(values):
    """The values as the other reduced width holds them.

    The layout is the upper half of the single-precision pattern, rounded
    to nearest with ties to even.
    """
    bits = np.array(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def to_values(t):
    """The elements as numbers."""
    return np.array(t.tolist(), dtype=np.float32)


def operands():
    rng = np.random.default_rng(3)
    self_np = rng.standard_normal(COUNT).astype(np.float16)
    first_np = rng.standard_normal(COUNT).astype(np.float16)
    second_np = rng.standard_normal(COUNT).astype(np.float16)
    weight_np = rng.random(COUNT).astype(np.float16)
    return self_np, first_np, second_np, weight_np


def scaled_product(self_np, first_np, second_np, alpha):
    """self + alpha * t1 * t2, carried through single precision."""
    return (self_np.astype(np.float32) +
            (alpha.astype(np.float32) * first_np.astype(np.float32) *
             second_np.astype(np.float32))).astype(np.float16)


def quotient(self_np, first_np, second_np, alpha):
    """self + alpha * t1 / t2, carried through single precision."""
    divisor = np.where(second_np == np.float16(0), np.float16(1),
                       second_np).astype(np.float16)
    return (self_np.astype(np.float32) +
            (alpha.astype(np.float32) * first_np.astype(np.float32) /
             divisor.astype(np.float32))).astype(np.float16)


@pytest.mark.parametrize("device", DEVICES)
def test_the_scaled_product_rounds_at_the_reduced_width(device):
    self_np, first_np, second_np, _ = operands()
    alpha = np.float16(0.75)

    got = tp.addcmul(tp.from_numpy(self_np).to(device),
                     tp.from_numpy(first_np).to(device),
                     tp.from_numpy(second_np).to(device),
                     value=0.75)

    want = scaled_product(self_np, first_np, second_np, alpha)
    assert np.array_equal(read_bits(got), want.view(np.uint16))


@pytest.mark.parametrize("device", DEVICES)
def test_the_quotient_rounds_at_the_reduced_width(device):
    self_np, first_np, second_np, _ = operands()
    alpha = np.float16(0.75)

    got = tp.addcdiv(tp.from_numpy(self_np).to(device),
                     tp.from_numpy(first_np).to(device),
                     tp.from_numpy(second_np).to(device),
                     value=0.75)

    want = quotient(self_np, first_np, second_np, alpha)
    assert np.array_equal(read_bits(got), want.view(np.uint16))


@pytest.mark.parametrize("device", DEVICES)
def test_the_interpolation_is_computed_in_single_precision(device):
    # The interpolation is computed in single precision and narrowed once:
    # the weight picks the nearer end when it is below a half.  Both
    # backends answer it that way.
    self_np, _, second_np, weight_np = operands()

    start = self_np.astype(np.float32)
    end = second_np.astype(np.float32)
    weight = weight_np.astype(np.float32)
    delta = end - start
    near = start + weight * delta
    far = end - delta * (np.float32(1) - weight)
    want = np.where(np.abs(weight) < 0.5, near, far).astype(np.float16)

    got = tp.lerp(tp.from_numpy(self_np).to(device),
                  tp.from_numpy(second_np).to(device),
                  tp.from_numpy(weight_np).to(device))

    assert np.array_equal(read_bits(got), want.view(np.uint16))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["addcmul", "addcdiv"])
def test_the_other_reduced_width_carries_the_formula_through(device, name):
    # The other reduced width carries the formula through single precision
    # and narrows once, the same way the half layout does.
    rng = np.random.default_rng(3)
    count = 4096
    self_np = rng.standard_normal(count).astype(np.float32)
    first_np = rng.standard_normal(count).astype(np.float32)
    second_np = rng.standard_normal(count).astype(np.float32)
    alpha = np.float32(0.75)

    if name == "addcmul":
        want = to_bf16(self_np + alpha * first_np * second_np)
        got = getattr(tp, name)(tp.from_numpy(self_np).to(tp.bfloat16).to(device),
                                tp.from_numpy(first_np).to(tp.bfloat16).to(device),
                                tp.from_numpy(second_np).to(tp.bfloat16).to(device),
                                value=0.75)
    else:
        divisor = np.where(second_np == np.float32(0), np.float32(1), second_np)
        want = to_bf16(self_np + alpha * first_np / divisor)
        got = getattr(tp, name)(tp.from_numpy(self_np).to(tp.bfloat16).to(device),
                                tp.from_numpy(first_np).to(tp.bfloat16).to(device),
                                tp.from_numpy(second_np).to(tp.bfloat16).to(device),
                                value=0.75)

    got_values = np.array(got.tolist(), dtype=np.float32)
    want_values = np.array(want.astype(np.uint32) << np.uint32(16),
                           dtype=np.uint32).view(np.float32)
    # The two backends answer the same; the formula stated in single
    # precision may sit one unit in the last place of the reduced width
    # away from it, where the rounding of the two widths differs.
    # The bound is the one the reduced width carries: a unit in its last
    # place is a larger share of a number than it is of a wider one.
    assert np.allclose(got_values, want_values, rtol=2e-2, atol=2e-2,
                       equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["hypot", "atan2", "logaddexp"])
def test_the_pairwise_maps_answer_in_single_precision(device, name):
    # The pairwise maps answer in single precision and narrow once, so they
    # are within the bound the reduced width carries of the answer stated at
    # the reduced width.
    self_np, first_np, _, _ = operands()

    def single(a, b):
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        if name == "hypot":
            return np.hypot(a32, b32)
        if name == "atan2":
            return np.arctan2(a32, b32)
        biggest = np.maximum(a32, b32)
        return biggest + np.log1p(np.exp(-np.abs(a32 - b32)))

    want = single(self_np, first_np).astype(np.float16)

    got = getattr(tp, name)(tp.from_numpy(self_np).to(device),
                            tp.from_numpy(first_np).to(device))

    assert np.allclose(to_values(got), np.array(want, dtype=np.float32),
                       rtol=BOUND, atol=BOUND, equal_nan=True)
