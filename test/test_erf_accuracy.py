"""Accuracy of the vectorized float32 ``erf``.

The float32 kernel evaluates two forms and selects between them: a series
near zero and a rational tail away from it.  The tail alone ends in
``1 - r``, which cancels away the low bits of a small result, so the split is
what keeps the *relative* error bounded near zero -- these tests pin that,
and the worst case over the whole line.
"""

import math

import numpy as np
import pytest

import tensorplay as tp

# The float32 kernel is a fast approximation, not a correctly-rounded one.
# Its measured worst case is 2.7 ulp; the budget leaves a little headroom.
FLOAT32_EPS = float(np.finfo(np.float32).eps)
ULP_BUDGET = 4.0

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def _relative_ulp(values, device):
    got = np.asarray(
        tp.tensor([float(v) for v in values], dtype=tp.float32)
        .to(device)
        .erf()
        .cpu()
        .tolist(),
        dtype=np.float64,
    )
    ref = np.array([math.erf(float(np.float32(v))) for v in values])
    keep = np.abs(ref) > 0.0
    return np.max(np.abs(got[keep] - ref[keep]) / np.abs(ref[keep])) / FLOAT32_EPS


@pytest.mark.parametrize("device", DEVICES)
def test_erf_stays_within_its_ulp_budget(device):
    values = np.linspace(-6.0, 6.0, 20001, dtype=np.float32)
    assert _relative_ulp(values, device) < ULP_BUDGET


@pytest.mark.parametrize("device", DEVICES)
def test_erf_keeps_relative_accuracy_near_zero(device):
    # erf(x) -> 2x/sqrt(pi) here, so an absolute-error bound says nothing:
    # a rational form that reaches zero as ``1 - r`` lands thousands of ulp
    # out on these inputs even while its absolute error looks fine.
    values = np.array(
        [1e-7, 3e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5],
        dtype=np.float32,
    )
    assert _relative_ulp(np.concatenate([values, -values]), device) < ULP_BUDGET


@pytest.mark.parametrize("device", DEVICES)
def test_erf_is_continuous_across_the_form_boundary(device):
    # The two forms meet at 0.7; neither side may step across the seam.
    values = np.linspace(0.6, 0.8, 4001, dtype=np.float32)
    assert _relative_ulp(np.concatenate([values, -values]), device) < ULP_BUDGET


@pytest.mark.parametrize("device", DEVICES)
def test_erf_corner_values(device):
    inf = float("inf")
    got = (
        tp.tensor([0.0, -0.0, inf, -inf, 30.0, -30.0], dtype=tp.float32)
        .to(device)
        .erf()
        .cpu()
        .tolist()
    )
    assert got == [0.0, -0.0, 1.0, -1.0, 1.0, -1.0]
    assert math.copysign(1.0, got[1]) == -1.0
    nan = tp.tensor([float("nan")], dtype=tp.float32).to(device).erf().cpu().tolist()
    assert math.isnan(nan[0])


@pytest.mark.parametrize("device", DEVICES)
def test_erf_is_odd(device):
    values = np.linspace(0.0, 5.0, 2001, dtype=np.float32)
    both = tp.tensor(
        [float(v) for v in values] + [float(-v) for v in values],
        dtype=tp.float32,
    ).to(device).erf().cpu().tolist()
    half = len(values)
    for positive, negative in zip(both[:half], both[half:]):
        assert positive == -negative


@pytest.mark.parametrize("device", DEVICES)
def test_erf_matches_across_the_vector_tail(device):
    # Lengths that land on, just below and just above a whole vector make the
    # masked tail evaluate the same expression as the full-width body.
    for count in (1, 3, 7, 8, 9, 15, 16, 17, 31, 33):
        values = np.linspace(-3.0, 3.0, count, dtype=np.float32)
        assert _relative_ulp(values, device) < ULP_BUDGET
