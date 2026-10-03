"""Reductions with few outputs over a long axis are computed in two layers.

The reduced extent is cut into pieces reduced side by side, then the pieces of
each output are combined; a cut that does not divide the extent masks the
positions past its end.  Each case is lowered strictly, so a region whose
reduction could not be built fails here instead of running op by op, and is
compared with the eager result.
"""

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")

CASES = {
    "contiguous axis, three outputs": (lambda x: x.sum(-1), (3, 1_000_003)),
    "whole tensor": (lambda x: x.sum(), (777, 1301)),
    "channels of a channels-last batch": (lambda x: x.sum((0, 1, 2)), (64, 31, 33, 48)),
    "mean, odd extent": (lambda x: x.mean(1), (5, 999_983)),
    "maximum of the whole tensor": (lambda x: x.amax(), (1003, 2011)),
    "half precision, accumulated wide": (lambda x: x.half().sum(0).float(), (100_003, 7)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_two_layer_reduction_matches_eager(name):
    fn, shape = CASES[name]
    tp.manual_seed(0)
    x = tp.randn(*shape, device="cuda")
    want = fn(x)
    got = tp.compile(fn, strict_native=True)(x)
    tol = 2e-3 if "half" in name else 1e-5
    assert ((got - want).abs().max() / (want.abs().max() + 1e-6)).item() < tol


@pytest.mark.parametrize("channels_last", [False, True])
def test_two_layer_mean_and_variance(channels_last):
    # The statistics of a batch norm: a mean and a variance per channel over
    # every other axis, combined across pieces by their counts.
    def stats(x):
        if channels_last:
            x = x.contiguous(memory_format=tp.channels_last)
        return tp.var_mean(x, dim=(0, 2, 3), correction=0)

    tp.manual_seed(0)
    x = tp.randn(64, 48, 31, 33, device="cuda")
    want = stats(x)
    got = tp.compile(stats, strict_native=True)(x)
    for a, b in zip(got, want):
        assert ((a - b).abs().max() / b.abs().max()).item() < 1e-5
