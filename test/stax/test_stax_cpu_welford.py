"""Variance reductions in generated CPU kernels.

A running mean and second moment are what a variance, a layer norm or a batch
norm reduces; the generated kernel folds them per vector lane, merges the
lanes, and keeps long reductions accurate by folding finished chunks in
balanced pairs.  Each case is lowered strictly -- a kernel that did not build
fails here -- and compared with float64.
"""

import pytest

import tensorplay as tp
import tensorplay.nn as nn


@pytest.mark.parametrize("shape, dim", [
    ((64, 1000), 1),
    ((8, 6, 35), (0, 2)),
    ((8, 6, 5, 7), (0, 2, 3)),
    ((1000, 64), 0),
    ((3, 100_003), 1),
])
def test_var_mean_matches_float64(shape, dim):
    tp.manual_seed(0)
    x = tp.randn(*shape) * 3 + 1
    var, mean = tp.compile(lambda t: tp.var_mean(t, dim=dim, correction=0), strict_native=True)(x)
    ref_var, ref_mean = tp.var_mean(x.double(), dim=dim, correction=0)
    assert ((var.double() - ref_var).abs().max() / ref_var.abs().max()).item() < 1e-5
    assert ((mean.double() - ref_mean).abs().max() / ref_mean.abs().max()).item() < 1e-5


def test_layer_norm_model_builds_both_halves():
    tp.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 8))
    ref = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 8))
    ref.load_state_dict(model.state_dict())
    x = tp.randn(32, 64)
    tp.compile(model, strict_native=True)(x).sum().backward()
    ref(x).sum().backward()
    for p, q in zip(model.parameters(), ref.parameters()):
        assert tp.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4)
