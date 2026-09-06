"""Mixed CPU regions: fusible runs compiled, other operators left as they are."""

import numpy as np
import pytest

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay._stax import stax as stax_mod


def _close(got, ref, rel=2e-5):
    got = np.asarray(got.tolist(), dtype=np.float64)
    ref = np.asarray(ref.tolist(), dtype=np.float64)
    assert got.shape == ref.shape
    scale = max(1e-6, float(np.max(np.abs(ref))) if ref.size else 1.0)
    assert float(np.max(np.abs(got - ref))) <= rel * scale


def _route(compiled):
    lowering = next(iter(compiled._tensorplay_cache.values()))
    return getattr(lowering, "_tensorplay_codegen", None), lowering


class Mlp(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))


# ---------------------------------------------------------------------------
# routing


def test_mixed_region_compiles_its_fusible_runs():
    tensorplay.manual_seed(0)
    model = Mlp().eval()
    x = tensorplay.randn(4, 9, 32)
    compiled = tensorplay.compile(model, backend="stax")
    with tensorplay.no_grad():
        _close(compiled(x), model(x))
        codegen, lowering = _route(compiled)
    assert codegen == "stax-fused-cpu-segments"
    # The gelu chain is one kernel; the norm and the two products stay calls.
    kinds = [step for step in lowering._steps]
    assert len(kinds) >= 5


def test_a_fully_fusible_region_keeps_the_whole_region_path():
    tensorplay.manual_seed(1)
    x = tensorplay.randn(8, 32)
    fn = lambda v: ((v * 2.0).tanh() + 1.0) / 3.0  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(x), fn(x))
    codegen, _ = _route(compiled)
    assert codegen == "stax-fused-cpu"


def test_a_region_of_native_operators_keeps_the_native_graph():
    tensorplay.manual_seed(2)
    model = nn.Linear(16, 16).eval()
    x = tensorplay.randn(4, 16)
    compiled = tensorplay.compile(model, backend="stax")
    with tensorplay.no_grad():
        _close(compiled(x), model(x))
        codegen, _ = _route(compiled)
    assert codegen == "stax-native"


# ---------------------------------------------------------------------------
# numerics across the shapes the wiring has to carry


MIXED_CASES = [
    ("erf-between-products", lambda v, w: (v @ w).erf() * 2.0),
    ("reduction-after-reshape", lambda v, w: (v @ w).reshape(-1).sum()),
    ("chain-then-extern", lambda v, w: ((v * 2.0).tanh() @ w).exp()),
    ("extern-last", lambda v, w: ((v + 1.0) * 0.5) @ w),
    ("shared-value", lambda v, w: (v @ w).erf() + (v @ w)),
]


@pytest.mark.parametrize(
    "name,fn", MIXED_CASES, ids=[case[0] for case in MIXED_CASES]
)
def test_mixed_regions_match_the_reference(name, fn):
    tensorplay.manual_seed(3)
    x = tensorplay.randn(6, 24)
    w = tensorplay.randn(24, 24)
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(x, w), fn(x, w))


def test_reduction_run_inside_a_mixed_region():
    tensorplay.manual_seed(4)
    x = tensorplay.randn(6, 24)
    w = tensorplay.randn(24, 24)
    fn = lambda v, m: ((v @ m) * 2.0).sum(dim=1)  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(x, w), fn(x, w))
    codegen, _ = _route(compiled)
    assert codegen == "stax-fused-cpu-segments"


def test_captured_parameters_reach_the_kernels():
    tensorplay.manual_seed(5)
    model = Mlp(16).eval()
    x = tensorplay.randn(3, 5, 16)
    compiled = tensorplay.compile(model, backend="stax")
    with tensorplay.no_grad():
        first = compiled(x)
        second = compiled(x)
        _close(first, model(x))
        _close(second, model(x))


def test_operator_keywords_survive_the_wiring():
    tensorplay.manual_seed(6)
    x = tensorplay.randn(4, 12)
    fn = lambda v: (v * 2.0).tanh().sum(dim=1, keepdim=True).exp()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(x), fn(x))


def test_grad_inputs_keep_the_uncompiled_route():
    tensorplay.manual_seed(7)
    x = tensorplay.randn(4, 8, requires_grad=True)
    w = tensorplay.randn(8, 8)
    fn = lambda v, m: (v @ m).erf().sum()  # noqa: E731
    result = tensorplay.compile(fn, backend="stax")(x, w)
    result.backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# planning


def _plan(fn, *args):
    from tensorplay.graph import symbolic_trace

    module = symbolic_trace(fn)
    return stax_mod._lower_cpu_segmented(module, list(args))


def test_planner_declines_a_region_with_no_fusible_run():
    tensorplay.manual_seed(8)
    x = tensorplay.randn(4, 8)
    w = tensorplay.randn(8, 8)
    assert _plan(lambda v, m: (v @ m).reshape(-1), x, w) is None


def test_planner_declines_grad_carrying_inputs():
    x = tensorplay.randn(4, 8, requires_grad=True)
    w = tensorplay.randn(8, 8)
    assert _plan(lambda v, m: (v @ m).erf(), x, w) is None


def test_segment_externals_keep_first_use_order():
    from tensorplay.graph import symbolic_trace

    module = symbolic_trace(lambda a, b, c: (a * b) + c)
    body = tuple(
        node
        for node in module.graph.nodes
        if node.op in {"call_function", "call_method"}
    )
    externals = stax_mod._segment_externals(body)
    assert [node.name for node in externals] == ["a", "b", "c"]
