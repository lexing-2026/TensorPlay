"""M5c training-state segmented VJP (multi-kernel backward chaining).

The orchestration contract is verified locally with fake launches that
execute the true per-segment math eagerly: forward chaining, tangent
expansion for sum/mean reduction segments, fan-out gradient accumulation,
and placeholder-aligned gradient returns.  Numeric behavior checks against eager
autograd on a real GPU is gated on ``runtime_available()``.
"""

import pytest

import tensorplay as tp
from tensorplay._stax.codegen import triton as st
from tensorplay.graph import Tracer


def _fake_runtime(monkeypatch, launches):
    """Enable the triton lane headlessly; launches[name] = eager callable."""

    monkeypatch.setattr(st, "HAS_TRITON", True)
    monkeypatch.setattr(
        st,
        "_supports_runtime_inputs",
        lambda *args, **kwargs: True,
    )

    def fake_autotune(name, program, constants, outputs, examples, **kwargs):
        def launch(values):
            return launches[name](*values)

        launch.name = name
        return launch

    monkeypatch.setattr(st, "_autotune_launch", fake_autotune)


# --- local orchestration contract -------------------------------------------------


def test_multi_segment_training_chains_vjps(monkeypatch):
    """(x*w).relu().sum() + sigmoid(y*w).sum() → [pw+red][pw+red][pw]."""

    launches = {}

    def fwd0(x, w):
        # a pw+red segment exports its reduction result
        return (x * w).relu().sum()

    def bwd0(x, w, go):
        # go arrives expanded to the reduction-input shape
        t = x * w
        mask = (t > 0).to(t.dtype)
        return (go * mask * w, go * mask * x)

    def fwd1(y, w):
        return tp.sigmoid(y * w).sum()

    def bwd1(y, w, go):
        q = y * w
        sig = tp.sigmoid(q)
        d = sig * (1.0 - sig)
        return (go * d * w, go * d * y)

    def fwd2(a, b):
        return a + b

    def bwd2(a, b, go):
        return (go, go)

    for name, fn in [
        ("fwd0", fwd0), ("bwd0", bwd0),
        ("fwd1", fwd1), ("bwd1", bwd1),
        ("fwd2", fwd2), ("bwd2", bwd2),
    ]:
        launches[name] = fn

    _fake_runtime(monkeypatch, launches)

    x = tp.randn(8, requires_grad=True)
    w = tp.randn(8, requires_grad=True)
    y = tp.randn(8, requires_grad=True)

    def fn(x, w, y):
        return (x * w).relu().sum() + tp.sigmoid(y * w).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": x, "w": w, "y": y})
    compiled = st.compile_graph_module(gm, [x, w, y])
    assert compiled is not None
    assert compiled._tensorplay_codegen == "triton"
    assert compiled._tensorplay_backward_codegen == "triton"
    segments = gm.meta["stax_segments"]
    assert [seg["kind"] for seg in segments] == ["pw+red", "pw+red", "pw"]

    out = compiled(x.detach(), w.detach(), y.detach())
    expected = fn(x.detach(), w.detach(), y.detach())
    assert tp.abs(out - expected).max().item() < 1e-6

    # backward through the chained VJP kernels vs eager reference
    xe = x.detach().requires_grad_(True)
    we = w.detach().requires_grad_(True)
    ye = y.detach().requires_grad_(True)
    out_e = fn(xe, we, ye)
    out_e.backward()
    ref_gx, ref_gw, ref_gy = xe.grad, we.grad, ye.grad

    xc = x.detach().requires_grad_(True)
    wc = w.detach().requires_grad_(True)
    yc = y.detach().requires_grad_(True)
    out_c = compiled(xc, wc, yc)
    out_c.backward(tp.tensor(1.0))
    assert tp.abs(out_c - out_e).max().item() < 1e-6
    assert tp.abs(xc.grad - ref_gx).max().item() < 1e-5
    assert tp.abs(wc.grad - ref_gw).max().item() < 1e-5
    assert tp.abs(yc.grad - ref_gy).max().item() < 1e-5


def test_fanout_gradient_accumulation(monkeypatch):
    """Two reduction chains over shared x sum their contributions."""

    launches = {
        "fwd0": lambda x: (x * 2.0).sum(),
        "bwd0": lambda x, go: (go * 2.0,),
        "fwd1": lambda x: (x * 3.0).sum(),
        "bwd1": lambda x, go: (go * 3.0,),
        "fwd2": lambda a, b: a + b,
        "bwd2": lambda a, b, go: (go, go),
    }

    _fake_runtime(monkeypatch, launches)

    x = tp.randn(6, requires_grad=True)

    def fn(x):
        return (x * 2.0).sum() + (x * 3.0).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": x})
    compiled = st.compile_graph_module(gm, [x])
    assert compiled is not None

    xc = x.detach().requires_grad_(True)
    out = compiled(xc)
    out.backward()
    assert tp.abs(out - fn(x.detach())).max().item() < 1e-6
    expected_grad = tp.ones(6) * 5.0
    assert tp.abs(xc.grad - expected_grad).max().item() < 1e-5


def test_untrainable_reduction_still_falls_back(monkeypatch):
    """amax has no uniform VJP: whole graph must fall back (M5f)."""

    calls = []
    _fake_runtime(monkeypatch, {})

    def fake_autotune(name, *args, **kwargs):
        calls.append(name)

        def launch(values):
            return None

        return launch

    monkeypatch.setattr(st, "_autotune_launch", fake_autotune)

    x = tp.rand(16, requires_grad=True)
    gm = Tracer().trace(lambda t: t.amax(dim=0), sample_inputs={"t": x})
    compiled = st.compile_graph_module(gm, [x])
    # amax training graphs keep the eager fallback for now
    assert compiled is None
    assert calls == []


# --- numeric checks on a real GPU -------------------------------------------------


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_multi_segment_training_matches_eager_gpu():
    from tensorplay.graph import Tracer as _Tracer

    device = tp.device("cuda", 0)

    def fn(x, w, y):
        return (x * w).relu().sum() + tp.sigmoid(y * w).sum()

    xs = [
        tp.rand(64, device=device, requires_grad=True),
        tp.rand(64, device=device, requires_grad=True),
        tp.rand(64, device=device, requires_grad=True),
    ]
    gm = _Tracer().trace(fn, sample_inputs=dict(zip("xwy", xs)))
    compiled = st.compile_graph_module(gm, list(xs))
    assert compiled is not None
    assert compiled._tensorplay_backward_codegen == "triton"

    ins = [v.detach().requires_grad_(True) for v in xs]
    out = compiled(*ins)
    out.backward()
    tp.cuda.synchronize()

    ref_ins = [v.detach().clone().requires_grad_(True) for v in xs]
    ref_out = fn(*ref_ins)
    ref_out.backward()

    assert tp.abs(out.cpu() - ref_out.cpu()).max().item() < 1e-5
    for got, want in zip(ins, ref_ins):
        assert tp.abs(got.grad.cpu() - want.grad.cpu()).max().item() < 1e-5
