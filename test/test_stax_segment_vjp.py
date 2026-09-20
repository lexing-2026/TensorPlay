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


def test_bare_extremum_reduction_trains_via_mask_vjp(monkeypatch):
    """A bare extremum reduction trains through the eager select-mask VJP:
    only the forward kernel is emitted, no backward kernel."""

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
    assert compiled is not None
    assert compiled._tensorplay_backward_codegen == "triton"
    assert calls == ["fwd0"]


def test_extremum_reduction_producer_chain_still_falls_back(monkeypatch):
    """A pointwise chain feeding the extremum has no eager VJP in v1: the
    whole region keeps the eager fallback."""

    calls = []
    _fake_runtime(monkeypatch, {})

    def fake_autotune(name, *args, **kwargs):
        calls.append(name)

        def launch(values):
            return None

        return launch

    monkeypatch.setattr(st, "_autotune_launch", fake_autotune)

    x = tp.rand(16, requires_grad=True)
    gm = Tracer().trace(lambda t: (t * 2).amax(dim=0), sample_inputs={"t": x})
    compiled = st.compile_graph_module(gm, [x])
    # the forward kernel is built before the plan gate rejects the sweep
    assert compiled is None
    assert calls == ["fwd0"]


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


# --- extern-segment analytic VJP rules (M5f) ---------------------------------------


def _vjp_rule(target, kwargs=None, args_after=()):
    """Build one extern tangent rule against a synthetic single-input node."""

    from types import SimpleNamespace

    from tensorplay.graph import Graph

    operand = Graph().placeholder("x")
    node = SimpleNamespace(
        target=target, args=(operand, *args_after), kwargs=kwargs or {}
    )
    return st._build_extern_analytic_vjp(
        node, lambda v: 0 if v is operand else None
    )


def test_extern_analytic_vjp_rules_match_engine():
    g = tp.randn(4, 8)
    cases = [
        ("softmax", {"dim": 1}, lambda t: t.softmax(dim=1), False),
        ("neg", {}, lambda t: -t, False),
        ("exp", {}, lambda t: t.exp(), False),
        ("sigmoid", {}, lambda t: t.sigmoid(), False),
        ("tanh", {}, lambda t: t.tanh(), False),
        ("sqrt", {}, lambda t: t.sqrt(), True),
        ("rsqrt", {}, lambda t: t.rsqrt(), True),
        ("abs", {}, lambda t: t.abs(), False),
        ("relu", {}, lambda t: t.relu(), False),
    ]
    for target, kwargs, fwd, positive in cases:
        # sqrt/rsqrt rules are only real on the positive domain
        x = (tp.rand(4, 8) + 0.5 if positive else tp.randn(4, 8)).requires_grad_(True)
        vjp = _vjp_rule(target, kwargs)
        assert vjp is not None, target
        (dx,) = vjp([x.detach()], g)
        ref = tp.autograd.grad(fwd(x), x, grad_outputs=g)[0]
        assert tp.abs(dx - ref).max().item() < 1e-5, target


def test_extern_reshape_rule_restores_input_shape():
    x = tp.randn(4, 8)
    g = tp.randn(2, 16)
    vjp = _vjp_rule("reshape", {})
    (dx,) = vjp([x], g)
    assert list(dx.shape) == [4, 8]
    assert tp.abs(dx - g.reshape([4, 8])).max().item() < 1e-6


def test_extern_transpose_rule_is_involution():
    x = tp.randn(2, 3, 4)
    g = tp.randn(2, 4, 3)
    vjp = _vjp_rule("transpose", {}, args_after=(1, 2))
    (dx,) = vjp([x], g)
    assert tp.abs(dx - g.transpose(1, 2)).max().item() < 1e-6


def test_extern_uncovered_operator_has_no_rule():
    from types import SimpleNamespace

    from tensorplay.graph import Graph

    operand = Graph().placeholder("x")
    node = SimpleNamespace(
        target="cumsum", args=(operand,), kwargs={"dim": 1}
    )
    assert (
        st._build_extern_analytic_vjp(
            node, lambda v: 0 if v is operand else None
        )
        is None
    )


# --- mixed-graph training on a real GPU --------------------------------------------


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_mixed_extern_segment_training_matches_eager_gpu():
    """Fused runs, an eager softmax between them, and a reduction tail
    train through the chained local VJPs with the closed-form extern
    tangent rule."""

    def fn(t):
        g = (t * 2.0).relu()
        s = t.softmax(dim=1)
        return (s * g).sum(dim=1)

    xc = tp.randn(4, 8, device=tp.device("cuda", 0), requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(xc)
    out.sum().backward()

    xr = xc.detach().clone().requires_grad_(True)
    ref = fn(xr)
    ref.sum().backward()

    assert tp.abs(out.cpu() - ref.cpu()).max().item() < 1e-5
    assert tp.abs(xc.grad.cpu() - xr.grad.cpu()).max().item() < 1e-5


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def _spy_canonical_launches(monkeypatch):
    """Record lowering launches; the frontend pipeline imports its own
    backend module instance, so the spy must target the canonical entry
    (fetched from sys.modules after a warmup compile, never imported
    directly at test scope)."""

    import sys

    device = tp.device("cuda", 0)
    tp.compile(lambda a: a * 2, fullgraph=True)(tp.rand(4, device=device))
    canonical = sys.modules["tensorplay.compiler.backends.stax.codegen.triton"]
    seen = []
    original = canonical._autotune_launch

    def spy(name, *args, **kwargs):
        seen.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(canonical, "_autotune_launch", spy)
    return seen


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_bare_amax_training_matches_eager_gpu(monkeypatch):
    """Bare extremum reduction: the select-mask VJP splits ties evenly."""

    def fn(t):
        return t.amax(dim=1)

    launches = _spy_canonical_launches(monkeypatch)
    x = tp.randn(4, 8, device=tp.device("cuda", 0), requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(x)
    out.sum().backward()
    tp.cuda.synchronize()
    assert launches and not [n for n in launches if n.startswith("bwd")]

    ref_in = x.detach().clone().requires_grad_(True)
    ref = fn(ref_in)
    ref.sum().backward()

    assert tp.abs(out.cpu() - ref.cpu()).max().item() < 1e-5
    assert tp.abs(x.grad.cpu() - ref_in.grad.cpu()).max().item() < 1e-5


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_extremum_mask_vjp_flows_between_segments_gpu(monkeypatch):
    """softmax extern -> bare amax -> sum: the reduction tangent flows back
    through the mask VJP into the extern segment's engine rule."""

    def fn(t):
        return t.softmax(dim=1).amax(dim=1).sum()

    launches = _spy_canonical_launches(monkeypatch)
    x = tp.randn(4, 8, device=tp.device("cuda", 0), requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(x)
    out.backward()
    tp.cuda.synchronize()
    assert [n for n in launches if n.startswith("bwd")], launches

    ref_in = x.detach().clone().requires_grad_(True)
    ref = fn(ref_in)
    ref.backward()

    assert tp.abs(out.cpu() - ref.cpu()).max().item() < 1e-5
    assert tp.abs(x.grad.cpu() - ref_in.grad.cpu()).max().item() < 1e-5


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_min_amin_training_matches_eager_gpu(monkeypatch):
    """amin fuses forward (tl.minimum) and trains through the mask VJP;
    full-reduction min() trains the same way."""

    def fn(t):
        return t.amin(dim=1).sum() + t.min()

    launches = _spy_canonical_launches(monkeypatch)
    x = tp.randn(4, 8, device=tp.device("cuda", 0), requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(x)
    out.backward()
    tp.cuda.synchronize()
    assert launches

    ref_in = x.detach().clone().requires_grad_(True)
    ref = fn(ref_in)
    ref.backward()

    assert tp.abs(out.cpu() - ref.cpu()).max().item() < 1e-5
    assert tp.abs(x.grad.cpu() - ref_in.grad.cpu()).max().item() < 1e-5


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_reduction_with_pointwise_tail_training_gpu(monkeypatch):
    """A reduction with a pointwise tail trains: the schedule splits the
    store-time epilogue into its own segment closed by a local VJP."""

    def fn(t, w):
        return ((t * w).relu().sum(dim=1) + 1.0).relu()

    launches = _spy_canonical_launches(monkeypatch)
    ts = [
        tp.randn(8, 8, device=tp.device("cuda", 0), requires_grad=True),
        tp.randn(8, 8, device=tp.device("cuda", 0), requires_grad=True),
    ]
    compiled = tp.compile(fn, fullgraph=True)
    ins = [v.detach().clone().requires_grad_(True) for v in ts]
    got = compiled(*ins)
    got.sum().backward()
    tp.cuda.synchronize()
    assert [n for n in launches if n.startswith("bwd")], launches

    ref_ins = [v.detach().clone().requires_grad_(True) for v in ts]
    ref = fn(*ref_ins)
    ref.sum().backward()

    assert tp.abs(got.cpu() - ref.cpu()).max().item() < 1e-4
    for g, want in zip(ins, ref_ins):
        assert tp.abs(g.grad.cpu() - want.grad.cpu()).max().item() < 1e-4


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_extern_uncovered_training_uses_engine_vjp(monkeypatch):
    """An eager operator without a closed-form rule trains through the
    engine rule (recompute + nested grad) inside the compiled region."""

    def fn(t):
        return (t * 2).cumsum(dim=1).sum()

    launches = _spy_canonical_launches(monkeypatch)
    x = tp.randn(4, 8, device=tp.device("cuda", 0), requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(x)
    out.backward()
    tp.cuda.synchronize()
    # fused pointwise and reduction kernels around the extern cumsum
    assert [name for name in launches if name.startswith("bwd")], launches

    ref_in = x.detach().clone().requires_grad_(True)
    ref = fn(ref_in)
    ref.backward()

    assert tp.abs(out.cpu() - ref.cpu()).max().item() < 1e-5
    assert tp.abs(x.grad.cpu() - ref_in.grad.cpu()).max().item() < 1e-5


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_extern_engine_vjp_trainable_between_fused_kernels(monkeypatch):
    """matmul extern segment between fused kernels: the engine rule returns
    gradients for every floating operand."""

    def fn(t, w):
        return (t * 2).matmul(w).relu().sum()

    launches = _spy_canonical_launches(monkeypatch)
    # same-shape operands: the region contract admits one reference shape
    ts = [
        tp.randn(8, 8, device=tp.device("cuda", 0), requires_grad=True),
        tp.randn(8, 8, device=tp.device("cuda", 0), requires_grad=True),
    ]
    compiled = tp.compile(fn, fullgraph=True)
    ins = [v.detach().clone().requires_grad_(True) for v in ts]
    got = compiled(*ins)
    got.backward()
    tp.cuda.synchronize()
    assert [name for name in launches if name.startswith("bwd")], launches

    ref_ins = [v.detach().clone().requires_grad_(True) for v in ts]
    ref = fn(*ref_ins)
    ref.backward()

    # fp32 matmul reassociates the accumulation order against cuBLAS
    assert tp.abs(got.cpu() - ref.cpu()).max().item() < 1e-4
    for g, want in zip(ins, ref_ins):
        assert tp.abs(g.grad.cpu() - want.grad.cpu()).max().item() < 1e-4


# --- M5f: broadcast operands train through sum-to-shape ---------------------------


def test_broadcast_operand_single_segment_trains(monkeypatch):
    """(x + b).relu().sum() with b (N,) against x (M, N): the bias partial
    comes back at the fused iteration space and is summed to (N,)."""

    def fwd0(x, b):
        return (x + b).relu().sum()

    def bwd0(x, b, go):
        mask = ((x + b) > 0).to(x.dtype)
        # both partials at the fused (M, N) space; the sweep must reduce
        # the second to b's own shape
        return (go * mask, go * mask)

    _fake_runtime(monkeypatch, {"fwd0": fwd0, "bwd0": bwd0})

    x = tp.randn(8, 16, requires_grad=True)
    b = tp.randn(16, requires_grad=True)

    def fn(x, b):
        return (x + b).relu().sum()

    gm = Tracer().trace(fn, sample_inputs={"x": x, "b": b})
    compiled = st.compile_graph_module(gm, [x, b])
    assert compiled is not None
    assert compiled._tensorplay_codegen == "triton"
    assert compiled._tensorplay_backward_codegen == "triton"

    xe = x.detach().requires_grad_(True)
    be = b.detach().requires_grad_(True)
    fn(xe, be).backward()
    ref_gx, ref_gb = xe.grad, be.grad
    assert tuple(ref_gb.shape) == (16,)

    xc = x.detach().requires_grad_(True)
    bc = b.detach().requires_grad_(True)
    out = compiled(xc, bc)
    out.backward()
    assert tp.abs(out - fn(x.detach(), b.detach())).max().item() < 1e-6
    assert tuple(bc.grad.shape) == (16,)
    assert tp.abs(xc.grad - ref_gx).max().item() < 1e-5
    assert tp.abs(bc.grad - ref_gb).max().item() < 1e-5


def test_broadcast_operand_multi_segment_trains(monkeypatch):
    """Two reduction chains over one broadcast bias: each segment's partial
    is summed to (N,) before the fan-out accumulation."""

    def fwd0(x, b):
        return (x + b).relu().sum()

    def bwd0(x, b, go):
        mask = ((x + b) > 0).to(x.dtype)
        return (go * mask, go * mask)

    def fwd1(x, b):
        return (x * b).sum()

    def bwd1(x, b, go):
        return (go * b, go * x)

    def fwd2(a, b):
        return a + b

    def bwd2(a, b, go):
        return (go, go)

    _fake_runtime(
        monkeypatch,
        {"fwd0": fwd0, "bwd0": bwd0, "fwd1": fwd1, "bwd1": bwd1, "fwd2": fwd2, "bwd2": bwd2},
    )

    x = tp.randn(8, 16, requires_grad=True)
    b = tp.randn(16, requires_grad=True)

    def fn(x, b):
        return (x + b).relu().sum() + (x * b).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": x, "b": b})
    compiled = st.compile_graph_module(gm, [x, b])
    assert compiled is not None
    segments = gm.meta["stax_segments"]
    assert [seg["kind"] for seg in segments] == ["pw+red", "pw+red", "pw"]

    xe = x.detach().requires_grad_(True)
    be = b.detach().requires_grad_(True)
    fn(xe, be).backward()
    ref_gx, ref_gb = xe.grad, be.grad

    xc = x.detach().requires_grad_(True)
    bc = b.detach().requires_grad_(True)
    out = compiled(xc, bc)
    out.backward()
    assert tp.abs(out - fn(x.detach(), b.detach())).max().item() < 1e-6
    assert tuple(bc.grad.shape) == (16,)
    assert tp.abs(xc.grad - ref_gx).max().item() < 1e-5
    assert tp.abs(bc.grad - ref_gb).max().item() < 1e-5


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_broadcast_operand_training_matches_eager_gpu():
    device = tp.device("cuda", 0)

    def fn(x, b):
        return (x + b).relu().sum()

    xs = [
        tp.randn(32, 48, device=device, requires_grad=True),
        tp.randn(48, device=device, requires_grad=True),
    ]
    compiled = tp.compile(fn, fullgraph=True)
    ins = [v.detach().clone().requires_grad_(True) for v in xs]
    got = compiled(*ins)
    got.backward()
    tp.cuda.synchronize()
    assert tuple(ins[1].grad.shape) == (48,)

    ref_ins = [v.detach().clone().requires_grad_(True) for v in xs]
    ref = fn(*ref_ins)
    ref.backward()
    assert tp.abs(got.cpu() - ref.cpu()).max().item() < 1e-3
    for g, want in zip(ins, ref_ins):
        assert tp.abs(g.grad.cpu() - want.grad.cpu()).max().item() < 1e-3


@pytest.mark.skipif(not st.runtime_available(), reason="Triton/CUDA unavailable")
def test_broadcast_operand_scalar_tangent_gpu():
    """A scalar hand-back expands against the output shape, not a broadcast
    input, and every operand still receives its own-shape gradient."""

    device = tp.device("cuda", 0)

    def fn(b, x):
        return ((x + b) * x).sum()

    xs = [
        tp.randn(48, device=device, requires_grad=True),
        tp.randn(32, 48, device=device, requires_grad=True),
    ]
    compiled = tp.compile(fn, fullgraph=True)
    ins = [v.detach().clone().requires_grad_(True) for v in xs]
    got = compiled(*ins)
    got.backward()
    tp.cuda.synchronize()

    ref_ins = [v.detach().clone().requires_grad_(True) for v in xs]
    ref = fn(*ref_ins)
    ref.backward()
    assert tp.abs(got.cpu() - ref.cpu()).max().item() < 1e-3
    for g, want in zip(ins, ref_ins):
        assert tp.abs(g.grad.cpu() - want.grad.cpu()).max().item() < 1e-3


def test_bare_pair_reduction_trains_via_scatter_vjp(monkeypatch):
    """x.max(dim=1).values.sum(): the pair segment's projection export is
    closed by the eager index-scatter VJP (no backward kernel)."""

    launches = {
        # the pair segment exports its values projection (one port)
        "fwd0": lambda x: x.max(dim=1).values,
        # the sum segment reduces the values export
        "fwd1": lambda v: v.sum(),
        # sum backward: the expanded tangent passes through unchanged
        "bwd1": lambda v, go: (go,),
    }

    _fake_runtime(monkeypatch, launches)

    x = tp.randn(6, 8, requires_grad=True)

    def fn(x):
        return x.max(dim=1).values.sum()

    gm = Tracer().trace(fn, sample_inputs={"x": x})
    compiled = st.compile_graph_module(gm, [x])
    assert compiled is not None
    assert compiled._tensorplay_backward_codegen == "triton"
    segments = gm.meta["stax_segments"]
    assert [seg["kind"] for seg in segments] == ["pw+red", "pw+red"]

    out = compiled(x.detach())
    expected = fn(x.detach())
    assert tp.abs(out - expected).max().item() < 1e-6

    xc = x.detach().requires_grad_(True)
    out_c = compiled(xc)
    out_c.backward()
    xe = x.detach().requires_grad_(True)
    fn(xe).backward()
    assert tp.abs(out_c - fn(xe)).max().item() < 1e-6
    assert tp.abs(xc.grad - xe.grad).max().item() < 1e-5


def test_argmax_segment_trains_without_gradient_path(monkeypatch):
    """A training region containing argmax compiles instead of falling
    back wholesale: the index reduction closes with the no-tangent rule and
    its integer output flows to an extern consumer (index_select) whose
    engine VJP skips the non-float operand."""

    table = tp.randn(8, 4)
    # segment order: [pw+red mul+sum] [pw+red argmax] [extern index_select]
    # [pw+red gathered sum] [pw add] — the extern segment runs eagerly, the
    # fused segments take fake launches keyed by segment index
    launches = {
        "fwd0": lambda x: (x * 2.0).sum(),
        "bwd0": lambda x, go: (go * 2.0,),
        "fwd1": lambda x: x.argmax(dim=1),
        "fwd3": lambda rows: rows.sum(),
        "bwd3": lambda rows, go: (go,),
        "fwd4": lambda a, b: a + b,
        "bwd4": lambda a, b, go: (go, go),
    }

    _fake_runtime(monkeypatch, launches)

    x = tp.randn(5, 8, requires_grad=True)

    def fn(x):
        return (x * 2.0).sum() + tp.index_select(
            table, 0, x.argmax(dim=1)
        ).sum()

    from types import SimpleNamespace

    gm = Tracer().trace(fn, sample_inputs={"x": x})
    # the direct trace skips the frontend's tensor-meta propagation; the
    # extern plan needs the index_select output signature
    for node in gm.graph.nodes:
        if node.op == "call_function" and getattr(
            node.target, "__name__", ""
        ) == "index_select":
            node.meta["tensor_meta"] = SimpleNamespace(
                shape=(5, 4), dtype=tp.float32
            )
    compiled = st.compile_graph_module(gm, [x])
    assert compiled is not None
    assert compiled._tensorplay_backward_codegen == "triton"
    segments = gm.meta["stax_segments"]
    assert "pw+red" in [seg["kind"] for seg in segments]

    xc = x.detach().requires_grad_(True)
    out_c = compiled(xc)
    out_c.backward()

    # eager: the argmax branch carries no gradient path
    xe = x.detach().requires_grad_(True)
    out_e = fn(xe)
    out_e.backward()
    assert tp.abs(out_c - out_e).max().item() < 1e-5
    assert tp.abs(xc.grad - xe.grad).max().item() < 1e-5


def test_get_attr_segment_trains_through_function_tail(monkeypatch):
    """A lifted module parameter trains: the get_attr extern segment serves
    the CURRENT attribute value and its identity VJP routes the export
    tangent back to the leaf via the trailing autograd input."""

    import operator

    from tensorplay.graph import Graph, GraphModule

    class _Root:
        pass

    root = _Root()
    scale = tp.randn(6, requires_grad=True)
    root.scale = scale

    g = Graph()
    x = g.placeholder("x")
    s = g.get_attr("scale")
    g.output(g.call_function(operator.mul, (x, s)))
    gm = GraphModule(root, g)

    # segment order: [extern get_attr] [pw mul] — the get_attr segment runs
    # its real plan, the fused mul takes the fake launch at index 1
    launches = {
        "fwd1": lambda x, s: x * s,
        "bwd1": lambda x, s, go: (go * s, go * x),
    }
    _fake_runtime(monkeypatch, launches)

    x = tp.randn(4, 6, requires_grad=True)
    compiled = st.compile_graph_module(gm, [x])
    assert compiled is not None
    assert compiled._tensorplay_backward_codegen == "triton"

    xc = x.detach().requires_grad_(True)
    out = compiled(xc)
    out.sum().backward()

    # the parameter rebind is picked up per call
    root.scale = tp.full_like(scale, 2.0)
    out2 = compiled(xc)
    assert tp.abs(out2 - xc * 2.0).max().item() < 1e-6

    xr = x.detach().requires_grad_(True)
    sr = scale.detach().clone().requires_grad_(True)
    (xr * sr).sum().backward()
    assert tp.abs(out - xr * sr).max().item() < 1e-6
    assert tp.abs(xc.grad - xr.grad).max().item() < 1e-5
    assert scale.grad is not None
    assert tp.abs(scale.grad - sr.grad).max().item() < 1e-5


def test_extern_epilogue_composes_chain_after_operator(monkeypatch):
    """The pointwise tail an extern segment carries replays as a tuned
    pointwise launch over the operator's eager output (inference)."""

    from types import SimpleNamespace

    launches = {}

    def fwd0ep(y):
        return (y.relu() + 1.0).sigmoid()

    launches["fwd0ep"] = fwd0ep
    _fake_runtime(monkeypatch, launches)

    w = tp.randn(6, 5)

    def fn(t):
        return ((t @ w).relu() + 1.0).sigmoid()

    x = tp.randn(4, 6)
    gm = Tracer(execute=True).trace(fn, sample_inputs={"t": x})
    for node in gm.graph.nodes:
        if node.op in {"call_function", "call_method"}:
            node.meta["tensor_meta"] = SimpleNamespace(
                shape=(4, 5), dtype=tp.float32
            )
    compiled = st.compile_graph_module(gm, [x])
    assert compiled is not None
    assert compiled._tensorplay_codegen == "triton"
    segments = gm.meta["stax_segments"]
    assert [seg["kind"] for seg in segments] == ["extern"]
    assert [seg["epilogue"] for seg in segments] == [
        ["relu", "add", "sigmoid"]
    ]

    out = compiled(x)
    expected = fn(x)
    assert tp.abs(out - expected).max().item() < 1e-6
