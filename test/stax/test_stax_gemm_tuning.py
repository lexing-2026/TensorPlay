"""GEMM candidate selection for extern matmul segments (max-autotune mode)."""

import json

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen import triton_gemm as tg


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    import tensorplay.compiler.backends.stax.codecache as cc

    monkeypatch.setattr(cc, "_default_caches", {})
    return tmp_path


# --- layout / key helpers -----------------------------------------------------


def test_standard_2d_accepts_both_major_orders():
    assert tg._standard_2d((32, 48), (48, 1))       # row-major
    assert tg._standard_2d((48, 32), (1, 48))       # column-major (t() view)
    assert not tg._standard_2d((2, 3, 4), (12, 4, 1))
    assert not tg._standard_2d((4, 4), (2, 8))      # overlapping stride


def test_decision_key_tracks_shape_device_and_precision():
    a = tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0", False)
    assert a == tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0", False)
    assert a != tg._decision_key(32, 64, 48, "tensorplay.float32", "cuda:0", False)
    assert a != tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:1", False)
    # the precision switch separates decision records: tf32 numerics must
    # never replay through an ieee decision or vice versa
    assert a != tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0", True)


def test_matmul_allow_tf32_gated_by_hardware_and_knob(monkeypatch):
    import tensorplay.backends.cuda as cuda_backends

    # hardware without native tf32 never opts in, whatever the knob says
    monkeypatch.setattr(tp.cuda, "is_tf32_supported", lambda: False)
    assert tg._matmul_allow_tf32() is False

    monkeypatch.setattr(tp.cuda, "is_tf32_supported", lambda: True)
    monkeypatch.setattr(cuda_backends.matmul, "allow_tf32", True)
    assert tg._matmul_allow_tf32() is True

    monkeypatch.setattr(cuda_backends.matmul, "allow_tf32", False)
    assert tg._matmul_allow_tf32() is False


def test_tuned_matmul_declines_without_cuda():
    def base(feed):
        raise AssertionError("never reached")

    literal = tp.randn(8, 8)
    specs = ((None, literal), (None, literal))
    assert tg.tuned_matmul_launch(base, [], specs, (8, 8)) is None


def test_tuned_matmul_declines_non_float32():
    def base(feed):
        raise AssertionError("never reached")

    a = tp.randn(8, 8, dtype=tp.float64)
    specs = ((None, a), (None, tp.randn(8, 8, dtype=tp.float64)))
    assert tg.tuned_matmul_launch(base, [], specs, (8, 8)) is None


# --- end-to-end (gated) -------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


@requires_cuda
def test_matmul_region_benches_and_persists_a_gemm_decision(cache_root):
    from tensorplay.compiler.backends.stax.codecache import default_cache

    w = tp.randn(64, 96, device="cuda")

    def fn(x):
        return x @ w.t()

    x = tp.randn(128, 96, device="cuda")
    compiled = tp.compile(fn, mode="max-autotune")
    assert tp.allclose(compiled(x), fn(x), rtol=1e-4, atol=1e-3)

    records = [
        json.loads(open(path, "rb").read().decode())
        for path in cache_root.rglob("*.json")
    ]
    gemm_records = [r for r in records if "choice" in r]
    assert gemm_records, "the matmul extern segment must persist a decision"
    assert gemm_records[0]["choice"] in ("native", "triton")


@requires_cuda
def test_gemm_decision_replay_skips_benchmarking(cache_root, monkeypatch):
    w = tp.randn(64, 96, device="cuda")

    def fn(x):
        return x @ w.t()

    x = tp.randn(128, 96, device="cuda")
    first = tp.compile(fn, mode="max-autotune")
    assert tp.allclose(first(x), fn(x), rtol=1e-4, atol=1e-3)

    from tensorplay.compiler.backends.stax.codegen import triton_gemm

    monkeypatch.setattr(
        triton_gemm,
        "_bench_candidates",
        lambda *a, **k: pytest.fail("a persisted decision must not re-bench"),
    )
    second = tp.compile(fn, mode="max-autotune")
    assert tp.allclose(second(x), fn(x), rtol=1e-4, atol=1e-3)


@requires_cuda
def test_module_region_with_parameters_matches_eager(cache_root):
    class M(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = tp.nn.Linear(48, 32, bias=True)

        def forward(self, x):
            return self.lin(x).relu().sum()

    module = M().cuda()
    x = tp.randn(64, 48, device="cuda")
    compiled = tp.compile(module, mode="max-autotune")
    assert tp.allclose(compiled(x), module(x))


@requires_cuda
def test_tf32_opt_in_persists_flag_and_stays_correct(cache_root, monkeypatch):
    import tensorplay.backends.cuda as cuda_backends

    monkeypatch.setattr(cuda_backends.matmul, "allow_tf32", True)
    w = tp.randn(64, 96, device="cuda")

    def fn(x):
        return x @ w.t()

    x = tp.randn(128, 96, device="cuda")
    compiled = tp.compile(fn, mode="max-autotune")
    got = compiled(x)
    # tf32 shortens the mantissa: compare against a float64 reference within
    # the accepted precision trade instead of the fp32 gate
    ref64 = x.double() @ w.t().double()
    assert tp.allclose(got.double(), ref64, rtol=2e-2, atol=2e-2)

    records = [
        json.loads(path.read_bytes().decode())
        for path in cache_root.rglob("*.json")
    ]
    gemm_records = [r for r in records if "choice" in r]
    assert gemm_records, "the matmul extern segment must persist a decision"
    assert gemm_records[0].get("tf32") is True


@requires_cuda
def test_gemm_decision_replay_skips_benchmarking_with_tf32(cache_root, monkeypatch):
    import tensorplay.backends.cuda as cuda_backends

    monkeypatch.setattr(cuda_backends.matmul, "allow_tf32", True)
    w = tp.randn(64, 96, device="cuda")

    def fn(x):
        return x @ w.t()

    x = tp.randn(128, 96, device="cuda")
    first = tp.compile(fn, mode="max-autotune")
    assert tp.allclose(first(x), fn(x), rtol=2e-2, atol=2e-2)

    from tensorplay.compiler.backends.stax.codegen import triton_gemm

    monkeypatch.setattr(
        triton_gemm,
        "_bench_candidates",
        lambda *a, **k: pytest.fail("a persisted decision must not re-bench"),
    )
    second = tp.compile(fn, mode="max-autotune")
    assert tp.allclose(second(x), fn(x), rtol=2e-2, atol=2e-2)


# --- epilogue-fused tiles -------------------------------------------------


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_epilogue_decision_key_separates_chains():
    """Different store-time chains on one shape never share a decision."""

    key_relu = tg._decision_key(
        64, 32, 48, "tensorplay.float32", "cuda:0", False, ([17, 0, -1], [], 0)
    )
    key_sig = tg._decision_key(
        64, 32, 48, "tensorplay.float32", "cuda:0", False, ([21, 0, -1], [], 0)
    )
    bare = tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0", False)
    assert len({key_relu, key_sig, bare}) == 3


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_emit_tile_epilogue_lines_pins_chain_on_register():
    """The shared emitter resolves the chain's single tensor input onto the
    accumulator register and stores the chain's final temporary."""

    from tensorplay.compiler.backends.stax.codegen.triton import (
        TritonProgramCodegen,
        emit_tile_epilogue_lines,
    )

    opcode = {
        name: code for code, name in TritonProgramCodegen._OP_NAMES.items()
    }
    # relu(acc); etmp1 * 0.5 (unused rhs slots point at spare constants)
    program = [opcode["relu"], 0, -1, opcode["mul"], 1, -2]
    lines, final = emit_tile_epilogue_lines(program, [0.0, 0.5], 0, "acc")
    assert lines == [
        "etmp1 = tl.maximum(acc, 0.0)",
        "etmp2 = etmp1 * 0.5",
    ]
    assert final == "etmp2"


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_fused_epilogue_matmul_matches_eager_gpu(cache_root):
    """A matmul with a single-user pointwise tail compiles to one fused
    tile under max-autotune and matches eager within fp32 tile-order
    noise; every benched candidate runs the full region."""

    import importlib

    canonical_gemm = importlib.import_module(
        "tensorplay.compiler.backends.stax.codegen.triton_gemm"
    )

    device = tp.device("cuda", 0)
    w = tp.randn(512, 512, device=device)

    def fn(t):
        return (t @ w).relu()

    x = tp.randn(512, 512, device=device)
    compiled = tp.compile(fn, mode="max-autotune", fullgraph=True)
    out = compiled(x)
    ref = fn(x)
    assert out.shape == ref.shape
    assert tp.allclose(out, ref, rtol=1e-4, atol=1e-4)
    # the fused tile candidates were built and benched (the region never
    # silently degrades to the composite native floor)
    assert len(canonical_gemm._EPI_KERNEL_MEMO) > 0

    def chain(t):
        return (((t @ w).relu() + 1.0) * 0.5).sigmoid()

    x2 = tp.randn(513, 512, device=device)
    compiled2 = tp.compile(chain, mode="max-autotune", fullgraph=True)
    out2 = compiled2(x2)
    ref2 = chain(x2)
    assert tp.allclose(out2, ref2, rtol=1e-4, atol=1e-4)

    records = [
        json.loads(path.read_bytes().decode())
        for path in cache_root.rglob("*.json")
    ]
    gemm_records = [r for r in records if "choice" in r]
    assert gemm_records, "the fused segment must persist a decision"


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_extern_epilogue_region_feeds_reduction_gpu(cache_root):
    """The folded chain's export wires into a following reduction
    segment; numerics stay on the eager schedule."""

    device = tp.device("cuda", 0)
    w = tp.randn(96, 128, device=device)

    def fn(t):
        return ((t @ w).relu()).sum(dim=1)

    x = tp.randn(64, 96, device=device)
    compiled = tp.compile(fn, mode="max-autotune", fullgraph=True)
    out = compiled(x)
    ref = fn(x)
    assert out.shape == ref.shape
    assert tp.allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_extern_epilogue_training_stays_segmented_gpu(cache_root):
    """Training schedules carry no store-time epilogues: the pointwise tail
    keeps its own segment and gradients match eager (the output magnitude
    is O(sum), so tolerances are relative)."""

    device = tp.device("cuda", 0)
    w = tp.randn(64, 96, device=device)

    def fn(t):
        return ((t @ w).relu()).sum()

    xc = tp.randn(8, 64, device=device, requires_grad=True)
    compiled = tp.compile(fn, fullgraph=True)
    out = compiled(xc)
    out.backward()

    xr = xc.detach().clone().requires_grad_(True)
    ref = fn(xr)
    ref.backward()
    assert tp.allclose(out, ref, rtol=1e-5, atol=1e-4)
    assert tp.allclose(xc.grad, xr.grad, rtol=1e-5, atol=1e-4)


# --- linear-form tiles (bias + optional chain) ---------------------------------


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_linear_decision_key_separates_forms():
    base = tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0", False)
    lin = tg._decision_key(
        64, 32, 48, "tensorplay.float32", "cuda:0", False, None, True, True
    )
    assert base != lin


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_linear_tile_launch_bias_and_chain_match_eager_gpu():
    """The generated linear tile (transposed weight view, row-broadcast
    bias, chain on the accumulator) matches eager directly, independent of
    which side the benchmark prefers."""

    from tensorplay.compiler.backends.stax.codegen.triton import (
        TritonProgramCodegen,
        emit_tile_epilogue_lines,
    )

    opcode = {
        name: code for code, name in TritonProgramCodegen._OP_NAMES.items()
    }
    chain = [opcode["relu"], 0, -1]

    device = tp.device("cuda", 0)
    K, N, M = 96, 128, 64
    x = tp.randn(M, K, device=device)
    w = tp.randn(N, K, device=device)
    b = tp.randn(N, device=device)

    def eager(feed):
        return (feed[0] @ feed[1].t() + feed[2]).relu()

    launch = tg._triton_launch_factory(
        (0, None), (1, None),
        M, N, K,
        (32, 64, 32, 4, 3),
        eager,
        epilogue=(chain, [], 0),
        bias_spec=(2, None),
        b_transposed=True,
    )
    out = launch([x, w, b])
    ref = (x @ w.t() + b).relu()
    assert tp.allclose(out, ref, rtol=1e-4, atol=1e-4)

    # bias without a chain: the store leaves the accumulator directly
    bare = tg._triton_launch_factory(
        (0, None), (1, None),
        M, N, K,
        (32, 64, 32, 4, 3),
        lambda feed: feed[0] @ feed[1].t() + feed[2],
        bias_spec=(2, None),
        b_transposed=True,
    )
    out2 = bare([x, w, b])
    ref2 = x @ w.t() + b
    assert tp.allclose(out2, ref2, rtol=1e-4, atol=1e-4)

    # a non-contiguous runtime bias takes the fallback launch
    bad_bias = tp.randn(N * 2, device=device)[::2]
    assert not bad_bias.is_contiguous() or bad_bias.stride(0) == 1
    hijacked = tg._triton_launch_factory(
        (0, None), (1, None),
        M, N, K,
        (32, 64, 32, 4, 3),
        lambda feed: "fallback",
        bias_spec=(2, None),
        b_transposed=True,
    )
    assert hijacked([x, w, tp.randn(N + 1, device=device)]) == "fallback"


@pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA unavailable"
)
def test_module_linear_region_with_chain_matches_eager_gpu(cache_root):
    """A Linear module region with a single-user pointwise tail compiles
    under max-autotune; every candidate runs the full region, so the
    output matches eager whichever side won the bench."""

    device = tp.device("cuda", 0)

    class M(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = tp.nn.Linear(512, 512, bias=True)

        def forward(self, x):
            return self.lin(x).relu()

    m = M().cuda()
    x = tp.randn(512, 512, device=device)
    compiled = tp.compile(m, mode="max-autotune", fullgraph=True)
    out = compiled(x)
    ref = m(x)
    assert tp.allclose(out, ref, rtol=1e-4, atol=1e-4)
    records = [
        json.loads(path.read_bytes().decode())
        for path in cache_root.rglob("*.json")
    ]
    assert any("choice" in r for r in records)
