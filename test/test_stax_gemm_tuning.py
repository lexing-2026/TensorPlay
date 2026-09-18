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


def test_decision_key_tracks_shape_and_device():
    a = tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0")
    assert a == tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:0")
    assert a != tg._decision_key(32, 64, 48, "tensorplay.float32", "cuda:0")
    assert a != tg._decision_key(64, 32, 48, "tensorplay.float32", "cuda:1")


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
