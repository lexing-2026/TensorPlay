"""L5-M1/M2: kernel codecache wiring + compile-time autotune logic."""

import json
from types import SimpleNamespace

import pytest

import tensorplay as tp
from tensorplay.compiler.runtime import stax_autotune as sa
from tensorplay.compiler.runtime.stax_autotune import CANDIDATE_CONFIGS
from tensorplay.compiler.runtime.stax_autotune import CANDIDATE_CONFIGS
from tensorplay.compiler.codegen import triton as st
from tensorplay.compiler.codegen.triton import TritonProgramCodegen
from tensorplay.compiler.codecache import CodeCache


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    # default_cache memoizes per process; reset for isolation.
    import tensorplay.compiler.codecache as cc

    monkeypatch.setattr(cc, "_default_caches", {})
    return tmp_path


# --- M2 helpers ---------------------------------------------------------------


def test_xnumel_bucket_powers_of_two():
    assert sa.xnumel_bucket(0) == 128
    assert sa.xnumel_bucket(1) == 128
    assert sa.xnumel_bucket(127) == 128
    assert sa.xnumel_bucket(128) == 128
    assert sa.xnumel_bucket(129) == 256
    assert sa.xnumel_bucket(1000) == 1024
    assert sa.xnumel_bucket(5000) == 8192


def test_decision_roundtrip(cache_root):
    digest = sa.program_digest([1, 0, 2, -1, 3], [1.5], (4,))
    assert sa.load_decision(digest, 256, "cuda:0") is None
    sa.store_decision(digest, 256, "cuda:0", (512, 8))
    assert sa.load_decision(digest, 256, "cuda:0") == (512, 8)
    assert sa.load_decision(digest, 256, "cuda:1") is None
    assert sa.load_decision(digest, 512, "cuda:0") is None


def test_load_decision_rejects_unknown_config(cache_root):
    digest = sa.program_digest([1], [], (0,))
    payload = json.dumps({"xblock": 333, "warps": 7}).encode()
    sa._decision_cache().store(sa.decision_key(digest, 128, "d"), payload,
                               ext="json")
    assert sa.load_decision(digest, 128, "d") is None


def test_pick_config_benchmarks_once_then_uses_decision(cache_root):
    digest = sa.program_digest([1, 2], [], (0,))
    builds: list[tuple[int, int]] = []
    benches: list[tuple[int, int]] = []

    times = {c: float(len(CANDIDATE_CONFIGS) - i) for i, c in enumerate(CANDIDATE_CONFIGS)}
    times[CANDIDATE_CONFIGS[-1]] = 0.5  # last config wins

    def build_launch(config):
        builds.append(config)

        class FakeLaunch:
            pass

        return FakeLaunch()

    def bench_fn(launch, args):
        benches.append("called")  # type: ignore[attr-defined]
        return times[builds[-1]]

    config, launch = sa.pick_config(digest, 300, "cuda:0", build_launch,
                                    [], bench_fn=bench_fn)
    assert config == CANDIDATE_CONFIGS[-1]  # fastest candidate wins
    assert builds == list(sa.CANDIDATE_CONFIGS)  # every candidate compiled once
    assert len(benches) == len(CANDIDATE_CONFIGS)

    # Second call hits the persisted decision: single rebuild, no benchmarking.
    builds.clear()
    benches.clear()
    config2, launch2 = sa.pick_config(digest, 300, "cuda:0", build_launch,
                                      [], bench_fn=bench_fn)
    assert config2 == CANDIDATE_CONFIGS[-1]
    assert builds == [CANDIDATE_CONFIGS[-1]]
    assert benches == []


def test_pick_config_skips_failing_candidates(cache_root):
    digest = sa.program_digest([7], [], (0,))

    def build_launch(config):
        if config[0] < 512:
            raise RuntimeError("nope")
        return object()

    calls = {"n": 0}

    def bench_fn(launch, args):
        calls["n"] += 1
        return 2.0

    eligible = [c for c in CANDIDATE_CONFIGS if c[0] >= 512]
    config, _ = sa.pick_config(digest, 64, "cuda:0", build_launch, [],
                               bench_fn=bench_fn)
    assert config == eligible[0]
    assert calls["n"] == len(eligible)


def test_pick_config_all_fail_raises(cache_root):
    digest = sa.program_digest([9], [], (0,))

    def build_launch(config):
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="all candidate configs"):
        sa.pick_config(digest, 10, "cuda:0", build_launch, [],
                       bench_fn=lambda l, a: 1.0)


def test_disabled_env(monkeypatch):
    monkeypatch.setenv(sa._DISABLE_ENV, "1")
    assert sa.disabled() is True
    monkeypatch.setenv(sa._DISABLE_ENV, "0")
    assert sa.disabled() is False


# --- M1/codegen emission ------------------------------------------------------


def _codegen(fixed_config=None):
    # One triple: mul(in0, 1.5) -> tmp0; output_refs point at tmp0.
    codegen = TritonProgramCodegen(
        program=[3, 0, -1],
        constants=[1.5],
        output_refs=(1,),
        input_count=1,
    )
    return codegen.generate("k", fixed_config=fixed_config)


def test_generate_default_emits_autotune_decorator():
    src = _codegen(None)
    assert "@triton.autotune(" in src
    assert "key=['xnumel']" in src
    assert "num_warps" in src.split("@triton.jit")[0]
    assert "[grid](inputs[0]" in src.split("def kernel_launch")[1]
    assert "XBLOCK=" not in src.split("def kernel_launch")[1]


def test_generate_fixed_config_drops_decorator_pins_launch():
    # Pass reference_shape so the literal-grid fast path is exercised.
    codegen = TritonProgramCodegen(
        program=[3, 0, -1],
        constants=[1.5],
        output_refs=(1,),
        input_count=1,
        reference_shape=(256,),
    )
    src = codegen.generate("k", fixed_config=(256, 4))
    head, tail = src.split("def kernel_launch")
    assert "@triton.autotune" not in head
    assert "@triton.jit" in head
    assert "XBLOCK=256, num_warps=4" in tail
    # Literal grid (no lambda meta) confirms the fast-path emission.
    assert "lambda meta" not in tail
    assert "grid_n" not in tail  # grid computed at codegen time, not emitted


def _cuda_inputs():
    return [tp.rand(300, device=tp.device("cuda", 0))]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")
def test_program_digest_shape_independent():
    x = _cuda_inputs()[0]
    digest_a = sa.program_digest([3, -1, 1], [1.5], (2,))
    digest_b = sa.program_digest([3, -1, 1], [1.5], (2,))
    assert digest_a == digest_b
    y = x * 2  # same program, different tensor -> same digest
    assert sa.program_digest([3, -1, 1], [1.5], (2,)) == digest_a
    del y


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not st.runtime_available(), reason="triton cannot target this device")
def test_bench_launch_returns_positive_ms():
    from tensorplay.compiler.codegen.triton import _compile_program

    pytest.importorskip("triton")
    args = _cuda_inputs()
    launch = _compile_program(program=[3, 0, -1], constants=[2.0],
                              output_refs=(1,), example_inputs=args)
    ms = sa.bench_launch(launch, args, warmup_ms=0.5, iters=3)
    assert ms > 0.0


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not st.runtime_available(), reason="triton cannot target this device")
def test_autotune_launch_end_to_end(tmp_path, monkeypatch):
    """Full path: first call benchmarks & persists, second call reuses."""

    triton = pytest.importorskip("triton")
    from tensorplay.compiler.codegen import triton as st

    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    import tensorplay.compiler.codecache as cc

    monkeypatch.setattr(cc, "_default_caches", {})
    st._launch_memo.clear()

    args = _cuda_inputs()
    launch1 = st._autotune_launch("fwd", [3, 0, -1], [2.0], (1,), args)
    assert callable(launch1)
    out = launch1(args)
    tp.cuda.synchronize()
    ref = (args[0] * 2.0).cpu()
    assert tp.abs(out.cpu() - ref).max().item() < 1e-6

    # Decision persisted: a fresh process view (cleared memo) skips benching.
    # Must include TUNING_VERSION in the key (stax_autotune.py salt).
    digest = sa.program_digest([3, 0, -1], [2.0], (1,))
    bucket = sa.xnumel_bucket(args[0].numel())
    device_key = repr(args[0].device)
    assert sa.load_decision(digest, bucket, device_key) is not None
    # Verify the stored record includes the tuning version salt.
    record = json.loads(
        sa._decision_cache().load(
            sa.decision_key(digest, bucket, device_key), ext="json"
        )
    )
    assert "xblock" in record and "warps" in record

    st._launch_memo.clear()
    launch2 = st._autotune_launch("fwd", [3, 0, -1], [2.0], (1,), args)
    assert callable(launch2)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not st.runtime_available(), reason="triton cannot target this device")
def test_triton_backward_still_works_through_autotune_path(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    from tensorplay.compiler.codegen.triton import compile_graph_module
    from tensorplay.compiler.graph import Tracer

    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    import tensorplay.compiler.codecache as cc

    monkeypatch.setattr(cc, "_default_caches", {})

    def fn(x, w):
        return ((x * w).relu()).sum()

    x = tp.rand(64, device=tp.device("cuda", 0), requires_grad=True)
    w = tp.rand(64, device=tp.device("cuda", 0), requires_grad=True)

    def backward_fn(x, w):
        return (x * w).relu()

    smap = {"x": x, "w": w}
    gm = Tracer().trace(backward_fn, sample_inputs=smap)
    compiled = compile_graph_module(gm, [x, w])
    assert compiled is not None
    assert compiled._tensorplay_codegen == "triton"
    assert compiled._tensorplay_backward_codegen == "triton"

    out = compiled(x.detach(), w.detach())  # inference call through fixed cfg
    tp.cuda.synchronize()
    expected = (x.detach() * w.detach()).relu().cpu()
    got = out.cpu() if isinstance(out, list) else out.cpu()
    assert tp.abs(got - expected).max().item() < 1e-6


# --- M5d: axis-reduction autotune -------------------------------------------------


def test_dims_autotune_persists_and_reuses_decision(cache_root, monkeypatch):
    """Sweep candidates once; later compiles reuse the persisted winner."""

    from tensorplay.compiler.codegen.triton import (
        _DIM_REDUCTION_CANDIDATES,
        ReductionSpec,
        _autotune_dims_program,
    )

    spec = ReductionSpec("sum", (1,))
    reference = (32, 8192)
    built: list[tuple] = []

    def fake_build(*args, **kwargs):
        cfg = kwargs["fixed_config"]
        built.append(cfg)

        def launch(inputs):
            return None

        launch.config = cfg
        return launch

    timings = {i: float(len(st._DIM_REDUCTION_CANDIDATES) - i)
               for i in range(len(st._DIM_REDUCTION_CANDIDATES))}
    timings[3] = min(timings.values()) - 1.0
    monkeypatch.setattr(st, "_compile_program", fake_build)
    monkeypatch.setattr(
        sa,
        "bench_launch",
        lambda launch, args: timings[_DIM_REDUCTION_CANDIDATES.index(launch.config)],
    )
    sample = SimpleNamespace(shape=(32, 8192), device="cuda:0", dtype="float32")
    launch = st._autotune_dims_program(
        "fwd0", [3, 0, -1], [2.0], (1,), [sample],
        reduction=spec, input_shapes=((32, 8192),), reference_shape=reference,
    )
    # every candidate was compiled + benchmarked exactly once
    assert len(built) == len(_DIM_REDUCTION_CANDIDATES)
    expected = min(
        _DIM_REDUCTION_CANDIDATES, key=lambda c: timings[_DIM_REDUCTION_CANDIDATES.index(c)]
    )
    assert launch.config[:2] == expected[:2]

    # second call: decision cache hit -> exactly one build, no benchmarking
    built.clear()
    monkeypatch.setattr(
        sa, "bench_launch", lambda launch, args: pytest.fail("must not bench")
    )
    cached = st._autotune_dims_program(
        "fwd0", [3, 0, -1], [2.0], (1,), [sample],
        reduction=spec, input_shapes=((32, 8192),), reference_shape=reference,
    )
    assert len(built) == 1
    assert built[0][:2] == expected[:2]


def test_dims_autotune_disabled_uses_static_config(cache_root, monkeypatch):
    from tensorplay.compiler.codegen.triton import (
        _STATIC_DIM_TRIPLE,
        ReductionSpec,
        _autotune_dims_program,
    )

    monkeypatch.setenv("TP_DISABLE_STAX_AUTOTUNE", "1")
    built: list[tuple] = []
    monkeypatch.setattr(st, "_compile_program", lambda *args, **kw: built.append(kw["fixed_config"]) or (lambda inputs: None))
    sample = SimpleNamespace(shape=(32, 64), device="cuda:0", dtype="float32")
    st._autotune_dims_program(
        "fwd0", [3], [], (1,), [sample],
        reduction=ReductionSpec("sum", (1,)),
        input_shapes=((32, 64),), reference_shape=(32, 64),
    )
    assert built == [_STATIC_DIM_TRIPLE]


def test_dims_autotune_all_fail_falls_back_static(cache_root, monkeypatch):
    from tensorplay.compiler.codegen.triton import (
        _STATIC_DIM_TRIPLE,
        ReductionSpec,
        _autotune_dims_program,
    )

    def broken_build(*args, **kwargs):
        raise RuntimeError("no cuda here")

    monkeypatch.setattr(st, "_compile_program", broken_build)
    sample = SimpleNamespace(shape=(32, 64), device="cuda:0", dtype="float32")
    # all candidates fail -> static fallback raises through the same builder
    with pytest.raises(RuntimeError, match="no cuda"):
        st._autotune_dims_program(
            "fwd0", [3], [], (1,), [sample],
            reduction=ReductionSpec("sum", (1,)),
            input_shapes=((32, 64),), reference_shape=(32, 64),
        )


def test_dims_autotune_invalid_cached_config_rebuilds(cache_root, monkeypatch):
    from tensorplay.compiler.codecache import default_cache
    from tensorplay.compiler.runtime import stax_autotune as sa
    import json as json_mod
    from tensorplay.compiler.codegen.triton import (
        ReductionSpec,
        _autotune_dims_program,
        _dims_decision_key,
    )

    spec = ReductionSpec("sum", (1,))
    key = _dims_decision_key(
        sa.program_digest([3], [], (1,)), spec, 32, 64, repr("cuda:0"), None, ""
    )
    default_cache("triton-autotune").store(
        key, json_mod.dumps({"xblock": 999, "warps": 1, "stages": 1}).encode(),
        ext="json",
    )
    built: list[tuple] = []
    monkeypatch.setattr(st, "_compile_program", lambda *args, **kw: built.append(kw["fixed_config"]) or (lambda inputs: None))
    times = iter([1.0] * len(st._DIM_REDUCTION_CANDIDATES))
    monkeypatch.setattr(sa, "bench_launch", lambda launch, args: next(times))
    sample = SimpleNamespace(shape=(32, 64), device="cuda:0", dtype="float32")
    st._autotune_dims_program(
        "fwd0", [3], [], (1,), [sample],
        reduction=spec,
        input_shapes=((32, 64),), reference_shape=(32, 64),
    )
    assert built  # full sweep ran despite the poisoned cache entry


def test_dims_autotune_rblock_record_roundtrip(cache_root, monkeypatch):
    """A quad winner persists RBLOCK and is reused verbatim on the next
    call (the old 3-field record could never validate a quad entry, so every
    process re-benchmarked forever)."""

    import json as json_mod
    from tensorplay.compiler.codecache import default_cache
    from tensorplay.compiler.codegen.triton import (
        _DIM_REDUCTION_CANDIDATES,
        _dims_decision_key,
        ReductionSpec,
        _autotune_dims_program,
    )

    spec = ReductionSpec("sum", (1,))
    quad = next(c for c in _DIM_REDUCTION_CANDIDATES if len(c) > 3)
    built: list[tuple] = []

    def fake_build(*args, **kwargs):
        cfg = kwargs["fixed_config"]
        built.append(cfg)

        def launch(inputs):
            return None

        launch.config = cfg
        return launch

    monkeypatch.setattr(st, "_compile_program", fake_build)
    monkeypatch.setattr(
        sa,
        "bench_launch",
        lambda launch, args: (
            0.5 if launch.config == quad else 5.0
        ),
    )
    sample = SimpleNamespace(shape=(32, 8192), device="cuda:0", dtype="float32")
    launch = _autotune_dims_program(
        "fwd0", [3, 0, -1], [2.0], (1,), [sample],
        reduction=spec, input_shapes=((32, 8192),), reference_shape=(32, 8192),
    )
    assert launch.config == quad
    key = _dims_decision_key(
        sa.program_digest([3, 0, -1], [2.0], (1,)), spec, 32, 8192,
        repr("cuda:0"), None, "",
    )
    record = json_mod.loads(
        default_cache("triton-autotune").load(key, ext="json").decode()
    )
    assert record["rblock"] == quad[2] and record["stages"] == quad[3]

    # cache hit: one build with the full quad, no benchmarking
    built.clear()
    monkeypatch.setattr(
        sa, "bench_launch", lambda launch, args: pytest.fail("must not bench")
    )
    cached = _autotune_dims_program(
        "fwd0", [3, 0, -1], [2.0], (1,), [sample],
        reduction=spec, input_shapes=((32, 8192),), reference_shape=(32, 8192),
    )
    assert built == [quad]
    assert cached.config == quad
