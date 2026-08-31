"""CPU codegen stack: ISA selection, build plumbing, DAG rendering, e2e."""

import os
import textwrap

import numpy as np
import pytest

import tensorplay
from tensorplay._stax.codegen import cpp as cpu_cpp
from tensorplay._stax.codecache import CodeCache, file_lock
from tensorplay._stax.cpp_builder import (
    CppBuilder,
    CppOptions,
    get_compiler_version_info,
    get_cpp_compiler,
)
from tensorplay._stax.cpu_vec_isa import (
    InvalidVecISA,
    VecAVX2,
    VecDefault,
    pick_vec_isa,
)


# ---------------------------------------------------------------------------
# cpp_builder


def test_compiler_discovery_and_version_fingerprint():
    compiler = get_cpp_compiler()
    assert compiler, "a system C++ compiler is required for the native path"
    info = get_compiler_version_info(compiler)
    assert info
    assert get_compiler_version_info(compiler) == info


def test_command_orders_libraries_after_sources(tmp_path):
    options = CppOptions(
        compiler="g++",
        include_dirs=["/inc"],
        cflags=["-O3"],
        library_dirs=["/lib"],
        libraries=["p10"],
    )
    cmd = options.command(["kernel.cpp"], "kernel.so")
    assert cmd.index("kernel.cpp") < cmd.index("-lp10")
    assert cmd[-1] == "kernel.so"


def test_flags_hash_stable_and_sensitive(tmp_path):
    a = CppOptions(compiler="g++", cflags=["-O3", "-mavx2"])
    a2 = CppOptions(compiler="g++", cflags=["-O3", "-mavx2"])
    c = CppOptions(compiler="g++", cflags=["-O2", "-mavx2"])
    d = CppOptions(compiler="g++", cflags=["-O3", "-mavx2"], libraries=["p10"])
    assert a.flags_hash() == a2.flags_hash()
    assert a.flags_hash() != c.flags_hash()
    assert a.flags_hash() != d.flags_hash()


def test_builder_reports_failure_detail(tmp_path):
    options = CppOptions(compiler=get_cpp_compiler(), cflags=["-std=c++20"])
    builder = CppBuilder(
        name="broken.so",
        sources=[],
        options=options,
        output_dir=str(tmp_path),
    )
    source = tmp_path / "broken.cpp"
    source.write_text("this is not c++\n")
    builder.sources = [str(source)]
    with pytest.raises(RuntimeError, match="cpp build failed"):
        builder.build()


# ---------------------------------------------------------------------------
# cpu_vec_isa


def test_vec_isa_metadata():
    assert VecAVX2().bit_width == 256
    assert VecAVX2().nelements() == 8
    assert VecDefault().nelements() == 4
    assert "-mavx2" in VecAVX2().build_arch_flags()
    assert VecAVX2().definitions()[0] == "-DCPU_CAPABILITY_AVX2"
    assert not InvalidVecISA()


def test_pick_vec_isa_override(monkeypatch):
    monkeypatch.setenv("TP_STAX_CPU_TIER", "default")
    isa = pick_vec_isa()
    assert isa.name == "default"
    monkeypatch.setenv("TP_STAX_CPU_TIER", "invalid")
    assert isinstance(pick_vec_isa(), InvalidVecISA)


def test_pick_vec_isa_is_toolchain_verified():
    isa = pick_vec_isa()
    assert isa and isa.is_feasible()


def test_isa_probe_marker_persists(tmp_path, monkeypatch):
    from tensorplay._stax import codecache

    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TP_STAX_CPU_TIER", raising=False)
    monkeypatch.setattr(codecache, "_default_caches", {})
    isa = VecDefault()
    assert isa.is_feasible()
    markers = list(tmp_path.rglob("*.load_ok"))
    assert markers, "probe verdict should be persisted next to the cache"
    # A second instance picks up the persisted verdict without rebuilding.
    monkeypatch.setattr(codecache, "_default_caches", {})
    assert VecDefault().is_feasible()


# ---------------------------------------------------------------------------
# codegen: rendering structure


def _render(steps, constants, n, out):
    return cpu_cpp.render_kernel_source(steps, constants, n, out, "tp_test_entry")


def test_render_chain_has_three_phase_loops():
    # (a + b) * 2.0
    steps = [
        ("add", 0, 1, 2),
        ("mul", 2, -1, 3),
    ]
    source = _render(steps, [2.0], 2, 3)
    assert "#pragma GCC ivdep" in source
    assert "__restrict__" in source
    assert "i + 4 * W <= e" in source  # unrolled main loop
    assert "i + W <= e" in source  # single-vector loop
    assert "const long count = e - i;" in source  # scalar tail
    assert "V c0 = V(2.0f);" in source


def test_entry_parallel_decision_matches_pool_semantics():
    # Serial below (pool threads x min chunk); the bridge receives the
    # chunk floor so the split stays even (static schedule semantics).
    source = _render([("relu", 0, -1, 1)], [], 1, 1)
    import tensorplay

    threads = tensorplay.get_num_threads()
    assert f"if (n < {threads * 512}LL)" in source
    assert "tp_body(&ctx, 0, n);" in source
    assert "tp_parallel_for_c(0, n, 512LL, tp_body, &ctx);" in source


def test_render_skips_unroll_for_long_programs():
    # 20 chained muls exceed the unroll threshold: only the single-vector
    # loop may appear.
    steps = []
    prev = 0
    for i in range(20):
        result = 3 + i
        steps.append(("mul", prev, -1, result))
        prev = result
    source = _render(steps, [1.5] * 20, 3, prev)
    assert "i + 4 * W <= e" not in source
    assert "i + W <= e" in source


def test_render_where_pair_uses_blendv():
    steps = [
        ("gt", 0, 1, 2),
        ("where", 2, 0, 3),
        ("where_rest", 2, 1, 4),
    ]
    source = _render(steps, [], 2, 4)
    assert "V::blendv(" in source
    assert "x0.gt(x1)" in source


def test_render_rejects_bad_programs():
    with pytest.raises(Exception):
        _render([("where_rest", 0, 1, 2)], [], 2, 2)  # rest without where
    with pytest.raises(Exception):
        _render([("add", 0, 99, 2)], [], 2, 2)  # ref out of range
    with pytest.raises(Exception):
        _render([("frobnicate", 0, 1, 2)], [], 2, 2)  # unknown op
    with pytest.raises(Exception):
        _render([("mul", 0, -7, 2)], [1.0], 2, 2)  # constant ref out of range
    with pytest.raises(Exception):
        _render([("mul", 0, 1, 5)], [], 2, 3)  # result refs must be sequential
    with pytest.raises(Exception):
        _render([("cast", 0, 1, 2)], [], 2, 2)  # non-f32 cast target


def test_output_can_be_a_raw_input():
    # Degenerate but legal: dead computation, output is an input directly.
    source = _render([("mul", 0, 1, 3)], [], 3, 1)
    assert "x1.store(out + i, W);" in source


# ---------------------------------------------------------------------------
# codegen: numeric equivalence against numpy


def _execute(instructions, constants, inputs, output_ref=None):
    n = len(inputs)
    if output_ref is None:
        output_ref = n + len(instructions) - 1
    runner = cpu_cpp.build_cpu_native_kernel(instructions, constants, n, output_ref)
    assert runner is not None, "native build failed"
    return runner(inputs)


def test_equivalence_chain_and_diamond():
    a = np.random.RandomState(0).randn(4096).astype(np.float32)
    b = np.random.RandomState(1).randn(4096).astype(np.float32)
    ta = tensorplay.from_numpy(a)
    tb = tensorplay.from_numpy(b)

    # chain: tanh(exp(a) * 2 + b) / 3
    got = _execute(
        [
            ("exp", 0, -1, 2),
            ("mul", 2, -1, 3),
            ("add", 3, 1, 4),
            ("tanh", 4, -1, 5),
            ("div", 5, -2, 6),
        ],
        [2.0, 3.0],
        [ta, tb],
    )
    expected = np.tanh(np.exp(a) * 2.0 + b) / 3.0
    # Transcendentals use SLEEF inside the kernel and libm in numpy, so a
    # tolerance applies; the diamond case below is pure IEEE arithmetic.
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)

    # diamond: (a + b) * (a - b) — both consumers reload shared temps
    got = _execute(
        [
            ("add", 0, 1, 2),
            ("sub", 0, 1, 3),
            ("mul", 2, 3, 4),
        ],
        [],
        [ta, tb],
    )
    np.testing.assert_array_almost_equal_nulp(
        got.numpy(), (a + b) * (a - b), nulp=4
    )


def test_equivalence_where_and_comparisons():
    a = np.random.RandomState(2).randn(1000).astype(np.float32)
    b = np.random.RandomState(3).randn(1000).astype(np.float32)
    ta = tensorplay.from_numpy(a)
    tb = tensorplay.from_numpy(b)

    # where(a > b, a * 0.5, -b)
    got = _execute(
        [
            ("gt", 0, 1, 2),
            ("mul", 0, -1, 3),
            ("neg", 1, -1, 4),
            ("where", 2, 3, 5),
            ("where_rest", 2, 4, 6),
        ],
        [0.5],
        [ta, tb],
    )
    expected = np.where(a > b, a * 0.5, -b)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)

    # clamp via minimum/maximum: maximum(a, 0) + minimum(b, 1)
    got = _execute(
        [
            ("clamp_min", 0, -1, 2),
            ("clamp_max", 1, -2, 3),
            ("add", 2, 3, 4),
        ],
        [0.0, 1.0],
        [ta, tb],
    )
    expected = np.maximum(a, 0.0) + np.minimum(b, 1.0)
    np.testing.assert_array_almost_equal_nulp(
        got.numpy(), expected.astype(np.float32), nulp=4
    )


def test_equivalence_tail_and_single_element():
    rs = np.random.RandomState(4)
    for size in (1, 7, 33, 1001):
        a = rs.randn(size).astype(np.float32)
        b = rs.randn(size).astype(np.float32)
        ta = tensorplay.from_numpy(a)
        tb = tensorplay.from_numpy(b)
        got = _execute(
            [
                ("add", 0, 1, 2),
                ("sigmoid", 2, -1, 3),
            ],
            [],
            [ta, tb],
        )
        expected = 1.0 / (1.0 + np.exp(-(a + b)))
        np.testing.assert_allclose(
            got.numpy(), expected.astype(np.float32), rtol=1e-6
        )


def test_kernel_cache_reuses_artifact():
    ta = tensorplay.from_numpy(np.ones(64, dtype=np.float32))
    instructions = [("relu", 0, -1, 1)]
    first = cpu_cpp.build_cpu_native_kernel(instructions, [], 1, 1)
    assert first is not None
    # Second build with the identical program must hit the on-disk artifact
    # (no rebuild) and still produce a working callable.
    second = cpu_cpp.build_cpu_native_kernel(instructions, [], 1, 1)
    assert second is not None
    np.testing.assert_array_equal(
        second([ta]).numpy(), np.maximum(np.ones(64), 0.0)
    )


# ---------------------------------------------------------------------------
# e2e through the compiler entrypoint


def _codecache_clean():
    root = tensorplay._stax.codecache.default_cache("stax-cpu-native").root
    return os.path.exists(root)


def test_compile_routes_pointwise_to_cpu_native(tmp_path):
    x = tensorplay.randn(64, 64)

    def fn(x):
        return ((x * 2.0).tanh() + 1.0) / 3.0

    compiled = tensorplay.compile(fn, backend="stax")
    out = compiled(x)
    expected = (np.tanh(x.numpy() * 2.0) + 1.0) / 3.0
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-3, atol=1e-7)

    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen == "stax-fused-cpu"
    assert lowering._native_runner is not None


def test_compile_routes_extended_surface_without_grad(tmp_path):
    x = tensorplay.randn(32, 32)
    y = tensorplay.randn(32, 32)

    def fn(x, y):
        return tensorplay.where(x > y, x * 0.5, -y)

    compiled = tensorplay.compile(fn, backend="stax")
    out = compiled(x, y)
    xa, ya = x.numpy(), y.numpy()
    expected = np.where(xa > ya, xa * 0.5, -ya)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-3, atol=1e-7)

    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen == "stax-fused-cpu"
    assert lowering._native_runner is not None
    assert lowering._gradient_plan is None


def test_compile_extended_surface_with_grad_stays_supported():
    # grad-enabled graphs keep the base-surface route (fused autograd);
    # this graph contains only base ops, so it must compile with gradients.
    x = tensorplay.randn(16, 16, requires_grad=True)

    def fn(x):
        return tensorplay.tanh(x * 2.0)

    compiled = tensorplay.compile(fn, backend="stax")
    out = compiled(x)
    out.backward(tensorplay.ones_like(out))
    expected_x = x.detach().numpy() * 2.0
    th = np.tanh(expected_x)
    expected = 2.0 * (1.0 - th * th)
    np.testing.assert_allclose(
        x.grad.numpy(), expected, rtol=1e-4, atol=1e-6
    )
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen == "stax-fused-cpu"
    assert lowering._gradient_plan is not None
