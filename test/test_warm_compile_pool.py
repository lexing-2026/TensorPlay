"""Warm-compile helper pool for the Triton autotuner.

Gate plumbing and the worker's source/launch mechanics are tested
everywhere; the checks that push compiles through the real helper pool
are gated on ``runtime_available()`` -- they need a live Triton/CUDA
runtime, as does anything the pool would compile.
"""

import linecache
import time

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen.triton import (
    _compile_program,
    _program_source,
    runtime_available,
)
from tensorplay.compiler.backends.stax.runtime import warm_compile


# --- gates -----------------------------------------------------------------------


@pytest.fixture
def fresh_pool_state():
    """Pin the process-persistent pool out of the gate checks.

    Other suites exercise the production autotune path, which may have
    already built the shared pool; the gates must observe a clean slate.
    """

    saved = warm_compile._executor
    warm_compile._executor = None
    yield
    warm_compile._executor = saved


def test_environment_switch_disables_everything(monkeypatch, fresh_pool_state):
    monkeypatch.setenv("TP_STAX_PARALLEL_COMPILE", "0")
    assert warm_compile.enabled() is False
    assert warm_compile.warm_sources([]) == 0
    assert warm_compile._executor is None


def test_single_task_is_not_worth_a_pool(fresh_pool_state):
    # One candidate has nothing to overlap; the gate declines without
    # building the executor regardless of runtime availability.
    assert warm_compile.warm_sources([("src", "<f>", ())]) == 0
    assert warm_compile._executor is None


def test_dtype_and_device_lookup_from_reprs():
    assert warm_compile._dtype_by_repr(tp, repr(tp.float32)) is tp.float32
    fallback = warm_compile._dtype_by_repr(tp, "not-a-dtype")
    assert fallback is tp.float32
    assert str(warm_compile._device_by_repr(tp, "cpu")) == "cpu"


# --- worker mechanics -------------------------------------------------------------


def _mul_program_source(x, config):
    # (mul, out, const) over input 0: the smallest program that exercises
    # the full generated launch path (alloc, jit call, store).
    return _program_source([3, 0, -1], [2.0], (1,), [x], fixed_config=config)


def _mul_meta(x):
    return (
        (tuple(int(size) for size in x.shape), repr(x.dtype), str(x.device)),
    )


def _parent_launch(src, fake_file, args):
    namespace = {}
    linecache.cache[fake_file] = (len(src), None, src.splitlines(True), fake_file)
    exec(compile(src, fake_file, "exec"), namespace, namespace)
    out = namespace["kernel_launch"](args)
    tp.cuda.synchronize()
    return out


def test_source_generation_is_deterministic():
    # The pool ships source text to helpers; regeneration in the parent
    # must produce byte-identical text or the helper compiles a different
    # cache key than the parent will request.
    tp.manual_seed(0)
    x = tp.rand(32, 64)
    first = _program_source([3, 0, -1], [2.0], (1,), [x], fixed_config=(128, 4))
    second = _program_source([3, 0, -1], [2.0], (1,), [x], fixed_config=(128, 4))
    assert first == second
    assert "kernel_launch" in first[0]


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_worker_exec_compiles_and_launches():
    tp.manual_seed(0)
    x = tp.rand(64, 128, device="cuda")
    src, fake_file = _mul_program_source(x, (128, 4))
    assert warm_compile._warm_one((src, fake_file, _mul_meta(x))) is True

    out = _parent_launch(src, fake_file, [x])
    assert bool(((out - x * 2.0).abs().max() < 1e-6))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_pool_warm_reaches_the_parent():
    tp.manual_seed(0)
    x = tp.rand(64, 128, device="cuda")
    configs = [(128, 4), (256, 4)]
    meta = _mul_meta(x)
    tasks = [
        (*_mul_program_source(x, config), meta) for config in configs
    ]

    # Pre-pay the parent's one-time launch-path initialization so the
    # timing assertion below measures the disk-cache hit alone.
    warm_src, warm_fake = _mul_program_source(x, (512, 4))
    _parent_launch(warm_src, warm_fake, [x])

    try:
        assert warm_compile.warm_sources(tasks) == 2

        for config, task in zip(configs, tasks):
            src, fake_file = task[0], task[1]
            start = time.perf_counter()
            out = _parent_launch(src, fake_file, [x])
            elapsed = time.perf_counter() - start
            # The helper already compiled this exact (source, config) pair
            # into the shared on-disk cache; the parent's first launch
            # loads it instead of compiling.  A serial in-parent compile of
            # the candidate would blow well past this bound.
            assert elapsed < 0.35, f"config {config} launched cold: {elapsed:.3f}s"
            assert bool(((out - x * 2.0).abs().max() < 1e-6))
    finally:
        warm_compile._shutdown()


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_warmed_candidates_bench_without_recompiling():
    from tensorplay.compiler.backends.stax.runtime import stax_autotune

    tp.manual_seed(0)
    x = tp.rand(128, 256, device="cuda")
    candidates = stax_autotune.CANDIDATE_CONFIGS[:3]
    meta = _mul_meta(x)
    tasks = [
        (*_mul_program_source(x, config), meta) for config in candidates
    ]
    try:
        assert warm_compile.warm_sources(tasks) == len(tasks)

        def build(config):
            return _compile_program([3, 0, -1], [2.0], (1,), [x], fixed_config=config)

        best_config, best_launch, best_time = stax_autotune.bench_candidates(
            build, candidates, [x]
        )
        assert best_config in candidates
        assert best_time < float("inf")
        out = best_launch([x])
        assert bool(((out - x * 2.0).abs().max() < 1e-6))
    finally:
        warm_compile._shutdown()
