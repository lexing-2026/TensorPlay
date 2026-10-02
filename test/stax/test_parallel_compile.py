"""Parallel generated-kernel builds: worker resolution, ordering, wiring."""

import os
import threading
import time

import numpy as np
import pytest

import tensorplay
from tensorplay.compiler.backends.stax import parallel_compile


def _close(got, ref, rel=2e-5):
    got = np.asarray(got.tolist(), dtype=np.float64)
    ref = np.asarray(ref.tolist(), dtype=np.float64)
    assert got.shape == ref.shape
    scale = max(1e-6, float(np.max(np.abs(ref))) if ref.size else 1.0)
    assert float(np.max(np.abs(got - ref))) <= rel * scale


@pytest.fixture
def thread_config(monkeypatch):
    """Pin the worker count for one test; restore the knob afterwards."""

    import tensorplay.compiler.config as config

    def pin(count):
        monkeypatch.setattr(config, "compile_threads", count, raising=False)

    return pin


# ---------------------------------------------------------------------------
# worker-count resolution


def test_threads_default_to_cpu_count_capped(monkeypatch, thread_config):
    monkeypatch.delenv("TP_COMPILE_THREADS", raising=False)
    thread_config(None)
    assert parallel_compile.resolve_compile_threads() == min(
        parallel_compile.MAX_BUILD_WORKERS, os.cpu_count() or 1
    )


def test_threads_env_override_beats_the_cpu_default(monkeypatch, thread_config):
    monkeypatch.setenv("TP_COMPILE_THREADS", "3")
    thread_config(None)
    assert parallel_compile.resolve_compile_threads() == 3


def test_threads_invalid_env_falls_back(monkeypatch, thread_config):
    monkeypatch.setenv("TP_COMPILE_THREADS", "not-a-number")
    thread_config(None)
    assert parallel_compile.resolve_compile_threads() >= 1


@pytest.mark.parametrize("value", [0, -2, 1.5, "four", True])
def test_threads_rejects_invalid_config_values(thread_config, value):
    thread_config(value)
    with pytest.raises(TypeError):
        parallel_compile.resolve_compile_threads()


# ---------------------------------------------------------------------------
# run_builds semantics


def test_results_keep_job_order_under_jitter():
    def job(index):
        time.sleep((index * 7) % 5 / 500)
        return index * 10

    results = parallel_compile.run_builds([lambda i=i: job(i) for i in range(16)])
    assert results == [i * 10 for i in range(16)]


def test_jobs_overlap_when_more_than_one_worker_is_available(thread_config):
    thread_config(4)
    # The first job parks until the second one signals: a serial run would
    # stall on the handshake until the timeout, so completing at all proves
    # the two builds ran concurrently.
    release = threading.Event()

    def park():
        assert release.wait(timeout=30), "builds did not overlap"
        return "parked"

    def signal():
        release.set()
        return "signaled"

    assert parallel_compile.run_builds([park, signal]) == ["parked", "signaled"]


def test_single_worker_runs_jobs_on_the_calling_thread(thread_config):
    thread_config(1)
    main = threading.get_ident()
    seen = []
    parallel_compile.run_builds(
        [lambda seen=seen: seen.append(threading.get_ident()) for _ in range(3)]
    )
    assert seen == [main, main, main]


def test_single_job_never_spawns_workers(thread_config):
    thread_config(8)
    assert parallel_compile.run_builds([lambda: threading.get_ident()]) == [
        threading.get_ident()
    ]


def test_first_raising_job_surfaces_to_the_caller(thread_config):
    thread_config(4)
    sentinel = RuntimeError("build exploded")

    def boom():
        raise sentinel

    with pytest.raises(RuntimeError) as caught:
        parallel_compile.run_builds([lambda: "ok", boom, lambda: "late"])
    assert caught.value is sentinel


# ---------------------------------------------------------------------------
# wiring into the segmented CPU lowering


@pytest.fixture
def cold_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    return tmp_path


def _two_kernel_region():
    # Two fusible runs separated by (and ending in) matrix products: the
    # scheduler turns each run into one generated kernel and each product
    # stays a call, so the region builds exactly two kernels.
    return lambda v, m: ((v * 2.0).tanh() @ m).erf().exp() @ m


def _compile_two_kernel_region():
    tensorplay.manual_seed(11)
    x = tensorplay.randn(6, 24)
    w = tensorplay.randn(24, 24)
    fn = _two_kernel_region()
    return tensorplay.compile(fn, backend="stax"), fn, x, w


def _codegen_route(compiled):
    lowering = next(iter(compiled._tensorplay_cache.values()))
    return getattr(lowering, "_tensorplay_codegen", None), lowering


def _spy_build_threads(monkeypatch):
    from tensorplay.compiler.backends.stax import backend as backend_mod

    idents = []
    real = backend_mod._compile_segment_kernel

    def spy(plan, device):
        idents.append(threading.get_ident())
        return real(plan, device)

    monkeypatch.setattr(backend_mod, "_compile_segment_kernel", spy)
    return idents


def test_multi_kernel_region_builds_serially_and_matches(
    cold_cache, monkeypatch, thread_config
):
    thread_config(1)
    idents = _spy_build_threads(monkeypatch)

    compiled, fn, x, w = _compile_two_kernel_region()
    _close(compiled(x, w), fn(x, w))
    codegen, _lowering = _codegen_route(compiled)
    assert codegen == "stax-fused-cpu-segments"
    assert len(idents) == 2
    assert set(idents) == {threading.get_ident()}


def test_multi_kernel_region_builds_on_worker_threads(
    cold_cache, monkeypatch, thread_config
):
    thread_config(4)
    idents = _spy_build_threads(monkeypatch)

    compiled, fn, x, w = _compile_two_kernel_region()
    _close(compiled(x, w), fn(x, w))
    codegen, _lowering = _codegen_route(compiled)
    assert codegen == "stax-fused-cpu-segments"
    assert len(idents) == 2
    assert threading.get_ident() not in set(idents), "builds stayed on the caller"


def test_parallel_and_serial_builds_agree(cold_cache, thread_config):
    thread_config(4)
    compiled, fn, x, w = _compile_two_kernel_region()
    parallel_result = np.asarray(compiled(x, w).tolist(), dtype=np.float64)

    thread_config(1)
    tensorplay.compiler.reset()
    serial_result = np.asarray(
        tensorplay.compile(fn, backend="stax")(x, w).tolist(), dtype=np.float64
    )

    assert np.array_equal(parallel_result, serial_result)
    _close(parallel_result, np.asarray(fn(x, w).tolist(), dtype=np.float64))
