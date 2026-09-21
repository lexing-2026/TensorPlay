"""Pre-bound native launcher for recorded Triton kernels.

Guard plumbing is exercised on stub kernels without a runtime; the
record/replay/numerics checks and the replay-vs-native A/B timing need a
live Triton/CUDA runtime and are gated on ``runtime_available()``.
"""

import time

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen.triton import (
    ReductionSpec,
    _program_source,
    runtime_available,
)
from tensorplay.compiler.backends.stax.runtime import fastlaunch as _fl


# --- stub kernels for the guard table (no runtime needed) ------------------------


class _StubSrc:
    def __init__(self, signature):
        self.signature = signature


class _StubMetadata:
    def __init__(self, **overrides):
        self.num_warps = 4
        self.shared = 0
        self.num_ctas = 1
        self.launch_pdl = False
        self.launch_cooperative_grid = False
        self.global_scratch_size = 0
        self.global_scratch_align = 1
        self.profile_scratch_size = 0
        self.profile_scratch_align = 1
        for key, value in overrides.items():
            setattr(self, key, value)


class _StubKernel:
    def __init__(self, signature, **metadata_overrides):
        self.src = _StubSrc(signature)
        self.metadata = _StubMetadata(**metadata_overrides)
        self.function = 0x51D0
        self.shared = 0


_FULL = {"0": "*fp32", "1": "i32", "2": "constexpr", "3": "i64"}


def test_launcher_type_matches_the_cuda_build():
    from tensorplay import _C

    assert _fl._static_launcher_type() is getattr(
        _C, "_StaxFastLauncher", None
    )


def test_guards_reject_unsupported_binaries(monkeypatch):
    if _fl._static_launcher_type() is None:
        pytest.skip("_StaxFastLauncher unavailable (CPU-only build)")

    # No live parameters at all: nothing a direct launch could pack.
    stub = _StubKernel({"0": "constexpr", "1": "constexpr"})
    assert _fl._native_for_kernel(stub, 2) is None
    # Scalar type outside the packing table.
    stub = _StubKernel({"0": "*fp32", "1": "fp8"})
    assert _fl._native_for_kernel(stub, 2) is None
    # Call-site argument count disagrees with the signature.
    stub = _StubKernel(_FULL)
    assert _fl._native_for_kernel(stub, 3) is None
    # Extra CTAs / PDL / cooperative launch need the generic launcher.
    for overrides in (
        {"num_ctas": 2},
        {"launch_pdl": True},
        {"launch_cooperative_grid": True},
    ):
        stub = _StubKernel(_FULL, **overrides)
        assert _fl._native_for_kernel(stub, 4) is None
    # Kernels that need scratch storage cannot run with NULL scratch slots.
    stub = _StubKernel(_FULL, global_scratch_size=64)
    assert _fl._native_for_kernel(stub, 4) is None
    stub = _StubKernel(_FULL, profile_scratch_size=16)
    assert _fl._native_for_kernel(stub, 4) is None
    # The surface is absent entirely.
    monkeypatch.setattr(_fl, "_static_launcher_type", lambda: None)
    stub = _StubKernel(_FULL)
    assert _fl._native_for_kernel(stub, 4) is None


def test_happy_path_maps_signature_and_keeps_kernel_alive():
    if _fl._static_launcher_type() is None:
        pytest.skip("_StaxFastLauncher unavailable (CPU-only build)")

    stub = _StubKernel(_FULL)
    record = _fl._native_for_kernel(stub, 4)
    assert record is not None
    launcher, function, packed = record
    assert function == stub.function
    assert packed is None
    # The launcher object holds the kernel; the type is the C surface.
    assert type(launcher).__name__ == "_StaxFastLauncher"
    assert launcher.__sizeof__() > 0  # constructed, not a bare surrogate


# --- recorded-path checks (need a live runtime) ----------------------------------


def _exec_program(program, constants, output_refs, example_inputs, **kwargs):
    """Exec one generated program source and return its module namespace.

    ``_compile_program`` memoizes and discards the namespace; the tests need
    to introspect the recorded record tuple, so they drive the exec
    themselves through the same :func:`_program_source` entry point.
    """

    source, fake_file = _program_source(
        program, constants, output_refs, example_inputs, **kwargs
    )
    import linecache

    linecache.cache[fake_file] = (
        len(source),
        None,
        source.splitlines(True),
        fake_file,
    )
    namespace: dict = {}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    return namespace


def _jit_function(namespace):
    import triton

    return next(
        value
        for value in namespace.values()
        if isinstance(value, triton.runtime.JITFunction)
    )


def _launcher_type_name(record):
    if record is None:
        return None
    return type(record[0]).__name__


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_pointwise_records_native_and_matches_eager():
    x = tp.rand(4096, device="cuda")
    namespace = _exec_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=(256, 4),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )
    launch = namespace["kernel_launch"]
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, x * 2.0, rtol=1e-5, atol=1e-5)
    assert _launcher_type_name(namespace["_rec"]) == "_StaxFastLauncher"
    before = _fl.FAST_CALLS
    out = launch([x])
    tp.cuda.synchronize()
    assert _fl.FAST_CALLS == before + 1
    assert tp.allclose(out, x * 2.0, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_pointwise_scalar_input_skips_dead_positions():
    """numel == 1 specializes xnumel away; the packed call must skip it."""

    x = tp.rand(1, device="cuda")
    namespace = _exec_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=(256, 4),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )
    launch = namespace["kernel_launch"]
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, x * 2.0, rtol=1e-5, atol=1e-5)
    assert _launcher_type_name(namespace["_rec"]) == "_StaxFastLauncher"
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, x * 2.0, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_dims_and_single_reductions_record_native():
    x = tp.rand(64, 4096, device="cuda")
    namespace = _exec_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=(1, 16, 4096, 3), reduction=ReductionSpec("sum", (1,)),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )
    launch = namespace["kernel_launch"]
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, (x * 2.0).sum(dim=1), rtol=1e-4, atol=1e-1)
    assert _launcher_type_name(namespace["_rec"]) == "_StaxFastLauncher"
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, (x * 2.0).sum(dim=1), rtol=1e-4, atol=1e-1)

    small = tp.rand(64, device="cuda")
    namespace = _exec_program(
        [3, 0, -1], [2.0], (1,), [small],
        fixed_config=(64, 4), reduction=ReductionSpec("sum"),
        input_shapes=(tuple(small.shape),), reference_shape=tuple(small.shape),
    )
    launch = namespace["kernel_launch"]
    out = launch([small])
    tp.cuda.synchronize()
    assert tp.allclose(out, (small * 2.0).sum(), rtol=1e-4, atol=1e-1)
    assert _launcher_type_name(namespace["_rec"]) == "_StaxFastLauncher"
    out = launch([small])
    tp.cuda.synchronize()
    assert tp.allclose(out, (small * 2.0).sum(), rtol=1e-4, atol=1e-1)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_replay_vs_native_launch_cost():
    """A/B the two record styles through the production fast-path call.

    Informational: prints both costs and only guards against the native
    path being pathologically slower (the point is the CPU cost between
    launches, so the timing is stable once the kernel is memory-bound).
    """

    # Small enough that the GPU drains faster than the CPU submits: the
    # loop then measures the per-call CPU cost instead of driver-queue
    # throughput (a large grid would block cuLaunchKernel and level the
    # two paths).
    x = tp.rand(4096, device="cuda")
    namespace = _exec_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=(256, 4),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )
    launch = namespace["kernel_launch"]
    launch([x])
    tp.cuda.synchronize()
    assert _launcher_type_name(namespace["_rec"]) == "_StaxFastLauncher"

    jitfn = _jit_function(namespace)
    snapshot = _fl.cache_size(jitfn)
    replay = _fl.take_kernel(jitfn, snapshot)
    native = _fl.native_wrap(jitfn, snapshot, 4)
    assert _launcher_type_name(native) == "_StaxFastLauncher"

    # Isolate the recorded callable: the launch() closure adds an output
    # allocation and Python-side alignment guards to every call, which
    # would drown the launcher difference.  Both records take the identical
    # positional shape, so the same loop body serves both.
    out = tp.empty_like(x)
    stream = _fl.current_stream()
    xnumel = x.numel()
    grid0 = -(-xnumel // 256)

    def timed(record, iters=5000):
        run, function, packed = record
        run(grid0, 1, 1, stream, function, packed, None, None, None,
            x, out, xnumel, 256)
        tp.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            run(grid0, 1, 1, stream, function, packed, None, None, None,
                x, out, xnumel, 256)
        tp.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e6

    replay_us = timed(replay)
    native_us = timed(native)
    print(
        f"\nreplay {replay_us:.2f} us/call  native {native_us:.2f}"
        f" us/call  speedup {replay_us / max(native_us, 1e-9):.2f}x"
    )
    assert native_us <= replay_us * 1.5 + 2.0
    # The direct calls wrote through `out`; results stay eager-matching.
    assert tp.allclose(out, x * 2.0, rtol=1e-5, atol=1e-5)
