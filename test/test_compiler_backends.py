from importlib.metadata import EntryPoint

import pytest

import tensorplay as tp
from tensorplay._stax import registry


@pytest.fixture
def fresh_entrypoints(monkeypatch):
    """Rerun entry-point discovery against a stubbed group listing."""

    def install(*points):
        def fake_entry_points(group=None):
            assert group == "tensorplay_compiler_backends"
            return list(points)

        monkeypatch.setattr(registry, "entry_points", fake_entry_points)
        monkeypatch.setattr(registry, "_entrypoints_loaded", False)

    yield install
    for name in ("ep_backend", "ep_broken", "ep_not_callable"):
        registry.unregister_backend(name)


def ep_backend(graph_module, example_inputs, **kwargs):
    return graph_module


EP_NOT_CALLABLE = 3


def test_invalid_backend_suggests_close_names():
    with pytest.raises(tp.compiler.InvalidBackend, match="did you mean: 'stax'"):
        tp.compiler.lookup_backend("stx")


def test_invalid_backend_is_value_and_runtime_error():
    with pytest.raises(ValueError):
        tp.compiler.lookup_backend("no_such_backend")
    with pytest.raises(RuntimeError, match="list_backends"):
        tp.compiler.lookup_backend("no_such_backend")


def test_entry_point_backend_loads_lazily(fresh_entrypoints):
    fresh_entrypoints(
        EntryPoint("ep_backend", f"{__name__}:ep_backend", "tensorplay_compiler_backends")
    )
    assert "ep_backend" in tp.compiler.list_backends()
    assert "ep_backend" not in registry._compiler_fns
    assert tp.compiler.lookup_backend("ep_backend") is ep_backend
    assert registry._is_registered_backend(ep_backend)

    compiled = tp.compile(lambda x: x * 2, backend="ep_backend")
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]


def test_entry_point_does_not_shadow_builtin(fresh_entrypoints):
    fresh_entrypoints(
        EntryPoint("stax", f"{__name__}:ep_backend", "tensorplay_compiler_backends")
    )
    assert tp.compiler.lookup_backend("stax") is not ep_backend


def test_entry_point_load_errors_name_the_entry_point(fresh_entrypoints):
    fresh_entrypoints(
        EntryPoint("ep_broken", "no_such_module_xyz:fn", "tensorplay_compiler_backends"),
        EntryPoint(
            "ep_not_callable", f"{__name__}:EP_NOT_CALLABLE", "tensorplay_compiler_backends"
        ),
    )
    with pytest.raises(RuntimeError, match="no_such_module_xyz:fn"):
        tp.compiler.lookup_backend("ep_broken")
    with pytest.raises(TypeError, match="expected a callable"):
        tp.compiler.lookup_backend("ep_not_callable")


def test_unregistered_callable_is_not_a_registered_backend():
    assert not registry._is_registered_backend(ep_backend)
    assert registry._is_registered_backend(tp.compiler.lookup_backend("stax"))


def test_reset_calls_backend_reset_hooks():
    calls = []

    def stateful(graph_module, example_inputs, **kwargs):
        return graph_module

    stateful.reset = lambda: calls.append("reset")
    tp.compiler.register_backend(stateful, name="stateful_reset_backend")
    try:
        tp.compiler.reset()
        assert calls == ["reset"]
    finally:
        tp.compiler.unregister_backend("stateful_reset_backend")


DEBUG_BACKENDS = (
    "eager",
    "eager_debug",
    "eager_noexcept",
    "non_leaf_compile_error_TESTING_ONLY",
    "relu_accuracy_error_TESTING_ONLY",
    "relu_compile_error_TESTING_ONLY",
    "relu_runtime_error_TESTING_ONLY",
)


def _relu_fn(x):
    return tp.relu(x) * 2 + x.relu()


def test_debug_backends_are_hidden_by_default():
    visible = tp.compiler.list_backends()
    everything = tp.compiler.list_backends(exclude_tags=None)
    for name in DEBUG_BACKENDS:
        assert name not in visible
        assert name in everything


@pytest.mark.parametrize("backend", ["eager", "eager_noexcept", "eager_debug"])
def test_eager_backends_match_uncompiled(backend):
    x = tp.tensor([-1.0, 2.0])
    assert tp.compile(_relu_fn, backend=backend)(x).tolist() == [0.0, 6.0]


def test_eager_noexcept_wraps_graph_errors():
    from tensorplay._stax.debugging import eager_noexcept

    class Boom:
        def __call__(self, *args, **kwargs):
            raise ValueError("boom")

    inner = eager_noexcept(Boom(), [])
    with pytest.raises(RuntimeError, match="Unexpected exception") as info:
        inner()
    assert isinstance(info.value.__cause__, ValueError)


def test_eager_debug_reports_failing_node():
    def fn(x, y):
        return tp.matmul(x, y) + 1

    compiled = tp.compile(fn, backend="eager_debug")
    assert tuple(compiled(tp.randn(2, 3), tp.randn(3, 2)).shape) == (2, 2)
    with pytest.raises(Exception, match="While executing"):
        compiled(tp.randn(2, 3), tp.randn(4, 2))


def test_relu_compile_error_backend():
    from tensorplay._stax.debugging import ReluCompileError

    with pytest.raises(ReluCompileError):
        tp.compile(_relu_fn, backend="relu_compile_error_TESTING_ONLY")(tp.randn(2))


def test_relu_runtime_error_backend():
    with pytest.raises(AssertionError, match="ReluRuntimeError"):
        tp.compile(_relu_fn, backend="relu_runtime_error_TESTING_ONLY")(tp.randn(2))


def test_relu_accuracy_error_backend():
    x = tp.tensor([-1.0, 2.0])
    compiled = tp.compile(_relu_fn, backend="relu_accuracy_error_TESTING_ONLY")
    # relu(x) is replaced by x + 1.
    assert compiled(x).tolist() == [0.0, 9.0]


def test_non_leaf_compile_error_backend():
    from tensorplay._stax.debugging import TestingOnlyCompileError

    def fn(x):
        return tp.sin(x)

    leaf = tp.tensor([1.0, 2.0], requires_grad=True)
    compiled = tp.compile(fn, backend="non_leaf_compile_error_TESTING_ONLY")
    compiled(leaf)
    fresh = tp.compile(fn, backend="non_leaf_compile_error_TESTING_ONLY")
    with pytest.raises(TestingOnlyCompileError):
        fresh(leaf * 2)


def test_tvm_backend_accepts_compile_options(monkeypatch):
    from tensorplay._stax import tvm as tvm_backend

    seen = {}

    def fake_lower(graph_module, example_inputs, *, target, parallel):
        seen.update(target=target, parallel=parallel)
        return None  # unsupported region -> interpreter fallback

    monkeypatch.setattr(tvm_backend, "_require_tvm", lambda: (object(), object()))
    monkeypatch.setattr(tvm_backend, "_lower_pointwise", fake_lower)

    compiled = tp.compile(
        lambda x: x + 1,
        backend="tvm",
        options={"target": "llvm -mcpu=generic", "parallel": True},
    )
    assert compiled(tp.ones(2)).tolist() == [2.0, 2.0]
    assert seen == {"target": "llvm -mcpu=generic", "parallel": True}

    with pytest.raises(RuntimeError, match="bogus"):
        tp.compile(lambda x: x + 1, backend="tvm", options={"bogus": 1})(tp.ones(2))


# --------------------------------------------------------------------------
# cudagraphs backend
# --------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


def test_cudagraphs_backend_is_listed():
    assert "cudagraphs" in tp.compiler.list_backends()
    assert tp.compiler.lookup_backend("cudagraphs").compiler_name == "cudagraphs"


def test_cudagraphs_skips_cpu_regions(caplog):
    compiled = tp.compile(lambda a: a * 2, backend="cudagraphs")
    with caplog.at_level("WARNING", logger="tensorplay._stax.cudagraphs"):
        assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert "skipping cudagraphs due to cpu device" in caplog.text


def test_cudagraphs_strict_native_rejects_skipped_regions():
    from tensorplay._stax import CudaGraphError

    compiled = tp.compile(lambda a: a * 2, backend="cudagraphs", strict_native=True)
    with pytest.raises(CudaGraphError, match="cpu device"):
        compiled(tp.tensor([1.0]))


@requires_cuda
def test_cudagraphs_replays_new_data_and_keeps_outputs():
    lin = tp.nn.Linear(16, 16).cuda()

    def fn(x, scale: float = 2.0):
        y = lin(x).relu() * scale
        return y.sum(dim=-1), y.shape[0], {"y": y}

    compiled = tp.compile(fn, backend="cudagraphs")
    with tp.no_grad():
        x1 = tp.randn(4, 16, device="cuda")
        first = compiled(x1)
        x2 = tp.randn(4, 16, device="cuda")
        second = compiled(x2)
        for got, x in ((first, x1), (second, x2)):
            expected = fn(x)
            assert (got[0] - expected[0]).abs().max().item() < 1e-5
            assert got[1] == 4
            assert (got[2]["y"] - expected[2]["y"]).abs().max().item() < 1e-5
        # Outputs are copies: a later replay must not overwrite them.
        assert (first[0] - fn(x1)[0]).abs().max().item() < 1e-5
        # A new input layout is captured separately.
        assert tuple(compiled(tp.randn(3, 16, device="cuda"))[0].shape) == (3,)


@requires_cuda
def test_cudagraphs_training_region_keeps_autograd():
    lin = tp.nn.Linear(8, 8).cuda()
    compiled = tp.compile(lambda x: lin(x).relu().sum(), backend="cudagraphs")
    x = tp.randn(2, 8, device="cuda", requires_grad=True)
    compiled(x).backward()
    assert x.grad is not None and lin.weight.grad is not None


@requires_cuda
@pytest.mark.parametrize(
    ("fn", "reason"),
    [
        (lambda a: a.add_(1), "mutated inputs (a)"),
        (lambda a: a.cpu() + 1, "cpu device"),
        (lambda a: a[a > 0], "incompatible op"),
        (lambda a: a.nonzero(), "incompatible op"),
    ],
)
def test_cudagraphs_skip_reasons(fn, reason, caplog):
    compiled = tp.compile(fn, backend="cudagraphs")
    with caplog.at_level("WARNING", logger="tensorplay._stax.cudagraphs"):
        compiled(tp.randn(4, device="cuda"))
    assert f"skipping cudagraphs due to {reason}" in caplog.text


@requires_cuda
def test_reset_releases_captured_cuda_graphs():
    from tensorplay._stax.cudagraphs import CudagraphsBackend

    compiled = tp.compile(lambda a: a * 3, backend="cudagraphs")
    with tp.no_grad():
        compiled(tp.randn(4, device="cuda"))
    assert any(manager._entries for manager in CudagraphsBackend._managers)
    tp.compiler.reset()
    assert not any(manager._entries for manager in CudagraphsBackend._managers)


# --------------------------------------------------------------------------
# backend capabilities and contract handshake
# --------------------------------------------------------------------------


def test_builtin_capabilities_are_declared():
    from tensorplay._stax.registry import BackendCapabilities

    caps = tp.compiler.get_backend_capabilities("onnxrt")
    assert caps.inference_only is True
    assert caps.handles_training is False
    assert "onnxruntime" in caps.optional_deps

    stax_caps = tp.compiler.get_backend_capabilities("stax")
    assert stax_caps.handles_training is True

    tvm_caps = tp.compiler.get_backend_capabilities("tvm")
    assert tvm_caps.inference_only is True

    assert isinstance(
        tp.compiler.get_backend_capabilities("cudagraphs"), BackendCapabilities
    )


def test_capabilities_via_decorator():
    from tensorplay._stax.registry import (
        BackendCapabilities,
        declares_capabilities,
        register_backend,
        unregister_backend,
    )

    @declares_capabilities(
        BackendCapabilities(inference_only=True, handles_training=False)
    )
    def caps_backend(graph_module, example_inputs, **kwargs):
        return graph_module.forward

    register_backend(caps_backend, name="caps_probe")
    try:
        assert tp.compiler.get_backend_capabilities("caps_probe").inference_only
    finally:
        unregister_backend("caps_probe")


def test_missing_optional_dependency_gives_install_guidance(monkeypatch):
    from tensorplay._stax import registry

    monkeypatch.setattr(registry, "missing_optional_deps", lambda caps: ("onnxruntime",))
    with pytest.raises(RuntimeError, match="pip install onnxruntime"):
        tp.compiler.lookup_backend("onnxrt")
    # The backend is hidden from the default listing while unavailable.
    assert "onnxrt" not in tp.compiler.list_backends()
    assert "onnxrt" in tp.compiler.list_backends(include_unavailable=True)


def test_contract_version_handshake(monkeypatch):
    from tensorplay._stax import registry
    from tensorplay._stax.registry import BackendCapabilities

    def future_backend(graph_module, example_inputs, **kwargs):
        return graph_module.forward

    registry.register_backend(
        future_backend,
        name="contract_probe",
        capabilities=BackendCapabilities(
            contract_version=registry.CORE_BACKEND_CONTRACT_VERSION + 1
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="upgrade TensorPlay"):
            tp.compiler.lookup_backend("contract_probe")
    finally:
        registry.unregister_backend("contract_probe")

    def stale_backend(graph_module, example_inputs, **kwargs):
        return graph_module.forward

    registry.register_backend(
        stale_backend,
        name="core_range_probe",
        capabilities=BackendCapabilities(max_core_version="0.0.1"),
    )
    try:
        with pytest.raises(RuntimeError, match="supports TensorPlay <="):
            tp.compiler.lookup_backend("core_range_probe")
    finally:
        registry.unregister_backend("core_range_probe")


def test_inference_only_backend_gets_aot_wrapped_for_training():
    from tensorplay._stax.common import AotAutograd
    from tensorplay._stax.registry import BackendCapabilities, declares_capabilities

    calls = []

    @declares_capabilities(
        BackendCapabilities(inference_only=True, handles_training=False)
    )
    def inference_backend(graph_module, example_inputs, **kwargs):
        calls.append(kwargs)
        return graph_module.forward

    def fn(x):
        return tp.exp(x) * x

    x = tp.randn(4, requires_grad=True)
    compiled = tp.compile(fn, backend=inference_backend)
    # The decision happens at compile time; running the artifact exercises the
    # parallel-line dispatcher machinery, so only assert the adaptation here.
    adapted = compiled._tensorplay_original is fn
    assert adapted
    from tensorplay._stax import api

    assert hasattr(api, "_adapt_backend_to_region")
    wrapper = api._adapt_backend_to_region(
        inference_backend, (x,), {}
    )
    assert isinstance(wrapper, AotAutograd)
    # Inference calls pass the backend through untouched.
    assert api._adapt_backend_to_region(inference_backend, (tp.randn(4),), {}) is inference_backend


def test_training_rejects_backend_without_capabilities():
    def rigid_backend(graph_module, example_inputs, **kwargs):
        return graph_module.forward

    from tensorplay._stax.registry import BackendCapabilities, declares_capabilities

    rigid_backend = declares_capabilities(
        BackendCapabilities(inference_only=False, handles_training=False)
    )(rigid_backend)

    x = tp.randn(4, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not support training regions"):
        tp.compile(lambda v: tp.exp(v), backend=rigid_backend)(x)


def test_onnxrt_backend_matches_eager():
    onnxruntime = pytest.importorskip("onnxruntime")

    def fn(x):
        return tp.tanh(x) * 2 + 1

    compiled = tp.compile(fn, backend="onnxrt")
    x = tp.randn(4, 8)
    out = compiled(x)
    ref = fn(x)
    assert float((out - ref).abs().max()) < 1e-5
