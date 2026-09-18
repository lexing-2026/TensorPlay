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
