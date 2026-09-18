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
