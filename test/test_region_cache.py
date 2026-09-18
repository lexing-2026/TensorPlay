"""Region capture cache and kernel cache concurrency."""

import pytest

import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.compiler.backends.stax.codecache import CodeCache
from tensorplay.compiler._core.region_cache import (
    load_region,
    region_key,
    store_region,
)
from tensorplay.graph import Graph, GraphModule

try:
    from tensorplay.compiler._core import registry as _registry
except ImportError:  # pragma: no cover - registry ships with the package
    _registry = None


def _mul(a, b):
    return a * b


def _mul_other(a, b):
    return a * b + 0.0


class _Holder(nn.Module):
    pass


def _build_gm():
    root = _Holder()
    root.scale = tp.tensor(2.0)
    g = Graph()
    x = g.placeholder("x")
    s = g.get_attr("scale")
    g.output(g.call_function(_mul, (x, s)))
    return GraphModule(root, g)


@pytest.fixture
def isolated_region_cache(tmp_path, monkeypatch):
    """Point the capture-region store at a fresh per-test cache root."""

    monkeypatch.setattr("tensorplay.compiler.backends.stax.codecache._default_caches", {})
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))


def test_region_key_requires_readable_source():
    assert region_key(len, None, ("sig",)) is None


def test_region_key_is_stable_and_sensitive():
    k1 = region_key(_mul, None, (("tensor", "f32"),))
    k2 = region_key(_mul, None, (("tensor", "f32"),))
    assert k1 == k2
    assert k1 != region_key(_mul, None, (("tensor", "f64"),))
    assert k1 != region_key(_mul_other, None, (("tensor", "f32"),))


def test_region_key_tracks_module_state():
    module = nn.Linear(4, 3)
    k1 = region_key(module.forward, module, ())
    with tp.no_grad():
        module.weight.copy_(tp.randn(3, 4))
    k2 = region_key(module.forward, module, ())
    assert k1 != k2
    # Unchanged state keeps the key stable.
    assert k2 == region_key(module.forward, module, ())


def test_region_key_tracks_module_state_without_direct_host_view():
    # dtypes without a direct host view go through the numeric bridge;
    # the key must still exist and still track value changes.
    module = nn.Linear(4, 3)
    module.weight.data = module.weight.data.to(tp.bfloat16)
    k1 = region_key(module.forward, module, ())
    assert k1 is not None
    with tp.no_grad():
        module.weight.copy_(tp.randn(3, 4).to(tp.bfloat16))
    assert k1 != region_key(module.forward, module, ())


def test_store_load_roundtrip_preserves_graph_and_meta(isolated_region_cache):
    gm = _build_gm()
    gm.meta["metadata_touches"] = (("x", "shape"),)
    key = region_key(_mul, None, ("spec",))
    store_region(key, gm)
    loaded = load_region(key)
    assert loaded is not None
    out = loaded(tp.tensor([1.0, 2.0, 3.0]))
    assert out.tolist() == [2.0, 4.0, 6.0]
    assert loaded.meta["metadata_touches"] == (("x", "shape"),)
    assert loaded.scale.tolist() == gm.scale.tolist()


def test_store_region_is_silent_on_unpicklable_state(isolated_region_cache):
    gm = _build_gm()
    gm.meta["unpicklable"] = lambda: None
    key = region_key(_mul, None, ("spec",))
    store_region(key, gm)
    assert load_region(key) is None


def test_region_cache_keys_missing_entries_return_none():
    assert load_region(None) is None
    assert load_region("") is None
    store_region(None, _build_gm())  # must not raise


def test_compile_or_load_compiles_once_per_key(tmp_path):
    cache = CodeCache("unit", root=str(tmp_path))
    calls = []

    def compile_fn(source):
        calls.append(source)
        return b"artifact:" + source.encode()

    first, path = cache.compile_or_load(compile_fn, "src-a")
    assert calls == ["src-a"]
    again, same_path = cache.compile_or_load(compile_fn, "src-a")
    assert calls == ["src-a"]
    assert first == again == b"artifact:src-a"
    assert path == same_path

    # A fresh instance over the same root reads the artifact from disk.
    disk = CodeCache("unit", root=str(tmp_path))
    loaded, _ = disk.compile_or_load(compile_fn, "src-a")
    assert calls == ["src-a"]
    assert loaded == b"artifact:src-a"


def test_compile_or_load_disabled_cache_compiles_every_time(tmp_path, monkeypatch):
    from tensorplay.compiler import config

    monkeypatch.setattr(config, "force_disable_caches", True)
    cache = CodeCache("unit", root=str(tmp_path))
    calls = []
    artifact, _ = cache.compile_or_load(lambda src: calls.append(src) or b"a", "s")
    assert artifact == b"a" and calls == ["s"]
    cache.compile_or_load(lambda src: calls.append(src) or b"a", "s")
    assert calls == ["s", "s"]


_probe_runs: list[int] = []


def _region_cache_probe(x):
    # The eager run happens only during capture; a region loaded from the
    # persistent store replays the graph without entering this function.
    _probe_runs.append(1)
    return x * 2.0 + 1.0


def test_persistent_region_cache_skips_recapture(tmp_path, monkeypatch):
    backend_name = "region_cache_noop_TESTING_ONLY"

    def noop_backend(graph_module, example_inputs, **kwargs):
        return graph_module

    _registry.register_backend(noop_backend, name=backend_name)
    monkeypatch.setattr("tensorplay.compiler.backends.stax.codecache._default_caches", {})
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    try:
        x = tp.tensor([1.0, 2.0])
        assert tp.compile(_region_cache_probe, backend=backend_name)(x).tolist() == [
            3.0,
            5.0,
        ]
        assert len(_probe_runs) == 1
        # A fresh process would hold no memoized cache instance and no
        # captured region: the region must come from the disk store.
        monkeypatch.setattr("tensorplay.compiler.backends.stax.codecache._default_caches", {})
        assert tp.compile(_region_cache_probe, backend=backend_name)(x).tolist() == [
            3.0,
            5.0,
        ]
        assert len(_probe_runs) == 1
    finally:
        _registry.unregister_backend(backend_name)


_gate_probe_runs: list[int] = []


def _region_cache_gate_probe(x):
    # Branch on data: capture bakes one outcome and records a guard replay;
    # a stored region must carry that replay so later calls key on it.
    _gate_probe_runs.append(1)
    if x.sum() > 2:
        return x * 10
    return x


def test_persistent_region_cache_keeps_gate_replay(tmp_path, monkeypatch):
    backend_name = "region_cache_gate_TESTING_ONLY"

    def noop_backend(graph_module, example_inputs, **kwargs):
        return graph_module

    _registry.register_backend(noop_backend, name=backend_name)
    monkeypatch.setattr("tensorplay.compiler.backends.stax.codecache._default_caches", {})
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    try:
        assert tp.compile(_region_cache_gate_probe, backend=backend_name)(
            tp.tensor([1.0, 2.0])
        ).tolist() == [10.0, 20.0]
        assert len(_gate_probe_runs) == 1
        # Same shapes, opposite branch outcome: the loaded region's guard
        # replay must route this to a re-specialization, not the stale graph.
        monkeypatch.setattr("tensorplay.compiler.backends.stax.codecache._default_caches", {})
        compiled = tp.compile(_region_cache_gate_probe, backend=backend_name)
        assert compiled(tp.tensor([1.0, 2.0])).tolist() == [10.0, 20.0]
        assert compiled(tp.tensor([0.5, 0.5])).tolist() == [0.5, 0.5]
        assert len(_gate_probe_runs) == 2
    finally:
        _registry.unregister_backend(backend_name)
