"""mode="reduce-overhead" replay shell around a stax-compiled artifact.

Logic tests run against a stand-in native surface and stand-in CUDA tensors;
the numerics of a real capture are covered by the CUDA-gated cases at the
bottom.
"""

from types import SimpleNamespace

import pytest

import tensorplay as tp
from tensorplay._stax.cudagraphs import (
    CudaGraphError,
    CudagraphCompiledCallable,
    cudagraph_wrap,
    wrap_skip_reason,
)


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------


class FakeDevice:
    def is_cuda(self):
        return True

    def __repr__(self):
        return "cuda:0"


class FakeTensor:
    """Duck-typed CUDA tensor delegating storage to a real CPU tensor."""

    def __init__(self, data):
        self.data = data
        self.device = FakeDevice()

    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def clone(self):
        return FakeTensor(self.data.clone())

    def copy_(self, other):
        self.data.copy_(other.data)
        return self

    def data_ptr(self):
        return self.data.data_ptr()

    def stride(self):
        return tuple(self.data.stride())

    def tolist(self):
        return self.data.tolist()


def _fake_is_tensor(value):
    return isinstance(value, (tp.Tensor, FakeTensor))


@pytest.fixture
def fake_cuda_probe(monkeypatch):
    """Pretend inputs are CUDA; drop traced metas (they stay CPU here)."""

    import tensorplay._stax.cudagraphs as cg

    monkeypatch.setattr(cg, "_is_tensor", _fake_is_tensor)
    monkeypatch.setattr(cg, "get_device_node_mapping", lambda gm: {})


class FakeGraph:
    def __init__(self):
        self.in_capture = False
        self.captured = False
        self.replays = 0
        self.resets = 0
        self.recompute = None

    def capture_begin(self, pool=0, capture_error_mode="global", stream=None):
        assert not self.in_capture
        self.in_capture = True

    def capture_end(self):
        assert self.in_capture
        self.in_capture = False
        self.captured = True

    def replay(self):
        assert self.captured
        # a real recorded graph re-executes its kernels against the captured
        # input storage; the stand-in recomputes its output the same way
        if self.recompute is not None:
            self.recompute()
        self.replays += 1

    def reset(self):
        self.resets += 1


class FakeNative:
    CUDAGraph = FakeGraph


class FakeNativeNoCapture(FakeNative):
    class CUDAGraph(FakeGraph):
        def capture_begin(self, *args, **kwargs):
            raise RuntimeError("capture rejected by the driver")


def _twice_artifact(graph):
    """Callable standing in for a compiled kernel: out = in * 2.

    During the capture window it registers a recompute hook so the stand-in
    replay refreshes the static output from the captured input storage, the
    way a real recorded graph re-executes its kernels against that storage.
    """

    def artifact(x):
        out = FakeTensor(x.data * 2)
        if graph.in_capture:
            def recompute():
                out.data = x.data * 2

            graph.recompute = recompute
        return out

    return artifact


def _trace_gm(fn, *args):
    """Trace ``fn`` and stamp tensor metadata (the default capture pipeline)."""

    from tensorplay.graph import Tracer
    from tensorplay.graph.passes import ShapeProp

    gm = Tracer().trace(
        fn, sample_inputs={name: value for name, value in zip(("a", "b", "c"), args)}
    )
    ShapeProp(list(args))(gm)
    return gm


def _clean_gm():
    return _trace_gm(lambda a: (a * 2).relu(), tp.tensor([1.0]))


def _capture_stub(entry):
    """Give wrap_skip_reason a truthy compiled artifact without behavior."""
    return entry


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------


def test_skip_reason_dynamic():
    gm = _clean_gm()
    reason = wrap_skip_reason(_capture_stub(1), gm, [tp.tensor([1.0])], dynamic=True)
    assert reason == "skipping cudagraphs due to dynamic shape specialization"


def test_skip_reason_cpu_input():
    gm = _clean_gm()
    reason = wrap_skip_reason(_capture_stub(1), gm, [tp.tensor([1.0])])
    assert reason == "skipping cudagraphs due to input on a non-CUDA device"


def test_skip_reason_requires_grad(fake_cuda_probe):
    gm = _clean_gm()
    value = FakeTensor(tp.tensor([1.0], requires_grad=True))
    reason = wrap_skip_reason(_capture_stub(1), gm, [value])
    assert reason == "skipping cudagraphs due to input requires grad"


def test_skip_reason_non_tensor_input(fake_cuda_probe):
    gm = _clean_gm()
    reason = wrap_skip_reason(_capture_stub(1), gm, [FakeTensor(tp.tensor([1.0])), [1, 2]])
    assert "unsupported input type" in reason


def test_skip_reason_mutated_inputs(fake_cuda_probe):
    gm = _trace_gm(lambda a: a.add_(1.0), tp.tensor([1.0]))
    reason = wrap_skip_reason(
        _capture_stub(1), gm, [FakeTensor(tp.tensor([1.0]))]
    )
    assert reason == "skipping cudagraphs due to mutated inputs (a)"


def test_skip_reason_incompatible_op(fake_cuda_probe):
    gm = _trace_gm(lambda a: a.nonzero(), tp.tensor([1.0, 0.0]))
    reason = wrap_skip_reason(
        _capture_stub(1), gm, [FakeTensor(tp.tensor([1.0, 0.0]))]
    )
    assert reason == "skipping cudagraphs due to incompatible op (nonzero)"


def test_skip_reason_clean_region_is_none(fake_cuda_probe):
    gm = _clean_gm()
    assert (
        wrap_skip_reason(_capture_stub(1), gm, [FakeTensor(tp.tensor([1.0]))])
        is None
    )


# --------------------------------------------------------------------------
# wrapper behavior (stand-in native)
# --------------------------------------------------------------------------


def test_wrap_and_replay_reads_captured_storage(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    artifact = _twice_artifact(graph)
    wrapped, reason = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert reason is None and isinstance(wrapped, CudagraphCompiledCallable)

    held = FakeTensor(tp.tensor([3.0]))
    out1 = wrapped(held)
    assert out1.tolist() == [6.0]
    held.data.copy_(tp.tensor([-4.0]))  # fresh data, same storage
    out2 = wrapped(held)
    # the replay re-reads the captured storage live: same graph, new values
    assert out2.tolist() == [-8.0]
    # steady state is a replay, not an artifact call: one launch per call
    assert graph.replays == 2  # first call replays once after capture
    assert graph.captured


def test_drifted_input_runs_artifact_directly(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    calls = []

    def artifact(x):
        calls.append(x.data.tolist())
        return FakeTensor(x.data * 2)

    wrapped, _ = cudagraph_wrap(artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native)
    held = FakeTensor(tp.tensor([3.0]))
    assert wrapped(held).tolist() == [6.0]
    assert graph.replays == 1  # capture + same-storage replay
    # a tensor at a different address cannot be seen by the graph: the call
    # runs the artifact directly instead
    other = FakeTensor(tp.tensor([-4.0]))
    assert wrapped(other).tolist() == [-8.0]
    assert graph.replays == 1
    # warmup and the capture window both ran on the first call's tensor
    assert calls == [[3.0], [3.0], [-4.0]]
    # drift is transient: the wrapper stays live and replays again for the
    # captured storage
    assert wrapped._state == "live"
    assert wrapped(held).tolist() == [6.0]
    assert graph.replays == 2


def test_replayed_outputs_alias_static_buffers(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    wrapped, _ = cudagraph_wrap(
        _twice_artifact(graph), gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    held = FakeTensor(tp.tensor([1.0]))
    first = wrapped(held)
    held.data.copy_(tp.tensor([5.0]))
    again = wrapped(held)
    assert first is again
    # the previous output is overwritten by the next replay
    assert first.tolist() == [10.0]


def test_capture_failure_falls_back_permanently(fake_cuda_probe):
    gm = _clean_gm()
    native = FakeNativeNoCapture()
    calls = []

    def artifact(x):
        calls.append(x.data.tolist())
        return FakeTensor(x.data * 2)

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert wrapped(FakeTensor(tp.tensor([2.0]))).tolist() == [4.0]
    assert wrapped(FakeTensor(tp.tensor([3.0]))).tolist() == [6.0]
    # capture warmup + the fallback call after the failed capture + the
    # second direct call
    assert len(calls) == 3


def test_kwargs_delegate_to_artifact(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    calls = []

    def artifact(x, scale=2.0):
        calls.append(scale)
        out = FakeTensor(x.data * scale)
        if graph.in_capture:
            def recompute():
                out.data = x.data * 2.0

            graph.recompute = recompute
        return out

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert wrapped(FakeTensor(tp.tensor([1.0]))).tolist() == [2.0]
    assert wrapped(FakeTensor(tp.tensor([1.0])), scale=3.0).tolist() == [3.0]
    # capture warmup + capture-window run used the default scale; the
    # keyword call skipped replay and ran the artifact directly
    assert calls == [2.0, 2.0, 3.0]


def test_reset_releases_wrapped_graphs(fake_cuda_probe):
    from tensorplay._stax.cudagraphs import CudagraphsBackend

    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    wrapped, _ = cudagraph_wrap(
        _twice_artifact(graph), gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    wrapped(FakeTensor(tp.tensor([1.0])))
    manager = wrapped._manager
    assert manager in CudagraphsBackend._managers
    CudagraphsBackend.reset()
    assert not manager._entries
    assert graph.resets == 1


# --------------------------------------------------------------------------
# mode wiring through tp.compile
# --------------------------------------------------------------------------


def test_reduce_overhead_mode_compiles_and_runs_on_cpu():
    compiled = tp.compile(lambda a: (a * 2).relu().sum(), mode="reduce-overhead")
    x = tp.tensor([1.0, -2.0, 3.0])
    assert compiled(x).item() == 8.0


def test_max_autotune_no_cudagraphs_mode_accepted():
    compiled = tp.compile(lambda a: (a * 2).relu(), mode="max-autotune-no-cudagraphs")
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]


def test_stax_cudagraphs_option_accepted():
    compiled = tp.compile(
        lambda a: (a * 2).relu(), options={"stax.cudagraphs": True}
    )
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]


def test_stax_cudagraphs_option_validated():
    # the backend lowers lazily: the option contract bites on first call
    compiled = tp.compile(lambda a: a, options={"stax.cudagraphs": "yes"})
    with pytest.raises(RuntimeError, match="must be bool values"):
        compiled(tp.tensor([1.0]))


# --------------------------------------------------------------------------
# real CUDA capture (gated)
# --------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


@requires_cuda
def test_reduce_overhead_replays_compiled_region():
    compiled = tp.compile(
        lambda a: (a * 2).relu().sum(dim=-1), mode="reduce-overhead"
    )
    with tp.no_grad():
        x = tp.randn(4, 8, device="cuda")
        first = compiled(x)
        expected1 = (x * 2).relu().sum(dim=-1)
        assert (first - expected1).abs().max().item() < 1e-5
        x.copy_(tp.randn(4, 8, device="cuda"))  # fresh data, same storage
        second = compiled(x)
        expected2 = (x * 2).relu().sum(dim=-1)
        assert (second - expected2).abs().max().item() < 1e-5
        # outputs alias the graph pool: the second replay overwrote the first
        assert first is second
        # a tensor at a new address cannot be seen by the captured graph;
        # the call still computes correctly, running the artifact directly
        y = tp.randn(4, 8, device="cuda")
        third = compiled(y)
        assert (third - (y * 2).relu().sum(dim=-1)).abs().max().item() < 1e-5


@requires_cuda
def test_reduce_overhead_skips_mutating_regions(caplog):
    compiled = tp.compile(lambda a: a.add_(1.0), mode="reduce-overhead")
    import logging

    with caplog.at_level(logging.WARNING, logger="tensorplay._stax.cudagraphs"):
        x = tp.zeros(4, device="cuda")
        compiled(x)
    assert "mutated inputs" in caplog.text


@requires_cuda
def test_reduce_overhead_falls_back_when_capture_fails(caplog, monkeypatch):
    import logging

    import tensorplay._stax.cudagraphs as cg

    class ExplodingGraph(FakeGraph):
        def capture_begin(self, *args, **kwargs):
            raise CudaGraphError("capture rejected by the driver")

    monkeypatch.setattr(
        cg, "_default_native", lambda: type("N", (), {"CUDAGraph": ExplodingGraph})
    )
    compiled = tp.compile(lambda a: (a * 2).relu().sum(), mode="reduce-overhead")
    with caplog.at_level(logging.WARNING, logger="tensorplay._stax.cudagraphs"):
        x = tp.randn(4, device="cuda")
        assert compiled(x).item() == (x * 2).relu().sum().item()
    assert "capture failed" in caplog.text


# --------------------------------------------------------------------------
# mode-to-options mapping
# --------------------------------------------------------------------------

def test_list_mode_options_mapping():
    from tensorplay.compiler import list_mode_options

    assert list_mode_options("default") == {}
    assert list_mode_options("reduce-overhead") == {"stax.cudagraphs": True}
    assert list_mode_options("max-autotune-no-cudagraphs") == {
        "stax.max_autotune": True,
        "stax.coordinate_descent_tuning": True,
    }
    assert list_mode_options("max-autotune") == {
        "stax.max_autotune": True,
        "stax.cudagraphs": True,
        "stax.coordinate_descent_tuning": True,
    }
    full = list_mode_options()
    assert set(full) == {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    with pytest.raises(RuntimeError, match="Unrecognized mode"):
        list_mode_options("fastest")


def test_stax_mode_patch_overlaid_by_explicit_options(monkeypatch):
    from tensorplay.compiler.backends.stax import backend as stax_backend
    from tensorplay.compiler.backends import cudagraphs as cg

    captured: dict = {}

    def fake_lower(graph_module, example_inputs, **kwargs):
        captured.update(kwargs)
        return lambda *args: "lowered"

    monkeypatch.setattr(stax_backend, "_lower_stax_region", fake_lower)

    wrapped: list = []

    def fake_wrap(compiled, graph_module, example_inputs, dynamic=None):
        wrapped.append(compiled)
        return compiled, None

    monkeypatch.setattr(cg, "cudagraph_wrap", fake_wrap)
    graph_like = SimpleNamespace()

    # the mode's patch lands as defaults and drives the replay wrap...
    stax_backend.stax(graph_like, [], mode="max-autotune")
    assert captured["max_autotune"] is True
    assert captured["coordinate_descent_tuning"] is True
    assert len(wrapped) == 1

    # ...and an explicit option wins over the mode patch per key
    stax_backend.stax(
        graph_like, [], mode="max-autotune", options={"stax.cudagraphs": False}
    )
    assert captured["max_autotune"] is True
    assert len(wrapped) == 1  # no further wrap: explicit opt-out

    # a mode without the autotune knobs leaves them off
    stax_backend.stax(
        graph_like, [], mode="reduce-overhead", options={"stax.native": False}
    )
    assert captured["max_autotune"] is False
    assert captured["coordinate_descent_tuning"] is False
    assert captured["use_native"] is False
    assert len(wrapped) == 2


def test_max_autotune_mode_compiles_and_matches_eager():
    def fn(a):
        return (a * 3 + 1).relu().sum()

    compiled = tp.compile(fn, mode="max-autotune")
    x = tp.randn(16, 64)
    assert tp.allclose(compiled(x), fn(x))


def test_max_autotune_no_cudagraphs_mode_compiles_and_matches_eager():
    def fn(a):
        return (a.sin() * a).sum()

    compiled = tp.compile(fn, mode="max-autotune-no-cudagraphs")
    x = tp.randn(32, 16)
    assert tp.allclose(compiled(x), fn(x))
