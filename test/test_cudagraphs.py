"""L5-M3: CUDA graph orchestration logic (fake native surface, no CUDA needed)."""

import pytest

import tensorplay as tp
from tensorplay._stax import CudaGraphError, CudaGraphManager


class FakeGraph:
    """Stand-in for tensorplay._C.CUDAGraph (new native surface)."""

    def __init__(self):
        self.in_capture = False
        self.captured = False
        self.replays = 0
        self.resets = 0
        self.pool = 0

    def capture_begin(self, pool=0, capture_error_mode="global", stream=None):
        assert not self.in_capture, "nested capture on one graph object"
        self.in_capture = True
        self.pool = pool

    def capture_end(self):
        assert self.in_capture, "capture_end without capture_begin"
        self.in_capture = False
        self.captured = True

    def replay(self):
        assert self.captured
        self.replays += 1

    def reset(self):
        self.resets += 1


class FakeNativeNoBulk:
    """Stand-in module exposing only the minimal pre-bulk surface."""

    CUDAGraph = FakeGraph


class FakeGraphBulk(FakeGraph):
    def stage_and_launch(self, static_inputs, inputs):
        assert self.captured
        for dst, src in zip(static_inputs, inputs):
            dst.copy_(src)
        self.replays += 1


class FakeNative(FakeNativeNoBulk):
    CUDAGraph = FakeGraphBulk


def _manager(native=FakeNative):
    return CudaGraphManager(native=native)


def test_missing_native_reports_surface(monkeypatch):
    import tensorplay._stax.cudagraphs as cg
    monkeypatch.setattr(cg, "_default_native", lambda: (_ for _ in ()).throw(
        NotImplementedError("CUDA graphs are not supported by this TensorPlay "
                            "build (tensorplay._C exposes no CUDAGraph class)")))
    mgr = CudaGraphManager()
    with pytest.raises(NotImplementedError) as ei:
        mgr.capture("k", lambda a: a, tp.tensor([1.0]))
    assert "CUDAGraph" in str(ei.value)


def test_capture_once_replay_copies_inputs():
    mgr = _manager()

    def fn(x, w):
        return (x * w).relu()

    x0, w0 = tp.tensor([1.0, -2.0]), tp.tensor([3.0, 4.0])
    entry = mgr.capture("mm", fn, x0, w0)
    out = mgr.replay("mm", tp.tensor([-1.0, 2.0]), tp.tensor([1.0, 1.0]))[0]
    # static buffers staged the replay inputs before launch
    assert entry.static_inputs[0].tolist() == [-1.0, 2.0]
    assert out is entry.static_outputs[0]
    with pytest.raises(CudaGraphError):
        mgr.replay("mm", tp.tensor([1.0]))  # arity mismatch
    with pytest.raises(CudaGraphError):
        mgr.replay("mm", tp.tensor([1.0]), tp.tensor([1.0, 2.0, 3.0]))  # shape drift
    assert entry.graph.replays == 1 and entry.replays == 1


def test_bulk_replay_routes_through_stage_and_launch():
    mgr = _manager()
    entry = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    assert entry.bulk is True
    out = mgr.replay("a", tp.tensor([5.0]))[0]
    assert entry.static_inputs[0].tolist() == [5.0]
    assert out is entry.static_outputs[0]
    assert entry.graph.replays == 1


def test_fallback_replay_without_bulk_path():
    mgr = _manager(native=FakeNativeNoBulk)
    entry = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    assert entry.bulk is False
    mgr.replay("a", tp.tensor([-3.0]))
    assert entry.static_inputs[0].tolist() == [-3.0]
    assert entry.graph.replays == 1


def test_same_key_same_signature_returns_entry():
    mgr = _manager()
    e1 = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    e2 = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    assert e1 is e2


def test_nested_capture_rejected():
    mgr = _manager()
    mgr.capturing = "outer"
    try:
        with pytest.raises(CudaGraphError):
            mgr.capture("inner", lambda x: x, tp.tensor([1.0]))
    finally:
        mgr.capturing = None


def test_clear_resets_graph_objects():
    mgr = _manager()
    entry = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    mgr.clear("a")
    assert entry.graph.resets == 1
    assert "a" not in mgr._entries
    with pytest.raises(CudaGraphError):
        mgr.replay("a", tp.tensor([1.0]))


def test_clear_all_resets_every_entry():
    mgr = _manager()
    e1 = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    e2 = mgr.capture("b", lambda x: x + 1, tp.tensor([1.0]))
    mgr.clear()
    assert e1.graph.resets == 1 and e2.graph.resets == 1
