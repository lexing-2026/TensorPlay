"""L5-M3: CUDA graph orchestration logic (fake native surface, no CUDA needed)."""

import pytest

import tensorplay as tp
from tensorplay.compiler import CudaGraphError, CudaGraphManager


class FakeNative:
    def __init__(self):
        self.launches = 0

    def cuda_graph_begin_capture(self):
        assert not getattr(self, "_in_capture", False)
        self._in_capture = True

    def cuda_graph_end_capture(self):
        self._in_capture = False
        return object()  # opaque graph

    def cuda_graph_instantiate(self, graph):
        return ("exec", graph)

    def cuda_graph_launch(self, executable):
        self.launches += 1


def _manager():
    return CudaGraphManager(native=FakeNative())


def test_missing_native_names_symbols(monkeypatch):
    class NoC:  # module stand-in without bindings
        pass

    import tensorplay.compiler.cudagraphs as cg
    monkeypatch.setattr(cg, "_default_native", lambda: (_ for _ in ()).throw(
        NotImplementedError("CUDA graph bindings not implemented yet in "
                            "tensorplay._C: cuda_graph_begin_capture")))
    mgr = CudaGraphManager()
    with pytest.raises(NotImplementedError) as ei:
        mgr.capture("k", lambda a: a, tp.tensor([1.0]))
    assert "cuda_graph_begin_capture" in str(ei.value)


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
    assert mgr.native.launches == 1 and entry.replays == 1


def test_same_key_same_signature_returns_entry():
    mgr = _manager()
    e1 = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    e2 = mgr.capture("a", lambda x: x * 2, tp.tensor([1.0]))
    assert e1 is e2


def test_nested_capture_rejected():
    mgr = _manager()
    native = mgr.native

    def inner():
        mgr.capture("inner", lambda x: x, tp.tensor([1.0]))

    def outer(x):
        native.cuda_graph_begin_capture()
        try:
            inner()
        finally:
            native._in_capture = False
        return x

    mgr.capturing = "outer"
    try:
        with pytest.raises(CudaGraphError):
            mgr.capture("inner", lambda x: x, tp.tensor([1.0]))
    finally:
        mgr.capturing = None
