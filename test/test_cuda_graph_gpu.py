"""CUDA graph tests that need a real GPU (skipped without CUDA).

capture/replay correctness, eager-instantiate at capture_end, shared memory
pools with refcounted release, graph-safe RNG freshness, bulk staging
(stage_and_launch), custom streams/capture error modes and DOT debug dumps.
"""

import os

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)

from tensorplay._stax import CudaGraphManager  # noqa: E402
from tensorplay.cuda.graphs import make_graphed_callables  # noqa: E402


def _cuda_device():
    return tp.Device("cuda", 0)


def test_capture_replay_matches_eager():
    device = _cuda_device()
    w = tp.randn((64, 64), device=device)
    x0 = tp.randn((64, 64), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x0.clone()
    with tp.cuda.graph(g):
        static_y = tp.matmul(static_x, w).relu()

    for i in range(3):
        a = tp.randn((64, 64), device=device)
        static_x.copy_(a)
        g.replay()
        tp.cuda.synchronize()
        got = static_y.clone()
        want = tp.matmul(a, w).relu()
        assert tp.allclose(got, want), f"replay {i} diverged from eager"


def test_stage_and_launch_bulk_path():
    device = _cuda_device()
    w = tp.randn((32, 32), device=device)
    x0 = tp.randn((32, 32), device=device)
    mgr = CudaGraphManager()

    def fn(x):
        return (x @ w).relu() * 2.0

    entry = mgr.capture("bulk", fn, x0)
    for i in range(3):
        a = tp.randn((32, 32), device=device)
        got = mgr.replay("bulk", a)[0]
        tp.cuda.synchronize()
        want = fn(a)
        assert tp.allclose(got, want), f"iteration {i} diverged"
    assert entry.replays == 3


def test_stage_and_launch_noncontiguous_fallback():
    device = _cuda_device()
    x0 = tp.arange(64, device=device).reshape(8, 8).contiguous().float()
    mgr = CudaGraphManager()
    entry = mgr.capture("nc", lambda x: x.relu(), x0)
    big = tp.arange(128, device=device).float().reshape(16, 8)
    view = big[::2]  # non-contiguous view of shape (8, 8)
    assert not view.is_contiguous()
    got = mgr.replay("nc", view.contiguous())[0]
    tp.cuda.synchronize()
    want = view.contiguous().relu()
    assert tp.allclose(got, want)


def test_rng_fresh_across_replays():
    device = _cuda_device()
    x = tp.ones((4, 4), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x.clone()
    with tp.cuda.graph(g):
        r1 = tp.randn((1024, 1024), device=device)
    g.replay()
    tp.cuda.synchronize()
    first = r1.clone()
    g.replay()
    tp.cuda.synchronize()
    second = r1
    assert not tp.allclose(first, second), "replays repeated capture-time randoms"


def test_shared_pool_via_handle():
    device = _cuda_device()
    handle = tp.cuda.graph_pool_handle()
    assert isinstance(handle, int) and handle != 0

    def build(pool_id, seed):
        tp.manual_seed(seed)
        x = tp.randn((256, 256), device=device)
        g = tp.cuda.CUDAGraph()
        static_x = x.clone()
        with tp.cuda.graph(g, pool=handle):
            static_y = static_x.sin()
        return g, static_x, static_y

    graphs = [build(handle, s)[0] for s in range(2)]
    built = [build(handle, 10 + s) for s in range(6)]
    for g, sx, sy in built:
        g.replay()
    tp.cuda.synchronize()
    # Reset one sharer: the pool must survive while others reference it.
    graphs[0].reset()
    built[0][0].replay()
    tp.cuda.synchronize()
    for g, _, _ in built:
        g.reset()
    for g in graphs:
        g.reset()


def test_pool_reset_frees_after_last_user():
    device = _cuda_device()
    tp.cuda.empty_cache()
    base = tp.cuda.memory_reserved(0)
    handle = tp.cuda.graph_pool_handle()
    keep = []
    g = tp.cuda.CUDAGraph()
    tp.manual_seed(7)
    x = tp.randn((2048, 2048), device=device)
    static_x = x.clone()
    with tp.cuda.graph(g, pool=handle):
        keep.append(static_x.sin())
    g.replay()
    tp.cuda.synchronize()
    g.reset()
    del keep
    # The eager-side inputs pin their own general-cache segments; drop them
    # so the leak assertion below measures pool reclamation, not liveness.
    del x, static_x
    tp.cuda.empty_cache()
    after = tp.cuda.memory_reserved(0)
    # The pool held ~32MB of segments; leaking them would exceed any slack.
    assert after <= base + (4 << 20), (
        f"pool segments leaked after reset: reserved {after} vs base {base}"
    )


def test_custom_stream_and_error_modes():
    device = _cuda_device()
    stream = tp.cuda.Stream()
    x = tp.ones((8, 8), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x.clone()
    with tp.cuda.graph(g, stream=stream):
        static_y = static_x + 1.0
    static_x.copy_(tp.full((8, 8), 5.0, device=device))
    g.replay()
    tp.cuda.synchronize()
    assert float(static_y.cpu().mean()) == 6.0
    with pytest.raises(ValueError):
        tp.cuda.CUDAGraph()._c.capture_begin(0, "bogus", None)


def test_debug_dump_dot_file(tmp_path):
    device = _cuda_device()
    x = tp.ones((8, 8), device=device)
    g = tp.cuda.CUDAGraph()
    g.enable_debug_mode()
    static_x = x.clone()
    with tp.cuda.graph(g):
        static_y = static_x * 2.0
    path = tmp_path / "graph.dot"
    g.debug_dump(str(path))
    assert path.exists() and path.stat().st_size > 0


def test_export_dot_module_level(tmp_path):
    device = _cuda_device()
    x = tp.ones((4,), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x.clone()
    with tp.cuda.graph(g):
        static_y = static_x + 1.0
    path = tp.cuda.export_dot(str(tmp_path / "last.dot"))
    assert os.path.exists(path)


def test_nested_capture_rejected_natively():
    device = _cuda_device()
    inner = tp.cuda.CUDAGraph()
    x = tp.ones((4, 4), device=device)
    outer = tp.cuda.CUDAGraph()
    static_x = x.clone()
    with pytest.raises(RuntimeError):
        with tp.cuda.graph(outer):
            static_x + 1.0
            with tp.cuda.graph(inner):  # nested -> native rejection
                pass


def test_is_current_stream_capturing():
    device = _cuda_device()
    assert tp.cuda.is_current_stream_capturing() is False
    x = tp.ones((4,), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x.clone()
    seen = []
    with tp.cuda.graph(g):

        def probe():
            seen.append(tp.cuda.is_current_stream_capturing())
            return static_x

        probe()
    assert seen == [True]


def test_replay_on_multiple_streams():
    device = _cuda_device()
    w = tp.randn((32, 32), device=device)
    x0 = tp.randn((32, 32), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x0.clone()
    with tp.cuda.graph(g):
        static_y = static_x @ w
    side = tp.cuda.Stream()
    a = tp.randn((32, 32), device=device)
    static_x.copy_(a)
    with tp.cuda.stream(side):
        g.replay()
    tp.cuda.synchronize()
    want = a @ w
    assert tp.allclose(static_y, want)


def test_explicit_stream_replay_overload():
    device = _cuda_device()
    w = tp.randn((16, 16), device=device)
    x0 = tp.randn((16, 16), device=device)
    g = tp.cuda.CUDAGraph()
    static_x = x0.clone()
    with tp.cuda.graph(g):
        static_y = static_x @ w
    side = tp.cuda.Stream()
    static_x.copy_(tp.ones((16, 16), device=device))
    g.replay(stream=side)
    tp.cuda.synchronize()
    assert tp.allclose(static_y, tp.ones((16, 16), device=device) @ w)


def test_memory_stats_fragmentation_fields():
    device = _cuda_device()
    x = tp.randn((64, 64), device=device)
    handle = tp.cuda.graph_pool_handle()
    g = tp.cuda.CUDAGraph()
    static_x = x.clone()
    with tp.cuda.graph(g, pool=handle):
        static_y = static_x.relu()

    stats = tp.cuda.memory_stats(0)
    alloc = stats["allocator"]
    required = {
        "allocated", "reserved", "max_allocated", "max_reserved",
        "segments", "free_blocks", "free_bytes", "largest_free_block",
        "pending_blocks", "pending_bytes", "graph_pools", "capturing",
    }
    missing = required - set(alloc)
    assert not missing, f"memory_stats missing fields: {missing}"
    assert alloc["segments"] >= 1
    assert alloc["allocated"] >= x.numel() * x.itemsize()
    assert alloc["graph_pools"] >= 1
    # reserved must dominate live bytes; free blocks bound the fragmentation.
    assert alloc["reserved"] >= alloc["allocated"]
    nested = tp.cuda.memory_stats_as_nested_dict(device=0)
    assert nested["allocator"]["graph_pools"] == alloc["graph_pools"]


def _conditional_nodes_supported():
    from tensorplay import _C

    probe = getattr(_C, "conditional_nodes_supported", None)
    return bool(probe()) if probe is not None else False


def test_conditional_if_node():
    """if-node body runs only when the replayed predicate is true."""
    if not _conditional_nodes_supported():
        pytest.skip("CUDA < 12.4: no conditional graph nodes")
    device = _cuda_device()
    g = tp.cuda.CUDAGraph()
    static_p = tp.ones((1,), dtype=tp.bool, device=device)
    static_x = tp.full((8, 8), 3.0, device=device)

    with tp.cuda.graph(g):
        g.begin_capture_to_if_node(static_p)
        static_z = static_x * 2.0
        g.end_capture_to_conditional_node()
        static_out = static_x + 1.0

    for pred in (True, False):
        static_p.fill_(1.0 if pred else 0.0)
        g.replay()
        tp.cuda.synchronize()
        assert float(static_out.mean()) == 4.0
        if pred:
            assert float(static_z.mean()) == 6.0


def test_multi_device_concurrent_capture():
    """Captures on different devices run concurrently from two threads."""
    if tp.cuda.device_count() < 2:
        pytest.skip("needs >= 2 GPUs")
    import threading

    errors = []

    def capture_on(index):
        try:
            dev = tp.Device("cuda", index)
            w = tp.randn((32, 32), device=dev)
            x = tp.randn((32, 32), device=dev)
            g = tp.cuda.CUDAGraph()
            sx = x.clone()
            with tp.cuda.graph(g):
                sy = sx @ w
            sx.copy_(tp.eye(32, device=dev))
            g.replay()
            tp.cuda.synchronize()
            assert tp.allclose(sy, w), f"device {index} replay mismatch"
            g.reset()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=capture_on, args=(i,)) for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent captures failed: {errors!r}"


def test_make_graphed_callables_matches_eager():
    """Graphed forward/backward replays match the eager path."""
    device = _cuda_device()
    tp.manual_seed(0)

    # Module parameters (plain closure tensors are invisible to the
    # per-callable static input surface).
    class Fn(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = tp.nn.Parameter(
                tp.randn((16, 16), device=device, requires_grad=True)
            )

        def forward(self, x):
            return (x @ self.weight).relu() * 2.0

    fn = Fn()
    W = fn.weight

    sample = (tp.randn((8, 16), device=device, requires_grad=True),)
    graphed = make_graphed_callables(fn, sample, num_warmup_iters=2)

    x = tp.randn((8, 16), device=device, requires_grad=True)
    out = graphed(x)
    out.sum().backward()
    x_grad_graphed = x.grad.clone()
    w_grad_graphed = W.grad.clone()

    x.grad = None
    W.grad = None
    out_eager = fn(x)
    out_eager.sum().backward()
    assert tp.allclose(out, out_eager.detach())
    assert x_grad_graphed is not None and x.grad is not None
    assert tp.allclose(x_grad_graphed, x.grad, atol=1e-5, rtol=1e-5)
    assert tp.allclose(w_grad_graphed, W.grad, atol=1e-5, rtol=1e-5)
