"""Captured-graph support for functional collectives.

A collective is a cross-process synchronization point, so a compiled region
must run it exactly once per call on every rank.  The capture guard keeps
the value symbolic while the region is traced (the trace pass performs no
second copy of a collective the recorded node will perform on replay), the
synchronous surface gives the graph a replayable node, and the extern
segment treats it like any other eager operator — inference composes the
following pointwise chain into the segment, training closes it with the
collective's closed-form tangent rule.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import tensorplay as tp

try:
    import tensorplay.distributed as dist
    from tensorplay.distributed import _functional_collectives as fc
    HAS_DIST = True
except Exception:  # noqa: BLE001 - distributed imports may fail without builds
    HAS_DIST = False


def _init_single_rank(backend):
    fd, store = tempfile.mkstemp(prefix=f"tp_stax_{backend}_")
    os.close(fd)
    os.unlink(store)
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{store}",
        rank=0,
        world_size=1,
    )


_INITIALIZED = False


def _ensure_single_rank(backend="gloo"):
    global _INITIALIZED
    if not _INITIALIZED:
        _init_single_rank(backend)
        _INITIALIZED = True


@unittest.skipUnless(HAS_DIST, "tensorplay.distributed unavailable")
class TestSyncCollectiveSurface(unittest.TestCase):
    """The synchronous eager surface the captured graph records."""

    @classmethod
    def setUpClass(cls):
        _ensure_single_rank()

    def test_all_reduce_sync_returns_reduced_copy(self):
        x = tp.randn(4, 8)
        out = fc.all_reduce_sync(x)
        self.assertTrue(tp.allclose(out, x))
        # the input stays untouched
        self.assertTrue(tp.allclose(x, out))

    def test_all_gather_sync_concatenates_leading_axis(self):
        x = tp.randn(4, 8)
        out = fc.all_gather_sync(x)
        self.assertEqual(tuple(out.shape), (4, 8))
        self.assertTrue(tp.allclose(out, x))

    def test_reduce_scatter_sync_splits_leading_axis(self):
        x = tp.randn(4, 8)
        out = fc.reduce_scatter_sync(x)
        self.assertEqual(tuple(out.shape), (4, 8))
        self.assertTrue(tp.allclose(out, x))


@unittest.skipUnless(HAS_DIST, "tensorplay.distributed unavailable")
class TestCapturedCollectives(unittest.TestCase):
    """Compilation captures the collective as one graph node."""

    @classmethod
    def setUpClass(cls):
        _ensure_single_rank()

    def _check(self, fn, x):
        import tempfile as _tf

        os.environ.setdefault("TP_CACHE_DIR", _tf.mkdtemp())
        from tensorplay.compiler.backends.stax.codegen import triton as tri

        self.addCleanup(
            lambda: (
                setattr(tri, "_default_caches", {}),
                os.environ.pop("TP_CACHE_DIR", None),
            )
        )
        import tensorplay.compiler.backends.stax.codecache as cc

        cc._default_caches = {}
        compiled = tp.compile(fn)
        reference = fn(x)
        out = compiled(x)
        self.assertTrue(tp.allclose(reference, out))

    def test_all_reduce_region_compiles_and_matches(self):
        self._check(
            lambda t: (fc.all_reduce(t) * 2.0).relu(), tp.randn(4, 8)
        )

    def test_all_gather_region_compiles_and_matches(self):
        self._check(
            lambda t: (fc.all_gather_single(t) * 1.5).sigmoid(), tp.randn(4, 8)
        )

    def test_reduce_scatter_region_compiles_and_matches(self):
        self._check(
            lambda t: (fc.reduce_scatter_single(t) * 0.5).tanh(), tp.randn(4, 8)
        )

    def test_collective_between_two_fused_runs(self):
        def fn(t):
            a = (t * 2.0).relu()
            r = fc.all_reduce(a)
            return (r * 3.0).sigmoid()

        self._check(fn, tp.randn(4, 8))

    def test_capture_rejects_non_sum_all_reduce(self):
        from tensorplay.graph import GraphCaptureError

        def fn(t):
            return fc.all_reduce(t, "max") * 2.0

        with self.assertRaises((ValueError, GraphCaptureError)):
            tp.compile(fn)(tp.randn(4, 8))

    def test_capture_rejects_non_leading_gather(self):
        from tensorplay.graph import GraphCaptureError

        def fn(t):
            return fc.all_gather_single(t, gather_dim=1) * 2.0

        with self.assertRaises((ValueError, GraphCaptureError)):
            tp.compile(fn)(tp.randn(4, 8))


# A rank worker exercising the captured regions end to end.  Each rank
# traces on its own real tensors; the guard keeps the collective symbolic so
# the trace performs no communication, and every replay launches it once.
# The shared-GPU case is not exercised: the collective library refuses two
# ranks on one device (duplicate bus id), so multi-rank runs use the CPU
# transport and the device transport runs one rank per GPU.
_WORKER = r"""
import os
import sys
import tempfile

sys.path.insert(0, "@ROOT@")
os.environ.setdefault("TP_CACHE_DIR", tempfile.mkdtemp())

import ctypes
@PRELOAD@

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed import _functional_collectives as fc

rank, world, backend, store = (
    int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
)
tp.cuda.set_device(rank % max(1, tp.cuda.device_count()))
device = "cuda:0" if backend == "nccl" else "cpu"
dist.init_process_group(backend=backend, init_method=f"file://{store}",
                        rank=rank, world_size=world)

base = tp.arange(world, dtype=tp.float32, device=device).reshape(world, 1)
x = base + rank  # distinct per rank

def region_ar(t):
    r = fc.all_reduce(t, "sum")
    return (r * 2.0).relu()

def region_ag(t):
    g = fc.all_gather_single(t)
    return (g * 1.0).sum()

def region_rs(t):
    s = fc.reduce_scatter_single(t)
    return (s * 1.0).sum()

compiled_ar = tp.compile(region_ar)
compiled_ag = tp.compile(region_ag)
compiled_rs = tp.compile(region_rs)

out_ar = compiled_ar(x)
# rank r holds base + r elementwise; the sum all-reduce yields
# world * base + sum(ranks) on every rank, then the region scales by two
expect = (base * world + float(sum(range(world)))) * 2.0
assert tp.allclose(out_ar, expect), (out_ar, expect)

out_ag = compiled_ag(x)
expect_ag = float(sum(r + c for r in range(world) for c in range(world)))
assert abs(float(out_ag) - expect_ag) < 1e-3, (float(out_ag), expect_ag)

out_rs = compiled_rs(x)
# leading axis has world rows; reduce-scatter hands rank r the summed row r
expect_rs = float(sum(rank + c for c in range(world)))
assert abs(float(out_rs) - expect_rs) < 1e-3, (float(out_rs), expect_rs)

# training: the tangent of a sum all-reduce is another sum all-reduce; the
# extern segment closes the collective with its closed-form rule
xt = x.clone().requires_grad_(True)

def train_region(t):
    r = fc.all_reduce(t, "sum")
    return (r * 2.0).sum()

ref = train_region(xt)
ref_grad = tp.autograd.grad(ref, xt)[0]
xc = x.clone().requires_grad_(True)
ct = tp.compile(train_region)
loss = ct(xc)
loss.backward()
assert tp.allclose(xc.grad, ref_grad, rtol=1e-5, atol=1e-4), (
    xc.grad, ref_grad)

print(f"RANK{rank}OK")
"""


@unittest.skipUnless(HAS_DIST, "tensorplay.distributed unavailable")
class TestMultiRankCapturedCollectives(unittest.TestCase):
    def _spawn(self, world, backend, preload=""):
        fd, store = tempfile.mkstemp(prefix="tp_stax_dist_")
        os.close(fd)
        os.unlink(store)
        script = _WORKER.replace(
            "@ROOT@",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).replace(
            "@PRELOAD@",
            preload,
        )
        try:
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(r), str(world), backend, store],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                for r in range(world)
            ]
            outs = []
            for p in procs:
                out, _ = p.communicate(timeout=300)
                outs.append(out.decode())
            for r, out in enumerate(outs):
                self.assertIn(f"RANK{r}OK", out, f"rank {r} failed:\n{out}")
        finally:
            if os.path.exists(store):
                os.unlink(store)

    def test_two_rank_gloo(self):
        self._spawn(2, "gloo")

    @unittest.skipUnless(tp.cuda.is_available(), "CUDA not available")
    def test_single_rank_nccl(self):
        # One rank per GPU: the device transport needs distinct bus ids, so
        # a single-GPU machine runs the device path with world size one.
        lib = (
            "/home/bluemoon/miniconda3/lib/python3.13/site-packages/"
            "nvidia/nccl/lib/libnccl.so.2"
        )
        preload = f"ctypes.CDLL({lib!r})" if os.path.exists(lib) else ""
        self._spawn(1, "nccl", preload=preload)


if __name__ == "__main__":
    unittest.main()
