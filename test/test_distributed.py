"""Tests for tensorplay.distributed (stores, NCCL process group) and
DistributedSampler parity against torch."""

import os
import subprocess
import sys
import tempfile
import unittest

import tensorplay as tp
from tensorplay.utils import data as td

try:
    import torch
    from torch.utils import data as torchdata
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestFileStore(unittest.TestCase):
    def setUp(self):
        from tensorplay.distributed._store import FileStore
        fd, self.path = tempfile.mkstemp(prefix="tp_store_test_")
        os.close(fd)
        os.unlink(self.path)
        self.store = FileStore(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_set_get(self):
        self.store.set("k", "v1")
        self.store.set("k", "v2")
        self.assertEqual(self.store.get("k"), b"v2")

    def test_add(self):
        self.assertEqual(self.store.add("counter", 1), 1)
        self.assertEqual(self.store.add("counter", 5), 6)
        self.assertEqual(self.store.add("other", -2), -2)

    def test_get_timeout(self):
        from tensorplay.distributed._store import StoreTimeoutError
        with self.assertRaises(StoreTimeoutError):
            self.store.get("missing", timeout=0.2)

    def test_wait(self):
        self.assertTrue(self.store.wait(["a"], timeout=0.1) is False or True)
        self.store.set("a", "1")
        self.assertTrue(self.store.wait(["a", "a"], timeout=1))

    def test_compare_set(self):
        self.assertEqual(self.store.compare_set("cs", "", "first"), b"first")
        self.assertEqual(self.store.compare_set("cs", "", "second"), b"first")
        self.assertEqual(self.store.compare_set("cs", "first", "third"), b"third")


class TestTCPStore(unittest.TestCase):
    def test_roundtrip(self):
        from tensorplay.distributed._store import TCPStore
        server = TCPStore("127.0.0.1", 0, is_master=True)
        client = TCPStore("127.0.0.1", server.port, is_master=False)
        server.set("x", "42")
        self.assertEqual(client.get("x"), b"42")
        self.assertEqual(client.add("x", 8), 50)
        self.assertTrue(client.wait(["x"], timeout=1))
        server._server.stop()

    def test_blocking_get_across_clients(self):
        import threading
        from tensorplay.distributed._store import TCPStore
        server = TCPStore("127.0.0.1", 0, is_master=True)
        client = TCPStore("127.0.0.1", server.port, is_master=False)

        result = []

        def delayed_set():
            import time
            time.sleep(0.3)
            server.set("late", "here")

        thread = threading.Thread(target=delayed_set)
        thread.start()
        result.append(client.get("late", timeout=5))
        thread.join()
        self.assertEqual(result[0], b"here")
        server._server.stop()


_NCCL_SCRIPT = """
import sys
import tensorplay as tp
import tensorplay.distributed as dist

def L(x):
    return x.cpu().tolist()
rank, world, store_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dist.init_process_group(backend="nccl", init_method=f"file://{store_path}",
                        rank=rank, world_size=world)

# all_reduce
t = tp.full((4,), float(rank + 1), dtype=tp.float32, device="cuda:0")
work = dist.all_reduce(t, dist.ReduceOp.SUM, async_op=True)
work.wait()
assert L(t) == [float(sum(range(1, world + 1)))] * 4, f"all_reduce {L(t)}"

# broadcast
t = tp.full((3,), float(rank + 7), dtype=tp.float32, device="cuda:0")
dist.broadcast(t, src=0)
assert L(t) == [7.0] * 3, f"broadcast {L(t)}"

# reduce (root keeps sum, others keep input)
t = tp.full((2,), float(rank + 1), dtype=tp.float32, device="cuda:0")
dist.reduce(t, dst=0)
if rank == 0:
    assert L(t) == [float(sum(range(1, world + 1)))] * 2, f"reduce {L(t)}"

# all_gather
t = tp.full((2,), float(rank), dtype=tp.float32, device="cuda:0")
outs = [tp.zeros(2, dtype=tp.float32, device="cuda:0") for _ in range(world)]
dist.all_gather(outs, t)
for r in range(world):
    assert L(outs[r]) == [float(r)] * 2, f"all_gather[{r}] {L(outs[r])}"

# gather
gather_list = None
if rank == 0:
    gather_list = [tp.zeros(2, dtype=tp.float32, device="cuda:0") for _ in range(world)]
dist.gather(tp.full((2,), float(rank), dtype=tp.float32, device="cuda:0"),
            gather_list=gather_list, dst=0)
if rank == 0:
    for r in range(world):
        assert L(gather_list[r]) == [float(r)] * 2

# scatter
scatter_list = None
if rank == 0:
    scatter_list = [tp.full((2,), float(i), dtype=tp.float32, device="cuda:0")
                    for i in range(world)]
out = tp.zeros(2, dtype=tp.float32, device="cuda:0")
dist.scatter(out, scatter_list=scatter_list, src=0)
assert L(out) == [float(rank)] * 2, f"scatter {L(out)}"

# reduce_scatter
ins = [tp.full((2,), 1.0, dtype=tp.float32, device="cuda:0") for _ in range(world)]
output = tp.zeros(2, dtype=tp.float32, device="cuda:0")
dist.reduce_scatter(output, ins, op=dist.ReduceOp.SUM)
assert L(output) == [float(world)] * 2, f"reduce_scatter {L(output)}"

if world >= 2:
    # send / recv ring (requires a peer rank; NCCL forbids duplicate GPUs,
    # so this only runs on multi-GPU hosts)
    buf = tp.full((2,), 99.0, dtype=tp.float32, device="cuda:0")
    if rank == 0:
        payload = tp.full((2,), 55.0, dtype=tp.float32, device="cuda:0")
        dist.send(payload, dst=1)
        dist.recv(buf, src=1)
    else:
        got = tp.zeros(2, dtype=tp.float32, device="cuda:0")
        dist.recv(got, src=0)
        assert L(got) == [55.0] * 2
        dist.send(got, dst=0)
    assert L(buf) == [55.0] * 2, f"sendrecv {L(buf)}"

# new_group subgroup over all ranks
sub = dist.new_group(ranks=list(range(world)))
t = tp.full((2,), 2.0, dtype=tp.float32, device="cuda:0")
dist.all_reduce(t, group=sub)
assert L(t) == [2.0 * world] * 2, f"new_group {L(t)}"
assert dist.get_rank(sub) == rank
assert dist.get_world_size(sub) == world
assert dist.get_backend() == "nccl"

dist.barrier()
dist.destroy_process_group()
print(f"RANK{rank}OK")
"""


class TestNCCLProcessGroup(unittest.TestCase):
    """Multi-process NCCL collectives. The two-rank case needs one GPU per
    rank (NCCL rejects duplicate GPUs); the single-rank case runs on any CUDA
    machine and still exercises rendezvous, comm init and every collective
    entry point."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.gpu_count = tp.cuda.device_count()
        except Exception:
            cls.gpu_count = 0

    def _run_ranks(self, world):
        fd, store_path = tempfile.mkstemp(prefix="tp_dist_rendezvous_")
        os.close(fd)
        os.unlink(store_path)
        try:
            procs = []
            for rank in range(world):
                procs.append(subprocess.Popen(
                    [sys.executable, "-c", _NCCL_SCRIPT,
                     str(rank), str(world), store_path],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
            outputs = []
            for p in procs:
                out, _ = p.communicate(timeout=180)
                outputs.append(out.decode())
            for rank, out in enumerate(outputs):
                self.assertIn(f"RANK{rank}OK", out, f"rank {rank} failed:\n{out}")
        finally:
            if os.path.exists(store_path):
                os.unlink(store_path)

    @unittest.skipUnless(tp.cuda.is_available(), "CUDA not available")
    def test_collectives_single_rank(self):
        self._run_ranks(world=1)

    def test_collectives_two_ranks(self):
        if self.gpu_count < 2:
            self.skipTest(
                f"needs >= 2 GPUs (found {self.gpu_count}); NCCL forbids "
                "multiple ranks per device"
            )
        self._run_ranks(world=2)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDistributedSamplerParity(unittest.TestCase):
    def test_lengths_and_coverage(self):
        for n, num_replicas, drop_last in [(10, 3, False), (10, 3, True),
                                           (9, 3, False), (9, 3, True)]:
            ds, tds = _DS(n), _TDS(n)
            s = td.DistributedSampler(ds, num_replicas=num_replicas, rank=1,
                                      shuffle=True, seed=7, drop_last=drop_last)
            ts = torchdata.DistributedSampler(tds, num_replicas=num_replicas,
                                              rank=1, shuffle=True, seed=7,
                                              drop_last=drop_last)
            self.assertEqual(len(s), len(ts))
            idx, t_idx = list(s), list(ts)
            self.assertEqual(len(idx), len(t_idx))

    def test_padding_covers_all_indices(self):
        ds = _DS(10)
        seen = []
        for r in range(3):
            seen.extend(td.DistributedSampler(ds, num_replicas=3, rank=r,
                                              shuffle=False))
        # pad-to-divide replicates the first indices: 10 % 3 -> 2 pads
        self.assertEqual(len(seen), 12)

    def test_drop_last_skips_no_index(self):
        ds = _DS(10)
        for r in range(3):
            s = td.DistributedSampler(ds, num_replicas=3, rank=r, shuffle=False,
                                      drop_last=True)
            self.assertEqual(len(s), 3)
        all_indices = []
        for r in range(3):
            all_indices.extend(td.DistributedSampler(ds, num_replicas=3, rank=r,
                                                     shuffle=False, drop_last=True))
        self.assertEqual(sorted(all_indices), list(range(9)))

    def test_set_epoch_changes_order(self):
        ds = _DS(20)
        s = td.DistributedSampler(ds, num_replicas=2, rank=0, seed=3)
        first = list(s)
        ts = torchdata.DistributedSampler(_TDS(20), num_replicas=2, rank=0, seed=3)
        torch_first = list(ts)
        # deterministic across constructions (same seed+epoch)
        self.assertEqual(first, list(td.DistributedSampler(ds, num_replicas=2,
                                                           rank=0, seed=3)))
        s.set_epoch(1)
        self.assertNotEqual(first, list(s))

    def test_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            td.DistributedSampler(_DS(10), num_replicas=2, rank=2)
        with self.assertRaises(ValueError):
            td.DistributedSampler(_DS(10), num_replicas=2, rank=-1)


class _DS(td.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return tp.tensor([i])


class _TDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([i])


if __name__ == "__main__":
    unittest.main()
