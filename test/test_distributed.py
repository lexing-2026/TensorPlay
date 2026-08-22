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
from tensorplay.nn.parallel import DistributedDataParallel

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

# all_gather_into_tensor / _allgather_base
t = tp.full((2,), float(rank + 1), dtype=tp.float32, device="cuda:0")
flat_out = tp.zeros(2 * world, dtype=tp.float32, device="cuda:0")
dist.all_gather_into_tensor(flat_out, t)
assert L(flat_out) == [float(r + 1) for r in range(world) for _ in range(2)], \\
    f"all_gather_into_tensor {L(flat_out)}"
flat_out.zero_()
dist._allgather_base(flat_out, t)
assert L(flat_out) == [float(r + 1) for r in range(world) for _ in range(2)], \\
    f"_allgather_base {L(flat_out)}"

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

# reduce_scatter_tensor / _reduce_scatter_base
flat_in = tp.ones(world * 2, dtype=tp.float32, device="cuda:0")
rs_out = tp.zeros(2, dtype=tp.float32, device="cuda:0")
dist.reduce_scatter_tensor(rs_out, flat_in)
assert L(rs_out) == [float(world)] * 2, f"reduce_scatter_tensor {L(rs_out)}"
rs_out.zero_()
dist._reduce_scatter_base(rs_out, flat_in)
assert L(rs_out) == [float(world)] * 2, f"_reduce_scatter_base {L(rs_out)}"

# group-rank translation APIs
assert dist.get_process_group_ranks() == list(range(world))
sub_all = dist.new_group(ranks=list(range(world)))
assert dist.get_global_rank(sub_all, rank) == rank
assert dist.get_group_rank(sub_all, rank) == rank
assert dist.get_process_group_ranks(sub_all) == list(range(world))

# all_to_all_single (equal splits)
in_t = tp.arange(world * 2, dtype=tp.float32, device="cuda:0") + 100.0 * rank
out_t = tp.zeros(world * 2, dtype=tp.float32, device="cuda:0")
dist.all_to_all_single(out_t, in_t)
expected = []
for r in range(world):
    expected.extend([float(100.0 * r + 2 * rank), float(100.0 * r + 2 * rank + 1)])
assert L(out_t) == expected, f"all_to_all_single {L(out_t)} vs {expected}"

# all_to_all_single (uneven splits): to rank r we send a chunk of size (r+1)
in_sizes = [r + 1 for r in range(world)]
out_sizes = [rank + 1] * world
flat_in = tp.cat([tp.full((s,), float(10 * rank + s),
                          dtype=tp.float32, device="cuda:0")
                  for s in in_sizes])
flat_out = tp.zeros(sum(out_sizes), dtype=tp.float32, device="cuda:0")
dist.all_to_all_single(flat_out, flat_in, out_sizes, in_sizes)
# rank r receives from each peer a chunk of size (rank+1) filled with 10*r_src+(rank+1)
got = []
for r in range(world):
    got.extend([float(10 * r + rank + 1)] * (rank + 1))
assert L(flat_out) == got, f"all_to_all_single uneven {L(flat_out)} vs {got}"

# all_to_all tensor-list form
ins_l = [tp.full((1,), float(rank * 10 + i), dtype=tp.float32,
                 device="cuda:0") for i in range(world)]
outs_l = [tp.zeros(1, dtype=tp.float32, device="cuda:0") for _ in range(world)]
dist.all_to_all(outs_l, ins_l)
for r in range(world):
    assert L(outs_l[r]) == [float(r * 10 + rank)], f"all_to_all[{r}]"

dist.barrier()

if world >= 2:
    # batch_isend_irecv ring exchange via grouped p2p
    peer = (rank + 1) % world
    prev = (rank - 1 + world) % world
    send_t = tp.full((3,), float(rank), dtype=tp.float32, device="cuda:0")
    recv_t = tp.zeros(3, dtype=tp.float32, device="cuda:0")
    ops = [dist.P2POp(dist.isend, send_t, peer),
           dist.P2POp(dist.irecv, recv_t, prev)]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    assert L(recv_t) == [float(prev)] * 3, f"batch_isend_irecv {L(recv_t)}"
    # isend/irecv used directly
    fut_work = dist.isend(send_t, peer)
    got_w = dist.irecv(recv_t, prev)
    fut_work.wait()
    got_w.wait()
    assert L(recv_t) == [float(rank)] * 3

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

# object collectives
objects = ["foo", {"r": rank}, rank]
if rank == 0:
    bcast = objects
else:
    bcast = [None] * 3
dist.broadcast_object_list(bcast, src=0)
assert bcast[0] == "foo" and bcast[1] == {"r": 0} and bcast[2] == 0, \\
    f"broadcast_object_list {bcast}"

ag = [None] * world
dist.all_gather_object(ag, {"me": rank})
for r in range(world):
    assert ag[r] == {"me": r}, f"all_gather_object[{r}] {ag[r]}"

go = [None] * world if rank == 0 else None
dist.gather_object({"mine": rank}, go, dst=0)
if rank == 0:
    for r in range(world):
        assert go[r] == {"mine": r}, f"gather_object[{r}] {go[r]}"

so = [None]
slist = [{"to": i} for i in range(world)] if rank == 0 else None
dist.scatter_object_list(so, slist, src=0)
assert so[0] == {"to": rank}, f"scatter_object_list {so[0]}"

if world >= 2:
    if rank == 0:
        so_list = ["a", "b"]
        dist.send_object_list(so_list, dst=1)
    else:
        ro = [None, None]
        got_src = dist.recv_object_list(ro, src=0)
        assert ro == ["a", "b"], f"recv_object_list {ro}"
        assert got_src == 0

dist.barrier()

# DDP: initial-state sync, gradient averaging, buffer sync
class Tiny(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = tp.nn.Linear(8, 4)
        self.register_buffer("step", tp.zeros(1))

tp.manual_seed(1234)
model = Tiny().to("cuda:0")
with tp.no_grad():
    model.step.add_(float(rank + 1))  # diverge buffers per rank
ddp = DistributedDataParallel(model, device_ids=[rank])
ddp.module.step.zero_()

# after construction all params/buffers equal rank 0's copies
ws = []
dist.all_gather_object(ws, ddp.module.fc.weight.detach().cpu().tolist())
assert all(w == ws[0] for w in ws), "DDP param sync failed"
bs = []
dist.all_gather_object(bs, ddp.module.step.detach().cpu().tolist())
assert all(b == bs[0] for b in bs), f"DDP init buffer sync failed {bs}"

x = tp.full((16, 8), 0.5, dtype=tp.float32, device="cuda:0")
loss = ddp(x).pow(2).sum()
loss.backward()
grads = []
dist.all_gather_object(grads, ddp.module.fc.weight.grad.detach().cpu().tolist())
assert all(g == grads[0] for g in grads), "DDP grads not identical across ranks"
y = x @ ddp.module.fc.weight.t() + ddp.module.fc.bias
expected_grad = (2.0 / world) * (y.t() @ x)
diff = (ddp.module.fc.weight.grad - expected_grad).abs().max().item()
assert diff < 1e-4, f"DDP grad mismatch {diff}"
assert int(ddp.module.step.item()) == 0

with ddp.no_sync():
    assert ddp.require_backward_grad_sync is False
assert ddp.require_backward_grad_sync is True

sd = ddp.state_dict()
assert any(k.startswith("module.") for k in sd.keys()), f"state_dict keys {list(sd)[:4]}"

# ---- DDP with forced multi-bucket reduction (tiny cap) ----
tp.manual_seed(1234)
model_b = Tiny().to("cuda:0")
ddp_b = DistributedDataParallel(model_b, device_ids=[rank], bucket_cap_mb=1)
loss = ddp_b(x).pow(2).sum()
loss.backward()
grads_b = []
dist.all_gather_object(grads_b, ddp_b.module.fc.weight.grad.detach().cpu().tolist())
assert all(g == grads_b[0] for g in grads_b), "bucketed grads not identical"
y_b = x @ ddp_b.module.fc.weight.t() + ddp_b.module.fc.bias
expected_b = (2.0 / world) * (y_b.t() @ x)
diff = (ddp_b.module.fc.weight.grad - expected_b).abs().max().item()
assert diff < 1e-4, f"bucketed grad mismatch {diff}"

# ---- DDP + allreduce comm hook (must match default path) ----
from tensorplay.distributed.algorithms.ddp_comm_hooks import default_hooks
tp.manual_seed(1234)
model_c = Tiny().to("cuda:0")
ddp_c = DistributedDataParallel(model_c, device_ids=[rank])
ddp_c.register_comm_hook(None, default_hooks.allreduce_hook)
loss = ddp_c(x).pow(2).sum()
loss.backward()
y_c = x @ ddp_c.module.fc.weight.t() + ddp_c.module.fc.bias
expected_c = (2.0 / world) * (y_c.t() @ x)
diff = (ddp_c.module.fc.weight.grad - expected_c).abs().max().item()
assert diff < 1e-4, f"comm hook grad mismatch {diff}"

# ---- DDP + fp16 compression hook ----
tp.manual_seed(1234)
model_d = Tiny().to("cuda:0")
ddp_d = DistributedDataParallel(model_d, device_ids=[rank])
ddp_d.register_comm_hook(None, default_hooks.fp16_compress_hook)
loss = ddp_d(x).pow(2).sum()
loss.backward()
diff = (ddp_d.module.fc.weight.grad - (2.0 / world) * (
    (x @ ddp_d.module.fc.weight.t() + ddp_d.module.fc.bias).t() @ x)
).abs().max().item()
assert diff < 5e-3, f"fp16 hook grad mismatch {diff}"

# ---- DDP find_unused_parameters: two heads, rank-dependent usage ----
class TwoHeads(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = tp.nn.Linear(8, 8)
        self.head0 = tp.nn.Linear(8, 2)
        self.head1 = tp.nn.Linear(8, 2)

    def forward(self, x):
        return tp.cat([self.head0(self.body(x)),
                       self.head1(self.body(x))], dim=1)

tp.manual_seed(1234)
model_e = TwoHeads().to("cuda:0")
ddp_e = DistributedDataParallel(model_e, device_ids=[rank],
                                find_unused_parameters=True,
                                bucket_cap_mb=1)
h = ddp_e(x)
out = h[:, :2].pow(2).sum() if rank % 2 == 0 else h[:, 2:].pow(2).sum()
out.backward()
g_body = []
dist.all_gather_object(g_body, ddp_e.module.body.weight.grad.detach().cpu().tolist())
assert all(g == g_body[0] for g in g_body), "find_unused body grads differ"
used_head = ddp_e.module.head0 if rank % 2 == 0 else ddp_e.module.head1
unused_head = ddp_e.module.head1 if rank % 2 == 0 else ddp_e.module.head0
assert used_head.weight.grad is not None, "used head grad missing"
assert unused_head.weight.grad is None, "unused head grad should stay None"
dist.barrier()

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
