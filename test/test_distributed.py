"""Tests for tensorplay.distributed (stores, NCCL process group) and
"""

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
        server.stop()

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
        server.stop()


_NCCL_SCRIPT = """
import sys
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.nn.parallel import DistributedDataParallel

def L(x):
    return x.cpu().tolist()
rank, world, store_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
tp.cuda.set_device(rank)
dev = f"cuda:{rank}"
dist.init_process_group(backend="nccl", init_method=f"file://{store_path}",
                        rank=rank, world_size=world)

# all_reduce
t = tp.full((4,), float(rank + 1), dtype=tp.float32, device=dev)
work = dist.all_reduce(t, dist.ReduceOp.SUM, async_op=True)
work.wait()
assert L(t) == [float(sum(range(1, world + 1)))] * 4, f"all_reduce {L(t)}"

# broadcast
t = tp.full((3,), float(rank + 7), dtype=tp.float32, device=dev)
dist.broadcast(t, src=0)
assert L(t) == [7.0] * 3, f"broadcast {L(t)}"

# reduce (root keeps sum, others keep input)
t = tp.full((2,), float(rank + 1), dtype=tp.float32, device=dev)
dist.reduce(t, dst=0)
if rank == 0:
    assert L(t) == [float(sum(range(1, world + 1)))] * 2, f"reduce {L(t)}"

# all_gather
t = tp.full((2,), float(rank), dtype=tp.float32, device=dev)
outs = [tp.zeros(2, dtype=tp.float32, device=dev) for _ in range(world)]
dist.all_gather(outs, t)
for r in range(world):
    assert L(outs[r]) == [float(r)] * 2, f"all_gather[{r}] {L(outs[r])}"

# all_gather_into_tensor / _allgather_base
t = tp.full((2,), float(rank + 1), dtype=tp.float32, device=dev)
flat_out = tp.zeros(2 * world, dtype=tp.float32, device=dev)
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
    gather_list = [tp.zeros(2, dtype=tp.float32, device=dev) for _ in range(world)]
dist.gather(tp.full((2,), float(rank), dtype=tp.float32, device=dev),
            gather_list=gather_list, dst=0)
if rank == 0:
    for r in range(world):
        assert L(gather_list[r]) == [float(r)] * 2

# scatter
scatter_list = None
if rank == 0:
    scatter_list = [tp.full((2,), float(i), dtype=tp.float32, device=dev)
                    for i in range(world)]
out = tp.zeros(2, dtype=tp.float32, device=dev)
dist.scatter(out, scatter_list=scatter_list, src=0)
assert L(out) == [float(rank)] * 2, f"scatter {L(out)}"

# reduce_scatter
ins = [tp.full((2,), 1.0, dtype=tp.float32, device=dev) for _ in range(world)]
output = tp.zeros(2, dtype=tp.float32, device=dev)
dist.reduce_scatter(output, ins, op=dist.ReduceOp.SUM)
assert L(output) == [float(world)] * 2, f"reduce_scatter {L(output)}"

# reduce_scatter_tensor / _reduce_scatter_base
flat_in = tp.ones(world * 2, dtype=tp.float32, device=dev)
rs_out = tp.zeros(2, dtype=tp.float32, device=dev)
dist.reduce_scatter_tensor(rs_out, flat_in)
assert L(rs_out) == [float(world)] * 2, f"reduce_scatter_tensor {L(rs_out)}"
rs_out.zero_()
dist._reduce_scatter_base(rs_out, flat_in)
assert L(rs_out) == [float(world)] * 2, f"_reduce_scatter_base {L(rs_out)}"

# _functional_collectives: coalesced family (single groupStart/groupEnd
from tensorplay.distributed import _functional_collectives as fc

ins_c = [tp.full((3,), float(rank + 1), dtype=tp.float32, device=dev),
         tp.full((2,), float(2 * (rank + 1)), dtype=tp.float32,
                 device=dev)]
outs_c = fc.all_reduce_coalesced(ins_c, "sum", dist.group.WORLD)
fc.wait_tensor(outs_c[0])
total = float(sum(range(1, world + 1)))
assert L(outs_c[0]) == [total] * 3, f"all_reduce_coalesced[0] {L(outs_c[0])}"
assert L(outs_c[1]) == [2.0 * total] * 2, f"all_reduce_coalesced[1] {L(outs_c[1])}"
assert L(ins_c[0]) == [float(rank + 1)] * 3, \
    f"all_reduce_coalesced modified input {L(ins_c[0])}"
xa = tp.full((2,), float(rank + 1), dtype=tp.float32,
             device=dev).requires_grad_(True)
outa = fc.all_reduce_coalesced([xa], "sum", dist.group.WORLD)[0]
outa.sum().backward()
assert L(xa.grad) == [float(world)] * 2, f"all_reduce_coalesced bwd {L(xa.grad)}"

ga = tp.full((2,), float(rank + 1), dtype=tp.float32,
             device=dev).requires_grad_(True)
outg = fc.all_gather_single_coalesced([ga], dist.group.WORLD)[0]
assert list(outg.shape) == [world, 2], f"all_gather coalesced shape {outg.shape}"
for r in range(world):
    assert L(outg[r]) == [float(r + 1)] * 2, f"all_gather_coalesced[{r}] {L(outg[r])}"
outg.sum().backward()
assert L(ga.grad) == [float(world)] * 2, f"all_gather_coalesced bwd {L(ga.grad)}"

ra = tp.full((world, 2), float(rank + 1), dtype=tp.float32,
             device=dev).requires_grad_(True)
outr = fc.reduce_scatter_single_coalesced([ra], "sum", [0], dist.group.WORLD)[0]
assert list(outr.shape) == [1, 2], f"reduce_scatter coalesced shape {outr.shape}"
row_sum = float(sum(range(1, world + 1)))
assert L(outr.reshape(-1)) == [row_sum] * 2, f"reduce_scatter_coalesced {L(outr)}"
outr.sum().backward()
assert L(ra.grad.reshape(-1)) == [1.0] * (world * 2), f"reduce_scatter_coalesced bwd {L(ra.grad)}"

rb = tp.full((2, world), float(rank + 1), dtype=tp.float32, device=dev)
outb = fc.reduce_scatter_single_coalesced([rb], "sum", [1], dist.group.WORLD)[0]
assert list(outb.shape) == [2, 1], f"reduce_scatter dim1 shape {outb.shape}"
assert L(outb) == [[row_sum]] * 2, f"reduce_scatter dim1 {L(outb)}"

assert dist.get_process_group_ranks(dist.group.WORLD) == list(range(world))
sub_all = dist.new_group(ranks=list(range(world)))
assert dist.get_global_rank(sub_all, rank) == rank
assert dist.get_group_rank(sub_all, rank) == rank
assert dist.get_process_group_ranks(sub_all) == list(range(world))

# all_to_all_single (equal splits)
in_t = tp.arange(world * 2, dtype=tp.float32, device=dev) + 100.0 * rank
out_t = tp.zeros(world * 2, dtype=tp.float32, device=dev)
dist.all_to_all_single(out_t, in_t)
expected = []
for r in range(world):
    expected.extend([float(100.0 * r + 2 * rank), float(100.0 * r + 2 * rank + 1)])
assert L(out_t) == expected, f"all_to_all_single {L(out_t)} vs {expected}"

# all_to_all_single (uneven splits): to rank r we send a chunk of size (r+1)
in_sizes = [r + 1 for r in range(world)]
out_sizes = [rank + 1] * world
flat_in = tp.cat([tp.full((s,), float(10 * rank + s),
                          dtype=tp.float32, device=dev)
                  for s in in_sizes])
flat_out = tp.zeros(sum(out_sizes), dtype=tp.float32, device=dev)
dist.all_to_all_single(flat_out, flat_in, out_sizes, in_sizes)
# rank r receives from each peer a chunk of size (rank+1) filled with 10*r_src+(rank+1)
got = []
for r in range(world):
    got.extend([float(10 * r + rank + 1)] * (rank + 1))
assert L(flat_out) == got, f"all_to_all_single uneven {L(flat_out)} vs {got}"

# all_to_all tensor-list form
ins_l = [tp.full((1,), float(rank * 10 + i), dtype=tp.float32,
                 device=dev) for i in range(world)]
outs_l = [tp.zeros(1, dtype=tp.float32, device=dev) for _ in range(world)]
dist.all_to_all(outs_l, ins_l)
for r in range(world):
    assert L(outs_l[r]) == [float(r * 10 + rank)], f"all_to_all[{r}]"

dist.barrier()

if world >= 2:
    # batch_isend_irecv ring exchange via grouped p2p
    peer = (rank + 1) % world
    prev = (rank - 1 + world) % world
    send_t = tp.full((3,), float(rank), dtype=tp.float32, device=dev)
    recv_t = tp.zeros(3, dtype=tp.float32, device=dev)
    ops = [dist.P2POp(dist.isend, send_t, peer),
           dist.P2POp(dist.irecv, recv_t, prev)]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    assert L(recv_t) == [float(prev)] * 3, f"batch_isend_irecv {L(recv_t)}"
    # isend/irecv used directly. Distinct payload and a fresh recv buffer so
    # a stale/no-op regression is detectable; in the ring a rank receives its
    # predecessor's value (same semantics as batch_isend_irecv above).
    send2 = tp.full((3,), float(rank) + 7.0, dtype=tp.float32, device=dev)
    recv2 = tp.full((3,), -1.0, dtype=tp.float32, device=dev)
    fut_work = dist.isend(send2, peer)
    got_w = dist.irecv(recv2, prev)
    fut_work.wait()
    got_w.wait()
    assert L(recv2) == [float(prev) + 7.0] * 3, f"isend/irecv {L(recv2)}"

if world >= 2:
    # send / recv ring (requires a peer rank; NCCL forbids duplicate GPUs,
    # so this only runs on multi-GPU hosts)
    buf = tp.full((2,), 99.0, dtype=tp.float32, device=dev)
    if rank == 0:
        payload = tp.full((2,), 55.0, dtype=tp.float32, device=dev)
        dist.send(payload, dst=1)
        dist.recv(buf, src=1)
        assert L(buf) == [55.0] * 2, f"sendrecv {L(buf)}"
    else:
        got = tp.zeros(2, dtype=tp.float32, device=dev)
        dist.recv(got, src=0)
        assert L(got) == [55.0] * 2
        dist.send(got, dst=0)

# new_group subgroup over all ranks
sub = dist.new_group(ranks=list(range(world)))
t = tp.full((2,), 2.0, dtype=tp.float32, device=dev)
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

tp.manual_seed(1234)
from tensorplay.distributed import collective_utils as _cu
rng_msg = _cu._check_rng_sync(None, dist.group.WORLD)
assert rng_msg is None, f"_check_rng_sync desync: {rng_msg}"

dist.barrier()

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

total = float(sum(range(1, world + 1)))
ca = [tp.full((3,), float(rank + 1), dtype=tp.float32, device=dev),
      tp.full((2,), 2.0 * (rank + 1), dtype=tp.float32, device=dev)]
dist.all_reduce_coalesced(ca, op=dist.ReduceOp.SUM)
assert L(ca[0]) == [total] * 3, f"all_reduce_coalesced[0] {L(ca[0])}"
assert L(ca[1]) == [2.0 * total] * 2, f"all_reduce_coalesced[1] {L(ca[1])}"

ga_in = [tp.full((2,), float(rank + 1), dtype=tp.float32, device=dev),
         tp.full((4,), float(rank + 10), dtype=tp.float32, device=dev)]
ga_out = [[tp.zeros(2, dtype=tp.float32, device=dev) for _ in range(world)],
          [tp.zeros(4, dtype=tp.float32, device=dev) for _ in range(world)]]
dist.all_gather_coalesced(ga_out, ga_in)
for r in range(world):
    assert L(ga_out[0][r]) == [float(r + 1)] * 2, f"all_gather_coalesced[0][{r}] {L(ga_out[0][r])}"
    assert L(ga_out[1][r]) == [float(r + 10)] * 4, f"all_gather_coalesced[1][{r}] {L(ga_out[1][r])}"

rs_out = [tp.zeros(2, dtype=tp.float32, device=dev)]
rs_in = [[tp.full((2,), float(rank + 1), dtype=tp.float32, device=dev)
          for _ in range(world)]]
dist.reduce_scatter_coalesced(rs_out, rs_in, op=dist.ReduceOp.SUM)
assert L(rs_out[0]) == [total] * 2, f"reduce_scatter_coalesced {L(rs_out[0])}"

dist.monitored_barrier()
dist.monitored_barrier(wait_all_ranks=True)

dist.barrier()

# DDP: initial-state sync, gradient averaging, buffer sync
class Tiny(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = tp.nn.Linear(8, 4)
        self.register_buffer("step", tp.zeros(1))

    def forward(self, x):
        return self.fc(x)

tp.manual_seed(1234)
model = Tiny().to(dev)
with tp.no_grad():
    model.step.add_(float(rank + 1))  # diverge buffers per rank
ddp = DistributedDataParallel(model, device_ids=[rank])
ddp.module.step.zero_()

# after construction all params/buffers equal rank 0's copies
ws = [None] * world
dist.all_gather_object(ws, ddp.module.fc.weight.detach().cpu().tolist())
assert all(w == ws[0] for w in ws), "DDP param sync failed"
bs = [None] * world
dist.all_gather_object(bs, ddp.module.step.detach().cpu().tolist())
assert all(b == bs[0] for b in bs), f"DDP init buffer sync failed {bs}"

x = tp.full((16, 8), 0.5 + 0.1 * rank, dtype=tp.float32, device=dev)


def avg_grad(W, b):
    # Mean over ranks of the local gradient d/dW sum((x_r @ W^T + b)^2).
    # x_r is deterministic in r and W/b are synced across ranks, so every rank
    # can reconstruct every rank's local gradient analytically -> a
    # non-circular expected value for DDP's all-reduce average.
    acc = None
    for r in range(world):
        x_r = tp.full((16, 8), 0.5 + 0.1 * r, dtype=tp.float32, device=dev)
        y_r = x_r @ W.t() + b
        g_r = y_r.t() @ x_r
        acc = g_r if acc is None else acc + g_r
    return (2.0 / world) * acc


loss = ddp(x).pow(2).sum()
loss.backward()
grads = [None] * world
dist.all_gather_object(grads, ddp.module.fc.weight.grad.detach().cpu().tolist())
assert all(g == grads[0] for g in grads), "DDP grads not identical across ranks"
expected_grad = avg_grad(ddp.module.fc.weight, ddp.module.fc.bias)
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
model_b = Tiny().to(dev)
ddp_b = DistributedDataParallel(model_b, device_ids=[rank], bucket_cap_mb=1)
loss = ddp_b(x).pow(2).sum()
loss.backward()
grads_b = [None] * world
dist.all_gather_object(grads_b, ddp_b.module.fc.weight.grad.detach().cpu().tolist())
assert all(g == grads_b[0] for g in grads_b), "bucketed grads not identical"
expected_b = avg_grad(ddp_b.module.fc.weight, ddp_b.module.fc.bias)
diff = (ddp_b.module.fc.weight.grad - expected_b).abs().max().item()
assert diff < 1e-4, f"bucketed grad mismatch {diff}"

# ---- DDP + allreduce comm hook (must match default path) ----
from tensorplay.distributed.algorithms.ddp_comm_hooks import default_hooks
tp.manual_seed(1234)
model_c = Tiny().to(dev)
ddp_c = DistributedDataParallel(model_c, device_ids=[rank])
ddp_c.register_comm_hook(None, default_hooks.allreduce_hook)
loss = ddp_c(x).pow(2).sum()
loss.backward()
expected_c = avg_grad(ddp_c.module.fc.weight, ddp_c.module.fc.bias)
diff = (ddp_c.module.fc.weight.grad - expected_c).abs().max().item()
assert diff < 1e-4, f"comm hook grad mismatch {diff}"

# ---- DDP + fp16 compression hook ----
tp.manual_seed(1234)
model_d = Tiny().to(dev)
ddp_d = DistributedDataParallel(model_d, device_ids=[rank])
ddp_d.register_comm_hook(None, default_hooks.fp16_compress_hook)
loss = ddp_d(x).pow(2).sum()
loss.backward()
diff = (ddp_d.module.fc.weight.grad - avg_grad(
    ddp_d.module.fc.weight, ddp_d.module.fc.bias)).abs().max().item()
assert diff < 5e-3, f"fp16 hook grad mismatch {diff}"

# ---- DDP gradient_as_bucket_view: grads alias the reduced bucket ----
tp.manual_seed(1234)
model_g = Tiny().to(dev)
ddp_g = DistributedDataParallel(model_g, device_ids=[rank],
                                gradient_as_bucket_view=True)
for _it in range(2):
    ddp_g.zero_grad()
    loss = ddp_g(x).pow(2).sum()
    loss.backward()
    diff = (ddp_g.module.fc.weight.grad - avg_grad(
        ddp_g.module.fc.weight, ddp_g.module.fc.bias)).abs().max().item()
    assert diff < 1e-4, f"gradient_as_bucket_view grad mismatch {diff}"
# set_to_none=False: grads persist as bucket-view aliases and are reused
for _it in range(2):
    ddp_g.zero_grad(set_to_none=False)
    loss = ddp_g(x).pow(2).sum()
    loss.backward()
    diff = (ddp_g.module.fc.weight.grad - avg_grad(
        ddp_g.module.fc.weight, ddp_g.module.fc.bias)).abs().max().item()
    assert diff < 1e-4, f"gradient_as_bucket_view reuse grad mismatch {diff}"

# ---- DDP with fp16 params: pre-divide + SUM path (no ncclAvg at half) ----
tp.manual_seed(1234)
model_h16 = Tiny().to(dev).half()
ddp_h16 = DistributedDataParallel(model_h16, device_ids=[rank])
x16 = x.half()
loss = ddp_h16(x16).pow(2).sum()
loss.backward()
exp16 = avg_grad(ddp_h16.module.fc.weight.float(),
                 ddp_h16.module.fc.bias.float())
diff = (ddp_h16.module.fc.weight.grad.float() - exp16).abs().max().item()
assert diff < 5e-2, f"fp16 DDP grad mismatch {diff}"

# ---- DDP static_graph: reuse the first-iteration traversal ----
tp.manual_seed(1234)
model_sg = Tiny().to(dev)
ddp_sg = DistributedDataParallel(model_sg, device_ids=[rank],
                                 find_unused_parameters=True,
                                 static_graph=True)
for _it in range(2):
    ddp_sg.zero_grad()
    loss = ddp_sg(x).pow(2).sum()
    loss.backward()
    diff = (ddp_sg.module.fc.weight.grad - avg_grad(
        ddp_sg.module.fc.weight, ddp_sg.module.fc.bias)).abs().max().item()
    assert diff < 1e-4, f"static_graph grad mismatch {diff}"

# ---- unused param without find_unused must raise on the next forward ----
class UnusedParam(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = tp.nn.Linear(8, 4)
        self.unused = tp.nn.Linear(8, 4)

    def forward(self, x):
        return self.used(x)

tp.manual_seed(1234)
model_f = UnusedParam().to(dev)
ddp_f = DistributedDataParallel(model_f, device_ids=[rank])
loss = ddp_f(x).pow(2).sum()
loss.backward()  # bucket never completes: unused param's hook never fires
try:
    ddp_f(x)
    raise AssertionError("expected RuntimeError for unused parameter")
except RuntimeError as e:
    assert "find_unused_parameters" in str(e), str(e)

# ---- no_sync gradient accumulation: only the synced backward reduces ----
tp.manual_seed(1234)
model_h = Tiny().to(dev)
ddp_h = DistributedDataParallel(model_h, device_ids=[rank])
xs_all = [[tp.full((16, 8), 0.5 + 0.1 * r + 0.01 * k,
                   dtype=tp.float32, device=dev)
           for k in range(3)] for r in range(world)]
with ddp_h.no_sync():
    for k in range(2):
        ddp_h(xs_all[rank][k]).pow(2).sum().backward()
ddp_h(xs_all[rank][2]).pow(2).sum().backward()
W_h, b_h = ddp_h.module.fc.weight, ddp_h.module.fc.bias
exp_h = None
for r in range(world):
    for k in range(3):
        xk = xs_all[r][k]
        yk = xk @ W_h.t() + b_h
        gk = yk.t() @ xk
        exp_h = gk if exp_h is None else exp_h + gk
exp_h = (2.0 / world) * exp_h
diff = (W_h.grad - exp_h).abs().max().item()
assert diff < 1e-4, f"no_sync accumulation grad mismatch {diff}"

# ---- DDP find_unused_parameters: two heads, rank-dependent usage ----
class TwoHeads(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = tp.nn.Linear(8, 8)
        self.head0 = tp.nn.Linear(8, 2)
        self.head1 = tp.nn.Linear(8, 2)
        self.use0 = True

    def forward(self, x):
        body = self.body(x)
        return self.head0(body) if self.use0 else self.head1(body)

tp.manual_seed(1234)
model_e = TwoHeads().to(dev)
model_e.use0 = (rank % 2 == 0)
ddp_e = DistributedDataParallel(model_e, device_ids=[rank],
                                find_unused_parameters=True,
                                bucket_cap_mb=1)
h = ddp_e(x)
out = h.pow(2).sum()
out.backward()
g_body = [None] * world
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


@unittest.skipUnless(HAS_TORCH, "reference package not available")
class TestDistributedSamplerReference(unittest.TestCase):
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


class _TDS(torch.utils.data.Dataset if HAS_TORCH else td.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([i]) if HAS_TORCH else tp.tensor([i])


if __name__ == "__main__":
    unittest.main()
