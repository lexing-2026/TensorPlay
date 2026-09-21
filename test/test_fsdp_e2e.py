"""End-to-end tests for fully sharded data parallel training.

Three layers of coverage:
  * in-process single-rank runs (no process group, default mesh) comparing
    FSDP-iterated parameters, gradients and losses against an unwrapped
    reference model;
  * a two-rank gloo/CPU run through spawned ranks exercising the real
    all-gather / reduce-scatter path with analytic gradient checks;
  * a single-rank CUDA/NCCL run through a spawned interpreter following the
    same rendezvous pattern as the collective tests.
"""

import copy
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp
from tensorplay import nn
from tensorplay.distributed.fsdp import FullyShardedDataParallel
from tensorplay.distributed.fsdp._fully_shard._fsdp_param import ShardedState
from tensorplay.distributed.fsdp.api import ShardingStrategy


class _Tiny(nn.Module):
    def __init__(self, din: int = 16, hidden: int = 32, dout: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(din, hidden)
        self.fc2 = nn.Linear(hidden, dout)

    def forward(self, x):
        return self.fc2(tp.nn.functional.relu(self.fc1(x)))


def _fsdp_params(module):
    state = module._get_fsdp_state()
    for group in state._all_param_groups():
        yield from group.fsdp_params


def _shard_states(module):
    return {p.module_info.fqn: p._state for p in _fsdp_params(module)}


class TestFSDPSingleRankInProcess(unittest.TestCase):
    """World-size-1 behaviour with the default mesh (no process group)."""

    def _reference_pair(self, device):
        tp.manual_seed(1234)
        model = _Tiny().to(device)
        tp.manual_seed(1234)
        ref = _Tiny().to(device)
        for a, b in zip(model.parameters(), ref.parameters()):
            tp.testing.assert_close(a, b)
        return model, ref

    def test_train_step_matches_unwrapped(self):
        device = tp.device("cpu")
        model, ref = self._reference_pair(device)
        wrapped = FullyShardedDataParallel(model, device_id=device)

        x = tp.randn(12, 16, device=device)
        loss = wrapped(x).pow(2).sum()
        ref_loss = ref(x).pow(2).sum()
        self.assertAlmostEqual(float(loss), float(ref_loss), places=5)

        loss.backward()
        ref_loss.backward()

        opt = tp.optim.SGD(wrapped.parameters(), lr=0.1)
        ref_opt = tp.optim.SGD(ref.parameters(), lr=0.1)
        opt.step()
        ref_opt.step()

        for w_param, r_param in zip(wrapped.parameters(), ref.parameters()):
            self.assertEqual(tuple(w_param.shape), tuple(r_param.shape))
            tp.testing.assert_close(w_param, r_param)
        for w_param, r_param in zip(wrapped.parameters(), ref.parameters()):
            self.assertIsNotNone(r_param.grad)
            tp.testing.assert_close(w_param.grad, r_param.grad)

    def test_resharding_lifecycle(self):
        device = tp.device("cpu")
        model, _ = self._reference_pair(device)
        wrapped = FullyShardedDataParallel(model, device_id=device)

        states = _shard_states(wrapped.module)
        self.assertTrue(all(s == ShardedState.SHARDED for s in states.values()))

        seen = {}
        def pre_hook(module, args):
            for name, p in module.named_parameters(recurse=False):
                seen[f"fc1.{name}"] = tuple(p.shape)
        wrapped.module.fc1.register_forward_pre_hook(pre_hook)

        x = tp.randn(4, 16, device=device)
        out = wrapped(x)
        self.assertEqual(tuple(out.shape), (4, 4))
        self.assertEqual(seen["fc1.weight"], (32, 16))

        states = _shard_states(wrapped.module)
        self.assertTrue(all(s == ShardedState.SHARDED for s in states.values()))
        out.pow(2).sum().backward()
        states = _shard_states(wrapped.module)
        self.assertTrue(all(s == ShardedState.SHARDED for s in states.values()))

    def test_summon_full_params_roundtrip(self):
        device = tp.device("cpu")
        model, ref = self._reference_pair(device)
        wrapped = FullyShardedDataParallel(model, device_id=device)

        with FullyShardedDataParallel.summon_full_params(
            wrapped.module, with_grads=False
        ):
            for name, value in wrapped.module.named_parameters():
                tp.testing.assert_close(value, dict(ref.named_parameters())[name])

        states = _shard_states(wrapped.module)
        self.assertTrue(all(s == ShardedState.SHARDED for s in states.values()))

    def test_no_sync_accumulates_gradients(self):
        device = tp.device("cpu")
        model, ref = self._reference_pair(device)
        wrapped = FullyShardedDataParallel(model, device_id=device)
        ref_opt = tp.optim.SGD(ref.parameters(), lr=0.0)

        xs = [tp.randn(6, 16, device=device) for _ in range(3)]
        for x in xs:
            ref(x).pow(2).sum().backward()
            ref_opt.step()
        expected = [p.grad.clone() for p in ref.parameters()]

        with wrapped.no_sync():
            for x in xs[:2]:
                wrapped(x).pow(2).sum().backward()
        wrapped(xs[2]).pow(2).sum().backward()

        for w_param, exp in zip(wrapped.parameters(), expected):
            tp.testing.assert_close(w_param.grad, exp)

    def test_multi_iteration_matches_reference(self):
        """Several fwd/bwd/step cycles must track the unwrapped reference.

        The optimizer is constructed once, before any cycle, so this also
        guards the sharded-parameter identity across unshard/reshard cycles:
        if resharding re-created the parameter objects the optimizer would
        silently step orphaned tensors with no gradients.
        """
        device = tp.device("cpu")
        model, ref = self._reference_pair(device)
        wrapped = FullyShardedDataParallel(model, device_id=device)
        opt = tp.optim.SGD(wrapped.parameters(), lr=0.05)
        ref_opt = tp.optim.SGD(ref.parameters(), lr=0.05)
        opt_params = [p for g in opt.param_groups for p in g["params"]]

        tp.manual_seed(8)
        xs = [tp.randn(8, 16, device=device) for _ in range(6)]
        losses = []
        for step, x in enumerate(xs):
            opt.zero_grad()
            ref_opt.zero_grad()
            loss = wrapped(x).pow(2).mean()
            ref_loss = ref(x).pow(2).mean()
            self.assertAlmostEqual(float(loss), float(ref_loss), places=5)
            loss.backward()
            ref_loss.backward()
            self.assertEqual(
                [id(p) for p in opt_params],
                [id(p) for _, p in wrapped.module.named_parameters()],
                f"optimizer lost the sharded parameters at step {step}",
            )
            self.assertTrue(all(p.grad is not None for p in opt_params))
            opt.step()
            ref_opt.step()
            for w_param, r_param in zip(wrapped.parameters(), ref.parameters()):
                tp.testing.assert_close(w_param, r_param)
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])

    def test_fully_shard_composable_api(self):
        from tensorplay.distributed.fsdp._fully_shard import fully_shard

        device = tp.device("cpu")
        tp.manual_seed(11)
        model, ref = self._reference_pair(device)
        fully_shard(model)

        x = tp.randn(5, 16, device=device)
        loss = model(x).pow(2).sum()
        ref_loss = ref(x).pow(2).sum()
        self.assertAlmostEqual(float(loss), float(ref_loss), places=5)
        loss.backward()
        ref_loss.backward()
        opt = tp.optim.SGD(model.parameters(), lr=0.1)
        ref_opt = tp.optim.SGD(ref.parameters(), lr=0.1)
        opt.step()
        ref_opt.step()
        for w_param, r_param in zip(model.parameters(), ref.parameters()):
            tp.testing.assert_close(w_param, r_param)

    def test_checkpoint_roundtrip_with_optimizer(self):
        from tensorplay.distributed import checkpoint as dcp
        from tensorplay.distributed.checkpoint.state_dict import (
            get_state_dict,
            set_state_dict,
        )

        device = tp.device("cpu")
        model, _ = self._reference_pair(device)
        wrapped = FullyShardedDataParallel(model, device_id=device)
        opt = tp.optim.AdamW(wrapped.parameters(), lr=1e-3)

        x = tp.randn(6, 16, device=device)
        wrapped(x).pow(2).sum().backward()
        opt.step()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "fsdp.dcp")
            model_sd, optim_sd = get_state_dict(wrapped, opt)
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt)

            tp.manual_seed(99)
            fresh = FullyShardedDataParallel(_Tiny().to(device), device_id=device)
            fresh_opt = tp.optim.AdamW(fresh.parameters(), lr=1e-3)
            fresh_model_sd, fresh_optim_sd = get_state_dict(fresh, fresh_opt)
            dcp.load(
                {"model": fresh_model_sd, "optim": fresh_optim_sd},
                checkpoint_id=ckpt,
            )
            set_state_dict(
                fresh,
                fresh_opt,
                model_state_dict=fresh_model_sd,
                optim_state_dict=fresh_optim_sd,
            )
            for a, b in zip(wrapped.parameters(), fresh.parameters()):
                tp.testing.assert_close(a, b)
            self.assertEqual(
                len(fresh_opt.state_dict()["state"]),
                len(opt.state_dict()["state"]),
            )


def _fsdp_two_rank_body(rank: int, world: int, port: int) -> None:
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.fsdp import FullyShardedDataParallel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", init_method="env://", rank=rank,
                            world_size=world)

    tp.manual_seed(1234)
    model = _Tiny()
    ref = copy.deepcopy(model)
    wrapped = FullyShardedDataParallel(
        model, process_group=dist.group.WORLD,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )

    states = set(p._state for p in
                 wrapped.module._get_fsdp_state()._all_param_groups()[0].fsdp_params)
    assert states == {ShardedState.SHARDED}, f"states {states}"

    def rank_grads(input_tensor):
        for p in ref.parameters():
            p.grad = None
        ref(input_tensor).pow(2).sum().backward()
        return [p.grad.detach().clone() for p in ref.parameters()]

    x = tp.full((12, 16), 0.5 + 0.25 * rank)
    loss = wrapped(x).pow(2).sum()
    ref_loss = ref(x).pow(2).sum()
    assert abs(float(loss) - float(ref_loss)) < 1e-4, (float(loss), float(ref_loss))
    loss.backward()

    # reduce-scatter averages gradients: expected local shard is the mean of
    # the per-rank gradients, sliced to this rank's dim-0 window.
    opt = tp.optim.SGD(wrapped.parameters(), lr=0.05)
    per_rank_grads = [
        rank_grads(tp.full((12, 16), 0.5 + 0.25 * r)) for r in range(world)
    ]
    for idx, (w_param, r_param) in enumerate(
        zip(wrapped.parameters(), ref.parameters())
    ):
        mean_grad = sum(grads[idx] for grads in per_rank_grads) / world
        rows = r_param.shape[0]
        lo, hi = rank * (rows // world), (rank + 1) * (rows // world)
        expected = mean_grad[lo:hi]
        assert w_param.grad is not None, f"rank {rank} param {idx} has no grad"
        diff = (w_param.grad - expected).abs().max().item()
        assert diff < 1e-5, f"rank {rank} param {idx} grad diff {diff}"

    opt.step()
    dist.barrier()

    # parameters stay in sync across ranks: shards hold different rows by
    # design, so the check evaluates one shared input on every rank and
    # requires identical losses
    common = tp.full((12, 16), 0.5)
    with tp.no_grad():
        losses = [tp.zeros(()) for _ in range(world)]
        dist.all_gather(losses, wrapped(common).pow(2).sum().detach())
    vals = [float(v) for v in losses]
    assert max(vals) - min(vals) < 1e-5, f"parameters desynced (losses {vals})"

    # a second iteration exercises the recycled all-gather storages again
    opt.zero_grad()
    x2 = tp.randn(6, 16)
    loss2 = wrapped(x2).pow(2).sum()
    loss2.backward()
    opt.step()
    dist.barrier()
    dist.destroy_process_group()
    print(f"RANK{rank}OK")


class TestFSDPTwoRankGloo(unittest.TestCase):
    def test_two_rank_training(self):
        import multiprocessing as mp
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_fsdp_two_rank_body, args=(rank, 2, port))
            for rank in range(2)
        ]
        for p in procs:
            p.start()
        try:
            codes = [p.join(timeout=180) or p.exitcode for p in procs]
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
        self.assertEqual(codes, [0, 0], f"two-rank FSDP failed: {codes}")


_FSDP_NCCL_SCRIPT = r'''
import sys

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed.fsdp import FullyShardedDataParallel
from tensorplay.distributed.fsdp._fully_shard._fsdp_param import ShardedState

rank, world, store_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
tp.cuda.set_device(0)
dev = tp.device("cuda:0")
dist.init_process_group("nccl", init_method=f"file://{store_path}", rank=rank,
                        world_size=world)

class Tiny(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = tp.nn.Linear(16, 32)
        self.fc2 = tp.nn.Linear(32, 4)
    def forward(self, x):
        return self.fc2(tp.nn.functional.relu(self.fc1(x)))

tp.manual_seed(1234)
ref = Tiny().to(dev)
tp.manual_seed(1234)
model = Tiny().to(dev)
wrapped = FullyShardedDataParallel(model, device_id=0)

state = wrapped.module._get_fsdp_state()
groups = state._all_param_groups()
assert all(p._state == ShardedState.SHARDED
           for g in groups for p in g.fsdp_params), "not sharded after init"

x = tp.randn(32, 16, device=dev)
loss = wrapped(x).pow(2).sum()
ref_loss = ref(x).pow(2).sum()
assert abs(float(loss) - float(ref_loss)) < 1e-3, (float(loss), float(ref_loss))
loss.backward()
ref_loss.backward()
for w_param, r_param in zip(wrapped.parameters(), ref.parameters()):
    assert r_param.grad is not None
    tp.testing.assert_close(w_param.grad, r_param.grad)

opt = tp.optim.SGD(wrapped.parameters(), lr=0.1)
ref_opt = tp.optim.SGD(ref.parameters(), lr=0.1)
opt.step()
ref_opt.step()
for w_param, r_param in zip(wrapped.parameters(), ref.parameters()):
    tp.testing.assert_close(w_param, r_param)

assert all(p._state == ShardedState.SHARDED
           for g in groups for p in g.fsdp_params), "not resharded after bwd"

# several more iterations through recycled unshard storages, tracked by the
# reference model so the final full-parameter comparison stays meaningful
for step in range(3):
    opt.zero_grad()
    ref_opt.zero_grad()
    xi = tp.randn(16, 16, device=dev)
    wrapped(xi).pow(2).mean().backward()
    ref(xi).pow(2).mean().backward()
    opt.step()
    ref_opt.step()

with FullyShardedDataParallel.summon_full_params(wrapped.module):
    for (n, a), b in zip(wrapped.module.named_parameters(), ref.parameters()):
        tp.testing.assert_close(a, b)

from tensorplay.distributed import checkpoint as dcp
from tensorplay.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict)
model_sd, optim_sd = get_state_dict(wrapped, opt)
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    ckpt = os.path.join(tmp, "fsdp.dcp")
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt)
    tp.manual_seed(5)
    fresh = Tiny().to(dev)
    fresh_wrapped = FullyShardedDataParallel(fresh, device_id=0)
    fresh_opt = tp.optim.SGD(fresh_wrapped.parameters(), lr=0.1)
    fresh_msd, fresh_osd = get_state_dict(fresh_wrapped, fresh_opt)
    dcp.load({"model": fresh_msd, "optim": fresh_osd}, checkpoint_id=ckpt)
    set_state_dict(fresh_wrapped, fresh_opt, model_state_dict=fresh_msd,
                   optim_state_dict=fresh_osd)
    for a, b in zip(wrapped.parameters(), fresh_wrapped.parameters()):
        tp.testing.assert_close(a, b)

dist.barrier()
dist.destroy_process_group()
print(f"RANK{rank}OK")
'''


class TestFSDPSingleRankNCCL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.gpu_count = tp.cuda.device_count()
        except Exception:
            cls.gpu_count = 0

    def _run_ranks(self, world):
        fd, store_path = tempfile.mkstemp(prefix="tp_fsdp_rendezvous_")
        os.close(fd)
        os.unlink(store_path)
        try:
            procs = []
            for rank in range(world):
                procs.append(subprocess.Popen(
                    [sys.executable, "-c", _FSDP_NCCL_SCRIPT,
                     str(rank), str(world), store_path],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
            outputs = []
            for p in procs:
                out, _ = p.communicate(timeout=300)
                outputs.append(out.decode())
            for rank, out in enumerate(outputs):
                self.assertIn(f"RANK{rank}OK", out, f"rank {rank} failed:\n{out}")
        finally:
            if os.path.exists(store_path):
                os.unlink(store_path)

    @unittest.skipUnless(tp.cuda.is_available(), "CUDA not available")
    def test_single_rank_nccl_training(self):
        self._run_ranks(world=1)


if __name__ == "__main__":
    unittest.main()
