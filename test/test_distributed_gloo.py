"""CPU-backend process-group tests: gloo over two spawned ranks, and MPI
under mpirun when a launcher is available.

Gloo test bodies live at module level so ``spawn`` can pickle them."""

import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spawn_ranks(target, world_size, args=()):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=target, args=(rank, world_size) + args)
        p.start()
        procs.append(p)
    codes = [p.join(timeout=120) or p.exitcode for p in procs]
    for code in codes:
        if code != 0:
            raise AssertionError(f"rank failed, exitcodes={codes}")


def _env_for(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def _init(rank, port, backend="gloo"):
    import tensorplay as tp
    import tensorplay.distributed as dist

    _env_for(rank, 2, port)
    dist.init_process_group(backend=backend, init_method="env://")
    return tp, dist


# --- gloo test bodies (module level for spawn pickling) --------------------

def body_init_and_metadata(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        assert dist.get_rank() == rank
        assert dist.get_world_size() == size
        assert dist.get_backend() == "gloo"
        assert dist.is_initialized()
    finally:
        dist.destroy_process_group()


def body_all_reduce(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        t = tp.full((3,), float(rank + 1), dtype=tp.float32)
        dist.all_reduce(t)
        assert t.tolist() == [3.0] * 3, t.tolist()
    finally:
        dist.destroy_process_group()


def body_all_reduce_avg(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        t = tp.full((2,), float(2 * (rank + 1)), dtype=tp.float32)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        assert t.tolist() == [3.0] * 2, t.tolist()
    finally:
        dist.destroy_process_group()


def body_broadcast(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        t = tp.full((2,), 99.0 if rank == 0 else 0.0, dtype=tp.float32)
        dist.broadcast(t, src=0)
        assert t.tolist() == [99.0] * 2, t.tolist()
    finally:
        dist.destroy_process_group()


def body_all_gather(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        tensor_list = [tp.zeros((2,)) for _ in range(size)]
        dist.all_gather(tensor_list, tp.full((2,), float(rank), dtype=tp.float32))
        for i in range(size):
            assert tensor_list[i].tolist() == [float(i)] * 2
    finally:
        dist.destroy_process_group()


def body_gather_scatter(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        gather_list = [tp.zeros((2,)) for _ in range(size)]
        dist.gather(tp.full((2,), float(rank), dtype=tp.float32), gather_list, dst=0)
        if rank == 0:
            for i in range(size):
                assert gather_list[i].tolist() == [float(i)] * 2
        out = tp.zeros((2,))
        scatter_list = [tp.full((2,), float(i), dtype=tp.float32) for i in range(size)]
        dist.scatter(out, scatter_list if rank == 0 else None, src=0)
        assert out.tolist() == [float(rank)] * 2, out.tolist()
    finally:
        dist.destroy_process_group()


def body_reduce_scatter(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        inputs = [tp.full((2,), 1.0, dtype=tp.float32) for _ in range(size)]
        out = tp.zeros((2,))
        dist.reduce_scatter(out, inputs)
        assert out.tolist() == [float(size)] * 2, out.tolist()
    finally:
        dist.destroy_process_group()


def body_all_gather_into_tensor(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        out = tp.zeros((size * 2,))
        dist.all_gather_into_tensor(out, tp.full((2,), float(rank + 1), dtype=tp.float32))
        assert out.tolist() == [
            float(i + 1) for i in range(size) for _ in range(2)], out.tolist()
    finally:
        dist.destroy_process_group()


def body_reduce_scatter_tensor(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        inp = tp.full((size * 2,), 1.0, dtype=tp.float32)
        out = tp.zeros((2,))
        dist.reduce_scatter_tensor(out, inp)
        assert out.tolist() == [float(size)] * 2, out.tolist()
    finally:
        dist.destroy_process_group()


def body_all_to_all_single(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        inp = tp.full((size * 2,), float(rank + 1), dtype=tp.float32)
        out = tp.zeros((size * 2,))
        dist.all_to_all_single(out, inp)
        assert out.tolist() == [
            float(i + 1) for i in range(size) for _ in range(2)], out.tolist()
    finally:
        dist.destroy_process_group()


def body_send_recv(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        dst = (rank + 1) % size
        src = (rank + size - 1) % size
        dist.isend(tp.full((2,), float(100 + rank), dtype=tp.float32), dst, tag=rank)
        buf = tp.zeros((2,))
        dist.recv(buf, src=src, tag=src)
        assert buf.tolist() == [float(100 + src)] * 2, buf.tolist()
    finally:
        dist.destroy_process_group()


def body_barrier_and_monitored(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        dist.barrier()
        dist.monitored_barrier(timeout=30)
    finally:
        dist.destroy_process_group()


def body_object_collectives(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        objs = dist.broadcast_object_list(
            [{"rank": 0, "payload": [1, 2, 3]}] if rank == 0 else None, src=0)
        assert objs[0]["rank"] == 0
        assert objs[0]["payload"] == [1, 2, 3]

        out = [None] * size
        dist.all_gather_object(out, {"me": rank})
        assert [o["me"] for o in out] == list(range(size))
    finally:
        dist.destroy_process_group()


def body_new_group_subsets(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        sub = dist.new_group(ranks=[0])
        if rank == 0:
            t = tp.full((1,), 7.0, dtype=tp.float32)
            dist.all_reduce(t, group=sub)
            assert t.tolist() == [7.0], t.tolist()
            assert dist.get_backend(sub) == "gloo"
    finally:
        dist.destroy_process_group()


def body_async_all_reduce(rank, size, port):
    tp, dist = _init(rank, port)
    try:
        t = tp.full((2,), float(rank + 1), dtype=tp.float32)
        work = dist.all_reduce(t, async_op=True)
        work.wait()
        assert t.tolist() == [3.0] * 2, t.tolist()
    finally:
        dist.destroy_process_group()


class TestGlooProcessGroup(unittest.TestCase):
    world_size = 2

    def _run(self, body):
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        _spawn_ranks(body, self.world_size, args=(port,))

    def test_init_and_metadata(self):
        self._run(body_init_and_metadata)

    def test_all_reduce(self):
        self._run(body_all_reduce)

    def test_all_reduce_avg(self):
        self._run(body_all_reduce_avg)

    def test_broadcast(self):
        self._run(body_broadcast)

    def test_all_gather(self):
        self._run(body_all_gather)

    def test_gather_scatter(self):
        self._run(body_gather_scatter)

    def test_reduce_scatter(self):
        self._run(body_reduce_scatter)

    def test_all_gather_into_tensor(self):
        self._run(body_all_gather_into_tensor)

    def test_reduce_scatter_tensor(self):
        self._run(body_reduce_scatter_tensor)

    def test_all_to_all_single(self):
        self._run(body_all_to_all_single)

    def test_send_recv(self):
        self._run(body_send_recv)

    def test_barrier_and_monitored_barrier(self):
        self._run(body_barrier_and_monitored)

    def test_object_collectives(self):
        self._run(body_object_collectives)

    def test_new_group_subsets(self):
        self._run(body_new_group_subsets)

    def test_async_all_reduce(self):
        self._run(body_async_all_reduce)


@unittest.skipUnless(shutil.which("mpirun"), "mpirun not available")
class TestMPIProcessGroup(unittest.TestCase):
    """Runs a two-rank MPI job through mpirun (separate OS processes)."""

    def _script(self, body_lines):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(prefix="tp_mpi_"), "rank.py")
        with open(path, "w") as f:
            f.write(
                "import sys\n"
                "sys.path.insert(0, %r)\n" % os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
                + "import tensorplay as tp\n"
                "import tensorplay.distributed as dist\n"
                "dist.init_process_group(backend='mpi')\n"
                + body_lines
                + "dist.destroy_process_group()\n"
            )
        return path

    def test_all_reduce(self):
        import socket
        import subprocess

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        path = self._script(
            "t = tp.full((3,), float(dist.get_rank() + 1), dtype=tp.float32)\n"
            "dist.all_reduce(t)\n"
            "assert t.tolist() == [3.0] * 3, t.tolist()\n"
            "print('rank', dist.get_rank(), 'OK', flush=True)\n"
        )
        env = dict(os.environ)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(free_port)
        result = subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", "2",
             sys.executable, path],
            capture_output=True, text=True, timeout=180, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("rank 0 OK", result.stdout)
        self.assertIn("rank 1 OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
