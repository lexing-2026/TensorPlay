"""Native process-group collective coverage for CPU backends."""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from test_distributed_gloo import _env_for, _spawn_ranks


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_raises(exception_type, function):
    try:
        function()
    except exception_type:
        return
    except Exception as error:
        raise AssertionError(
            f"expected {exception_type.__name__}, got {type(error).__name__}"
        ) from error
    raise AssertionError(f"expected {exception_type.__name__}")


def _expected_rows(world_size, start):
    return [[float(start + rank * 10), float(start + rank * 10 + 1)]
            for rank in range(world_size)]


def _expected_flat(world_size, start):
    return [value for row in _expected_rows(world_size, start) for value in row]


def _expected_reduce_scatter(rank, world_size, process_offset=0):
    process_sum = sum(range(world_size))
    return [
        float(
            world_size * (rank * 2 + column)
            + world_size * process_offset
            + 100 * process_sum
        )
        for column in range(2)
    ]


def _initialize(backend, rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist

    if port is not None:
        _env_for(rank, world_size, port)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return tp, dist


def _exercise_collectives(
    rank,
    world_size,
    backend,
    port=None,
    include_coalesced=True,
):
    tp, dist = _initialize(backend, rank, world_size, port)
    try:
        input_tensor = tp.arange(2, dtype=tp.float32)

        _assert_raises(
            RuntimeError,
            lambda: dist.all_gather_single(
                tp.zeros((world_size * 2 - 1,), dtype=tp.float32),
                input_tensor,
            ),
        )
        _assert_raises(
            RuntimeError,
            lambda: dist.reduce_scatter_single(
                tp.zeros((2,), dtype=tp.float32),
                tp.zeros((world_size * 2 - 1,), dtype=tp.float32),
            ),
        )
        _assert_raises(
            ValueError,
            lambda: dist.gather_single(input_tensor, None, dst=world_size),
        )

        if include_coalesced:
            _assert_raises(
                RuntimeError,
                lambda: dist.all_gather_coalesced(
                    [[]], [input_tensor]
                ),
            )

        gathered = tp.zeros((world_size, 2), dtype=tp.float32)
        gathered_input = tp.arange(2, dtype=tp.float32) + float(rank * 10)
        self_result = dist.all_gather_single(gathered, gathered_input)
        assert self_result is None
        assert gathered.tolist() == _expected_rows(world_size, 0)

        gathered_flat = tp.zeros((world_size * 2,), dtype=tp.float32)
        async_input = tp.arange(2, dtype=tp.float32) + float(100 + rank * 10)
        work = dist.all_gather_single(
            gathered_flat, async_input, async_op=True
        )
        assert work.wait() is True
        assert work.is_completed()
        assert gathered_flat.tolist() == _expected_flat(world_size, 100)

        root = world_size - 1
        gather_input = tp.arange(2, dtype=tp.float32) + float(rank * 10)
        gather_output = (
            tp.zeros((world_size, 2), dtype=tp.float32) if rank == root else None
        )
        self_result = dist.gather_single(gather_input, gather_output, dst=root)
        assert self_result is None
        if rank == root:
            assert gather_output.tolist() == _expected_rows(world_size, 0)
        else:
            assert gather_output is None

        reduce_input = (
            tp.arange(world_size * 2, dtype=tp.float32)
            .reshape((world_size, 2))
            + float(rank * 100)
        )
        reduce_output = tp.zeros((2,), dtype=tp.float32)
        self_result = dist.reduce_scatter_single(reduce_output, reduce_input)
        assert self_result is None
        assert reduce_output.tolist() == _expected_reduce_scatter(rank, world_size)

        future_input = (
            tp.arange(world_size * 2, dtype=tp.float32)
            .reshape((world_size, 2))
            + float(1000 + rank * 100)
        )
        future_output = tp.zeros((2,), dtype=tp.float32)
        future_work = dist.reduce_scatter_single(
            future_output, future_input, async_op=True
        )
        future = future_work.get_future()
        resolved = future.wait()
        assert future.done()
        assert future_work.is_completed()
        assert len(resolved) == 1
        assert resolved[0] is future_output
        assert future_output.tolist() == _expected_reduce_scatter(
            rank, world_size, process_offset=1000
        )

        if include_coalesced:
            coalesced_inputs = [
                tp.arange(2, dtype=tp.float32) + float(rank),
                tp.arange(3, dtype=tp.float32) + float(10 + rank),
            ]
            coalesced_outputs = [
                [tp.zeros((2,), dtype=tp.float32) for _ in range(world_size)],
                [tp.zeros((3,), dtype=tp.float32) for _ in range(world_size)],
            ]
            self_result = dist.all_gather_coalesced(
                coalesced_outputs, coalesced_inputs
            )
            assert self_result is None
            for source_rank in range(world_size):
                assert coalesced_outputs[0][source_rank].tolist() == [
                    float(source_rank),
                    float(source_rank + 1),
                ]
                assert coalesced_outputs[1][source_rank].tolist() == [
                    float(10 + source_rank),
                    float(11 + source_rank),
                    float(12 + source_rank),
                ]
    finally:
        dist.destroy_process_group()


def _gloo_body(rank, world_size, port):
    _exercise_collectives(rank, world_size, "gloo", port)


def _mpi_test_available():
    if shutil.which("mpirun") is None:
        return False
    try:
        import tensorplay.distributed as dist
    except Exception:
        return False
    return bool(dist.is_mpi_available())


class TestGlooNativeCollectives(unittest.TestCase):
    def _run(self, world_size):
        _spawn_ranks(_gloo_body, world_size, args=(_free_port(),))

    def test_single_rank_collectives(self):
        self._run(1)

    def test_multi_rank_collectives(self):
        self._run(2)


@unittest.skipUnless(_mpi_test_available(), "MPI launcher or backend unavailable")
class TestMPINativeCollectives(unittest.TestCase):
    def test_multi_rank_collectives(self):
        world_size = 2
        with tempfile.TemporaryDirectory(
            prefix="tp_native_collectives_mpi_"
        ) as directory:
            script_path = os.path.join(directory, "rank.py")
            with open(script_path, "w", encoding="utf-8") as script_file:
                script_file.write(
                    f"import os, sys\n"
                    f"sys.path.insert(0, {TEST_DIR!r})\n"
                    "from test_distributed_native_collectives import "
                    "_exercise_collectives\n"
                    "rank = int(os.environ.get(\"OMPI_COMM_WORLD_RANK\", "
                    "os.environ.get(\"PMI_RANK\", \"0\")))\n"
                    "world_size = int(os.environ.get(\"OMPI_COMM_WORLD_SIZE\", "
                    "os.environ.get(\"PMI_SIZE\", \"2\")))\n"
                    "_exercise_collectives(rank, world_size, \"mpi\", "
                    "include_coalesced=False)\n"
                    "print(f\"mpi rank {rank} native collectives OK\", flush=True)\n"
                )
            environment = dict(os.environ)
            environment["MASTER_ADDR"] = "127.0.0.1"
            environment["MASTER_PORT"] = str(_free_port())
            result = subprocess.run(
                [
                    "mpirun",
                    "--allow-run-as-root",
                    "-np",
                    str(world_size),
                    sys.executable,
                    script_path,
                ],
                capture_output=True,
                text=True,
                timeout=180,
                env=environment,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        for rank in range(world_size):
            self.assertIn(f"mpi rank {rank} native collectives OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
