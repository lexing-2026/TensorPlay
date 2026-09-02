"""Functional collective coverage for the native process-group path."""

import os
import socket
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from test_distributed_gloo import _env_for, _spawn_ranks


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _body(rank, world_size, port):
    import datetime
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed import _functional_collectives as funcol

    _env_for(rank, world_size, port)
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        assert dist.get_backend_config() == "cpu:gloo"
        assert dist.get_default_backend_for_device("cpu") == "gloo"
        dist.set_timeout(datetime.timedelta(seconds=30))
        with dist.record_comm("functional-test"):
            marked_work = dist.all_reduce(tp.ones((1,), dtype=tp.float32),
                                          async_op=True)
        assert getattr(marked_work, "_profiling_name", None) == "functional-test"
        marked_work.wait()

        value = tp.full((2,), float(rank + 1), dtype=tp.float32,
                        requires_grad=True)
        reduced = funcol.all_reduce(value)
        assert reduced.tolist() == [3.0, 3.0]
        (reduced.sum()).backward()
        assert value.grad.tolist() == [2.0, 2.0]

        complex_value = tp.tensor(
            [complex(rank + 1, rank + 2)], dtype=tp.complex64
        )
        complex_reduced = funcol.all_reduce(complex_value)
        assert complex_reduced.tolist() == [3 + 5j]

        gathered_input = tp.full((2,), float(rank), dtype=tp.float32)
        gathered = funcol.all_gather_single(gathered_input)
        assert gathered.tolist() == [0.0, 0.0, 1.0, 1.0]

        gathered_2d = funcol.all_gather_single(
            tp.tensor([[rank * 2.0, rank * 2.0 + 1.0]]), 1
        )
        assert gathered_2d.tolist() == [[0.0, 1.0, 2.0, 3.0]]

        scattered_input = tp.arange(4, dtype=tp.float32) + float(rank * 10)
        scattered = funcol.reduce_scatter_single(scattered_input)
        expected = [
            float(sum(source * 10 + rank * 2 + column
                      for source in range(world_size)))
            for column in range(2)
        ]
        assert scattered.tolist() == expected, scattered.tolist()

        scattered_2d_input = tp.arange(8, dtype=tp.float32).reshape(2, 4)
        scattered_2d = funcol.reduce_scatter_single(
            scattered_2d_input + float(rank * 10), "sum", 1
        )
        assert scattered_2d.tolist() == (
            [[10.0, 12.0], [18.0, 20.0]] if rank == 0
            else [[14.0, 16.0], [22.0, 24.0]]
        )

        exchanged_input = tp.tensor([[rank * 10.0], [rank * 10.0 + 1.0]])
        exchanged = funcol.all_to_all_single(exchanged_input)
        expected = [float(source * 10 + rank) for source in range(world_size)]
        assert exchanged.tolist() == [[value] for value in expected]

        variable_input_sizes = ([1, 2] if rank == 0 else [2, 1])
        variable_base = tp.arange(3, dtype=tp.float32).reshape(3, 1)
        variable_base.requires_grad = True
        variable_input = variable_base + float(rank * 10)
        variable_output_sizes = variable_input_sizes
        variable = funcol.all_to_all_single(
            variable_input, variable_output_sizes, variable_input_sizes
        )
        expected_variable = (
            [[0.0], [10.0], [11.0]] if rank == 0
            else [[1.0], [2.0], [12.0]]
        )
        assert variable.tolist() == expected_variable
        variable.sum().backward()
        assert variable_base.grad.tolist() == [[1.0], [1.0], [1.0]]

        broadcast_input = tp.tensor([float(rank + 1)], requires_grad=True)
        broadcast_output = funcol.broadcast(broadcast_input, src=0)
        assert broadcast_output.tolist() == [1.0]
        broadcast_output.sum().backward()
        assert broadcast_input.grad.tolist() == ([2.0] if rank == 0 else [0.0])

        coalesced = funcol.all_reduce_coalesced([
            tp.full((1,), float(rank + 1), dtype=tp.float32),
            tp.full((2,), float(rank + 2), dtype=tp.float32),
        ])
        assert [tensor.tolist() for tensor in coalesced] == [
            [3.0], [5.0, 5.0]
        ]

        gathered_coalesced = funcol.all_gather_single_coalesced([
            tp.full((1,), float(rank), dtype=tp.float32),
            tp.full((2,), float(rank + 10), dtype=tp.float32),
        ])
        assert [tensor.tolist() for tensor in gathered_coalesced] == [
            [0.0, 1.0], [10.0, 10.0, 11.0, 11.0]
        ]

        scattered_coalesced = funcol.reduce_scatter_single_coalesced(
            [
                tp.arange(4, dtype=tp.float32) + float(rank * 10),
                tp.arange(8, dtype=tp.float32) + float(rank * 20),
            ],
            "sum",
            [0, 0],
        )
        assert [tensor.tolist() for tensor in scattered_coalesced] == [
            ([10.0, 12.0] if rank == 0 else [14.0, 16.0]),
            ([20.0, 22.0, 24.0, 26.0] if rank == 0
             else [28.0, 30.0, 32.0, 34.0]),
        ]
    finally:
        dist.destroy_process_group()


def _split_body(rank, world_size, port):
    import tensorplay.distributed as dist

    _env_for(rank, world_size, port)
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        subgroup = dist.split_group(split_ranks=[[0], [1]])
        assert subgroup.size() == 1
        assert subgroup.rank() == 0
        dist.destroy_process_group(subgroup)
    finally:
        dist.destroy_process_group()


def _shrink_body(rank, world_size, port):
    import tensorplay.distributed as dist

    _env_for(rank, world_size, port)
    dist.init_process_group(backend="gloo", init_method="env://")
    if rank == 0:
        subgroup = dist.shrink_group([1])
        assert subgroup.size() == 1
        assert dist.get_rank() == 0
        assert dist.get_world_size() == 1
        dist.destroy_process_group()
    else:
        dist.destroy_process_group()


class TestFunctionalCollectives(unittest.TestCase):
    def test_two_rank_native_path(self):
        _spawn_ranks(_body, 2, args=(_free_port(),))

    def test_split_group(self):
        _spawn_ranks(_split_body, 2, args=(_free_port(),))

    def test_shrink_group(self):
        _spawn_ranks(_shrink_body, 2, args=(_free_port(),))


if __name__ == "__main__":
    unittest.main()
