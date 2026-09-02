"""Multi-rank coverage for distributed tensor layouts and mesh groups."""

import os
import socket
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from test_distributed_gloo import _env_for, _spawn_ranks


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import Replicate, Shard
    from tensorplay.distributed.tensor import distribute_tensor, zeros

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh(
            "cpu", (1, 2), mesh_dim_names=("dp", "tp")
        )
        assert mesh.get_coordinate() == (0, rank)
        assert mesh.get_group("dp").size() == 1
        tp_group = mesh.get_group("tp")
        assert tp_group.size() == world_size
        assert mesh.get_local_rank("tp") == rank

        value = tp.arange(8, dtype=tp.float32).reshape(4, 2)
        sharded = distribute_tensor(
            value,
            mesh,
            [Replicate(), Shard(0)],
        )
        expected_local = value[rank * 2 : (rank + 1) * 2]
        assert sharded.shape == (4, 2)
        assert tuple(sharded.to_local().shape) == (2, 2)
        assert sharded.to_local().tolist() == expected_local.tolist()
        assert sharded.full_tensor().tolist() == value.tolist()

        replicated = sharded.redistribute(
            placements=[Replicate(), Replicate()]
        )
        assert replicated.to_local().tolist() == value.tolist()
        restored = replicated.redistribute(
            placements=[Replicate(), Shard(0)]
        )
        assert restored.to_local().tolist() == expected_local.tolist()

        created = zeros(
            4,
            2,
            dtype=tp.float32,
            device_mesh=mesh,
            placements=[Replicate(), Shard(0)],
        )
        assert tuple(created.to_local().shape) == (2, 2)
    finally:
        dist.destroy_process_group()


def _ops_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import DTensor, Shard, distribute_tensor

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        value = tp.arange(8, dtype=tp.float32).reshape(4, 2)
        sharded = distribute_tensor(value, mesh, [Shard(0)])

        pointwise = tp.add(sharded, 1)
        assert isinstance(pointwise, DTensor)
        assert pointwise.shape == (4, 2)
        assert pointwise.full_tensor().tolist() == (value + 1).tolist()

        transposed = sharded.transpose(0, 1)
        assert isinstance(transposed, DTensor)
        assert transposed.shape == (2, 4)
        assert transposed.full_tensor().tolist() == value.transpose(0, 1).tolist()

        reduced = sharded.sum(0)
        assert isinstance(reduced, DTensor)
        assert reduced.full_tensor().tolist() == value.sum(0).tolist()
    finally:
        dist.destroy_process_group()


class TestDistributedTensorNative(unittest.TestCase):
    def test_mesh_and_redistribution(self):
        _spawn_ranks(_body, 2, args=(_free_port(),))

    def test_dtensor_operation_dispatch(self):
        _spawn_ranks(_ops_body, 2, args=(_free_port(),))


if __name__ == "__main__":
    unittest.main()
