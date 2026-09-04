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
    from tensorplay.distributed.tensor import (
        DTensor,
        Replicate,
        Shard,
        distribute_tensor,
    )

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

        viewed = sharded.view(2, 4)
        assert isinstance(viewed, DTensor)
        assert viewed.shape == (2, 4)
        assert viewed.full_tensor().tolist() == value.view(2, 4).tolist()

        functional_view = tp.view(sharded, (2, 4))
        assert isinstance(functional_view, DTensor)
        assert functional_view.shape == (2, 4)
        assert functional_view.full_tensor().tolist() == value.view(2, 4).tolist()

        reshaped = tp.reshape(sharded, (2, 4))
        assert isinstance(reshaped, DTensor)
        assert reshaped.shape == (2, 4)
        assert reshaped.full_tensor().tolist() == value.reshape(2, 4).tolist()

        complex_view = tp.view_as_complex(sharded)
        assert isinstance(complex_view, DTensor)
        assert complex_view.shape == (4,)
        assert complex_view.full_tensor().tolist() == tp.view_as_complex(value).tolist()

        real_view = tp.view_as_real(complex_view)
        assert isinstance(real_view, DTensor)
        assert real_view.shape == (4, 2)
        assert real_view.full_tensor().tolist() == value.tolist()

        expanded = sharded.unsqueeze(1)
        assert isinstance(expanded, DTensor)
        assert expanded.shape == (4, 1, 2)
        assert expanded.full_tensor().tolist() == value.unsqueeze(1).tolist()

        reduced = sharded.sum(0)
        assert isinstance(reduced, DTensor)
        assert reduced.full_tensor().tolist() == value.sum(0).tolist()

    finally:
        dist.destroy_process_group()


def _nested_redistribution_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import (
        Partial,
        Replicate,
        Shard,
        distribute_tensor,
    )
    from tensorplay.distributed.tensor.experimental._context_parallel._attention import (
        _context_parallel_shard,
    )
    from tensorplay.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
    from tensorplay.distributed.tensor.placement_types import _StridedShard

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh(
            "cpu", (2, 2), mesh_dim_names=("dp", "tp")
        )
        uneven_cases = (
            ((10, 5), (Shard(0), Shard(0))),
            ((9, 5), (Shard(0), Shard(0))),
            ((3, 5), (Shard(0), Shard(0))),
            ((1, 3), (Shard(0), Shard(1))),
        )
        for shape, placements in uneven_cases:
            value = tp.arange(shape[0] * shape[1], dtype=tp.float32).reshape(*shape)
            sharded = distribute_tensor(value, mesh, placements)
            assert sharded.full_tensor().tolist() == value.tolist()

        value = tp.arange(64, dtype=tp.float32).reshape(8, 8)
        nested = distribute_tensor(value, mesh, [Shard(0), Shard(0)])
        mesh._flatten("dp_tp")
        replicated = nested.redistribute(
            placements=[Replicate(), Replicate()]
        )
        assert replicated.to_local().tolist() == value.tolist()

        changed = nested.redistribute(placements=[Shard(1), Shard(1)])
        assert changed.full_tensor().tolist() == value.tolist()
        restored = replicated.redistribute(placements=[Shard(0), Shard(0)])
        assert restored.full_tensor().tolist() == value.tolist()

        partial = replicated.redistribute(
            placements=[Partial("sum"), Partial("sum")]
        )
        assert partial.full_tensor().tolist() == value.tolist()
        for reduce_op in ("avg", "min", "max"):
            reduced = replicated.redistribute(
                placements=[Partial(reduce_op), Partial(reduce_op)]
            )
            assert reduced.full_tensor().tolist() == value.tolist()
        sharded_partial = nested.redistribute(
            placements=[Partial("sum"), Partial("sum")]
        )
        assert sharded_partial.full_tensor().tolist() == value.tolist()

        async_replicated = nested.redistribute(
            placements=[Replicate(), Replicate()], async_op=True
        )
        assert async_replicated.to_local().tolist() == value.tolist()
        async_partial = replicated.redistribute(
            placements=[Partial("sum"), Partial("sum")], async_op=True
        )
        assert async_partial.full_tensor().tolist() == value.tolist()
        async_sharded = partial.redistribute(
            placements=[Shard(0), Shard(0)], async_op=True
        )
        assert async_sharded.full_tensor().tolist() == value.tolist()

        strided = distribute_tensor(
            value, mesh, [_StridedShard(0, split_factor=2), Replicate()]
        )
        assert strided.full_tensor().tolist() == value.tolist()
        async_strided = strided.redistribute(
            placements=[Replicate(), Replicate()], async_op=True
        )
        assert async_strided.to_local().tolist() == value.tolist()

        cp_mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("cp",))

        def causal_mask(batch, head, query, key):
            return query >= key

        block_mask = create_block_mask(causal_mask, 1, 1, 512, 512)
        local_block_mask = _context_parallel_shard(
            cp_mesh, [block_mask], [2]
        )[0]
        assert isinstance(local_block_mask, BlockMask)
        assert local_block_mask.shape == (1, 1, 128, 512)
        output = flex_attention(
            tp.randn((1, 1, 128, 4)),
            tp.randn((1, 1, 512, 4)),
            tp.randn((1, 1, 512, 4)),
            block_mask=local_block_mask,
        )
        assert output.shape == (1, 1, 128, 4)
    finally:
        dist.destroy_process_group()


def _embedding_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import Replicate, Shard, distribute_tensor

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        weight = tp.arange(24, dtype=tp.float32).reshape(6, 4)
        indices = tp.tensor([[1, 3], [4, 0]], dtype=tp.int64)
        expected = tp.embedding(weight, indices)
        cases = (
            ([Shard(1)], [Replicate()]),
            ([Replicate()], [Shard(0)]),
            ([Shard(0)], [Replicate()]),
        )
        for weight_placements, index_placements in cases:
            distributed_weight = distribute_tensor(
                weight, mesh, weight_placements
            )
            distributed_indices = distribute_tensor(
                indices, mesh, index_placements
            )
            output = tp.embedding(distributed_weight, distributed_indices)
            assert output.shape == expected.shape
            assert output.full_tensor().tolist() == expected.tolist()
    finally:
        dist.destroy_process_group()


def _alltoall_autograd_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor._collective_utils import shard_dim_alltoall

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        value = tp.arange(16, dtype=tp.float32).reshape(4, 4)
        value.requires_grad_(True)
        output = shard_dim_alltoall(value, 0, 1, mesh, 0)
        assert tuple(output.shape) == (8, 2)
        (output * output).sum().backward()
        assert value.grad is not None
        assert tuple(value.grad.shape) == (4, 4)
    finally:
        dist.destroy_process_group()


def _tp_convolution_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import Shard, distribute_tensor
    from tensorplay.distributed.tensor._tp_conv import (
        _ring_send_recv_aggregate,
        _ring_send_recv_construct,
        tp_convolution,
    )

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        value = tp.arange(8, dtype=tp.float32).reshape(1, 1, 1, 8)
        local = distribute_tensor(value, mesh, [Shard(3)]).to_local()
        reconstructed = _ring_send_recv_construct(
            local, 1, 1, (rank - 1 + world_size) % world_size,
            (rank + 1) % world_size, rank, world_size,
        )
        expected_input = (
            value[:, :, :, :5] if rank == 0 else value[:, :, :, 3:]
        )
        assert reconstructed.tolist() == expected_input.tolist()

        gradient = tp.full((1, 1, 1, 5), float(rank + 1), dtype=tp.float32)
        aggregated = _ring_send_recv_aggregate(
            gradient, 1, 1, (rank - 1 + world_size) % world_size,
            (rank + 1) % world_size, rank, world_size,
        )
        expected_gradient = (
            [[[[1.0, 1.0, 1.0, 3.0]]]]
            if rank == 0
            else [[[[3.0, 2.0, 2.0, 2.0]]]]
        )
        assert aggregated.tolist() == expected_gradient

        weight = tp.ones((1, 1, 1, 3), dtype=tp.float32)
        args = (
            local,
            weight,
            None,
            [1, 1],
            [0, 1],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        result = tp_convolution(tp.functional.convolution, args, {})
        expected = tp.functional.convolution(
            value, weight, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1
        )
        expected = expected[:, :, :, :4] if rank == 0 else expected[:, :, :, 4:]
        assert result.tolist() == expected.tolist()
    finally:
        dist.destroy_process_group()


def _loss_parallel_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    import tensorplay.nn.functional as F
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import DTensor, Shard
    from tensorplay.distributed.tensor.parallel.loss import loss_parallel

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        data = [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ]
        target = tp.tensor([3, 0, 1, 2], dtype=tp.int64)
        full = tp.tensor(data, dtype=tp.float32, requires_grad=True)
        expected = F.cross_entropy(full, target)
        expected.backward()
        local = tp.tensor(
            [row[rank * 2 : (rank + 1) * 2] for row in data],
            dtype=tp.float32,
            requires_grad=True,
        )
        value = DTensor.from_local(
            local,
            mesh,
            [Shard(1)],
            run_check=False,
            shape=(4, 4),
            stride=(4, 1),
        )
        with loss_parallel():
            result = F.cross_entropy(value, target)
            result.backward()
        expected_grad = full.grad[:, rank * 2 : (rank + 1) * 2]
        error = max(
            abs(actual - wanted)
            for actual_row, wanted_row in zip(
                local.grad.tolist(), expected_grad.tolist()
            )
            for actual, wanted in zip(actual_row, wanted_row)
        )
        assert abs(result.to_local().item() - expected.item()) < 1e-5
        assert error < 1e-5
    finally:
        dist.destroy_process_group()


def _input_reshard_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    import tensorplay.nn as nn
    from tensorplay.autograd.function import Function
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import DTensor, Replicate, Shard
    from tensorplay.distributed.tensor.parallel.input_reshard import (
        _pack_hook_tp,
        _unpack_hook_tp,
        input_reshard,
    )

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        full = tp.arange(8, dtype=tp.float32).reshape(2, 4)
        replicated = DTensor.from_local(
            full,
            mesh,
            [Replicate()],
            run_check=False,
            shape=(2, 4),
            stride=(4, 1),
        )
        packed = _pack_hook_tp(mesh, 1, replicated)
        restored = _unpack_hook_tp(mesh, 1, packed)
        plain_packed = _pack_hook_tp(mesh, 1, full)
        plain_restored = _unpack_hook_tp(mesh, 1, plain_packed)
        assert packed.placements == (Shard(1),)
        assert restored.placements == (Replicate(),)
        assert plain_restored.tolist() == full.tolist()

        class SaveInput(Function):
            @staticmethod
            def forward(context, value):
                context.save_for_backward(value)
                return value * value

            @staticmethod
            def backward(context, grad_output):
                (value,) = context.saved_tensors
                return grad_output * (value + value)

        class Module(nn.Module):
            def forward(self, value):
                return SaveInput.apply(value)

        module = input_reshard(Module(), mesh, 1)
        value = tp.arange(8, dtype=tp.float32).reshape(2, 4)
        value.requires_grad_(True)
        module(value).sum().backward()
        assert value.grad.tolist() == (value + value).tolist()
    finally:
        dist.destroy_process_group()


def _experimental_mapping_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    from tensorplay.distributed.device_mesh import init_device_mesh
    from tensorplay.distributed.tensor import (
        DTensor,
        Replicate,
        Shard,
        distribute_tensor,
    )
    from tensorplay.distributed.tensor.experimental import (
        local_map,
        register_sharding,
    )

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        full = tp.arange(8, dtype=tp.float32).reshape(4, 2)
        value = distribute_tensor(full, mesh, [Shard(0)])

        @local_map(
            out_placements=([Shard(0)], None),
            in_placements=([Shard(0)], None),
            device_mesh=mesh,
        )
        def local_pair(local_value, scalar):
            return local_value + scalar, 7

        pair = local_pair(value, 1)
        assert isinstance(pair[0], DTensor)
        assert pair[0].placements == (Shard(0),)
        assert pair[0].full_tensor().tolist() == (full + 1).tolist()
        assert pair[1] == 7

        replicated = distribute_tensor(full, mesh, [Replicate()])
        redistributed = local_map(
            lambda local_value: local_value * 2,
            out_placements=[Shard(0)],
            in_placements=([Shard(0)],),
            device_mesh=mesh,
            redistribute_inputs=True,
        )(replicated)
        assert redistributed.full_tensor().tolist() == (full * 2).tolist()

        def custom_increment(local_value):
            return local_value + 1

        @register_sharding([custom_increment])
        def custom_increment_rule(local_value):
            return [([Shard(0)], [Shard(0)])]

        result = DTensor._op_dispatcher.dispatch(
            custom_increment, (value,), {}
        )
        assert isinstance(result, DTensor)
        assert result.placements == (Shard(0),)
        assert result.full_tensor().tolist() == (full + 1).tolist()
    finally:
        dist.destroy_process_group()


def _tp_transform_body(rank, world_size, port):
    import tensorplay as tp
    import tensorplay.distributed as dist
    import tensorplay.nn as nn
    from tensorplay.distributed.tensor.experimental._tp_transform import (
        tensor_parallel_transformation,
    )
    from tensorplay.distributed.tensor.parallel.style import (
        ColwiseParallel,
        RowwiseParallel,
    )
    from tensorplay.export import export

    _env_for(rank, world_size, port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 5)
                self.register_buffer(
                    "boundaries", tp.tensor([-1.0, 0.0, 1.0])
                )

            def forward(self, value):
                return tp.searchsorted(self.boundaries, self.fc(value))

        class MLPModel(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                self.mlp0 = nn.Sequential(
                    nn.Linear(6, 18),
                    nn.ReLU(),
                    nn.Linear(18, 6, bias=bias),
                )
                self.mlp1 = nn.Sequential(
                    nn.Linear(6, 18),
                    nn.ReLU(),
                    nn.Linear(18, 6, bias=bias),
                )

            def forward(self, value):
                value = tp.chunk(value, 2, dim=1)[0]
                value = self.mlp0(value)
                value = self.mlp1(value)
                return value + tp.ones_like(value)

        cases = (
            (
                LinearModel,
                (7, 3),
                {"fc": ColwiseParallel},
            ),
            (
                lambda: MLPModel(True),
                (10, 12),
                {
                    "mlp0.0": ColwiseParallel,
                    "mlp0.2": RowwiseParallel,
                    "mlp1.0": ColwiseParallel,
                    "mlp1.2": RowwiseParallel,
                },
            ),
            (
                lambda: MLPModel(False),
                (10, 12),
                {
                    "mlp0.0": ColwiseParallel,
                    "mlp0.2": RowwiseParallel,
                    "mlp1.0": ColwiseParallel,
                    "mlp1.2": RowwiseParallel,
                },
            ),
        )
        for model_factory, shape, strategies in cases:
            tp.manual_seed(0)
            model = model_factory()
            value = tp.randn(*shape)
            expected = model(value)
            program = tensor_parallel_transformation(
                export(model, value),
                rank,
                world_size,
                "cpu",
                strategies,
            )
            actual = program.module()(value)
            assert float((actual - expected).abs().max()) < 1e-5
    finally:
        dist.destroy_process_group()


class TestDistributedTensorNative(unittest.TestCase):
    def test_placement_chunk_and_graph_representation(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor.placement_types import (
            Partial,
            Replicate,
            Shard,
            _MaskPartial,
            _StridedShard,
            _explicit_or_backed_hint,
            _guarded_hint_int,
            _hint_proves_even_shard,
        )

        value = tp.arange(10, dtype=tp.float32)
        for count in (1, 2, 4, 16):
            chunks = Shard._custom_chunk(value, count, 0)
            self.assertEqual(len(chunks), count)
            self.assertEqual(sum(int(chunk.shape[0]) for chunk in chunks), 10)
        self.assertEqual(
            Shard(0).__fx_repr__()[0],
            "tensorplay.distributed.tensor.placement_types.Shard(dim=0)",
        )
        self.assertEqual(
            Replicate().__fx_repr__()[0],
            "tensorplay.distributed.tensor.placement_types.Replicate()",
        )
        self.assertEqual(
            Partial("sum").__fx_repr__()[0],
            "tensorplay.distributed.tensor.placement_types.Partial('sum')",
        )
        self.assertEqual(str(_StridedShard(0, split_factor=2)), "_S(0, 2)")
        hinted_size = tp.SymInt.symbolic("placement_size", hint=8)
        self.assertEqual(_explicit_or_backed_hint(hinted_size), 8)
        self.assertTrue(_hint_proves_even_shard(hinted_size, 2))
        self.assertEqual(
            _guarded_hint_int(
                tp.SymInt.symbolic("chunk_count", hint=2),
                reason="chunk count",
            ),
            2,
        )
        with self.assertRaisesRegex(RuntimeError, "concrete hint"):
            _guarded_hint_int(
                tp.SymInt.symbolic("unresolved_chunk_count"),
                reason="chunk count",
            )
        symbolic_factor = tp.SymInt.symbolic("placement_factor", hint=2)
        self.assertEqual(_StridedShard(0, sf=symbolic_factor)._split_factor_int(), 2)
        self.assertTrue(_MaskPartial().__fx_repr__()[0].endswith("offset_dim=0)"))

    def test_redistribute_autograd_protocol(self):
        import tensorplay as tp
        from tensorplay.autograd import grad
        from tensorplay.distributed.device_mesh import DeviceMesh
        from tensorplay.distributed.tensor import DTensor, Replicate, Shard

        mesh = DeviceMesh("cpu", [0])
        local = tp.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        value = DTensor.from_local(
            local,
            mesh,
            [Replicate()],
            run_check=False,
            shape=(4,),
            stride=(1,),
        )
        result = value.redistribute(placements=[Shard(0)])
        loss = (result.to_local() * result.to_local()).sum()
        first = grad(loss, (local,), create_graph=True)[0]
        self.assertEqual(first.tolist(), [2.0, 4.0, 6.0, 8.0])
        second = grad(first.sum(), (local,))[0]
        self.assertEqual(second.tolist(), [2.0, 2.0, 2.0, 2.0])

    def test_op_schema_protocol(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import (
            OpSchema,
            OpSpec,
            OpStrategy,
            OutputSharding,
            PlacementStrategy,
            TupleStrategy,
        )

        mesh = object()
        meta = TensorMeta((2, 3), (3, 1), tp.float32)
        spec = DTensorSpec(mesh, (), meta)
        first = OpSpec(spec, [spec], [[0.0]])
        second = PlacementStrategy(spec, [spec], [[0.0]])
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        grouped = TupleStrategy([OpStrategy([first])])
        self.assertEqual(grouped.childs, grouped.children)
        self.assertEqual(hash(grouped), hash(TupleStrategy(grouped.children)))
        schema = OpSchema("demo", (spec, 2), {"dim": 1})
        schema_hash = hash(schema)
        schema._recompute_comparison_key()
        self.assertEqual(hash(schema), schema_hash)
        self.assertIs(OutputSharding(spec).mesh, mesh)
        self.assertEqual(tuple(schema.gen_fake_args()[0].shape), (2, 3))

    def test_tensor_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import TensorMeta
        from tensorplay.distributed.tensor import Replicate
        from tensorplay.distributed.tensor._ops._experimental_ops import (
            slice_backward_rules,
        )
        from tensorplay.distributed.tensor._ops._tensor_ops import (
            gather_single_dim_strategy,
            index_put_single_dim_strategy,
            index_select_single_dim_strategy,
            scatter_add_single_dim_strategy,
            scatter_single_dim_strategy,
            select_int_single_dim_strategy,
            slice_single_dim_strategy,
            split_single_dim_strategy,
            register_tensor_ops,
        )

        meta = TensorMeta((8, 4), (4, 1), tp.float32)
        indices = TensorMeta((8, 4), (4, 1), tp.int64)
        index_vector = TensorMeta((2,), (1,), tp.int64)
        self.assertTrue(select_int_single_dim_strategy("select", (meta, 0, 1), {}))
        self.assertTrue(slice_single_dim_strategy("slice", (meta, 0, 1, 7, 1), {}))
        self.assertTrue(gather_single_dim_strategy("gather", (meta, 1, indices), {}))
        self.assertTrue(scatter_single_dim_strategy("scatter", (meta, 1, indices, meta), {}))
        self.assertTrue(scatter_add_single_dim_strategy("scatter_add", (meta, 1, indices, meta), {}))
        self.assertTrue(index_select_single_dim_strategy("index_select", (meta, 1, index_vector), {}))
        self.assertTrue(index_put_single_dim_strategy("index_put", (meta, (index_vector, None), meta), {}))
        self.assertTrue(split_single_dim_strategy("split", (meta, 2, 0), {}))

        register_tensor_ops()
        propagator = DTensor._op_dispatcher.sharding_propagator
        for name in ("select", "slice", "gather", "scatter", "index_select", "index", "split"):
            self.assertIn(name, propagator.op_single_dim_strategy_funcs)
        self.assertIn("stack", propagator.op_strategy_funcs)
        self.assertIn("unbind", propagator.op_strategy_funcs)
        self.assertNotIn("slice_backward", propagator.op_single_dim_strategy_funcs)
        mesh = type("Mesh", (), {"ndim": 2})()
        strategy = slice_backward_rules(mesh, None)
        self.assertEqual(len(strategy.strategies), 1)
        self.assertEqual(
            strategy.strategies[0].output_spec.placements,
            (Replicate(), Replicate()),
        )

    def test_view_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
        from tensorplay.distributed.tensor._ops._view_ops import (
            _StridedShard,
            dim_atleast_3d,
            dim_flatten,
            dim_movedim,
            dim_pad_left,
            dim_squeeze,
            dim_unsqueeze,
            expand,
            propagate_shape_and_sharding,
            register_view_ops,
            view_as_complex_single_dim_strategy,
            view_groups,
        )
        from tensorplay.distributed.tensor.placement_types import Partial, Replicate, Shard

        class Mesh:
            ndim = 1
            shape = (2,)

            @staticmethod
            def size(_dim=None):
                return 2

        self.assertEqual(dim_pad_left(2, 3)[0].__class__.__name__, "Singleton")
        self.assertEqual(len(dim_atleast_3d(1)), 3)
        self.assertEqual(dim_flatten(3, 1, 2)[1].__class__.__name__, "Flatten")
        self.assertEqual(dim_movedim(3, 0, 2)[2].input_dim, 0)
        self.assertEqual(len(dim_squeeze((1, 3, 1))), 1)
        self.assertEqual(len(dim_unsqueeze(2, 1)), 3)
        self.assertEqual(expand((2, 1), (2, 4))[1].__class__.__name__, "Broadcast")

        rule = view_groups((2, 4, 4), (8, 4))
        input_target, output = propagate_shape_and_sharding(
            (Shard(1),), (2, 4, 4), rule, (2,), True
        )
        self.assertEqual(input_target, (Shard(1),))
        self.assertEqual(output, [_StridedShard(0, split_factor=2)])
        unflatten_rule = view_groups((6, 4), (2, 3, 4))
        _, unflattened = propagate_shape_and_sharding(
            (_StridedShard(0, split_factor=2),), (6, 4), unflatten_rule, (2,), True
        )
        self.assertEqual(unflattened, [Shard(1)])

        meta = TensorMeta((2, 4, 2), (8, 2, 1), tp.float32)
        complex_strategies = view_as_complex_single_dim_strategy(
            "view_as_complex", (meta,), {}
        )
        self.assertEqual(len(complex_strategies), 4)
        self.assertEqual(complex_strategies[-2], [Partial("sum"), Partial("sum")])

        mesh = Mesh()
        view_meta = TensorMeta((2, 4, 4), (16, 4, 1), tp.float32)
        spec = DTensorSpec(mesh, (Shard(1),), view_meta)
        strategy = OpStrategy([OpSpec(spec)])
        register_view_ops()
        propagator = DTensor._op_dispatcher.sharding_propagator
        for name in (
            "view",
            "view_copy",
            "reshape",
            "expand",
            "permute",
            "repeat",
            "squeeze",
            "unsqueeze",
            "view_as_real",
        ):
            self.assertIn(name, propagator.op_strategy_funcs)
        self.assertIn("view_as_complex", propagator.op_single_dim_strategy_funcs)
        result = propagator.propagate_op_sharding_non_cached(
            OpSchema("view", (spec, (8, 4)), schema_info=None)
        )
        self.assertEqual(result.output_spec.placements, (_StridedShard(0, 2),))

    def test_parallel_mesh_validation(self):
        from tensorplay.distributed.tensor.parallel.api import parallelize_module
        from tensorplay.distributed.tensor.parallel._utils import _validate_tp_mesh_dim

        two_dimensional = type("Mesh", (), {"ndim": 2})()
        with self.assertRaises(ValueError):
            _validate_tp_mesh_dim(two_dimensional)
        with self.assertRaises(ValueError):
            parallelize_module(object(), two_dimensional, None)

        one_dimensional = type(
            "Mesh",
            (),
            {
                "ndim": 1,
                "_get_root_mesh": lambda self: self,
            },
        )()
        _validate_tp_mesh_dim(one_dimensional)
        module = object()
        self.assertIs(parallelize_module(module, one_dimensional, None), module)

    def test_strategy_validation_core(self):
        from dataclasses import dataclass

        import tensorplay as tp
        from tensorplay.distributed.tensor import Partial, Replicate, Shard
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import OpSpec, OpStrategy
        from tensorplay.distributed.tensor._ops.strategy_validation import (
            _checkerboard_mask,
            _create_partial_input,
            _extract_rules_from_op_strategy,
            _shard_tensors,
            create_fully_negated_sample,
            extract_tensors_from_sample,
            get_1d_input_placements_for_tensor,
            normalize_combo_key,
            parse_placement,
        )

        @dataclass
        class Sample:
            input: object
            args: tuple = ()
            kwargs: dict = None

            def __post_init__(self):
                if self.kwargs is None:
                    self.kwargs = {}

        value = tp.arange(12, dtype=tp.float32).reshape(3, 4)
        sample = Sample(value, args=(value + 1,), kwargs={"other": value})
        self.assertEqual([name for name, _ in extract_tensors_from_sample(sample)], [
            "tensor_0", "tensor_1", "tensor_2"
        ])
        self.assertIsNone(parse_placement("S(bad)"))
        self.assertEqual(repr(parse_placement("P(sum)")), "Partial('sum')")
        self.assertEqual(
            normalize_combo_key(
                (("S(0)",), ("S(0)",)), ((1, 4),), ((1, 4),)
            ),
            (("R",), ("R",)),
        )
        self.assertEqual(
            [repr(item) for item in get_1d_input_placements_for_tensor(value)],
            ["Replicate()", "Shard(dim=0)", "Shard(dim=1)"],
        )
        self.assertEqual(
            _checkerboard_mask(value).tolist(),
            [True, False, True, False, False, True, False, True, True, False, True, False],
        )
        partial = _create_partial_input(value, Partial("sum"), 2)
        self.assertTrue(tp.allclose(
            partial._local_tensors[0] + partial._local_tensors[1], value
        ))
        shards = _shard_tensors(
            [("value", value)], (Shard(0),), 2, mesh=None
        )[0]
        self.assertEqual([tuple(item.shape) for item in shards._local_tensors.values()], [(2, 4), (1, 4)])
        negative = create_fully_negated_sample(sample)
        self.assertTrue(tp.allclose(negative.input, -value))
        self.assertTrue(tp.allclose(negative.args[0], -(value + 1)))

        class Mesh:
            shape = (2,)
            ndim = 1

            def size(self, _dim=None):
                return 2

        input_spec = DTensorSpec(
            Mesh(), (Shard(0),), TensorMeta((3, 4), (4, 1), tp.float32)
        )
        output_spec = DTensorSpec(
            Mesh(), (Shard(0),), TensorMeta((3, 4), (4, 1), tp.float32)
        )
        rules = _extract_rules_from_op_strategy(
            OpStrategy([OpSpec(output_spec, (input_spec,))]),
            ((3, 4),),
            ((3, 4),),
        )
        self.assertIn((("S(0)",), ("S(0)",)), rules)

        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._ops.autogen import auto_register_op_variants
        from tensorplay.distributed.tensor._ops.single_dim_strategy import _SingleDimStrategyInfo

        class Field:
            def __init__(self, name, write=False):
                self.name = name
                self.type = "Tensor"
                self.kwarg_only = False
                self.is_out = False
                self.alias_info = type("Alias", (), {"is_write": write})() if write else None

        class Schema:
            def __init__(self, mutable=False):
                self.arguments = [Field("input", write=mutable)]
                self.returns = [Field("output")]
                self.is_mutable = mutable

        class Packet:
            def __init__(self, operation):
                self.operation = operation

            def overloads(self):
                return ["default"]

            def __getattr__(self, name):
                return self.operation if name == "default" else None

        class Operation:
            def __init__(self, name, mutable=False):
                self._name = name
                self._schema = Schema(mutable)
                self._variant_packets = {}

            def name(self):
                return self._name

        propagator = DTensor._op_dispatcher.sharding_propagator
        base = Operation("validation_base")
        inplace = Operation("validation_base_", mutable=True)
        base._variant_packets["validation_base_"] = Packet(inplace)
        info = _SingleDimStrategyInfo(
            lambda _operation, _args, _kwargs: [[Shard(0), Shard(0)]]
        )
        propagator.register_single_dim_op_strategy(base, info)
        try:
            auto_register_op_variants()
            self.assertIn(inplace, propagator.op_single_dim_strategy_funcs)
            self.assertEqual(
                propagator.op_single_dim_strategy_funcs[inplace].func(
                    inplace, (), {}
                ),
                [[Shard(0), Shard(0)]],
            )
        finally:
            propagator.op_single_dim_strategy_funcs.pop(base, None)
            propagator.op_single_dim_strategy_funcs.pop(inplace, None)
            propagator.propagate_op_sharding.cache_clear()

    def test_parallel_ddp_tensor_bridge(self):
        import tensorplay as tp
        import tensorplay.nn as nn
        from tensorplay.distributed.tensor import Replicate
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor.parallel._data_parallel_utils import (
            _flatten_tensor,
            _unflatten_tensor,
            sync_grad_hook,
        )
        from tensorplay.distributed.tensor.parallel.ddp import (
            _localize_dtensor,
            _reconstruct_dtensor,
            pre_dp_module_transform,
        )

        class Mesh:
            ndim = 1
            device_type = "cpu"
            shape = (1,)

            def size(self, mesh_dim=None):
                del mesh_dim
                return 1

            def get_coordinate(self):
                return (0,)

        class Leaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(tp.ones((2, 2), requires_grad=True))

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Leaf()

        mesh = Mesh()
        root = Root()
        dtensor = DTensor(
            tp.ones((2, 2), requires_grad=True),
            mesh,
            [Replicate()],
            shape=(2, 2),
            stride=(2, 1),
        )
        root.inner._parameters["weight"] = dtensor
        local, spec = _flatten_tensor(dtensor)
        self.assertEqual(spec.shape, (2, 2))
        self.assertEqual(spec.stride, (2, 1))
        restored = _unflatten_tensor(local, spec)
        self.assertIsInstance(restored, DTensor)
        self.assertEqual(restored.stride(), (2, 1))
        self.assertIs(sync_grad_hook(local), local)

        pre_dp_module_transform(root)
        self.assertIsInstance(root.inner.weight, nn.Parameter)
        _reconstruct_dtensor(root, ())
        self.assertIsInstance(root.inner.weight, DTensor)
        _localize_dtensor(root)
        self.assertIsInstance(root.inner.weight, nn.Parameter)
        self.assertIsNotNone(root.inner.weight._st_info)

    def test_parallel_fsdp_tensor_extension(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor import DTensor, Replicate, Shard
        from tensorplay.distributed.tensor.parallel.fsdp import DTensorExtensions

        class RootMesh:
            ndim = 2
            device_type = "cpu"
            shape = (1, 1)

            def size(self, mesh_dim=None):
                del mesh_dim
                return 1

            def get_coordinate(self):
                return (0, 0)

            def get_local_rank(self, mesh_dim=None):
                del mesh_dim
                return 0

            def _get_root_mesh(self):
                return self

            def get_group(self, mesh_dim=None):
                del mesh_dim
                return None

        class TensorParallelMesh:
            ndim = 1
            device_type = "cpu"
            shape = (1,)

            def size(self, mesh_dim=None):
                del mesh_dim
                return 1

            def get_coordinate(self):
                return (0,)

            def get_local_rank(self, mesh_dim=None):
                del mesh_dim
                return 0

            def _get_root_mesh(self):
                return root_mesh

            def get_group(self, mesh_dim=None):
                del mesh_dim
                return None

        root_mesh = RootMesh()
        tp_mesh = TensorParallelMesh()
        value = DTensor(
            tp.arange(8, dtype=tp.float32).reshape(4, 2),
            tp_mesh,
            [Replicate()],
            shape=(4, 2),
            stride=(2, 1),
        )
        extension = DTensorExtensions()
        local, spec = extension.pre_flatten_transform(value)
        self.assertEqual(spec.shape, (4, 2))
        self.assertEqual(spec.stride, (2, 1))
        rebuilt = extension.post_unflatten_transform(local, spec)
        self.assertEqual(rebuilt.placements, (Replicate(),))
        chunked = extension.chunk_dtensor(value, 0, tp_mesh)
        self.assertEqual(chunked.placements, (Shard(0), Replicate()))

        root_value = DTensor(
            value.to_local(),
            root_mesh,
            [Shard(0), Replicate()],
            shape=(4, 2),
            stride=(2, 1),
        )
        gathered = extension.all_gather_dtensor(root_value, root_mesh)
        self.assertEqual(tuple(gathered.shape), (4, 2))

    def test_parallel_style_runtime_layouts(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor import DTensor, Replicate, Shard
        from tensorplay.distributed.tensor.parallel.style import (
            PrepareModuleInput,
            PrepareModuleInputOutput,
            PrepareModuleOutput,
            SequenceParallel,
        )

        class MeshNDimensionality(int):
            def __call__(self):
                return int(self)

        class Mesh:
            ndim = MeshNDimensionality(1)
            device_type = "cpu"
            shape = (1,)

            def size(self, mesh_dim=None):
                del mesh_dim
                return 1

            def get_coordinate(self):
                return (0,)

            def get_local_rank(self, mesh_dim=None):
                del mesh_dim
                return 0

            def get_group(self, mesh_dim=None):
                del mesh_dim
                return None

        mesh = Mesh()
        value = DTensor(
            tp.arange(6, dtype=tp.float32).reshape(2, 3),
            mesh,
            [Shard(0)],
            shape=(2, 3),
            stride=(3, 1),
        )
        prepared = SequenceParallel(sequence_dim=1)._input(None, (value,), mesh)
        self.assertEqual(prepared[0].placements, (Shard(1),))
        with self.assertRaises(ValueError):
            SequenceParallel()._input(None, (7,), mesh)

        input_style = PrepareModuleInput(
            input_layouts=Shard(0), desired_input_layouts=Replicate()
        )
        output_style = PrepareModuleOutput(
            output_layouts=Replicate(), desired_output_layouts=Shard(0)
        )
        combined_style = PrepareModuleInputOutput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            output_layouts=Replicate(),
            desired_output_layouts=Shard(0),
        )
        self.assertIn("input_layouts", repr(input_style))
        self.assertIn("desired_output_layouts", repr(output_style))
        self.assertIn("use_local_input", repr(combined_style))

    def test_parallel_loss_cross_entropy(self):
        import tensorplay.distributed.tensor.parallel.loss as loss_module
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor.parallel.loss import loss_parallel

        names = (
            "log_softmax",
            "_log_softmax",
            "_log_softmax_backward_data",
            "nll_loss",
            "nll_loss_forward",
            "nll_loss2d",
            "nll_loss2d_forward",
            "nll_loss_backward",
            "nll_loss2d_backward",
        )
        self.assertFalse(hasattr(loss_module, "is_loss_parallel_enabled"))
        self.assertTrue(
            all(
                name not in DTensor._op_dispatcher._custom_op_handlers
                for name in names
            )
        )
        with loss_parallel():
            self.assertTrue(
                all(
                    name in DTensor._op_dispatcher._custom_op_handlers
                    for name in names
                )
            )
        self.assertTrue(
            all(
                name not in DTensor._op_dispatcher._custom_op_handlers
                for name in names
            )
        )
        _spawn_ranks(_loss_parallel_body, 2, args=(_free_port(),))

    def test_parallel_input_reshard(self):
        import tensorplay.nn as nn
        from tensorplay.distributed.tensor.parallel.input_reshard import input_reshard

        module = nn.ReLU()
        self.assertIs(input_reshard(module, None, None), module)
        _spawn_ranks(_input_reshard_body, 2, args=(_free_port(),))

    def test_parallel_experimental_mapping_and_registration(self):
        _spawn_ranks(_experimental_mapping_body, 2, args=(_free_port(),))

    def test_tensor_parallel_transformation(self):
        _spawn_ranks(_tp_transform_body, 2, args=(_free_port(),))

    def test_math_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import TensorMeta
        from tensorplay.distributed.tensor._ops._math_ops import (
            _NormPartial,
            global_median_single_dim_strategy,
            layer_norm_bwd_single_dim_strategy,
            layer_norm_single_dim_strategy,
            linalg_replicate_single_dim_strategy,
            nll_loss_backward_single_dim_strategy,
            nll_loss_forward_single_dim_strategy,
            rms_norm_bwd_single_dim_strategy,
            rms_norm_single_dim_strategy,
            scan_single_dim_strategy,
            register_math_ops,
        )

        meta = TensorMeta((8, 4), (4, 1), tp.float32)
        target = TensorMeta((8,), (1,), tp.int64)
        self.assertTrue(scan_single_dim_strategy("cumsum", (meta, 1), {}))
        self.assertEqual(global_median_single_dim_strategy("median", (meta,), {}), [])
        self.assertEqual(linalg_replicate_single_dim_strategy("linalg_svd", (meta,), {}), [])
        self.assertTrue(nll_loss_forward_single_dim_strategy("nll_loss_forward", (meta, target, None, "none", None), {}))
        self.assertTrue(nll_loss_backward_single_dim_strategy("nll_loss_backward", (meta, meta, target, None, "none", None, target), {}))
        self.assertTrue(layer_norm_single_dim_strategy("native_layer_norm", (meta, (4,), None, None), {}))
        self.assertTrue(rms_norm_single_dim_strategy("fused_rms_norm", (meta, (4,), None), {}))
        self.assertTrue(layer_norm_bwd_single_dim_strategy("native_layer_norm_backward", (meta, meta, (4,), meta, meta, None, None, (True, False, False)), {}))
        self.assertTrue(rms_norm_bwd_single_dim_strategy("fused_rms_norm_backward", (meta, meta, (4,), meta, None, (True, False)), {}))
        self.assertEqual(_NormPartial(2), _NormPartial(2))

        register_math_ops()
        propagator = DTensor._op_dispatcher.sharding_propagator
        for name in ("cumsum", "nll_loss_forward", "native_layer_norm", "fused_rms_norm"):
            self.assertIn(name, propagator.op_single_dim_strategy_funcs)

    def test_convolution_propagation_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import OpSchema, OutputSharding
        from tensorplay.distributed.tensor._ops._conv_ops import (
            convolution_backward_rules,
            convolution_rules,
        )
        from tensorplay.distributed.tensor.placement_types import Replicate, Shard

        class Mesh:
            ndim = 1

            @staticmethod
            def size(_dim=None):
                return 2

        mesh = Mesh()
        input_spec = DTensorSpec(
            mesh,
            (Shard(0),),
            TensorMeta((2, 3, 8, 8), (192, 64, 8, 1), tp.float32),
        )
        weight_spec = DTensorSpec(
            mesh,
            (Replicate(),),
            TensorMeta((4, 3, 3, 3), (27, 9, 3, 1), tp.float32),
        )
        bias_spec = DTensorSpec(
            mesh,
            (Replicate(),),
            TensorMeta((4,), (1,), tp.float32),
        )
        forward = convolution_rules(
            OpSchema(
                "convolution",
                (input_spec, weight_spec, bias_spec, [1, 1], [1, 1], [1, 1], False, [0, 0], 1),
            )
        )
        self.assertIsInstance(forward, OutputSharding)
        self.assertIn("convolution", DTensor._op_dispatcher._custom_op_handlers)
        self.assertIn(
            "convolution_backward", DTensor._op_dispatcher._custom_op_handlers
        )
        self.assertEqual(forward.output_spec.shape, (2, 4, 8, 8))
        self.assertEqual(forward.output_spec.placements, (Shard(0),))

        backward = convolution_backward_rules(
            OpSchema(
                "convolution_backward",
                (
                    forward.output_spec,
                    input_spec,
                    weight_spec,
                    [4],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    [True, True, True],
                ),
            )
        )
        self.assertIsInstance(backward, OutputSharding)
        self.assertEqual(len(backward.output_spec), 3)
        self.assertEqual(backward.output_spec[0], input_spec)
        self.assertTrue(backward.output_spec[1].placements[0].is_partial())
        self.assertTrue(backward.output_spec[2].placements[0].is_partial())

        no_bias_forward = convolution_rules(
            OpSchema(
                "convolution",
                (
                    input_spec,
                    weight_spec,
                    None,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                ),
            )
        )
        self.assertEqual(no_bias_forward.output_spec.shape, (2, 4, 8, 8))
        no_bias_backward = convolution_backward_rules(
            OpSchema(
                "convolution_backward",
                (
                    forward.output_spec,
                    input_spec,
                    weight_spec,
                    None,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    [True, True, False],
                ),
            )
        )
        self.assertIsNone(no_bias_backward.output_spec[2])

        for input_shape, weight_shape, spatial_args, expected_shape in (
            ((2, 3, 8), (4, 3, 3), ([1], [1], [1]), (2, 4, 8)),
            (
                (2, 3, 4, 5, 6),
                (4, 3, 3, 3, 3),
                ([1, 1, 1], [1, 1, 1], [1, 1, 1]),
                (2, 4, 4, 5, 6),
            ),
        ):
            input_ndim = len(input_shape)
            input_ndim_stride = []
            running = 1
            for size in reversed(input_shape):
                input_ndim_stride.append(running)
                running *= size
            input_ndim_stride.reverse()
            weight_ndim_stride = []
            running = 1
            for size in reversed(weight_shape):
                weight_ndim_stride.append(running)
                running *= size
            weight_ndim_stride.reverse()
            shape_input = DTensorSpec(
                mesh,
                (Shard(0),),
                TensorMeta(input_shape, tuple(input_ndim_stride), tp.float32),
            )
            shape_weight = DTensorSpec(
                mesh,
                (Replicate(),),
                TensorMeta(weight_shape, tuple(weight_ndim_stride), tp.float32),
            )
            shape_output = convolution_rules(
                OpSchema(
                    "convolution",
                    (
                        shape_input,
                        shape_weight,
                        None,
                        *spatial_args,
                        False,
                        [0] * (input_ndim - 2),
                        1,
                    ),
                )
            )
            self.assertEqual(shape_output.output_spec.shape, expected_shape)

    def test_convolution_single_dim_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import OpSchema
        from tensorplay.distributed.tensor._ops._conv_ops import (
            _convolution_full_mesh_strategy_filter,
            convolution_backward_single_dim_strategy,
            convolution_single_dim_strategy,
        )
        from tensorplay.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
        )
        from tensorplay.distributed.tensor.placement_types import (
            Partial,
            Replicate,
            Shard,
            _StridedShard,
        )

        class Mesh:
            ndim = 1
            shape = (2,)

            @staticmethod
            def size(_dim=None):
                return 2

        mesh = Mesh()
        input_meta = TensorMeta((2, 3, 8, 8), (192, 64, 8, 1), tp.float32)
        weight_meta = TensorMeta((4, 3, 1, 1), (3, 1, 1, 1), tp.float32)
        bias_meta = TensorMeta((4,), (1,), tp.float32)
        forward_args = (
            input_meta,
            weight_meta,
            bias_meta,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        forward_strategies = convolution_single_dim_strategy(
            "convolution", forward_args, {}
        )
        self.assertEqual(len(forward_strategies), 2)
        self.assertEqual(
            forward_strategies[0],
            [_ShardingPlaceholder(0), _ShardingPlaceholder(0), Replicate(), Replicate()],
        )
        self.assertEqual(
            forward_strategies[1],
            [Shard(3), Shard(3), Replicate(), Replicate()],
        )
        no_bias_strategies = convolution_single_dim_strategy(
            "convolution", (*forward_args[:2], None, *forward_args[3:]), {}
        )
        self.assertEqual(len(no_bias_strategies[0]), 3)
        self.assertEqual(len(no_bias_strategies[1]), 3)

        backward_args = (
            input_meta,
            input_meta,
            weight_meta,
            [4],
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
        backward_strategies = convolution_backward_single_dim_strategy(
            "convolution_backward", backward_args, {}
        )
        self.assertEqual(len(backward_strategies), 2)
        self.assertEqual(backward_strategies[0][1], Partial("sum"))
        self.assertEqual(backward_strategies[0][2], Partial("sum"))
        self.assertEqual(backward_strategies[1][0], Shard(3))
        self.assertEqual(backward_strategies[1][3], Shard(3))
        no_bias_backward_strategies = convolution_backward_single_dim_strategy(
            "convolution_backward", (*backward_args[:3], None, *backward_args[4:]), {}
        )
        self.assertIsNone(no_bias_backward_strategies[0][2])

        input_spec = DTensorSpec(mesh, (Shard(3),), input_meta)
        weight_spec = DTensorSpec(mesh, (Replicate(),), weight_meta)
        output_spec = DTensorSpec(mesh, (Shard(3),), input_meta)
        schema = OpSchema("convolution", forward_args)
        self.assertTrue(
            _convolution_full_mesh_strategy_filter(
                mesh, schema, [input_spec, weight_spec], output_spec
            )
        )
        strided_spec = DTensorSpec(
            mesh, (_StridedShard(3, split_factor=2),), input_meta
        )
        self.assertFalse(
            _convolution_full_mesh_strategy_filter(
                mesh, schema, [strided_spec, weight_spec], output_spec
            )
        )

        class TwoDimMesh:
            ndim = 2

            @staticmethod
            def size(_dim=None):
                return 2

        multi_axis_spec = DTensorSpec(
            TwoDimMesh(), (Shard(3), Shard(3)), input_meta
        )
        self.assertFalse(
            _convolution_full_mesh_strategy_filter(
                TwoDimMesh(), schema, [multi_axis_spec, weight_spec], output_spec
            )
        )

        propagator = DTensor._op_dispatcher.sharding_propagator
        self.assertIn("convolution", propagator.op_single_dim_strategy_funcs)
        self.assertIn(
            "convolution_backward", propagator.op_single_dim_strategy_funcs
        )
        self.assertEqual(
            propagator.op_to_schema_info_for_single_dim_strategy[
                "convolution"
            ].static_argnum,
            2,
        )
        expanded = propagator.propagate_op_sharding_non_cached(
            OpSchema(
                "convolution",
                (
                    input_spec,
                    weight_spec,
                    None,
                    *forward_args[3:],
                ),
            )
        )
        self.assertEqual(expanded.output_spec.placements, (Shard(3),))

    def test_attention_single_dim_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import (
            DTensorSpec,
            TensorMeta,
        )
        from tensorplay.distributed.tensor._op_schema import (
            OpSchema,
            OpSpec,
            OpStrategy,
        )
        from tensorplay.distributed.tensor._ops._matrix_ops import (
            _scaled_dot_product_cudnn_attention_backward_base_strategies,
            _scaled_dot_product_cudnn_attention_base_strategies,
            _scaled_dot_product_efficient_attention_backward_base_strategies,
            _scaled_dot_product_efficient_attention_base_strategies,
            _scaled_dot_product_flash_attention_backward_base_strategies,
            _scaled_dot_product_flash_attention_base_strategies,
            scaled_dot_product_cudnn_attention_backward_single_dim_strategy,
            scaled_dot_product_cudnn_attention_single_dim_strategy,
            scaled_dot_product_efficient_attention_backward_single_dim_strategy,
            scaled_dot_product_efficient_attention_single_dim_strategy,
            scaled_dot_product_flash_attention_backward_single_dim_strategy,
            scaled_dot_product_flash_attention_single_dim_strategy,
        )
        from tensorplay.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
        )
        from tensorplay.distributed.tensor.placement_types import Replicate, Shard

        class Mesh:
            ndim = 1
            shape = (2,)

            @staticmethod
            def size(_dim=None):
                return 2

        meta = TensorMeta((2, 4, 8, 16), (512, 128, 16, 1), tp.float32)
        bias_meta = TensorMeta((2, 4, 8, 8), (256, 64, 8, 1), tp.float32)

        flash_args = (meta, meta, meta, 0.0, False, False, None)
        flash = scaled_dot_product_flash_attention_single_dim_strategy(
            "_scaled_dot_product_flash_attention", flash_args, {}
        )
        self.assertEqual([len(item) for item in flash], [12, 12])
        self.assertEqual(flash[0][8], Replicate())
        debug_flash = scaled_dot_product_flash_attention_single_dim_strategy(
            "_scaled_dot_product_flash_attention", (*flash_args[:5], True, None), {}
        )
        self.assertEqual(debug_flash[0][8], _ShardingPlaceholder(1))
        flash_backward = scaled_dot_product_flash_attention_backward_single_dim_strategy(
            "_scaled_dot_product_flash_attention_backward",
            (meta, meta, meta, meta, meta, meta, None, None, 0, 0, 0.0, False, None, None, None),
            {},
        )
        self.assertEqual([len(item) for item in flash_backward], [9, 9])

        efficient_args = (meta, meta, meta, None, False, 0.0, False, None)
        efficient = scaled_dot_product_efficient_attention_single_dim_strategy(
            "_scaled_dot_product_efficient_attention", efficient_args, {}
        )
        self.assertEqual([len(item) for item in efficient], [7, 7])
        self.assertEqual(efficient[0][1], Replicate())
        efficient_bias = scaled_dot_product_efficient_attention_single_dim_strategy(
            "_scaled_dot_product_efficient_attention",
            (*efficient_args[:3], bias_meta, *efficient_args[4:]),
            {},
        )
        self.assertEqual([len(item) for item in efficient_bias], [8, 8])
        efficient_backward = scaled_dot_product_efficient_attention_backward_single_dim_strategy(
            "_scaled_dot_product_efficient_attention_backward",
            (meta, meta, meta, meta, None, meta, meta, None, None, 0.0, (True, True, True), False, None),
            {},
        )
        self.assertEqual([len(item) for item in efficient_backward], [12, 12])
        self.assertIsNone(efficient_backward[0][3])
        efficient_backward_bias = scaled_dot_product_efficient_attention_backward_single_dim_strategy(
            "_scaled_dot_product_efficient_attention_backward",
            (meta, meta, meta, meta, bias_meta, meta, meta, None, None, 0.0, (True, True, True), False, None),
            {},
        )
        self.assertEqual([len(item) for item in efficient_backward_bias], [13, 13])
        self.assertEqual(
            efficient_backward_bias[0][3], _ShardingPlaceholder(1)
        )

        cudnn_args = (meta, meta, meta, None, False, 0.0, False, False, None)
        cudnn = scaled_dot_product_cudnn_attention_single_dim_strategy(
            "_scaled_dot_product_cudnn_attention", cudnn_args, {}
        )
        self.assertEqual([len(item) for item in cudnn], [12, 12])
        self.assertIsNone(cudnn[0][8])
        cudnn_debug = scaled_dot_product_cudnn_attention_single_dim_strategy(
            "_scaled_dot_product_cudnn_attention",
            (*cudnn_args[:7], True, None),
            {},
        )
        self.assertEqual(cudnn_debug[0][8], _ShardingPlaceholder(1))
        cudnn_backward_args = (
            meta,
            meta,
            meta,
            meta,
            meta,
            meta,
            TensorMeta((1,), (1,), tp.int64),
            TensorMeta((1,), (1,), tp.int64),
            None,
            None,
            None,
            0,
            0,
            0.0,
            False,
            None,
        )
        cudnn_backward = scaled_dot_product_cudnn_attention_backward_single_dim_strategy(
            "_scaled_dot_product_cudnn_attention_backward",
            cudnn_backward_args,
            {},
        )
        self.assertEqual([len(item) for item in cudnn_backward], [11, 11])
        cudnn_backward_bias = scaled_dot_product_cudnn_attention_backward_single_dim_strategy(
            "_scaled_dot_product_cudnn_attention_backward",
            (*cudnn_backward_args[:8], bias_meta, *cudnn_backward_args[9:]),
            {},
        )
        self.assertEqual([len(item) for item in cudnn_backward_bias], [12, 12])

        mesh = Mesh()
        spec = DTensorSpec(mesh, (Replicate(),), meta)
        strategy = OpStrategy([OpSpec(spec)])
        self.assertEqual(
            [len(item) for item in _scaled_dot_product_flash_attention_base_strategies(
                OpSchema("flash", (strategy, strategy, strategy, 0.0, False, False))
            )],
            [12, 12, 12],
        )
        self.assertEqual(
            [len(item) for item in _scaled_dot_product_flash_attention_backward_base_strategies(
                OpSchema("flash_backward", (strategy,) * 6)
            )],
            [9, 9, 9],
        )
        self.assertEqual(
            [len(item) for item in _scaled_dot_product_efficient_attention_base_strategies(
                OpSchema("efficient", (strategy, strategy, strategy, None, False))
            )],
            [7, 7, 7],
        )
        self.assertEqual(
            [len(item) for item in _scaled_dot_product_efficient_attention_backward_base_strategies(
                OpSchema("efficient_backward", (strategy, strategy, strategy, strategy, None))
            )],
            [12, 12, 12],
        )
        cudnn_base_args = [None] * 15
        cudnn_base_args[1] = strategy
        self.assertEqual(
            [len(item) for item in _scaled_dot_product_cudnn_attention_backward_base_strategies(
                OpSchema("cudnn_backward", tuple(cudnn_base_args))
            )],
            [18, 18, 18],
        )
        self.assertEqual(
            [len(item) for item in _scaled_dot_product_cudnn_attention_base_strategies(
                OpSchema("cudnn", (strategy, strategy, strategy, None, False, 0.0, False, False))
            )],
            [12, 12, 12],
        )

        propagator = DTensor._op_dispatcher.sharding_propagator
        for name in (
            "_scaled_dot_product_flash_attention",
            "_scaled_dot_product_flash_attention_backward",
            "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_efficient_attention_backward",
            "_scaled_dot_product_cudnn_attention",
            "_scaled_dot_product_cudnn_attention_backward",
        ):
            self.assertIn(name, propagator.op_single_dim_strategy_funcs)

    def test_pointwise_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
        from tensorplay.distributed.tensor._ops._pointwise_ops import (
            pointwise_strategy,
            register_pointwise_ops,
        )
        from tensorplay.distributed.tensor.placement_types import Partial, Replicate, Shard

        class Mesh:
            ndim = 1

            @staticmethod
            def size(_dim=None):
                return 2

        mesh = Mesh()
        left = DTensorSpec(
            mesh,
            (Shard(1),),
            TensorMeta((2, 4), (4, 1), tp.float32),
        )
        right = DTensorSpec(
            mesh,
            (Replicate(),),
            TensorMeta((4,), (1,), tp.float32),
        )
        result = pointwise_strategy(
            mesh,
            OpSchema(
                "add",
                (OpStrategy([OpSpec(left)]), OpStrategy([OpSpec(right)])),
            ),
        )
        selected = result.strategies[0]
        self.assertEqual(selected.output_spec.placements, (Shard(1),))
        self.assertEqual(selected.input_specs[1].placements, (Shard(0),))

        partial = DTensorSpec(
            mesh,
            (Partial(),),
            TensorMeta((2, 4), (4, 1), tp.float32),
        )
        result = pointwise_strategy(
            mesh,
            OpSchema("relu", (OpStrategy([OpSpec(partial)]),)),
        )
        self.assertTrue(result.strategies[0].output_spec.placements[0].is_replicate())
        register_pointwise_ops()
        propagator = DTensor._op_dispatcher.sharding_propagator
        for name in ("add", "bitwise_and", "_foreach_add", "_fused_adam"):
            self.assertIn(name, propagator.op_strategy_funcs)

    def test_random_strategy_family(self):
        import tensorplay as tp
        from tensorplay.distributed.tensor._api import DTensor
        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
        from tensorplay.distributed.tensor._ops._random_ops import (
            random_op_strategy,
            register_random_ops,
        )
        from tensorplay.distributed.tensor.placement_types import Partial, Shard

        class Mesh:
            ndim = 1

            @staticmethod
            def size(_dim=None):
                return 2

        mesh = Mesh()
        spec = DTensorSpec(
            mesh,
            (Shard(0),),
            TensorMeta((4, 2), (2, 1), tp.float32),
        )
        result = random_op_strategy(
            mesh,
            OpSchema("normal_", (OpStrategy([OpSpec(spec)]),)),
        )
        self.assertEqual(result.strategies[0].output_spec, spec)

        partial = DTensorSpec(
            mesh,
            (Partial(),),
            TensorMeta((4, 2), (2, 1), tp.float32),
        )
        with self.assertRaisesRegex(RuntimeError, "Partial"):
            random_op_strategy(
                mesh,
                OpSchema("uniform_", (OpStrategy([OpSpec(partial)]),)),
            )
        register_random_ops()
        propagator = DTensor._op_dispatcher.sharding_propagator
        for name in ("normal_", "uniform_", "native_dropout", "bernoulli_", "bernoulli"):
            self.assertIn(name, propagator.op_strategy_funcs)

    def test_random_tracker_offsets(self):
        import tensorplay as tp
        from unittest.mock import patch

        from tensorplay.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from tensorplay.distributed.tensor._random import (
            OffsetBasedRNGTracker,
            is_rng_supported_mesh,
        )
        from tensorplay.distributed.tensor.placement_types import Shard

        class Handle:
            def __init__(self):
                self.state = tp.zeros(16, dtype=tp.uint8)

            def is_available(self):
                return True

            def current_device(self):
                return 0

            def get_rng_state(self):
                return self.state.clone()

            def set_rng_state(self, state):
                self.state = state.clone()

        class Mesh:
            ndim = 1
            shape = (2,)
            device_type = "cpu"

            def get_coordinate(self):
                return (1,)

            def size(self, _dim=0):
                return 2

        mesh = Mesh()
        handle = Handle()
        spec = DTensorSpec(
            mesh,
            (Shard(0),),
            TensorMeta((8,), (1,), tp.float32),
        )
        with patch(
            "tensorplay.distributed.tensor._random._get_device_handle",
            return_value=handle,
        ):
            self.assertTrue(is_rng_supported_mesh(mesh))
            tracker = OffsetBasedRNGTracker("cpu", run_state_sync=False)
            tracker.set_seed("parallel-rng", 123)
            self.assertEqual(tracker.get_seed("parallel-rng"), 123)
            with tracker._distribute_region(spec):
                self.assertEqual(tracker.get_offset("parallel-rng"), 4)
            self.assertEqual(tracker.get_offset("parallel-rng"), 8)
            self.assertEqual(tracker._calc_shard_linear_idx([1], [2]), 1)

    def test_mesh_and_redistribution(self):
        _spawn_ranks(_body, 2, args=(_free_port(),))

    def test_dtensor_operation_dispatch(self):
        _spawn_ranks(_ops_body, 2, args=(_free_port(),))

    def test_nested_redistribution(self):
        _spawn_ranks(_nested_redistribution_body, 4, args=(_free_port(),))

    def test_embedding_placements(self):
        _spawn_ranks(_embedding_body, 2, args=(_free_port(),))

    def test_alltoall_autograd(self):
        _spawn_ranks(_alltoall_autograd_body, 2, args=(_free_port(),))

    def test_tp_convolution_ring(self):
        _spawn_ranks(_tp_convolution_body, 2, args=(_free_port(),))


if __name__ == "__main__":
    unittest.main()
