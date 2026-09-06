import pytest

import tensorplay as tp


def test_index_tensor_cpu_dispatches_advanced_indices():
    source = tp.arange(12).reshape(3, 4)
    rows = tp.tensor([2, 0], dtype=tp.int64)
    got = tp.index(source, [rows, None])
    assert got.tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]


def test_compressed_constructor_family_infers_shapes_and_layouts():
    csc = tp.sparse_csc_tensor(
        tp.tensor([0, 2, 3, 5], dtype=tp.int64),
        tp.tensor([0, 2, 2, 0, 2], dtype=tp.int64),
        tp.tensor([1.0, 3.0, 4.0, 2.0, 5.0]),
    )
    assert csc.layout == tp.sparse_csc
    assert csc.is_sparse_csc()
    assert tuple(csc.shape) == (3, 3)
    assert csc.to_dense().tolist() == [
        [1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 5.0],
    ]

    blocks = tp.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 4.0]],
        ]
    )
    bsr = tp.sparse_bsr_tensor(
        tp.tensor([0, 1, 3], dtype=tp.int64),
        tp.tensor([0, 0, 1], dtype=tp.int64),
        blocks,
    )
    bsc = tp.sparse_bsc_tensor(
        tp.tensor([0, 2, 3], dtype=tp.int64),
        tp.tensor([0, 1, 1], dtype=tp.int64),
        blocks,
    )
    expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 4.0],
    ]
    for tensor, layout in ((bsr, tp.sparse_bsr), (bsc, tp.sparse_bsc)):
        assert tensor.layout == layout
        assert tensor.is_sparse_compressed()
        assert tuple(tensor.shape) == (4, 4)
        assert tensor.to_dense().tolist() == expected


def test_generic_compressed_constructor_selects_column_layout():
    ccol = tp.tensor([0, 2, 3, 5], dtype=tp.int32)
    rows = tp.tensor([0, 2, 2, 0, 2], dtype=tp.int32)
    values = tp.tensor([1.0, 3.0, 4.0, 2.0, 5.0])
    tensor = tp.sparse_compressed_tensor(
        ccol, rows, values, layout=tp.sparse_csc
    )
    assert tensor.layout == tp.sparse_csc
    assert tensor.ccol_indices().dtype == tp.int32
    assert tensor.row_indices().dtype == tp.int32
    assert tensor.to_dense().tolist()[2] == [3.0, 4.0, 5.0]


def test_batched_and_hybrid_compressed_shapes():
    batched = tp.sparse_csr_tensor(
        tp.tensor([[0, 1, 2], [0, 1, 2]], dtype=tp.int64),
        tp.tensor([[0, 1], [1, 0]], dtype=tp.int64),
        tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert tuple(batched.shape) == (2, 2, 2)
    assert batched.to_dense().tolist() == [
        [[1.0, 0.0], [0.0, 2.0]],
        [[0.0, 3.0], [4.0, 0.0]],
    ]

    hybrid = tp.sparse_csr_tensor(
        tp.tensor([0, 1, 2], dtype=tp.int64),
        tp.tensor([0, 1], dtype=tp.int64),
        tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
        size=[2, 2, 2],
    )
    assert hybrid.dense_dim() == 1
    assert tuple(hybrid.values().shape) == (2, 2)
    assert hybrid.to_dense().tolist() == [
        [[1.0, 2.0], [0.0, 0.0]],
        [[0.0, 0.0], [3.0, 4.0]],
    ]


def test_compressed_constructor_rejects_mismatched_layout_and_geometry():
    crow = tp.tensor([0, 1, 1], dtype=tp.int64)
    col = tp.tensor([0], dtype=tp.int64)
    values = tp.tensor([1.0])
    with pytest.raises(ValueError):
        tp.sparse_csr_tensor(crow, col, values, [2, 2], layout=tp.sparse_csc)

    with pytest.raises(Exception):
        tp.sparse_bsr_tensor(crow, col, values, [2, 2])


def test_compressed_with_dims_preserves_batch_and_dense_shape():
    tensor = tp._C._sparse_compressed_tensor_with_dims(
        7,
        1,
        [2, 4, 6, 5],
        [2, 3],
        tp.int64,
        dtype=tp.float32,
        layout=tp.sparse_bsr,
    )
    assert tuple(tensor.shape) == (2, 4, 6, 5)
    assert tuple(tensor.crow_indices().shape) == (2, 3)
    assert tuple(tensor.col_indices().shape) == (2, 7)
    assert tuple(tensor.values().shape) == (2, 7, 2, 3, 5)
