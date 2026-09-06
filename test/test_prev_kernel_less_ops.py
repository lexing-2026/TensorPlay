"""Coverage for the previously kernel-less ops: segment_reduce, hash_tensor,
_trilinear, _transform_bias_rescale_qkv, smm, _spdiags, to_sparse_csc/bsr/bsc,
_sparse_softmax/_sparse_log_softmax, _sparse_sum, native_norm, the slow/THNN
convolution spellings, pad/_pad_enum and cross_entropy_loss.

Each check pins values computed from the op's own definition (or an
equivalent spelled-out formula on dense tensors); no other framework's
outputs are consulted.
"""

import math

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F


def _flatten(x):
    if isinstance(x, tp.Tensor):
        return tp.reshape(x, [-1]).tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [x]


def _allclose(a, b, tol=1e-5):
    flat_a = _flatten(a)
    flat_b = _flatten(b)
    assert len(flat_a) == len(flat_b), (flat_a, flat_b)
    for x, y in zip(flat_a, flat_b):
        if math.isinf(y) or math.isinf(x):
            assert x == y or (math.isnan(x - y) and x * y < 0), (flat_a, flat_b)
        elif math.isnan(y):
            assert math.isnan(x), (flat_a, flat_b)
        else:
            assert abs(x - y) <= tol * max(1.0, abs(y)), (flat_a, flat_b)


# --------------------------------------------------------------------- pad


def test_pad_constant_matches_narrow_fill_formula():
    d = tp.arange(6.0).reshape(2, 3)
    out = tp._C.pad(d, [1, 1])
    assert out.tolist() == [[0, 0, 1, 2, 0], [0, 3, 4, 5, 0]]
    _allclose(tp._C.pad(d, [1, 1], 'constant', 9.0)[0], [9, 0, 1, 2, 9])


def test_pad_enum_modes_and_negative_pads():
    d = tp.arange(4.0).reshape(1, 4)
    # mode numbering: reflect=0, replicate=1, circular=2, constant=3;
    # each spelling must agree with the string-mode pad entry point
    _allclose(tp._C._pad_enum(d, [1, 1], 0)[0], tp._C.pad(d, [1, 1], 'reflect')[0])
    _allclose(tp._C._pad_enum(d, [1, 1], 1)[0], tp._C.pad(d, [1, 1], 'replicate')[0])
    _allclose(tp._C._pad_enum(d, [1, 1], 2)[0], tp._C.pad(d, [1, 1], 'circular')[0])
    _allclose(tp._C._pad_enum(d, [1, 1], 3, 5.0)[0], [5, 0, 1, 2, 3, 5])
    # negative pads shrink the output
    assert tp._C.pad(d, [-1, -1]).tolist() == [[1, 2]]


def test_pad_rejects_odd_lengths_and_bad_modes():
    d = tp.ones(2, 2)
    with pytest.raises(Exception):
        tp._C.pad(d, [1])
    with pytest.raises(Exception):
        tp._C.pad(d, [1, 1], 'edge')


# ---------------------------------------------------------- cross_entropy_loss


def _log_softmax_ref(row):
    m = max(row)
    logs = [x - m - math.log(sum(math.exp(v - m) for v in row)) for x in row]
    return logs


def test_cross_entropy_loss_hard_targets_matches_nll_definition():
    x = tp.tensor([[1.0, 2.0], [3.0, 1.0]])
    t = tp.tensor([0, 1])
    ls = [_log_softmax_ref([1.0, 2.0]), _log_softmax_ref([3.0, 1.0])]
    want = -(ls[0][0] + ls[1][1]) / 2
    got = tp.cross_entropy_loss(x, t).item()
    assert abs(got - want) < 1e-5


def test_cross_entropy_loss_soft_targets_and_smoothing():
    x = tp.tensor([[1.0, 2.0], [3.0, 1.0]])
    p = tp.tensor([[0.25, 0.75], [0.9, 0.1]])
    ls = [_log_softmax_ref([1.0, 2.0]), _log_softmax_ref([3.0, 1.0])]
    want = 0.0
    for b in range(2):
        for c in range(2):
            want -= ls[b][c] * p.tolist()[b][c]
    want /= 2
    assert abs(tp.cross_entropy_loss(x, p).item() - want) < 1e-5

    # label smoothing mixes the hard nll with the uniform-mixture term
    t = tp.tensor([0, 1])
    smooth = 0.2
    base = tp.cross_entropy_loss(x, t, label_smoothing=0.0).item()
    n_classes = 2
    logs = ls
    uniform = -(logs[0][0] + logs[0][1] + logs[1][0] + logs[1][1]) / 4
    want = (1 - smooth) * base + smooth * uniform
    got = tp.cross_entropy_loss(x, t, label_smoothing=smooth).item()
    assert abs(got - want) < 1e-5


def test_cross_entropy_loss_is_differentiable():
    x = tp.tensor([[1.0, 2.0], [3.0, 1.0]], requires_grad=True)
    tp.cross_entropy_loss(x, tp.tensor([0, 1])).backward()
    ls0 = _log_softmax_ref([1.0, 2.0])
    ls1 = _log_softmax_ref([3.0, 1.0])
    # mean reduction divides the softmax-minus-onehot gradient by the batch
    want = [
        [(math.exp(ls0[0]) - 1) / 2, math.exp(ls0[1]) / 2],
        [math.exp(ls1[0]) / 2, (math.exp(ls1[1]) - 1) / 2],
    ]
    _allclose(x.grad, want)


# ---------------------------------------------------------------- segment_reduce


def test_segment_reduce_lengths_all_reductions():
    data = tp.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    lengths = tp.tensor([2, 0, 3])
    _allclose(tp.segment_reduce(data, 'sum', lengths=lengths), [3.0, 0.0, 12.0])
    _allclose(tp.segment_reduce(data, 'max', lengths=lengths), [2.0, float('-inf'), 5.0])
    _allclose(tp.segment_reduce(data, 'min', lengths=lengths), [1.0, float('inf'), 3.0])
    out = tp.segment_reduce(data, 'mean', lengths=lengths)
    assert math.isnan(out.tolist()[1])
    _allclose([out.tolist()[0], out.tolist()[2]], [1.5, 4.0])


def test_segment_reduce_offsets_and_initial():
    data = tp.tensor([1.0, 2.0, 3.0, 4.0])
    offsets = tp.tensor([0, 2, 4])
    _allclose(tp.segment_reduce(data, 'sum', offsets=offsets), [3.0, 7.0])
    got = tp.segment_reduce(data, 'mean', offsets=offsets, initial=1.0)
    # initial participates as an additive identity seed for mean/sum
    _allclose(got, [(1.0 + 1.0 + 2.0) / 2, (1.0 + 3.0 + 4.0) / 2])


def test_segment_reduce_requires_boundaries_and_validates_lengths():
    data = tp.tensor([1.0, 2.0, 3.0])
    with pytest.raises(Exception):
        tp.segment_reduce(data, 'sum')
    with pytest.raises(Exception):
        tp.segment_reduce(data, 'sum', lengths=tp.tensor([1, 1]))
    with pytest.raises(Exception):
        tp.segment_reduce(data, 'sum', lengths=tp.tensor([-1, 4]))


def test_segment_reduce_backward_matches_definitions():
    data = tp.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    lengths = tp.tensor([2, 2])
    out = tp.segment_reduce(data, 'sum', lengths=lengths)
    out.sum().backward()
    _allclose(data.grad, [1.0, 1.0, 1.0, 1.0])

    data2 = tp.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    out2 = tp.segment_reduce(data2, 'max', lengths=lengths)
    out2.sum().backward()
    # each segment's maximum receives the gradient
    _allclose(data2.grad, [0.0, 1.0, 0.0, 1.0])


# ------------------------------------------------------------------ _trilinear


def test_trilinear_matches_bilinear_einsum():
    b, n, m, out = 2, 3, 4, 5
    input1 = tp.randn(b, n)
    weight = tp.randn(out, n, m)
    input2 = tp.randn(b, m)
    got = tp._C._trilinear(input1, weight, input2, [1, 3], [0], [1, 2], [2, 3], 1)
    want = tp.einsum('bi,kij,bj->bk', input1, weight, input2)
    _allclose(got.reshape([-1]), want.reshape([-1]), tol=1e-4)


def test_trilinear_is_differentiable():
    input1 = tp.ones(2, 3, requires_grad=True)
    weight = tp.ones(4, 3, 3, requires_grad=True)
    input2 = tp.ones(2, 3, requires_grad=True)
    tp._C._trilinear(input1, weight, input2, [1, 3], [0], [1, 2], [2, 3], 1).sum().backward()
    # d/dweight of sum_k sum_b b_i w_kij b_j with all ones: each entry sums
    # b_i*b_j over the 2 batches -> 2
    _allclose(weight.grad, tp.full((4, 3, 3), 2.0))


# --------------------------------------------------- _transform_bias_rescale_qkv


def test_transform_bias_rescale_qkv_layout_and_scale():
    b, t, heads, dph = 2, 3, 2, 4
    d = heads * dph
    qkv = tp.arange(float(b * t * 3 * d)).reshape(b, t, 3 * d)
    bias = tp.zeros(3 * d)
    q, k, v = tp._C._transform_bias_rescale_qkv(qkv, bias, heads)
    assert tuple(q.shape) == (b, heads, t, dph)
    # q picks the first third, rescaled by 1/sqrt(dim_per_head)
    scale = 1.0 / math.sqrt(dph)
    assert abs(q.reshape(-1)[0].item() - 0.0 * scale) < 1e-6
    assert abs(q.reshape(-1)[1].item() - 1.0 * scale) < 1e-6
    # k picks the middle third unchanged; v the last third
    assert abs(k.reshape(-1)[0].item() - d) < 1e-6
    assert abs(v.reshape(-1)[0].item() - 2 * d) < 1e-6


# -------------------------------------------------------------------------- smm


def test_smm_matches_dense_product_and_keeps_rows():
    dense = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    rhs = tp.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = tp.smm(tp.to_sparse(dense), rhs)
    assert out.is_sparse
    _allclose(out.to_dense(), dense @ rhs)


# ------------------------------------------------------- to_sparse_csc/bsr/bsc


def test_to_sparse_csc_components_and_roundtrip():
    d = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    csc = d.to_sparse_csc()
    assert csc.ccol_indices().tolist() == [0, 2, 3, 5]
    assert csc.row_indices().tolist() == [0, 2, 2, 0, 2]
    assert csc.values().tolist() == [1.0, 3.0, 4.0, 2.0, 5.0]
    _allclose(csc.to_dense(), d)


def test_to_sparse_bsr_block_components_and_roundtrip():
    b = tp.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 4.0]])
    bsr = b.to_sparse_bsr((2, 2))
    assert bsr.crow_indices().tolist() == [0, 1, 3]
    assert bsr.col_indices().tolist() == [0, 0, 1]
    assert tuple(bsr.values().shape) == (3, 2, 2)
    assert bsr.values()[2].tolist() == [[0.0, 0.0], [0.0, 4.0]]
    _allclose(bsr.to_dense(), b)


def test_to_sparse_bsc_roundtrip_and_batched_error_paths():
    d = tp.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 4.0]])
    bsc = d.to_sparse_bsc((2, 2))
    assert bsc.ccol_indices().tolist() == [0, 2, 3]
    _allclose(bsc.to_dense(), d)
    bad = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    with pytest.raises(Exception):
        bad.to_sparse_bsr((2, 2))  # 3 not divisible by 2
    with pytest.raises(Exception):
        bad.to_sparse_bsc((2, 2))


def test_to_sparse_compressed_batched_joins_batches():
    batch = tp.stack([tp.eye(2), 2 * tp.eye(2)])
    bsr = batch.to_sparse_bsr((2, 2))
    assert tuple(bsr.shape) == (2, 2, 2)
    assert bsr.crow_indices().tolist() == [[0, 1], [0, 1]]
    _allclose(bsr.to_dense(), batch)


def test_to_sparse_csc_rejects_uneven_batch_nnz():
    batch = tp.stack([tp.tensor([[1.0, 0.0], [0.0, 1.0]]),
                      tp.tensor([[1.0, 0.0], [0.0, 0.0]])])
    with pytest.raises(Exception):
        batch.to_sparse_csc()


# --------------------------------------------------- _sparse_softmax family


def test_sparse_softmax_pools_over_sparse_dim():
    d = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    sp = d.to_sparse()
    got = tp._C._sparse_softmax(sp, 1).to_dense()
    # unspecified entries act as negative infinities, so a pool's softmax
    # runs over its stored coordinates only
    e = math.exp(1.0)
    _allclose(got.tolist()[0][0], 1.0 / (1.0 + e))
    _allclose(got.tolist()[0][2], e / (1.0 + e))
    _allclose([got.tolist()[2][c] for c in range(3)],
              tp.softmax(d, 1).to_dense().tolist()[2])
    # rows without stored entries keep no output coordinates
    assert got.tolist()[1] == [0.0, 0.0, 0.0]


def test_sparse_softmax_dim_in_dense_part_uses_dense_path():
    # hybrid tensor: 2 sparse dims + a 3-wide dense payload
    idx = tp.tensor([[0, 0], [0, 1]], dtype=tp.int64)
    vals = tp.randn(2, 3)
    sp = tp._C.sparse_coo_tensor(idx, vals, (1, 2, 3))
    got = tp._C._sparse_softmax(sp, 2)
    _allclose(got.values(), tp.softmax(vals, 1))


def test_sparse_softmax_backward_matches_dense_reference_on_matching_coords():
    d = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    sp = d.to_sparse()
    out = tp._C._sparse_softmax(sp, 1)
    grad = tp.to_sparse(tp.tensor([[1.0, 0.0], [0.0, 1.0]]))
    g = tp._C._sparse_softmax_backward_data(grad, out, 1, sp)
    dense_out = tp.softmax(d, 1)
    dense_grad = tp.tensor([[1.0, 0.0], [0.0, 1.0]])
    dense_in = d
    dense_g = dense_out * (dense_grad - (dense_out * dense_grad).sum(1, True))
    assert tuple(g.shape) == tuple(sp.shape)
    _allclose(g.to_dense(), dense_g)


def test_sparse_log_softmax_and_backward():
    d = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    sp = d.to_sparse()
    out = tp._C._sparse_log_softmax(sp, 1)
    # values follow the coordinate order of the coalesced input
    dense_ls = tp.log_softmax(d, 1).to_dense()
    coords = sp.coalesce()._indices().tolist()
    vals = out.values().tolist()
    for n in range(sp._nnz()):
        assert abs(vals[n] - dense_ls.tolist()[coords[0][n]][coords[1][n]]) < 1e-6
    grad = tp.to_sparse(tp.ones(2, 2))
    g = tp._C._sparse_log_softmax_backward_data(grad, out, 1, sp)
    # reference formula per coordinate: gI_i = grad_i - softmax_i * sum_j grad_j
    dense_g = tp.ones(2, 2) - tp.softmax(d, 1) * 2.0
    _allclose(g.to_dense(), dense_g)


# ------------------------------------------------------------- _sparse_sum family


def test_sparse_sum_dim_keeps_partial_result_sparse():
    d = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    sp = d.to_sparse()
    got = tp._C._sparse_sum(sp, [0])
    assert got.is_sparse
    assert got.to_dense().tolist() == [4.0, 4.0, 7.0]
    total = tp._C._sparse_sum(sp)
    assert not total.is_sparse
    assert abs(total.item() - 15.0) < 1e-6


def test_sparse_sum_dtype_accumulates_in_requested_dtype():
    sp = tp.tensor([[1.5, 0.0], [0.0, 2.5]]).to_sparse()
    out = tp._C._sparse_sum(sp, dtype=tp.float64)
    assert out.dtype == tp.float64
    assert abs(out.item() - 4.0) < 1e-6


def test_sparse_sum_backward_scatters_to_input_coordinates():
    d = tp.tensor([[1.0, 0.0], [0.0, 2.0]])
    sp = d.to_sparse()
    got = tp._C._sparse_sum(sp, [0])
    grad = tp.to_sparse(tp.tensor([3.0, 4.0]))
    g = tp._C._sparse_sum_backward(grad, sp, [0])
    assert g.is_sparse
    _allclose(g.to_dense(), tp.tensor([[3.0, 0.0], [0.0, 4.0]]))


# ------------------------------------------------------------------ native_norm


def test_native_norm_full_reduction_matches_definition():
    d = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    sp = d.to_sparse()
    assert abs(tp.native_norm(sp).item() - 30.0 ** 0.5) < 1e-6
    assert abs(tp.native_norm(sp, 1.0).item() - 10.0) < 1e-6


def test_native_norm_rejects_partial_reductions_and_keepdim():
    sp = tp.ones(2, 2).to_sparse()
    with pytest.raises(Exception):
        tp.native_norm(sp, tp.tensor(2.0), [0], False)
    with pytest.raises(Exception):
        tp.native_norm(sp, tp.tensor(2.0), [0, 1], True)


# ------------------------------------------------------------------ hash_tensor


def test_hash_tensor_is_order_independent_and_empty_safe():
    d = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    h = tp.hash_tensor(d, [1])
    assert h.dtype == tp.uint64
    # XOR folding is exchange-safe: reordering elements keeps the hash
    assert h.tolist() == tp.hash_tensor(d.flip(1).contiguous(), [1]).tolist()
    # empty input reduces to an empty tensor (shape (0,) for dim=[1])
    assert tuple(tp.hash_tensor(tp.zeros(0, 2), [1]).shape) == (0,)


def test_hash_tensor_rejects_unknown_mode_and_zero_dims():
    d = tp.ones(2, 2)
    with pytest.raises(Exception):
        tp.hash_tensor(d, [1], mode=7)
    with pytest.raises(Exception):
        tp.hash_tensor(d, [2])  # reduction dim out of range


# --------------------------------------------------------------------- _spdiags


def test_spdiags_private_spelling_matches_public_one():
    diagonals = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    offsets = tp.tensor([0, 1])
    a = tp._C._spdiags(diagonals, offsets, [3, 3])
    b = tp.spdiags(diagonals, offsets, [3, 3])
    _allclose(a.to_dense(), b.to_dense())


# --------------------------------------------------------- slow / THNN convolutions


def test_thnn_conv2d_matches_conv2d():
    x = tp.randn(2, 3, 5, 5)
    w = tp.randn(4, 3, 3, 3)
    bias = tp.randn(4)
    got = tp.thnn_conv2d(x, w, [3, 3], bias, [1, 1], [1, 1])
    want = tp.conv2d(x, w, bias, [1, 1], [1, 1], [1, 1], 1)
    _allclose(got, want, tol=1e-3)


def test_thnn_conv2d_out_writes_into_buffer():
    x = tp.randn(1, 1, 4, 4)
    w = tp.randn(1, 1, 2, 2)
    out = tp.zeros(1, 1, 3, 3)
    tp.thnn_conv2d(x, w, [2, 2], None, [1, 1], [0, 0], out=out)
    _allclose(out, tp.conv2d(x, w, None, [1, 1], [0, 0], [1, 1], 1))


def test_slow_conv3d_matches_conv3d():
    x = tp.randn(1, 2, 4, 4, 4)
    w = tp.randn(3, 2, 2, 2, 2)
    got = tp.slow_conv3d(x, w, [2, 2, 2], None, [1, 1, 1], [0, 0, 0])
    want = tp.conv3d(x, w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1)
    _allclose(got, want, tol=1e-3)


def test_slow_conv_transpose2d_matches_conv_transpose2d():
    x = tp.randn(1, 2, 3, 3)
    w = tp.randn(2, 4, 2, 2)
    got = tp.slow_conv_transpose2d(x, w, [2, 2], None, [1, 1], [0, 0], [0, 0], [1, 1])
    want = tp.conv_transpose2d(x, w, None, [1, 1], [0, 0], [0, 0], 1, [1, 1])
    _allclose(got, want, tol=1e-3)


def test_slow_conv2d_forward_backward_flow():
    x = tp.ones(1, 1, 3, 3, requires_grad=True)
    w = tp.ones(1, 1, 2, 2, requires_grad=True)
    got = tp._C._slow_conv2d_forward(x, w, [2, 2], None, [1, 1], [0, 0])
    got.sum().backward()
    # each input pixel receives one gradient per overlapping kernel position
    _allclose(x.grad, tp.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]))
    # each weight tap sees the full 2x2 output window
    _allclose(w.grad, tp.full(w.shape, 4.0))


# ------------------------------------------------- histogramdd (regression pin)


def test_histogramdd_still_working():
    pts = tp.tensor([[0.5, 0.5], [1.5, 1.5]])
    hist, edges = tp.histogramdd(pts, [2, 2])
    assert hist.tolist()[0][0] == 1.0
    assert hist.tolist()[1][1] == 1.0
    assert len(edges) == 2
