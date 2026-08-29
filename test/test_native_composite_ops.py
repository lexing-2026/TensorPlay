"""Behavior checks for the composite native ops (src/backend/composite/*.cpp).

kernels are built from TensorPlay's own primitive ops, so agreement here also
exercises the underlying primitives and the dispatcher's Composite fallthrough.
"""

import numpy as np
import pytest

import torch
import torch.nn.functional as torch_F

import tensorplay as tp
import tensorplay.functional as tp_F


def _np(value):
    return value.detach().cpu().numpy()


def _tp(value):
    return tp.tensor(value.detach().cpu().numpy(), device="cpu")


def _assert_close(got, expected, *, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(_np(got), _np(expected) if isinstance(expected, torch.Tensor) else expected,
                               rtol=rtol, atol=atol)


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


# ---------------------------------------------------------------------------
# Index factories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row,col,offset", [
    (3, 3, 0), (4, 5, 1), (5, 2, -1), (0, 3, 0), (3, 0, 0), (1, 1, 5), (6, 6, -7),
])
def test_tril_triu_indices(row, col, offset):
    for tp_fn, torch_fn in ((tp_F.tril_indices, torch.tril_indices),
                            (tp_F.triu_indices, torch.triu_indices)):
        got = tp_fn(row, col, offset).numpy()
        expected = torch_fn(row, col, offset).numpy()
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [tp.int32, tp.int64])
def test_tril_indices_dtypes(dtype):
    torch_dtype = torch.int32 if dtype == tp.int32 else torch.int64
    got = tp_F.tril_indices(4, 4, dtype=dtype)
    expected = torch.tril_indices(4, 4, dtype=torch_dtype)
    assert _dtype_name(got.dtype) == _dtype_name(expected.dtype)
    np.testing.assert_array_equal(got.numpy(), expected.numpy())


# ---------------------------------------------------------------------------
# Linear algebra / itertools
# ---------------------------------------------------------------------------

def test_chain_matmul():
    mats_np = [np.random.randn(3, 4), np.random.randn(4, 2), np.random.randn(2, 5)]
    got = tp_F.chain_matmul(*[_tp(torch.from_numpy(m)) for m in mats_np])
    expected = torch.chain_matmul(*[torch.from_numpy(m) for m in mats_np])
    _assert_close(got, expected, rtol=1e-4)


def test_chain_matmul_single():
    m = torch.randn(3, 3)
    _assert_close(tp_F.chain_matmul(_tp(m)), m)


@pytest.mark.parametrize("r,with_replacement", [(2, False), (3, False), (2, True)])
def test_combinations(r, with_replacement):
    x = torch.tensor([1, 2, 3, 4])
    got = tp_F.combinations(_tp(x), r, with_replacement)
    expected = torch.combinations(x, r, with_replacement)
    np.testing.assert_array_equal(got.numpy(), expected.numpy())


# ---------------------------------------------------------------------------
# Integration / histograms / membership
# ---------------------------------------------------------------------------

def test_trapz_x():
    y = torch.tensor([[1.0, 4.0, 9.0], [2.0, 3.0, 5.0]])
    x = torch.tensor([0.0, 1.0, 3.0])
    _assert_close(tp_F.trapz(_tp(y), _tp(x)), torch.trapz(y, x))
    x0 = torch.tensor([0.0, 2.0])
    _assert_close(tp_F.trapz(_tp(y), _tp(x0), dim=0), torch.trapz(y, x0, dim=0))


def test_trapz_dx():
    y = torch.tensor([1.0, 4.0, 9.0])
    _assert_close(tp_F.trapz(_tp(y)), torch.trapz(y))
    _assert_close(tp_F.trapz(_tp(y), dx=0.5), torch.trapz(y, dx=0.5))


def test_histc():
    x = torch.tensor([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
    _assert_close(tp_F.histc(_tp(x), 4, 1.0, 5.0), torch.histc(x, 4, 1.0, 5.0))
    _assert_close(tp_F.histc(_tp(x)), torch.histc(x))


def test_isin_overloads():
    elements = torch.tensor([1, 2, 3, 4, 5])
    test_elements = torch.tensor([2, 4, 6])
    np.testing.assert_array_equal(
        tp_F.isin(_tp(elements), _tp(test_elements)).numpy(),
        torch.isin(elements, test_elements).numpy())
    np.testing.assert_array_equal(
        tp_F.isin(_tp(elements), 3).numpy(), torch.isin(elements, 3).numpy())
    np.testing.assert_array_equal(
        tp_F.isin(3, _tp(test_elements)).numpy(), torch.isin(3, test_elements).numpy())
    np.testing.assert_array_equal(
        tp_F.isin(_tp(elements), _tp(test_elements), invert=True).numpy(),
        torch.isin(elements, test_elements, invert=True).numpy())


def test_unique_consecutive():
    x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 1])
    got = tp_F.unique_consecutive(_tp(x))
    expected = torch.unique_consecutive(x)
    np.testing.assert_array_equal(got.numpy(), expected.numpy())

    got_o, got_inv, got_counts = tp_F.unique_consecutive(_tp(x), True, True)
    exp_o, exp_inv, exp_counts = torch.unique_consecutive(x, True, True)
    np.testing.assert_array_equal(got_o.numpy(), exp_o.numpy())
    np.testing.assert_array_equal(got_inv.numpy(), exp_inv.numpy())
    np.testing.assert_array_equal(got_counts.numpy(), exp_counts.numpy())


def test_unique_consecutive_dim():
    x = torch.tensor([[1, 1], [1, 1], [2, 3], [2, 3], [1, 1]])
    got = tp_F.unique_consecutive(_tp(x), dim=0)
    expected = torch.unique_consecutive(x, dim=0)
    np.testing.assert_array_equal(got.numpy(), expected.numpy())


# ---------------------------------------------------------------------------
# Scalars, predicates, dtype promotion
# ---------------------------------------------------------------------------

def test_scalar_tensor():
    t = tp_F.scalar_tensor(3.5)
    assert t.shape == () and abs(t.item() - 3.5) < 1e-6
    t = tp_F.scalar_tensor(7, dtype=tp.int32)
    assert _dtype_name(t.dtype) == "int32" and t.item() == 7


def test_tensor_predicates():
    x = tp.tensor([1.0, 2.0])
    assert tp_F.is_conj(x) is False or tp_F.is_conj(x) == False  # noqa: E712
    assert tp_F.is_neg(x) == False  # noqa: E712
    assert tp_F.is_nonzero(tp.tensor([1.0])) == True  # noqa: E712
    assert tp_F.is_nonzero(tp.tensor([0.0])) == False  # noqa: E712
    assert tp_F.is_same_size(x, tp.tensor([3.0, 4.0])) == True  # noqa: E712
    assert tp_F.is_same_size(x, tp.tensor([3.0])) == False  # noqa: E712
    assert tp_F.get_device(x) == -1


def test_can_cast_and_promote():
    pairs = [(torch.float32, torch.float64), (torch.int32, torch.float16),
             (torch.float64, torch.int64), (torch.bool, torch.float32),
             (torch.complex64, torch.float32)]
    for a, b in pairs:
        ta = getattr(tp, _dtype_name(a))
        tb = getattr(tp, _dtype_name(b))
        assert tp_F.can_cast(ta, tb) == torch.can_cast(a, b)
        assert _dtype_name(tp_F.promote_types(ta, tb)) == _dtype_name(torch.promote_types(a, b))


def test_result_type_overloads():
    tf32, ti64 = torch.tensor([1.0]), torch.tensor([1])
    assert _dtype_name(tp_F.result_type(_tp(tf32), _tp(ti64))) == \
        _dtype_name(torch.result_type(tf32, ti64))
    assert _dtype_name(tp_F.result_type(_tp(tf32), 1)) == \
        _dtype_name(torch.result_type(tf32, 1))
    assert _dtype_name(tp_F.result_type(1.0, _tp(ti64))) == \
        _dtype_name(torch.result_type(1.0, ti64))
    assert _dtype_name(tp_F.result_type(1, 2.0)) == \
        _dtype_name(torch.result_type(1, 2.0))


def test_put():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx = torch.tensor([0, 2])
    src = torch.tensor([9.0, 8.0])
    _assert_close(tp_F.put(_tp(x), _tp(idx), _tp(src)), torch.put(x, idx, src))
    _assert_close(tp_F.put(_tp(x), _tp(idx), _tp(src), True),
                  torch.put(x, idx, src, True))


def test_resolve_conj_neg():
    x = torch.tensor([1.0, 2.0])
    np.testing.assert_array_equal(tp_F.resolve_conj(_tp(x)).numpy(), x.numpy())
    np.testing.assert_array_equal(tp_F.resolve_neg(_tp(x)).numpy(), x.numpy())


# ---------------------------------------------------------------------------
# The *_copy / view-materialization family
# ---------------------------------------------------------------------------

def test_copy_family():
    x = torch.arange(24.0).reshape(2, 3, 4)
    tx = _tp(x)

    _assert_close(tp_F.alias_copy(tx), x)
    _assert_close(tp_F.t_copy(_tp(x[0])), x[0].t())
    _assert_close(tp_F.permute_copy(tx, [2, 0, 1]), x.permute(2, 0, 1))
    _assert_close(tp_F.transpose_copy(tx, 0, 2), x.transpose(0, 2))
    _assert_close(tp_F.squeeze_copy(_tp(x[:, :1, :])), x[:, :1, :].squeeze())
    _assert_close(tp_F.squeeze_copy(_tp(x[:, :1, :]), 1), x[:, :1, :].squeeze(1))
    _assert_close(tp_F.unsqueeze_copy(tx, 0), x.unsqueeze(0))
    _assert_close(tp_F.select_copy(tx, 1, 2), x.select(1, 2))
    _assert_close(tp_F.slice_copy(tx, 2, 1, 3), x[:, :, 1:3])
    _assert_close(tp_F.narrow_copy(tx, 1, 1, 2), x.narrow(1, 1, 2))
    _assert_close(tp_F.diagonal_copy(_tp(x[0])), x[0].diagonal())
    _assert_close(tp_F.diagonal_copy(_tp(x[0]), 1), x[0].diagonal(1))

    got = tp_F.unbind_copy(tx, 1)
    expected = x.unbind(1)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        _assert_close(g, e)

    got = tp_F.split_copy(_tp(torch.arange(10.0)), 3)
    expected = torch.arange(10.0).split(3)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        _assert_close(g, e)

    _assert_close(tp_F.view_copy(tx, [6, 4]), x.view(6, 4))
    _assert_close(tp_F.expand_copy(_tp(torch.ones(1, 3)), [4, 3]),
                  torch.ones(1, 3).expand(4, 3))
    _assert_close(tp_F.unfold_copy(_tp(torch.arange(7.0)), 0, 3, 2),
                  torch.arange(7.0).unfold(0, 3, 2))
    _assert_close(tp_F.reshape_as(tx, _tp(torch.zeros(4, 6))), x.reshape(4, 6))

    got = tp_F.unsafe_chunk(tx, 2, 0)
    expected = torch.chunk(x, 2, 0)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        _assert_close(g, e)

    got = tp_F.unsafe_split(_tp(torch.arange(10.0)), 4)
    expected = torch.split(torch.arange(10.0), 4)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        _assert_close(g, e)


def test_view_copy_dtype():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    got = tp_F.view_copy(_tp(x), tp.int32)
    expected = x.view(torch.int32)
    np.testing.assert_array_equal(got.numpy(), expected.numpy())


# ---------------------------------------------------------------------------
# Similarity / regularization / activation
# ---------------------------------------------------------------------------

def test_cosine_similarity():
    a = torch.randn(5, 8)
    b = torch.randn(5, 8)
    _assert_close(tp_F.cosine_similarity(_tp(a), _tp(b)),
                  torch_F.cosine_similarity(a, b))
    _assert_close(tp_F.cosine_similarity(_tp(a), _tp(b), dim=0),
                  torch_F.cosine_similarity(a, b, dim=0))


def test_dropout_eval_is_identity():
    x = torch.randn(16)
    for fn in (tp_F.dropout, tp_F.alpha_dropout, tp_F.feature_dropout,
               tp_F.feature_alpha_dropout):
        np.testing.assert_array_equal(fn(_tp(x), 0.5, False).numpy(), x.numpy())


def test_dropout_train_statistics():
    torch.manual_seed(0)
    x = torch.ones(4096)
    out = tp_F.dropout(_tp(x), 0.5, True).numpy()
    frac_zero = float((out == 0).mean())
    assert 0.4 < frac_zero < 0.6
    nonzero = out[out != 0]
    np.testing.assert_allclose(nonzero, 2.0, rtol=1e-6)


def test_rrelu_eval():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    _assert_close(tp_F.rrelu(_tp(x), 0.125, 0.5, False),
                  torch_F.rrelu(x, 0.125, 0.5, False))


def test_rrelu_train_bounds():
    torch.manual_seed(0)
    x = -torch.ones(512)
    out = tp_F.rrelu(_tp(x), 0.125, 0.5, True).numpy()
    assert out.min() >= -0.5 - 1e-6 and out.max() <= -0.125 + 1e-6


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def test_bilinear():
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 7)
    w = torch.randn(5, 6, 7)
    bias = torch.randn(5)
    _assert_close(tp_F.bilinear(_tp(x1), _tp(x2), _tp(w), _tp(bias)),
                  torch_F.bilinear(x1, x2, w, bias), rtol=1e-4)
    _assert_close(tp_F.bilinear(_tp(x1), _tp(x2), _tp(w)),
                  torch_F.bilinear(x1, x2, w), rtol=1e-4)


def test_bilinear_backward():
    x1 = torch.randn(3, 4, requires_grad=True)
    x2 = torch.randn(3, 5, requires_grad=True)
    w = torch.randn(2, 4, 5, requires_grad=True)
    bias = torch.randn(2, requires_grad=True)
    torch_F.bilinear(x1, x2, w, bias).sum().backward()

    t1 = _tp(x1.detach()); t1.requires_grad_(True)
    t2 = _tp(x2.detach()); t2.requires_grad_(True)
    tw = _tp(w.detach()); tw.requires_grad_(True)
    tb = _tp(bias.detach()); tb.requires_grad_(True)
    tp_F.bilinear(t1, t2, tw, tb).sum().backward()

    _assert_close(t1.grad, x1.grad, rtol=1e-4)
    _assert_close(t2.grad, x2.grad, rtol=1e-4)
    _assert_close(tw.grad, w.grad, rtol=1e-4)
    _assert_close(tb.grad, bias.grad, rtol=1e-4)


def test_conv_tbc():
    x = torch.randn(7, 2, 3)          # (time, batch, channels)
    w = torch.randn(3, 3, 4)          # (width, in, out)
    b = torch.randn(4)
    for pad in (0, 1):
        _assert_close(tp_F.conv_tbc(_tp(x), _tp(w), _tp(b), pad),
                      torch_F.conv_tbc(x, w, b, pad), rtol=1e-4)


def test_lstm_cell():
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    hx = (torch.randn(2, 8), torch.randn(2, 8))
    w_ih, w_hh = torch.randn(32, 4), torch.randn(32, 8)
    b_ih, b_hh = torch.randn(32), torch.randn(32)
    got = tp_F.lstm_cell(_tp(x), _tp(hx[0]), _tp(hx[1]),
                         _tp(w_ih), _tp(w_hh), _tp(b_ih), _tp(b_hh))
    expected = torch._VF.lstm_cell(x, hx, w_ih, w_hh, b_ih, b_hh)
    _assert_close(got[0], expected[0], rtol=1e-4)
    _assert_close(got[1], expected[1], rtol=1e-4)


@pytest.mark.parametrize("fn,torch_fn", [
    (tp_F.rnn_relu_cell, torch._VF.rnn_relu_cell),
    (tp_F.rnn_tanh_cell, torch._VF.rnn_tanh_cell),
])
def test_rnn_cells(fn, torch_fn):
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    hx = torch.randn(2, 8)
    w_ih, w_hh = torch.randn(8, 4), torch.randn(8, 8)
    b_ih, b_hh = torch.randn(8), torch.randn(8)
    _assert_close(fn(_tp(x), _tp(hx), _tp(w_ih), _tp(w_hh), _tp(b_ih), _tp(b_hh)),
                  torch_fn(x, hx, w_ih, w_hh, b_ih, b_hh), rtol=1e-4)


def test_native_channel_shuffle():
    x = torch.randn(2, 6, 4, 4)
    for groups in (2, 3):
        _assert_close(tp_F.native_channel_shuffle(_tp(x), groups),
                      torch.native_channel_shuffle(x, groups))


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def test_kaiser_window():
    for length in (0, 1, 5, 8):
        _assert_close(tp_F.kaiser_window(length), torch.kaiser_window(length))
    _assert_close(tp_F.kaiser_window(6, False), torch.kaiser_window(6, False))
    _assert_close(tp_F.kaiser_window(6, True, 8.0), torch.kaiser_window(6, True, 8.0))
    got = tp_F.kaiser_window(5, dtype=tp.float64)
    assert _dtype_name(got.dtype) == "float64"
    _assert_close(got, torch.kaiser_window(5, dtype=torch.float64))
