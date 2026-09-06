"""Coverage for operators that previously resolved to no kernel at all."""

import math

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F


def _allclose(a, b, tol=1e-5):
    flat_a = tp.reshape(a, [-1]).tolist()
    flat_b = tp.reshape(b, [-1]).tolist() if isinstance(b, tp.Tensor) else b
    assert len(flat_a) == len(flat_b)
    for x, y in zip(flat_a, flat_b):
        assert abs(x - y) <= tol * max(1.0, abs(y)), (flat_a, flat_b)


# --------------------------------------------------------------------------- prelu


def test_prelu_kernel_backward_matches_definition():
    x = tp.tensor([[-2.0, 3.0], [4.0, -5.0]])
    w = tp.tensor([[0.25, 0.25], [0.25, 0.25]])
    g = tp.ones(2, 2)

    grad_input, grad_weight = tp._C._prelu_kernel_backward(g, x, w)

    assert grad_input.tolist() == [[0.25, 1.0], [1.0, 0.25]]
    assert grad_weight.tolist() == [[-2.0, 0.0], [0.0, -5.0]]


def test_prelu_is_differentiable_in_both_inputs():
    x = tp.tensor([[[-2.0, 1.0]], [[3.0, -4.0]]], requires_grad=True)
    w = tp.tensor([0.5], requires_grad=True)

    F.prelu(x, w).sum().backward()

    assert x.grad is not None and w.grad is not None
    assert x.grad.tolist() == [[[0.5, 1.0]], [[1.0, 0.5]]]
    # only the negative entries carry weight gradient
    assert w.grad.tolist() == [-6.0]


def test_prelu_per_channel_weight_gradient_is_per_channel():
    x = tp.tensor([[[-1.0, -2.0], [-3.0, -4.0]]], requires_grad=True)
    w = tp.tensor([0.5, 0.5], requires_grad=True)

    F.prelu(x, w).sum().backward()

    assert w.grad.shape == (2,)
    assert w.grad.tolist() == [-3.0, -7.0]


# ------------------------------------------------------------------ determinants


def test_det_slogdet_logdet_agree():
    a = tp.tensor([[2.0, 0.0], [0.0, 3.0]])

    assert abs(tp.det(a).item() - 6.0) < 1e-5
    sign, logabsdet = tp.slogdet(a)
    assert sign.item() == 1.0
    assert abs(logabsdet.item() - math.log(6.0)) < 1e-5
    assert abs(tp.logdet(a).item() - math.log(6.0)) < 1e-5


def test_logdet_is_nan_for_negative_determinant():
    a = tp.tensor([[1.0, 0.0], [0.0, -1.0]])
    assert math.isnan(tp.logdet(a).item())


def test_linalg_det_reports_the_factorization():
    a = tp.tensor([[4.0, 3.0], [6.0, 3.0]])
    result, lu, pivots = tp._C._linalg_det(a)

    assert abs(result.item() - (-6.0)) < 1e-5
    assert lu.shape == (2, 2)
    assert pivots.numel() == 2


# ------------------------------------------------------------- matrix exponential


def test_matrix_exp_of_zero_is_identity():
    out = tp.matrix_exp(tp.zeros(3, 3))
    _allclose(out, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])


def test_matrix_exp_of_diagonal_is_elementwise_exp():
    a = tp.tensor([[1.0, 0.0], [0.0, -2.0]])
    out = tp.matrix_exp(a)
    _allclose(out, [math.exp(1.0), 0.0, 0.0, math.exp(-2.0)])


def test_matrix_exp_of_nilpotent_is_exact():
    # N^2 == 0, so exp(N) == I + N
    a = tp.tensor([[0.0, 5.0], [0.0, 0.0]])
    _allclose(tp.matrix_exp(a), [1.0, 5.0, 0.0, 1.0])


def test_matrix_exp_needs_scaling_and_squaring():
    # ||A||_1 well past the largest Pade threshold forces the squaring path
    a = tp.tensor([[10.0, 0.0], [0.0, 10.0]])
    _allclose(tp.matrix_exp(a), [math.exp(10.0), 0.0, 0.0, math.exp(10.0)], tol=1e-4)


def test_matrix_exp_inverse_property():
    a = tp.tensor([[0.3, 1.2], [-0.7, 0.4]])
    product = tp.matmul(tp.matrix_exp(a), tp.matrix_exp(tp.neg(a)))
    _allclose(product, [1.0, 0.0, 0.0, 1.0], tol=1e-4)


def test_matrix_exp_batched_mixed_magnitudes():
    a = tp.stack([tp.tensor([[0.0, 0.0], [0.0, 0.0]]),
                  tp.tensor([[8.0, 0.0], [0.0, 8.0]])])
    out = tp.matrix_exp(a)
    _allclose(out[0], [1.0, 0.0, 0.0, 1.0])
    _allclose(out[1], [math.exp(8.0), 0.0, 0.0, math.exp(8.0)], tol=1e-4)


def test_matrix_exp_is_differentiable():
    a = tp.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    tp.matrix_exp(a).sum().backward()
    assert a.grad is not None
    assert a.grad.shape == (2, 2)


# ------------------------------------------------------------------- weight norm


def test_norm_except_dim_reduces_every_other_axis():
    v = tp.tensor([[3.0, 4.0], [5.0, 12.0]])
    out = tp._C.norm_except_dim(v, 2, 0)
    assert out.shape == (2, 1)
    _allclose(out, [5.0, 13.0])


def test_weight_norm_rescales_each_slice():
    v = tp.tensor([[3.0, 4.0], [5.0, 12.0]])
    g = tp.tensor([[2.0], [1.0]])

    w = tp._C._weight_norm(v, g, 0)
    _allclose(w, [3.0 * 2 / 5, 4.0 * 2 / 5, 5.0 / 13, 12.0 / 13])

    fused, norm = tp._C._weight_norm_interface(v, g, 0)
    _allclose(fused, tp.reshape(w, [-1]).tolist())
    assert norm.shape == (2, 1)


def test_weight_norm_backward_spellings_agree():
    v = tp.tensor([[3.0, 4.0], [5.0, 12.0]])
    g = tp.tensor([[2.0], [1.0]])
    _, norm = tp._C._weight_norm_interface(v, g, 0)
    grad_w = tp.tensor([[1.0, -1.0], [0.5, 0.25]])

    fused = tp._C._weight_norm_interface_backward(grad_w, v, g, norm, 0)
    diff = tp._C._weight_norm_differentiable_backward(grad_w, v, g, norm, 0)

    _allclose(fused[0], tp.reshape(diff[0], [-1]).tolist())
    _allclose(fused[1], tp.reshape(diff[1], [-1]).tolist())
    assert fused[1].shape == g.shape


# ------------------------------------------------------------------ metadata ops


def test_detach_shares_storage_and_drops_history():
    x = tp.randn(3, requires_grad=True)
    d = x.detach()
    assert d.requires_grad is False
    assert d.data_ptr() == x.data_ptr()


def test_as_strided_and_storage_offset():
    base = tp.arange(6, dtype=tp.float32)
    view = tp._C.as_strided(base, [2, 2], [2, 1], 1)
    assert view.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert view.storage_offset() == 1
    assert base.storage_offset() == 0


def test_view_dtype_reinterprets_storage():
    x = tp.zeros(4, dtype=tp.float32)
    assert x.view(tp.int32).tolist() == [0, 0, 0, 0]


# ----------------------------------------------------------------------- eye


@pytest.mark.parametrize("dtype", [tp.float32, tp.float64, tp.int32, tp.int64,
                                  tp.int16, tp.int8, tp.uint8, tp.bool,
                                  tp.float16, tp.bfloat16])
def test_eye_is_the_identity_for_every_dtype(dtype):
    out = tp.eye(3, dtype=dtype)
    assert out.to(tp.float32).tolist() == [[1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0]]


def test_eye_complex():
    out = tp.eye(2, dtype=tp.complex64)
    assert out.real.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert out.imag.tolist() == [[0.0, 0.0], [0.0, 0.0]]


# -------------------------------------------------------------------- scatter


def test_scatter_reduce_variant_adds_into_self():
    self_ = tp.zeros(1, 4)
    index = tp.tensor([[0, 0, 0, 0]])
    src = tp.tensor([[1.0, 2.0, 3.0, 4.0]])

    out = tp._C.scatter(self_, 1, index, src, reduce="add")
    assert out.tolist() == [[10.0, 0.0, 0.0, 0.0]]


def test_scatter_reduce_variant_multiplies_into_self():
    self_ = tp.ones(1, 3)
    index = tp.tensor([[1, 1]])
    src = tp.tensor([[3.0, 4.0]])

    out = tp._C.scatter(self_, 1, index, src, reduce="multiply")
    assert out.tolist() == [[1.0, 12.0, 1.0]]


def test_scatter_value_reduce_variant():
    self_ = tp.zeros(1, 3)
    index = tp.tensor([[0, 2]])

    out = tp._C.scatter(self_, 1, index, 5.0, reduce="add")
    assert out.tolist() == [[5.0, 0.0, 5.0]]


def test_scatter_out_writes_into_the_given_buffer():
    self_ = tp.zeros(1, 3)
    index = tp.tensor([[0]])
    src = tp.tensor([[7.0]])
    out = tp.zeros(1, 3)

    result = tp._C.scatter(self_, 1, index, src, out=out)
    assert result is out
    assert out.tolist() == [[7.0, 0.0, 0.0]]


# --------------------------------------------------------------- shape / misc


def test_stack_internal_spelling():
    a = tp.tensor([1.0, 2.0])
    out = tp._C._stack([a, a], 0)
    assert out.tolist() == [[1.0, 2.0], [1.0, 2.0]]


def test_stack_out_reuses_the_buffer():
    a = tp.tensor([1.0, 2.0])
    out = tp.zeros(2, 2)
    ptr = out.data_ptr()
    result = tp._C._stack([a, a], 0, out=out)
    assert result is out
    assert out.data_ptr() == ptr
    assert out.tolist() == [[1.0, 2.0], [1.0, 2.0]]


def test_nonzero_numpy_splits_per_axis():
    x = tp.tensor([[0.0, 1.0], [2.0, 0.0]])
    rows, cols = tp._C.nonzero_numpy(x)
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 0]


def test_nonzero_numpy_on_a_scalar():
    assert tp._C.nonzero_numpy(tp.tensor(5.0))[0].tolist() == [0]
    assert tp._C.nonzero_numpy(tp.tensor(0.0))[0].tolist() == []


def test_type_as_adopts_the_other_dtype():
    x = tp.tensor([1.5, 2.5])
    other = tp.zeros(1, dtype=tp.int32)
    assert x.type_as(other).dtype == tp.int32


def test_unsafe_view_and_reshape_copy():
    x = tp.arange(6, dtype=tp.float32)
    assert tp._C._unsafe_view(x, [2, 3]).shape == (2, 3)
    copied = tp._C._reshape_copy(x, [2, 3])
    assert copied.shape == (2, 3)
    assert copied.data_ptr() != x.data_ptr()


def test_empty_permuted_gives_the_requested_physical_order():
    out = tp._C.empty_permuted([2, 3, 4, 5], [0, 2, 3, 1])
    assert out.shape == (2, 3, 4, 5)
    # channels-last: the logical axis 1 is the fastest moving one
    assert out.stride(1) == 1
    assert out.stride(3) == 3


def test_safe_softmax_zeroes_fully_masked_rows():
    inf = float("-inf")
    x = tp.tensor([[inf, inf], [0.0, 0.0]])
    out = tp._C._safe_softmax(x, -1)
    assert out.tolist() == [[0.0, 0.0], [0.5, 0.5]]


# ------------------------------------------------------------ reductions / misc


def test_logcumsumexp_internal_spelling():
    x = tp.tensor([0.0, 0.0, 0.0])
    out = tp._C._logcumsumexp(x, 0)
    _allclose(out, [0.0, math.log(2.0), math.log(3.0)])


def test_euclidean_dist_matches_the_direct_formula():
    a = tp.tensor([[0.0, 0.0], [3.0, 4.0]])
    b = tp.tensor([[0.0, 0.0], [1.0, 0.0]])
    out = tp._C._euclidean_dist(a, b)
    _allclose(out, [0.0, 1.0, 5.0, math.hypot(2.0, 4.0)], tol=1e-4)


def test_pdist_and_cdist_internal_spellings():
    a = tp.tensor([[0.0, 0.0], [3.0, 4.0]])
    _allclose(tp._C._pdist_forward(a, 2.0), [5.0], tol=1e-4)
    _allclose(tp._C._cdist_forward(a, a, 2.0, None), [0.0, 5.0, 5.0, 0.0], tol=1e-4)


def test_inplace_gamma_spellings():
    x = tp.tensor([1.0, 2.0])
    assert x.polygamma_(0) is x
    y = tp.tensor([1.0, 2.0])
    assert y.igamma_(tp.tensor([1.0, 1.0])) is y
    z = tp.tensor([1.0, 2.0])
    assert z.igammac_(tp.tensor([1.0, 1.0])) is z


# ------------------------------------------------------------------- lu_unpack


def test_lu_unpack_reconstructs_the_matrix():
    a = tp.tensor([[4.0, 3.0], [6.0, 3.0]])
    lu, pivots = tp.linalg.lu_factor(a)
    P, L, U = tp.lu_unpack(lu, pivots)

    assert P.shape == (2, 2)
    assert L.shape == (2, 2)
    assert U.shape == (2, 2)
    _allclose(tp.matmul(P, tp.matmul(L, U)), tp.reshape(a, [-1]).tolist(), tol=1e-4)


def test_lu_unpack_unit_diagonal_and_triangles():
    a = tp.tensor([[4.0, 3.0], [6.0, 3.0]])
    lu, pivots = tp.linalg.lu_factor(a)
    _, L, U = tp.lu_unpack(lu, pivots)

    values = L.tolist()
    assert values[0][0] == 1.0 and values[1][1] == 1.0
    assert values[0][1] == 0.0
    assert U.tolist()[1][0] == 0.0


def test_lu_unpack_can_skip_either_half():
    a = tp.tensor([[4.0, 3.0], [6.0, 3.0]])
    lu, pivots = tp.linalg.lu_factor(a)

    P, L, U = tp.lu_unpack(lu, pivots, unpack_data=False)
    assert P.numel() == 4 and L.numel() == 0 and U.numel() == 0

    P, L, U = tp.lu_unpack(lu, pivots, unpack_pivots=False)
    assert P.numel() == 0 and L.numel() == 4 and U.numel() == 4


def test_lu_unpack_batched():
    a = tp.stack([tp.tensor([[4.0, 3.0], [6.0, 3.0]]),
                  tp.tensor([[1.0, 2.0], [3.0, 4.0]])])
    lu, pivots = tp.linalg.lu_factor(a)
    P, L, U = tp.lu_unpack(lu, pivots)
    _allclose(tp.matmul(P, tp.matmul(L, U)), tp.reshape(a, [-1]).tolist(), tol=1e-4)
