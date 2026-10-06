"""Vector products read strided operands correctly, and the BLAS-style ops serve every dtype.

dot walks each vector by its own stride; vdot of complex vectors returns on
the device it ran on; addmv, addr and addbmm compose complex, whole-number
and truth-value operands from their products.
"""

import math

import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def column(n, dtype, device, col=1):
    """A non-contiguous vector: one column of an (n, 3) matrix."""
    tp.manual_seed(n)
    return tp.randn(n, 3, dtype=dtype, device=device)[:, col]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64, tp.complex128])
def test_dot_reads_strided_vectors_through_their_strides(device, dtype):
    a, b = column(7, dtype, device, 0), column(7, dtype, device, 2)
    want = (a * b).sum()
    assert tp.allclose(tp.dot(a, b), want, atol=1e-5)
    stepped = tp.arange(14, dtype=tp.float64, device=device)[::2]
    plain = tp.arange(7, dtype=tp.float64, device=device)
    assert tp.dot(stepped, plain).item() == (stepped * plain).sum().item()


@pytest.mark.parametrize("device", DEVICES)
def test_dot_of_an_expanded_vector(device):
    a = tp.tensor([2.0], dtype=tp.float64, device=device).expand(5)
    b = tp.arange(5, dtype=tp.float64, device=device)
    assert tp.dot(a, b).item() == 20.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [tp.complex64, tp.complex128])
def test_vdot_of_complex_vectors_conjugates_the_first(device, dtype):
    a = tp.tensor([1 + 2j, 3 - 1j], dtype=dtype, device=device)
    b = tp.tensor([2 - 1j, 1j], dtype=dtype, device=device)
    got = tp.vdot(a, b)
    assert got.device == a.device
    # conj(1+2j)(2-1j) + conj(3-1j)(1j) = (-5j) + (-1+3j) = -1-2j
    assert got.cpu().tolist() == pytest.approx(-1 - 2j)


@pytest.mark.parametrize("device", DEVICES)
def test_complex_addmv_addr_and_addbmm_match_their_compositions(device):
    tp.manual_seed(0)
    c = tp.complex128
    s_v, m, v = (tp.randn(3, dtype=c, device=device), tp.randn(3, 4, dtype=c, device=device),
                 tp.randn(4, dtype=c, device=device))
    got = tp.addmv(s_v, m, v, beta=2 - 1j, alpha=0.5j)
    assert tp.allclose(got, (2 - 1j) * s_v + 0.5j * tp.mv(m, v))
    s_m, v1, v2 = (tp.randn(3, 4, dtype=c, device=device), tp.randn(3, dtype=c, device=device),
                   tp.randn(4, dtype=c, device=device))
    assert tp.allclose(tp.addr(s_m, v1, v2, beta=3, alpha=1j), 3 * s_m + 1j * tp.outer(v1, v2))
    b1, b2 = tp.randn(2, 3, 2, dtype=c, device=device), tp.randn(2, 2, 4, dtype=c, device=device)
    assert tp.allclose(tp.addbmm(s_m, b1, b2), s_m + tp.bmm(b1, b2).sum(0))


def test_whole_number_addmv():
    m = tp.tensor([[1, 2], [3, 4]])
    v = tp.tensor([5, 6])
    s = tp.tensor([1, 1])
    assert tp.addmv(s, m, v, beta=2, alpha=3).tolist() == [2 + 3 * 17, 2 + 3 * 39]
    with pytest.raises(RuntimeError, match="must not be a floating point number"):
        tp.addmv(s, m, v, alpha=0.5)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="needs CUDA")
def test_whole_number_matrix_products_on_cuda_say_so():
    m = tp.tensor([[1, 2], [3, 4]], device="cuda")
    v = tp.tensor([5, 6], device="cuda")
    with pytest.raises(NotImplementedError, match="not implemented for 'Long'"):
        tp.addmv(v, m, v)


@pytest.mark.parametrize("device", DEVICES)
def test_whole_number_addr(device):
    v = tp.tensor([5, 6], device=device)
    s = tp.zeros(2, 2, dtype=tp.int64, device=device)
    assert tp.addr(s, v, v).tolist() == [[25, 30], [30, 36]]
    with pytest.raises(RuntimeError, match="must not be a floating point number"):
        tp.addr(s, v, v, beta=0.5)


@pytest.mark.parametrize("device", DEVICES)
def test_truth_value_addr_is_or_of_and(device):
    s = tp.tensor([[True, False], [False, False]], device=device)
    v1 = tp.tensor([True, False], device=device)
    v2 = tp.tensor([False, True], device=device)
    assert tp.addr(s, v1, v2).tolist() == [[True, True], [False, False]]
    assert tp.addr(s, v1, v2, beta=False).tolist() == [[False, True], [False, False]]
    with pytest.raises(RuntimeError, match="Boolean beta only supported for Boolean results"):
        tp.addr(s.float(), v1.float(), v2.float(), beta=True)


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_floating_precisions_stay_on_the_floating_kernels(device):
    tp.manual_seed(3)
    s = tp.randn(3, dtype=tp.float64, device=device)
    m = tp.randn(3, 4, dtype=tp.float32, device=device)
    v = tp.randn(4, dtype=tp.float32, device=device)
    want = 2 * s + 0.5 * tp.mv(m.double(), v.double())
    got = tp.addmv(s, m, v, beta=2, alpha=0.5)
    assert got.dtype == tp.float64
    assert tp.allclose(got, want, atol=1e-5)
    got = tp.addmv(s.float(), m.double(), v.double(), beta=2, alpha=0.5)
    assert got.dtype == tp.float64
    assert tp.allclose(got, want, atol=1e-5)
    b1, b2 = tp.randn(2, 3, 2, device=device), tp.randn(2, 2, 4, device=device)
    seed = tp.randn(3, 4, dtype=tp.float64, device=device)
    assert tp.addbmm(seed, b1, b2).dtype == tp.float32


@pytest.mark.parametrize("device", DEVICES)
def test_a_zero_beta_ignores_self_even_where_it_is_not_finite(device):
    c = tp.complex128
    s = tp.full((2,), complex(math.nan, 0), dtype=c, device=device)
    m = tp.eye(2, dtype=c, device=device)
    v = tp.tensor([1 + 1j, 2], dtype=c, device=device)
    assert tp.addmv(s, m, v, beta=0).cpu().tolist() == [1 + 1j, 2 + 0j]
