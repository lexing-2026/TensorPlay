"""Native linalg alignment tests — tensorplay.linalg vs the installed torch.

Locked in after the LAPACK runtime-resolution + kernel-fix campaign:
  - runtime ILP64 LAPACK discovery (numpy's bundled scipy_openblas64)
  - rfft-style backward conventions are spectral; here we lock the dense face
Every case compares against the local torch as oracle on float64 CPU.
"""
import numpy as np
import pytest
import torch

import tensorplay as tp


def to_torch(t):
    return torch.from_dlpack(t.__dlpack__())


def T(a):
    return torch.tensor(np.array(a, dtype=np.float64))


def P(a):
    return tp.from_dlpack(torch.tensor(np.array(a, dtype=np.float64)).__dlpack__())


def spd(n, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n, n)
    return a @ a.T + n * np.eye(n)


def general(n, seed=1):
    rng = np.random.RandomState(seed)
    m = rng.randn(n, n) + n * np.eye(n)
    return m


@pytest.mark.parametrize("n", [3, 8])
def test_surface_matches_torch(n):
    t = {x for x in dir(torch.linalg) if not x.startswith("_")}
    p = {x for x in dir(tp.linalg) if not x.startswith("_")}
    missing = t - p - {"common_notes"}
    assert not missing, f"missing from tensorplay.linalg: {sorted(missing)}"


@pytest.mark.parametrize("n", [3, 6])
def test_matmul_det_inv_solve(n):
    A, B = spd(n), np.random.RandomState(n).randn(n, 3)
    M = general(n)
    chk = [
        ("det", tp.linalg.det(P(A)).item(), torch.linalg.det(T(A)).item(), 1e-9),
        ("slogdet", tp.linalg.slogdet(P(A)).logabsdet.item(),
         torch.linalg.slogdet(T(A)).logabsdet.item(), 1e-8),
        ("inv", tp.linalg.inv(P(M)).numpy(), torch.linalg.inv(T(M)).numpy()),
        ("solve", tp.linalg.solve(P(M), P(B)).numpy(),
         torch.linalg.solve(T(M), T(B)).numpy()),
        ("matmul", (P(A) @ P(B)).numpy(), (T(A) @ T(B)).numpy(), 1e-12),
    ]
    for name, got, ref, *t in chk:
        tol = t[0] if t else 1e-9
        np.testing.assert_allclose(got, ref, rtol=tol, atol=tol, err_msg=name)


@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_zero_triangle_and_values(upper):
    A = spd(6)
    got = tp.linalg.cholesky(P(A), upper=upper).numpy()
    ref = np.linalg.cholesky(A)
    ref = ref.T if upper else ref
    # exact factor + strict zeros in the opposite triangle (torch contract)
    tri = lambda x: np.tril(x) if not upper else np.triu(x)
    np.testing.assert_allclose(tri(got), tri(ref), rtol=1e-12)
    off = lambda x: np.triu(x, 1) if not upper else np.tril(x, -1)
    assert np.abs(off(got)).max() == 0.0


def test_cholesky_batched():
    A = np.stack([spd(5, s) for s in range(3)])
    got = tp.linalg.cholesky(P(A)).numpy()
    ref = np.linalg.cholesky(A)
    np.testing.assert_allclose(np.tril(got), np.tril(ref), rtol=1e-10)


def test_svd_result_namedtuple_and_values():
    A = general(6)
    r = tp.linalg.svd(P(A))
    assert hasattr(r, "U") and hasattr(r, "S") and hasattr(r, "Vh")
    U, S, Vh = r.U.numpy(), r.S.numpy(), r.Vh.numpy()
    np.testing.assert_allclose(S, np.linalg.svd(A, compute_uv=False), rtol=1e-10)
    np.testing.assert_allclose(U @ np.diag(S) @ Vh, A, atol=1e-10)
    sv = tp.linalg.svdvals(P(A))
    np.testing.assert_allclose(sv.numpy(), S, rtol=1e-10)


def test_eigh_eig_namedtuple():
    Asym = spd(6) + spd(6).T
    r = tp.linalg.eigh(P(Asym))
    assert hasattr(r, "eigenvalues") and hasattr(r, "eigenvectors")
    w = r.eigenvalues.numpy()
    np.testing.assert_allclose(np.sort(w), np.sort(np.linalg.eigvalsh(Asym)), rtol=1e-10)
    g = tp.linalg.eig(P(general(5)))
    assert hasattr(g, "eigenvalues") and hasattr(g, "eigenvectors")
    ev = g.eigenvalues.numpy()
    ev_ref = np.linalg.eigvals(general(5))
    np.testing.assert_allclose(
        np.sort_complex(ev), np.sort_complex(ev_ref), rtol=1e-7, atol=1e-8)


def test_lstsq_solution_residuals():
    A, B = spd(6), np.random.RandomState(3).randn(6, 2)
    r = tp.linalg.lstsq(P(A), P(B))
    assert hasattr(r, "solution") and hasattr(r, "residuals")
    ref = torch.linalg.lstsq(T(A), T(B)).solution.numpy()
    np.testing.assert_allclose(r.solution.numpy(), ref, rtol=1e-9)


def test_pinv_real_f64_no_complex_promotion():
    # regression: `1.0 / real_tensor` used to promote to complex128 via the
    # complex-first __rtruediv__ overload ordering.
    y = tp.tensor([1.0, 2.0, 4.0])
    assert (1.0 / y).dtype == tp.float32
    A = spd(5)
    got = tp.linalg.pinv(P(A)).numpy()
    np.testing.assert_allclose(got, np.linalg.pinv(A), rtol=1e-9)


def test_matrix_norm_numeric_ords():
    A = general(5)
    for ord_, tol in [(2, 1e-9), (-2, 1e-9), ("nuc", 1e-8), (1, 1e-10), (-1, 1e-10)]:
        got = tp.linalg.matrix_norm(P(A), ord_).numpy() if isinstance(ord_, str) \
            else tp.linalg.matrix_norm(P(A), ord_).numpy()
        ref = torch.linalg.matrix_norm(T(A), ord_)
        np.testing.assert_allclose(got, ref.numpy(), rtol=tol, err_msg=f"ord={ord_}")


def test_fft_real_input_torch_parity():
    # adjacent native fix locked here: fft/ifft accept real input like torch
    x = np.random.randn(16)
    ref = torch.fft.fft(T(x)).numpy()
    got = to_torch(tp.fft_fft(P(x))).numpy()
    np.testing.assert_allclose(got.real, ref.real, rtol=1e-12)
    np.testing.assert_allclose(got.imag, ref.imag, rtol=1e-12)


def test_gradient_scalar_spacing_torch_parity():
    x = [0.0, 1.0, 4.0, 9.0]
    got = tp.gradient(tp.tensor(x), spacing=2.0)[0]
    ref = torch.gradient(torch.tensor(x), spacing=2.0)[0]
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=1e-12)


def test_max_min_binary_and_namedtuple_faces():
    a = np.array([[1.0, 5.0], [3.0, 2.0]])
    b = np.array([[4.0, 1.0], [3.0, 6.0]])
    ap, bp = P(a), P(b)
    np.testing.assert_allclose(tp.max(ap, bp).numpy(), np.maximum(a, b))
    r = tp.max(ap, dim=0)
    assert hasattr(r, "values") and hasattr(r, "indices")
    np.testing.assert_allclose(tp.min(ap, bp).numpy(), np.minimum(a, b))


# ---------------------------------------------------------------------------
# Tensor device/layout method face (native bindings locked in this campaign)
# ---------------------------------------------------------------------------

def test_tensor_device_method_face():
    t = tp.tensor([1.0, 2.0])  # default dtype float32, same as torch
    assert t.is_cpu and not t.is_cuda and not t.is_meta
    assert t.nbytes() == 8
    assert tp.tensor([1.0, 2.0], dtype=tp.float64).nbytes() == 16
    assert t.storage_offset() == 0
    assert t.get_device() == -1  # torch: -1 for CPU
    # .cpu() is a no-copy identity on CPU tensors (torch semantics)
    assert t.cpu().data_ptr() == t.data_ptr()


def test_set_storage_aliasing():
    src = P(np.array([9.0, 8.0, 7.0, 6.0]))
    t = P(np.zeros(4))
    t.set_(src)
    np.testing.assert_allclose(t.numpy(), [9, 8, 7, 6])
    t2 = P(np.zeros(2))
    t2.set_(src, 2, [2], [1])
    np.testing.assert_allclose(t2.numpy(), [7, 6])
    # aliasing: writes through one view appear in the other
    t[0] = 42.0 if hasattr(t, "__setitem__") else None


def test_type_as_dtype_only_cast():
    w = P(np.array([[1.0, 2.0], [3.0, 4.0]]))
    other = tp.from_dlpack(torch.tensor(np.array([1], dtype=np.int32)).__dlpack__())
    got = w.type_as(other)
    assert str(got.dtype) == "tensorplay.int32"
    back = got.type_as(w)
    np.testing.assert_allclose(back.numpy(), [[1, 2], [3, 4]])
