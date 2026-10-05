"""Gradients of the matrix factorizations and solvers.

Every user-facing spelling forwards to the core operator that carries the
derivative, so the gradient reaches the input whichever one the caller used.
Expected values are worked out by hand for small matrices; a factorization
whose factors are multiplied back together must hand back the gradient of
the matrix itself.
"""
import pytest

import tensorplay as tp
from tensorplay.autograd.gradcheck import gradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

# A = [[2, 1], [1, 3]]: det 5, inverse [[0.6, -0.2], [-0.2, 0.4]].
SPD = [[2.0, 1.0], [1.0, 3.0]]
SPD_INV = [[0.6, -0.2], [-0.2, 0.4]]


def _leaf(values, device, dtype=tp.float64):
    return tp.tensor(values, dtype=dtype, device=device, requires_grad=True)


def _close(got, expected, tol=1e-6):
    expected = tp.tensor(expected, dtype=got.dtype)
    assert tp.allclose(got.detach().cpu(), expected, rtol=tol, atol=tol), got


@pytest.mark.parametrize("device", DEVICES)
def test_determinant_gradient_is_the_cofactor_matrix(device):
    A = _leaf(SPD, device)
    tp.linalg.det(A).backward()
    # det(A) A^-T
    _close(A.grad, [[3.0, -1.0], [-1.0, 2.0]])

    # A singular matrix still has its cofactors: d det/dA = [[d, -c], [-b, a]].
    S = _leaf([[1.0, 2.0], [2.0, 4.0]], device)
    tp.det(S).backward()
    _close(S.grad, [[4.0, -2.0], [-2.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
def test_log_determinant_gradient_is_the_inverse_transpose(device):
    A = _leaf(SPD, device)
    sign, logabsdet = tp.linalg.slogdet(A)
    logabsdet.backward()
    _close(A.grad, SPD_INV)
    assert sign.item() == 1.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spelling", ["inv", "inv_ex", "pinv", "pinverse"])
def test_inverse_spellings_share_one_gradient(device, spelling):
    A = _leaf(SPD, device)
    if spelling == "inv":
        out = tp.linalg.inv(A)
    elif spelling == "inv_ex":
        out = tp.linalg.inv_ex(A)[0]
    elif spelling == "pinv":
        out = tp.linalg.pinv(A)
    else:
        out = tp.pinverse(A)
    out.sum().backward()
    # -A^-T 1 1^T A^-T
    _close(A.grad, [[-0.16, -0.08], [-0.08, -0.04]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spelling", ["solve", "solve_ex", "lu_solve"])
def test_solve_spellings_differentiate_in_the_matrix_and_the_right_side(device, spelling):
    A = _leaf(SPD, device)
    b = _leaf([1.0, 2.0], device)
    if spelling == "solve":
        x = tp.linalg.solve(A, b)
    elif spelling == "solve_ex":
        x = tp.linalg.solve_ex(A, b)[0]
    else:
        LU, pivots = tp.linalg.lu_factor(A)
        x = tp.linalg.lu_solve(LU, pivots, b.unsqueeze(-1)).squeeze(-1)
    _close(x, [0.2, 0.6])
    x.sum().backward()
    # gb = A^-T 1, gA = -gb x^T
    _close(b.grad, [0.4, 0.2])
    _close(A.grad, [[-0.08, -0.24], [-0.04, -0.12]])


@pytest.mark.parametrize("device", DEVICES)
def test_solve_from_the_right_differentiates(device):
    A = _leaf(SPD, device)
    B = _leaf([[1.0, 2.0]], device)
    # X A = B, X = B A^-1 = [[0.2, 0.6]]; gB = 1 A^-T, gA = -X^T gB
    X = tp.linalg.solve(A, B, left=False)
    _close(X, [[0.2, 0.6]])
    X.sum().backward()
    _close(B.grad, [[0.4, 0.2]])
    _close(A.grad, [[-0.08, -0.04], [-0.24, -0.12]])


@pytest.mark.parametrize("device", DEVICES)
def test_triangular_solves_keep_the_triangle(device):
    A = _leaf([[2.0, 1.0], [0.0, 4.0]], device)
    b = _leaf([[1.0], [2.0]], device)
    # The legacy spelling takes the upper triangle by default.
    x, coefficient = tp.triangular_solve(b, A)
    _close(x, [[0.25], [0.5]])
    _close(coefficient, [[2.0, 1.0], [0.0, 4.0]])
    x.sum().backward()
    _close(b.grad, [[0.5], [0.125]])
    _close(A.grad, [[-0.125, -0.25], [0.0, -0.0625]])

    A2 = _leaf([[2.0, 1.0], [0.0, 4.0]], device)
    b2 = _leaf([[1.0], [2.0]], device)
    tp.linalg.solve_triangular(A2, b2, upper=True).sum().backward()
    _close(b2.grad, [[0.5], [0.125]])
    _close(A2.grad, [[-0.125, -0.25], [0.0, -0.0625]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spelling", ["linalg", "legacy", "ex"])
def test_cholesky_factor_multiplied_back_returns_the_matrix_gradient(device, spelling):
    A = _leaf([[4.0, 2.0], [2.0, 3.0]], device)
    if spelling == "linalg":
        L = tp.linalg.cholesky(A)
    elif spelling == "legacy":
        L = tp.cholesky(A)
    else:
        L = tp.linalg.cholesky_ex(A)[0]
    (L @ L.mT).sum().backward()
    _close(A.grad, [[1.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
def test_cholesky_inverse_and_solve_differentiate_through_the_factor(device):
    # A = [[4, 2], [2, 3]], A^-1 = [[0.375, -0.25], [-0.25, 0.5]], A^-1 1 = [0.125, 0.25].
    A = _leaf([[4.0, 2.0], [2.0, 3.0]], device)
    tp.cholesky_inverse(tp.cholesky(A)).sum().backward()
    _close(A.grad, [[-0.015625, -0.03125], [-0.03125, -0.0625]])

    A = _leaf([[4.0, 2.0], [2.0, 3.0]], device)
    b = _leaf([[1.0], [2.0]], device)
    x = tp.cholesky_solve(b, tp.cholesky(A))
    _close(x, [[-0.125], [0.75]])
    x.sum().backward()
    _close(b.grad, [[0.125], [0.25]])
    # Symmetric part of -A^-1 1 x^T.
    _close(A.grad, [[0.015625, -0.03125], [-0.03125, -0.1875]])


@pytest.mark.parametrize("device", DEVICES)
def test_eigenvalues_of_a_symmetric_matrix_differentiate_as_projectors(device):
    # [[2, 1], [1, 2]] has eigenvalues 1 and 3 along [1, -1] and [1, 1].
    A = _leaf([[2.0, 1.0], [1.0, 2.0]], device)
    values = tp.linalg.eigvalsh(A)
    _close(values, [1.0, 3.0])
    values[1].backward()
    _close(A.grad, [[0.5, 0.5], [0.5, 0.5]])

    A = _leaf([[2.0, 1.0], [1.0, 2.0]], device)
    L, V = tp.linalg.eigh(A)
    (V @ tp.diag_embed(L) @ V.mT).sum().backward()
    _close(A.grad, [[1.0, 1.0], [1.0, 1.0]])


def test_eigenvalues_of_a_general_matrix_differentiate_through_the_trace():
    A = _leaf([[2.0, 1.0], [0.0, 3.0]], "cpu")
    tp.linalg.eigvals(A).real.sum().backward()
    _close(A.grad, [[1.0, 0.0], [0.0, 1.0]])

    A = _leaf([[2.0, 1.0], [0.5, 3.0]], "cpu")
    L, V = tp.linalg.eig(A)
    (V @ tp.diag_embed(L) @ tp.linalg.inv(V)).real.sum().backward()
    _close(A.grad, [[1.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
def test_singular_values_differentiate_as_the_outer_vectors(device):
    # The nuclear norm of a symmetric positive definite matrix is its trace.
    A = _leaf([[2.0, 1.0], [1.0, 2.0]], device)
    tp.linalg.svdvals(A).sum().backward()
    _close(A.grad, [[1.0, 0.0], [0.0, 1.0]])

    A = _leaf([[3.0, 1.0], [1.0, 2.0], [0.0, 1.0]], device)
    U, S, Vh = tp.linalg.svd(A, full_matrices=False)
    (U @ tp.diag_embed(S) @ Vh).sum().backward()
    _close(A.grad, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    A = _leaf([[3.0, 1.0], [1.0, 2.0], [0.0, 1.0]], device)
    U, S, Vh = tp.linalg.svd(A)  # full factors
    (U[:, :2] @ tp.diag_embed(S) @ Vh).sum().backward()
    _close(A.grad, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_svd_returns_v_and_honors_some_and_compute_uv(device):
    A = _leaf([[3.0, 1.0], [1.0, 2.0], [0.0, 1.0]], device)
    U, S, V = tp.svd(A)
    assert tuple(U.shape) == (3, 2) and tuple(V.shape) == (2, 2)
    (U @ tp.diag_embed(S) @ V.mT).sum().backward()
    _close(A.grad, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    U, S, V = tp.svd(A.detach(), some=False)
    assert tuple(U.shape) == (3, 3) and tuple(V.shape) == (2, 2)

    U, S, V = tp.svd(A.detach(), compute_uv=False)
    assert tuple(U.shape) == (3, 3) and tuple(V.shape) == (2, 2)
    assert float(U.abs().sum()) == 0.0 and float(V.abs().sum()) == 0.0
    _close(S, tp.linalg.svdvals(A.detach()).cpu().tolist())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["reduced", "complete"])
def test_qr_factors_multiplied_back_return_the_matrix_gradient(device, mode):
    A = _leaf([[3.0, 1.0, 2.0], [1.0, 2.0, 0.0]], device) if mode == "complete" \
        else _leaf([[3.0, 1.0], [1.0, 2.0], [0.0, 1.0]], device)
    Q, R = tp.linalg.qr(A, mode=mode)
    (Q @ R).sum().backward()
    _close(A.grad, [[1.0] * A.shape[1]] * A.shape[0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3)])
def test_lu_factors_multiplied_back_return_the_matrix_gradient(device, shape):
    values = [[4.0, 3.0, 1.0], [6.0, 3.0, 2.0], [1.0, 5.0, 2.0]]
    rows, cols = shape
    A = _leaf([row[:cols] for row in values[:rows]], device)
    P, L, U = tp.linalg.lu(A)
    (P @ L @ U).sum().backward()
    _close(A.grad, [[1.0] * cols] * rows)

    A = _leaf([row[:cols] for row in values[:rows]], device)
    LU, pivots = tp.linalg.lu_factor_ex(A)[:2]
    P, L, U = tp.lu_unpack(LU, pivots)
    (P @ L @ U).sum().backward()
    _close(A.grad, [[1.0] * cols] * rows)


@pytest.mark.parametrize("device", DEVICES)
def test_least_squares_on_a_square_system_matches_the_solve(device):
    A = _leaf(SPD, device)
    B = _leaf([[1.0], [2.0]], device)
    solution = tp.linalg.lstsq(A, B).solution
    _close(solution, [[0.2], [0.6]])
    solution.sum().backward()
    _close(B.grad, [[0.4], [0.2]])
    _close(A.grad, [[-0.08, -0.24], [-0.04, -0.12]])


def test_factorizations_pass_the_numerical_gradient_check():
    A = tp.tensor([[1.5, 0.3, -0.2], [0.4, 2.0, 0.1], [-0.3, 0.2, 1.0]],
                  dtype=tp.float64, requires_grad=True)
    tall = tp.tensor([[1.5, 0.3], [0.4, 2.0], [-0.3, 0.2]],
                     dtype=tp.float64, requires_grad=True)
    assert gradcheck(lambda x: tp.linalg.det(x), A)
    assert gradcheck(lambda x: tp.linalg.inv(x), A)
    assert gradcheck(lambda x: tp.linalg.pinv(x), tall)
    assert gradcheck(lambda x: tp.linalg.svdvals(x), tall)
    assert gradcheck(lambda x: tp.linalg.qr(x).R, tall)
    assert gradcheck(lambda x: tp.linalg.lu(x)[2], A)
    assert gradcheck(lambda x: tp.linalg.lstsq(x, x.sum(-1, keepdim=True)).solution, tall)


def test_recorded_backward_differentiates_again():
    A = _leaf(SPD, "cpu")
    b = tp.tensor([1.0, 2.0], dtype=tp.float64)
    x = tp.linalg.solve(A, b)
    (gA,) = tp.autograd.grad(x.sum(), A, create_graph=True)
    _close(gA, [[-0.08, -0.24], [-0.04, -0.12]])
    gA.sum().backward()
    assert A.grad is not None and bool(tp.isfinite(A.grad).all())


def test_vectors_alone_are_required_for_the_eigenvalue_and_svd_gradients():
    A = _leaf([[2.0, 1.0], [1.0, 2.0]], "cpu")
    with tp.no_grad():
        assert tp.linalg.eigvalsh(A).requires_grad is False
    _, S, _ = tp._C._linalg_svd(A, False, False)
    with pytest.raises(RuntimeError, match="compute_uv"):
        S.sum().backward()
