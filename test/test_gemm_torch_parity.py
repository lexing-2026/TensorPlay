"""Torch-parity tests for the full GEMM family: mm/matmul/addmm/bmm/baddbmm/
mv/dot/inner/outer/einsum on CPU and CUDA.

Covers shapes, dtypes, numerics, autograd and torch's exact error wording.
"""

import numpy as np
import pytest
import torch

import tensorplay as tp


DEVICES = ["cpu", "cuda"]


def _np(t):
    """TensorPlay tensor -> numpy (host)."""
    return t.cpu().numpy() if str(t.device).startswith("cuda") else t.numpy()


def _mk(array, device):
    t = tp.tensor(np.ascontiguousarray(array))
    return t.to(tp.device(device)) if device == "cuda" else t


def _torch_mk(array, device):
    return torch.tensor(np.ascontiguousarray(array), device=device)


# ---------------------------------------------------------------------------
# Numerics: every op must match torch (bit-exact for fp32/fp64 where BLAS
# orders coincide; tolerance for reduced precision).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_blas_family_numerics(device):
    if device == "cuda" and not tp.device("cuda"):
        pytest.skip("no cuda")
    rng = np.random.RandomState(0)
    a = rng.randn(4, 5).astype(np.float32)
    b = rng.randn(5, 3).astype(np.float32)
    v5 = rng.randn(5).astype(np.float32)
    v4 = rng.randn(4).astype(np.float32)
    A = rng.randn(2, 3, 5).astype(np.float32)
    B = rng.randn(2, 5, 4).astype(np.float32)

    ta, tb, tv5, tv4 = (_mk(x, device) for x in (a, b, v5, v4))
    tA, tB = _mk(A, device), _mk(B, device)
    ha, hb, hv5, hv4 = (_torch_mk(x, device) for x in (a, b, v5, v4))
    hA, hB = _torch_mk(A, device), _torch_mk(B, device)

    cases = {
        "mm": (lambda: ta @ tb, lambda: ha @ hb),
        "matmul_batched": (lambda: tA @ tB, lambda: hA @ hB),
        "matmul_vec_mat": (lambda: tv5 @ tb,
                           lambda: hv5 @ hb),
        "mv": (lambda: tp.mv(ta, tv5), lambda: torch.mv(ha, hv5)),
        "dot": (lambda: tp.dot(tv5, tv5), lambda: torch.dot(hv5, hv5)),
        "outer": (lambda: tp.outer(tv4, tv5), lambda: torch.outer(hv4, hv5)),
        "inner": (lambda: tp.inner(ta, tb.transpose(-2, -1)), lambda: torch.inner(ha, hb.t())),
        "bmm": (lambda: tp.bmm(tA, tB), lambda: torch.bmm(hA, hB)),
        "addmm": (lambda: tp.addmm(_mk(v4, device), ta, tb, 2.0, 3.0),
                  lambda: torch.addmm(_torch_mk(v4, device), ha, hb, beta=2.0, alpha=3.0)),
        "baddbmm": (lambda: tp.baddbmm(_mk(A, device), tA, tB, 0.5, 2.0),
                    lambda: torch.baddbmm(_torch_mk(A, device), hA, hB, beta=0.5, alpha=2.0)),
    }
    for name, (tp_fn, torch_fn) in cases.items():
        got = _np(tp_fn())
        want = torch_fn().cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=name)


@pytest.mark.parametrize("device", DEVICES)
def test_addmm_broadcast_inputs(device):
    """torch accepts any input broadcastable to (M, N): 0-dim, (N,), (M,1), (1,N)."""
    if device == "cuda" and not tp.device("cuda"):
        pytest.skip("no cuda")
    rng = np.random.RandomState(1)
    a = rng.randn(3, 4).astype(np.float32)
    b = rng.randn(4, 2).astype(np.float32)
    for shape in [(), (2,), (3, 1), (1, 2), (3, 2)]:
        bias = (rng.randn(*shape) if shape else rng.randn()).astype(np.float32)
        got = _np(tp.addmm(_mk(bias, device), _mk(a, device), _mk(b, device)))
        want = torch.addmm(_torch_mk(bias, device),
                           _torch_mk(a, device), _torch_mk(b, device)).cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=str(shape))


@pytest.mark.parametrize("device", DEVICES)
def test_baddbmm_broadcast_inputs(device):
    if device == "cuda" and not tp.device("cuda"):
        pytest.skip("no cuda")
    rng = np.random.RandomState(2)
    A = rng.randn(2, 3, 4).astype(np.float32)
    B = rng.randn(2, 4, 5).astype(np.float32)
    for shape in [(), (5,), (3, 5), (1, 3, 5)]:
        bias = (rng.randn(*shape) if shape else rng.randn()).astype(np.float32)
        got = _np(tp.baddbmm(_mk(bias, device), _mk(A, device), _mk(B, device)))
        want = torch.baddbmm(_torch_mk(bias, device),
                             _torch_mk(A, device), _torch_mk(B, device)).cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=str(shape))


# ---------------------------------------------------------------------------
# Error messages must match torch verbatim.
# ---------------------------------------------------------------------------

def _err(fn):
    try:
        fn()
    except Exception as e:
        return str(e)
    return "<no error>"


@pytest.mark.parametrize("device", DEVICES)
def test_error_messages_match_torch(device):
    if device == "cuda" and not tp.device("cuda"):
        pytest.skip("no cuda")
    f32, f64 = np.float32, np.float64

    pairs = [
        # (name, tp_fn, torch_fn)
        ("mm dtype", lambda: tp.mm(_mk(np.ones((2, 3), f32), device), _mk(np.ones((3, 4), f64), device)),
         lambda: torch.mm(torch.ones(2, 3, device=device), torch.ones(3, 4, dtype=torch.float64, device=device))),
        ("mm shape", lambda: tp.mm(_mk(np.ones((2, 3), f32), device), _mk(np.ones((5, 4), f32), device)),
         lambda: torch.mm(torch.ones(2, 3, device=device), torch.ones(5, 4, device=device))),
        ("matmul fold", lambda: tp.matmul(_mk(np.ones((2, 3, 4), f32), device), _mk(np.ones((5, 6), f32), device)),
         lambda: torch.matmul(torch.ones(2, 3, 4, device=device), torch.ones(5, 6, device=device))),
        ("matmul batched K", lambda: tp.matmul(_mk(np.ones((2, 3, 4), f32), device), _mk(np.ones((2, 5, 6), f32), device)),
         lambda: torch.matmul(torch.ones(2, 3, 4, device=device), torch.ones(2, 5, 6, device=device))),
        ("dot size", lambda: tp.dot(_mk(np.ones(4, f32), device), _mk(np.ones(5, f32), device)),
         lambda: torch.dot(torch.ones(4, device=device), torch.ones(5, device=device))),
        ("dot ndim", lambda: tp.dot(_mk(np.ones((2, 3), f32), device), _mk(np.ones(3, f32), device)),
         lambda: torch.dot(torch.ones(2, 3, device=device), torch.ones(3, device=device))),
        ("dot dtype", lambda: tp.dot(_mk(np.ones(3, f32), device), _mk(np.ones(3, f64), device)),
         lambda: torch.dot(torch.ones(3, device=device), torch.ones(3, dtype=torch.float64, device=device))),
        ("mv", lambda: tp.mv(_mk(np.ones((4, 3), f32), device), _mk(np.ones(5, f32), device)),
         lambda: torch.mv(torch.ones(4, 3, device=device), torch.ones(5, device=device))),
        ("bmm ndim", lambda: tp.bmm(_mk(np.ones((2, 4), f32), device), _mk(np.ones((2, 4, 5), f32), device)),
         lambda: torch.bmm(torch.ones(2, 4, device=device), torch.ones(2, 4, 5, device=device))),
        ("bmm batch", lambda: tp.bmm(_mk(np.ones((2, 3, 4), f32), device), _mk(np.ones((3, 4, 5), f32), device)),
         lambda: torch.bmm(torch.ones(2, 3, 4, device=device), torch.ones(3, 4, 5, device=device))),
        ("bmm dtype", lambda: tp.bmm(_mk(np.ones((2, 3, 4), f32), device), _mk(np.ones((2, 4, 5), f64), device)),
         lambda: torch.bmm(torch.ones(2, 3, 4, device=device), torch.ones(2, 4, 5, dtype=torch.float64, device=device))),
        ("addmm expand", lambda: tp.addmm(_mk(np.ones(5, f32), device), _mk(np.ones((2, 3), f32), device), _mk(np.ones((3, 4), f32), device)),
         lambda: torch.addmm(torch.ones(5, device=device), torch.ones(2, 3, device=device), torch.ones(3, 4, device=device))),
        ("expand mismatch", lambda: _mk(np.ones(5, f32), device).expand((2, 4)),
         lambda: torch.ones(5, device=device).expand(2, 4)),
    ]
    for name, tp_fn, torch_fn in pairs:
        got = _err(tp_fn)
        want = _err(torch_fn)
        assert got == want, f"{name}:\n  tp   = {got!r}\n  torch= {want!r}"


# ---------------------------------------------------------------------------
# einsum battery.
# ---------------------------------------------------------------------------

EINSUM_CASES = [
    ("ij,jk->ik", [(2, 3), (3, 4)]),
    ("ij,jk", [(2, 3), (3, 4)]),                # implicit output
    ("ii->i", [(4, 4)]),                        # diagonal
    ("ii", [(4, 4)]),                           # trace
    ("ij->ji", [(3, 4)]),                       # transpose
    ("i,i->", [(5,), (5,)]),                    # dot
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),   # batched matmul
    ("ik,jk->ij", [(2, 4), (3, 4)]),            # bilinear
    ("bn,anm,bm->ba", [(2, 5), (3, 5, 4), (2, 4)]),
    ("...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)]),
    ("...ij,jk->...ik", [(2, 3, 4), (4, 5)]),   # ellipsis broadcast with shared rhs
    ("bij,bj->bi", [(2, 3, 4), (2, 4)]),        # batched matvec
    ("ij,kj->ik", [(3, 4), (5, 4)]),
    ("ki,jk->ij", [(4, 3), (5, 4)]),            # transposed contraction dims
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("equation,shapes", EINSUM_CASES)
def test_einsum_matches_torch(device, equation, shapes):
    if device == "cuda" and not tp.device("cuda"):
        pytest.skip("no cuda")
    rng = np.random.RandomState(hash(equation) % (2**31))
    arrays = [rng.randn(*s).astype(np.float32) for s in shapes]
    got = _np(tp.einsum(equation, *[_mk(a, device) for a in arrays]))
    want = torch.einsum(equation, *[_torch_mk(a, device) for a in arrays]).cpu().numpy()
    assert got.shape == want.shape, equation
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4, err_msg=equation)


def test_einsum_sublist_format():
    rng = np.random.RandomState(7)
    A = rng.randn(2, 3, 4).astype(np.float32)
    B = rng.randn(2, 4, 5).astype(np.float32)
    got = _np(tp.einsum(_mk(A, "cpu"), [..., 0, 1], _mk(B, "cpu"), [..., 1, 2], [..., 0, 2]))
    want = torch.einsum(torch.tensor(A), [..., 0, 1], torch.tensor(B), [..., 1, 2], [..., 0, 2]).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Autograd.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_autograd_new_ops(device):
    if device == "cuda" and not tp.device("cuda"):
        pytest.skip("no cuda")
    rng = np.random.RandomState(3)
    a = rng.randn(3, 4).astype(np.float32)
    b = rng.randn(4, 5).astype(np.float32)
    v4 = rng.randn(4).astype(np.float32)
    v5 = rng.randn(5).astype(np.float32)
    A = rng.randn(2, 3, 4).astype(np.float32)
    B = rng.randn(2, 4, 5).astype(np.float32)

    def check(tp_fn, torch_fn, inputs):
        tp_inputs = [_mk(x, device).clone() for x in inputs]
        for t in tp_inputs:
            t.requires_grad_(True)
        th_inputs = [torch.tensor(x, device=device, requires_grad=True) for x in inputs]
        tp_out = tp_fn(*tp_inputs)
        th_out = torch_fn(*th_inputs)
        grad = torch.rand_like(th_out).detach()
        gsum = float(grad.sum())
        tp_g = tp.tensor(grad.cpu().numpy() if device == "cpu" else grad.cpu().numpy()).to(tp_out.device)
        tp_out.backward(tp_g)
        th_out.backward(grad)
        for i, (t, th) in enumerate(zip(tp_inputs, th_inputs)):
            np.testing.assert_allclose(
                _np(t.grad), th.grad.cpu().numpy(),
                rtol=1e-4, atol=1e-4, err_msg=f"{check.__name__} grad[{i}]")

    check(lambda x, y: tp.mm(x, y), lambda x, y: torch.mm(x, y), [a, b])
    check(lambda x, y: tp.bmm(x, y), lambda x, y: torch.bmm(x, y), [A, B])
    check(lambda x, y: tp.mv(x, y), lambda x, y: torch.mv(x, y), [a, v4])
    check(lambda x, y: tp.dot(x, y), lambda x, y: torch.dot(x, y), [v4, v4])
    check(lambda x, y: tp.outer(x, y), lambda x, y: torch.outer(x, y), [v4, v5])
    check(lambda x, y: tp.inner(x, y), lambda x, y: torch.inner(x, y), [a, b.T.copy()])
    check(lambda x, y, z: tp.baddbmm(x, y, z), lambda x, y, z: torch.baddbmm(x, y, z), [A, A, B])
    check(lambda x, y: tp.einsum("ij,jk->ik", x, y), lambda x, y: torch.einsum("ij,jk->ik", x, y), [a, b])
    check(lambda x: tp.einsum("ii->i", x), lambda x: torch.einsum("ii->i", x), [np.eye(4, dtype=np.float32)])
