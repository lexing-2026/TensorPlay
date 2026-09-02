"""
mv/dot/inner/outer/einsum on CPU and CUDA.

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
# orders coincide; tolerance for reduced precision).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_blas_family_numerics(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.RandomState(0)
    a = rng.randn(4, 5).astype(np.float32)
    b = rng.randn(5, 3).astype(np.float32)
    v5 = rng.randn(5).astype(np.float32)
    v4 = rng.randn(4).astype(np.float32)
    v3 = rng.randn(3).astype(np.float32)
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
        "addmm": (lambda: tp.addmm(_mk(v3, device), ta, tb, 2.0, 3.0),
                  lambda: torch.addmm(_torch_mk(v3, device), ha, hb, beta=2.0, alpha=3.0)),
        "baddbmm": (lambda: tp.baddbmm(_mk(np.zeros((2, 3, 4), np.float32), device), tA, tB, beta=0.5, alpha=2.0),
                    lambda: torch.baddbmm(_torch_mk(np.zeros((2, 3, 4), np.float32), device), hA, hB, beta=0.5, alpha=2.0)),
    }
    for name, (tp_fn, torch_fn) in cases.items():
        got = _np(tp_fn())
        want = torch_fn().cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=name)


@pytest.mark.parametrize("device", DEVICES)
def test_addmm_broadcast_inputs(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.RandomState(1)
    a = rng.randn(3, 4).astype(np.float32)
    b = rng.randn(4, 2).astype(np.float32)
    for shape in [(), (2,), (3, 1), (1, 2), (3, 2)]:
        bias = np.asarray(rng.randn(*shape)).astype(np.float32)
        got = _np(tp.addmm(_mk(bias, device), _mk(a, device), _mk(b, device)))
        want = torch.addmm(_torch_mk(bias, device),
                           _torch_mk(a, device), _torch_mk(b, device)).cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=str(shape))


@pytest.mark.parametrize("device", DEVICES)
def test_baddbmm_broadcast_inputs(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.RandomState(2)
    A = rng.randn(2, 3, 4).astype(np.float32)
    B = rng.randn(2, 4, 5).astype(np.float32)
    for shape in [(), (5,), (3, 5), (1, 3, 5)]:
        bias = np.asarray(rng.randn(*shape)).astype(np.float32)
        got = _np(tp.baddbmm(_mk(bias, device), _mk(A, device), _mk(B, device)))
        want = torch.baddbmm(_torch_mk(bias, device),
                             _torch_mk(A, device), _torch_mk(B, device)).cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=str(shape))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _err(fn):
    try:
        fn()
    except Exception as e:
        return str(e)
    return "<no error>"


@pytest.mark.parametrize("device", DEVICES)
def test_error_messages_match_torch(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("no cuda")
    f32, f64 = np.float32, np.float64

    pairs = [
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
        assert got == want, f"{name}:\n  tp   = {got!r}\n  reference= {want!r}"


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
    # -- two-operand fast paths (single BLAS call + view assembly) --
    ("i,j->ij", [(6,), (7,)]),                  # outer
    ("ij,ji->", [(3, 3), (3, 3)]),              # both sides contracted
    ("i,ij->j", [(3,), (3, 4)]),                # vec @ mat
    ("ji,i->j", [(4, 3), (3,)]),
    ("ab,bc->ca", [(2, 3), (3, 4)]),            # output reorder
    ("bi,ci->bc", [(4, 3), (5, 3)]),
    ("bij,bjk->bki", [(2, 3, 4), (2, 4, 5)]),   # batched with reordered output
    ("bi,bij->bj", [(3, 4), (3, 4, 5)]),        # batched vec @ mat
    ("abc,cd->abd", [(2, 3, 4), (4, 5)]),       # multi-dim free on one side
    ("ij,jk->ik", [(0, 3), (3, 4)]),            # zero-sized contraction dim
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("equation,shapes", EINSUM_CASES)
def test_einsum_matches_torch(device, equation, shapes):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.RandomState(hash(equation) % (2**31))
    arrays = [rng.randn(*s).astype(np.float32) for s in shapes]
    got = _np(tp.einsum(equation, *[_mk(a, device) for a in arrays]))
    want = torch.einsum(equation, *[_torch_mk(a, device) for a in arrays]).cpu().numpy()
    assert got.shape == want.shape, equation
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4, err_msg=equation)


def test_einsum_fast_path_noncontig_inputs():
    """Transposed / strided views entering the two-operand fast path."""
    rng = np.random.RandomState(11)
    a0 = tp.tensor(rng.randn(5, 4).astype(np.float32))    # a0.t() = (4,5)
    b = tp.tensor(rng.randn(5, 6).astype(np.float32))
    m0 = tp.tensor(rng.randn(5, 7).astype(np.float32))    # (K=5, N=7)
    v5 = tp.tensor(rng.randn(5).astype(np.float32))
    B0 = tp.tensor(rng.randn(2, 5, 4).astype(np.float32))  # transpose -> (2,4,5)
    th = lambda t: torch.from_numpy(t.numpy())
    got = _np(tp.einsum("ij,jk->ik", a0.t(), b))
    want = torch.einsum("ij,jk->ik", th(a0).t(), th(b)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    got = _np(tp.einsum("i,ij->j", v5[::1], m0[:, ::2]))
    want = torch.einsum("i,ij->j", th(v5), th(m0)[:, ::2]).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    got = _np(tp.einsum("bij,bjk->bik", B0.transpose(1, 2), b.unsqueeze(0)))
    want = torch.einsum("bij,bjk->bik", th(B0).transpose(1, 2), th(b).unsqueeze(0)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_einsum_native_path_reorders_contraction():
    """With >2 operands and no explicit path, the kernel plans an order whose
    result still equals numpy's path-independent ground truth -- and avoids
"""
    rng = np.random.RandomState(13)
    # Left-to-right materializes a 40x40x40 intermediate here; the planned
    # order contracts the small pairs first.
    eq = "ab,ac,bd,cd->"
    arrays = [rng.randn(20, 20).astype(np.float64) for _ in range(4)]
    got = _np(tp.einsum(eq, *[_mk(a, "cpu") for a in arrays]))
    want = np.einsum(eq, *arrays)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    th = [torch.from_numpy(a) for a in arrays]
    twant = torch.einsum(eq, *th).numpy()
    np.testing.assert_allclose(got, twant, rtol=1e-6, atol=1e-6)

    eq2 = "bn,anm,bm->ba"
    arrays2 = [rng.randn(*s).astype(np.float32)
               for s in [(9, 12), (9, 12, 9), (9, 9)]]
    got2 = _np(tp.einsum(eq2, *[_mk(a, "cpu") for a in arrays2]))
    np.testing.assert_allclose(
        got2, np.einsum(eq2, *arrays2), rtol=1e-4, atol=1e-4)


def test_einsum_explicit_path_kwarg():
    rng = np.random.RandomState(17)
    xs = [rng.randn(3, 3).astype(np.float32) for _ in range(4)]
    got = _np(tp.functional.einsum("ab,bc,cd,de->ae", *[_mk(x, "cpu") for x in xs],
                                   path=[0, 1, 0, 1, 0, 1]))
    want = np.einsum("ab,bc,cd,de->ae", *xs)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    with pytest.raises(Exception) as exc:
        tp.functional.einsum("ab,bc,cd,de->ae", *[_mk(x, "cpu") for x in xs],
                             path=[0, 0])
    assert "contraction" in str(exc.value)


def test_einsum_fuzz_matches_numpy_and_torch():
    """Random equations (ellipsis, repeats, implicit output, broadcasts-free)
"""
    from collections import Counter
    rng = np.random.RandomState(42)
    LET = "abcde"

    def gen_eq(nops):
        labels = [[LET[rng.randint(0, 5)] for _ in range(rng.randint(1, 4))]
                  for _ in range(nops)]
        for i in range(1, nops):
            src = labels[0][rng.randint(len(labels[0]))]
            if rng.rand() < 0.5 and labels[i]:
                labels[i][rng.randint(len(labels[i]))] = src
            else:
                labels[i].append(src)
        sizes = {l: int(rng.randint(2, 4)) for l in LET}
        aligned, arrays = [], []
        use_ell = rng.rand() < 0.25
        ell_dims = {j: int(rng.randint(2, 4)) for j in range(rng.randint(1, 3))} if use_ell else {}
        for labs in labels:
            core = [sizes[l] for l in labs]
            if use_ell:
                k = rng.randint(0, len(ell_dims) + 1)
                arrays.append(rng.randn(*([ell_dims[j] for j in range(k)] + core)))
                aligned.append("..." + "".join(labs))
            else:
                arrays.append(rng.randn(*core))
                aligned.append("".join(labs))
        cnt = Counter(l for labs in labels for l in labs)
        keep = [l for l in LET if cnt[l] == 1]
        rng.shuffle(keep)
        batched = [l for l in LET if cnt[l] >= 2]
        if batched and rng.rand() < 0.4:
            keep.append(batched[rng.randint(len(batched))])
        return ",".join(aligned) + "->" + "".join(keep), arrays

    checked = 0
    for _ in range(220):
        nops = rng.randint(1, 6)
        eq, arrays = gen_eq(nops)
        try:
            want_np = np.einsum(eq, *[a.astype(np.float64) for a in arrays])
        except Exception:
            continue
        checked += 1
        got = _np(tp.einsum(eq, *[_mk(a, "cpu") for a in
                                  [a.astype(np.float32) for a in arrays]]))
        assert got.shape == want_np.shape, (eq, got.shape, want_np.shape)
        np.testing.assert_allclose(got, want_np, rtol=2e-3, atol=2e-3, err_msg=eq)
        want_th = torch.einsum(eq, *[torch.from_numpy(a.copy()) for a in arrays]).numpy()
        np.testing.assert_allclose(got, want_th, rtol=2e-3, atol=2e-3, err_msg=eq)
    assert checked > 120


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
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("no cuda")
    rng = np.random.RandomState(3)
    a = rng.randn(3, 4).astype(np.float32)
    b = rng.randn(4, 5).astype(np.float32)
    v4 = rng.randn(4).astype(np.float32)
    v5 = rng.randn(5).astype(np.float32)
    v3 = rng.randn(3).astype(np.float32)
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
    check(lambda x, y, z: tp.baddbmm(x, y, z), lambda x, y, z: torch.baddbmm(x, y, z),
          [np.zeros((2, 3, 5), np.float32), A, B])
    check(lambda x, y: tp.einsum("ij,jk->ik", x, y), lambda x, y: torch.einsum("ij,jk->ik", x, y), [a, b])
    check(lambda x: tp.einsum("ii->i", x), lambda x: torch.einsum("ii->i", x), [np.eye(4, dtype=np.float32)])
    # Fast-path variants: mv / vec@mat / batched / reordered output.
    check(lambda x, y: tp.einsum("ij,j->i", x, y),
          lambda x, y: torch.einsum("ij,j->i", x, y), [a, v4])
    check(lambda x, y: tp.einsum("i,ij->j", x, y),
          lambda x, y: torch.einsum("i,ij->j", x, y), [v3, a])
    check(lambda x, y: tp.einsum("bij,bjk->bik", x, y),
          lambda x, y: torch.einsum("bij,bjk->bik", x, y), [A, B])
    check(lambda x, y: tp.einsum("ik,jk->ij", x, y),
          lambda x, y: torch.einsum("ik,jk->ij", x, y), [a, b.T.copy()])
    check(lambda x, y, z: tp.einsum("bn,anm,bm->ba", x, y, z),
          lambda x, y, z: torch.einsum("bn,anm,bm->ba", x, y, z),
          [rng.randn(2, 5).astype(np.float32), rng.randn(3, 5, 4).astype(np.float32),
           rng.randn(2, 4).astype(np.float32)])
