"""Dynamic-program and COO backward derivatives, including saved inputs."""

import itertools

import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradcheck, gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def leaf(data, device):
    return tp.tensor(data, dtype=tp.float64, device=device).requires_grad_(True)


def array(tensor):
    return tensor.detach().cpu().numpy()


def check_sparse(fn, inputs):
    # Perturb the stored values directly; sparse extents are not storage.
    rng = np.random.default_rng(23)
    cotangent = leaf(rng.normal(size=tuple(fn(*inputs).shape)), str(inputs[0].device)).detach()
    directions = [leaf(rng.normal(size=tuple(x._values().shape)), str(x.device)).detach() for x in inputs]
    gradients = tp.autograd.grad(fn(*inputs), inputs, cotangent, create_graph=True)
    total = sum((g.to_dense() * tp.sparse_coo_tensor(x._indices(), d, tuple(x.shape),
                    is_coalesced=True).to_dense()).sum() for g, x, d in zip(gradients, inputs, directions))
    second = tp.autograd.grad(total, inputs)

    def objectives(args):
        y = fn(*args)
        g = tp.autograd.grad(y, args, cotangent)
        return float((y * cotangent).sum()), sum(float(np.sum(array(v._values()) * array(d)))
                                                for v, d in zip(g, directions))

    eps = 1e-5
    for i, x in enumerate(inputs):
        data = array(x._values())
        expected = np.empty_like(data)
        expected_second = np.empty_like(data)
        for position in np.ndindex(data.shape):
            trials = []
            for step in [-eps, eps]:
                perturbed = data.copy()
                perturbed[position] += step
                values = leaf(perturbed, str(x.device)).detach()
                arg = tp.sparse_coo_tensor(x._indices(), values, tuple(x.shape),
                                           is_coalesced=True).detach().requires_grad_(True)
                args = list(inputs)
                args[i] = arg
                trials.append(objectives(args))
            expected[position] = (trials[1][0] - trials[0][0]) / (2 * eps)
            expected_second[position] = (trials[1][1] - trials[0][1]) / (2 * eps)
        np.testing.assert_allclose(array(gradients[i]._values()), expected, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(array(second[i]._values()), expected_second, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_sparse_backward_independent_slots(device, logarithmic, dim):
    indices = tp.tensor([[0, 0, 1, 1], [0, 2, 0, 1]], device=device)
    grad_indices = tp.tensor([[0, 1, 1, 2], [2, 0, 1, 3]], device=device)
    shape = (3, 4, 2)
    go = tp.sparse_coo_tensor(grad_indices, leaf([[1, 2], [3, 4], [-1, 2], [7, 8]], device),
                             shape, is_coalesced=True).detach().requires_grad_(True)
    output = tp.sparse_coo_tensor(indices, leaf([[.2, .4], [.8, .6], [.3, .5], [.7, .5]], device),
                                 shape, is_coalesced=True).detach().requires_grad_(True)
    structural = output.detach().clone().requires_grad_(True)
    kernel = tp._C._sparse_log_softmax_backward_data if logarithmic else tp._C._sparse_softmax_backward_data
    fn = lambda g, o: kernel(g, o, dim, structural).to_dense()
    check_sparse(fn, (go, output))
    y = kernel(go, output, dim, structural)
    assert tp.autograd.grad(y.to_dense().sum(), structural, allow_unused=True)[0] is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("logarithmic", [False, True])
def test_sparse_softmax_full_second_pass(device, logarithmic):
    indices = tp.tensor([[0, 0, 1, 1], [0, 2, 0, 1]], device=device)
    x = tp.sparse_coo_tensor(indices, leaf([-.4, .7, .2, -.3], device), (2, 3),
                            is_coalesced=True).detach().requires_grad_(True)
    kernel = tp._C._sparse_log_softmax if logarithmic else tp._C._sparse_softmax
    fn = lambda s: kernel(s, 1, False).to_dense() ** 2
    check_sparse(fn, (x,))


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_single_dimension_and_large_extents(device):
    x = tp.sparse_coo_tensor(tp.tensor([[1, 99, 900000000]], device=device),
                            leaf([.2, .3, .5], device), (1000000000,),
                            is_coalesced=True).detach().requires_grad_(True)
    go = x.detach().clone().requires_grad_(True)
    y = tp._C._sparse_softmax_backward_data(go, x, 0, x)
    v = x.detach().clone()
    g = tp.autograd.grad(y, (go, x), v, create_graph=True)
    for tensor in g:
        assert tensor.is_sparse and tensor._nnz() == 3
        assert np.isfinite(array(tensor._values())).all()
    third = tp.autograd.grad(g[0], x, v)[0]
    assert third.is_sparse and third._nnz() == 3


def ctc_arguments(device, repeated=False):
    targets = tp.tensor([[1, 1] if repeated else [1, 2], [2, 0]], device=device)
    return targets, tp.tensor([4, 3], device=device), tp.tensor([2, 1], device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_ctc_full_second_pass(device, repeated, reduction):
    tp.manual_seed(18)
    x = tp.randn(4, 2, 3, dtype=tp.float64, device=device).requires_grad_(True)
    args = ctc_arguments(device, repeated)
    fn = lambda z: F.ctc_loss(z.log_softmax(2), *args, reduction=reduction)
    assert gradgradcheck(fn, (x,), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_ctc_backward_saved_inputs(device):
    tp.manual_seed(19)
    args = ctc_arguments(device, True)
    lp = tp.randn(4, 2, 3, dtype=tp.float64, device=device).log_softmax(2).detach().requires_grad_(True)
    go = leaf([.4, -.7], device)
    nll = leaf([1.2, .9], device)
    alpha = tp.randn(2, 4, 5, dtype=tp.float64, device=device).requires_grad_(True)
    fn = lambda g, p, n, a: tp._C._ctc_loss_backward(g, p, *args, n, a, 0, False)
    assert gradcheck(fn, (go, lp, nll, alpha), atol=1e-6, rtol=1e-4)
    assert gradgradcheck(fn, (go, lp, nll, alpha), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_ctc_alpha_output_derivatives(device):
    x = leaf(np.linspace(-1.5, -.1, 12).reshape(4, 1, 3), device)
    targets = tp.tensor([[1, 2]], device=device)
    lengths = tp.tensor([4], device=device)
    target_lengths = tp.tensor([2], device=device)
    fn = lambda p: tp._C._ctc_loss(p, targets, lengths, target_lengths, 0, False)[1][0, 3, 4]
    assert gradcheck(fn, (x,))
    assert gradgradcheck(fn, (x,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("blank,target", [(0, [1, 1]), (2, [1, 0])])
def test_ctc_hessian_matches_alignment_enumeration(device, blank, target):
    paths = []
    for path in itertools.product(range(3), repeat=4):
        collapsed = [label for i, label in enumerate(path) if i == 0 or label != path[i - 1]]
        if [label for label in collapsed if label != blank] == target:
            paths.append(path)
    data = np.linspace(-.8, .7, 12).reshape(4, 1, 3)
    x = leaf(data, device)
    v = leaf(np.linspace(.3, -.2, 12).reshape(4, 1, 3), device).detach()
    lp = x.log_softmax(2)
    loss = F.ctc_loss(lp, tp.tensor([target], device=device), tp.tensor([4], device=device),
                      tp.tensor([2], device=device), blank=blank, reduction="sum")
    g = tp.autograd.grad(loss, x, create_graph=True)[0]
    hv = tp.autograd.grad((g * v).sum(), x)[0]
    ref_x = leaf(data, device)
    ref_lp = ref_x.log_softmax(2)
    scores = tp.stack([sum(ref_lp[t, 0, label] for t, label in enumerate(path)) for path in paths])
    ref_loss = -scores.logsumexp(0)
    ref_g = tp.autograd.grad(ref_loss, ref_x, create_graph=True)[0]
    ref_hv = tp.autograd.grad((ref_g * v).sum(), ref_x)[0]
    np.testing.assert_allclose(array(g), array(ref_g), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(array(hv), array(ref_hv), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_ctc_zero_infinity_and_empty_lengths(device):
    data = np.full((3, 3, 3), -np.log(3))
    data[2, 0] = -np.inf
    data[:, 1] = -np.inf
    lp = leaf(data, device)
    targets = tp.tensor([[1, 1], [0, 0], [1, 0]], device=device)
    il = tp.tensor([2, 0, 3], device=device)
    tl = tp.tensor([2, 0, 1], device=device)
    loss = F.ctc_loss(lp, targets, il, tl, zero_infinity=True, reduction="sum")
    g = tp.autograd.grad(loss, lp, create_graph=True)[0]
    hv = tp.autograd.grad(g.sum(), lp)[0]
    assert np.isfinite(array(g)).all() and np.isfinite(array(hv)).all()
    np.testing.assert_array_equal(array(g)[:, :2], 0)
    np.testing.assert_array_equal(array(hv)[:, :2], 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_ctc_recording_preserves_first_gradient(device, dtype):
    lp = leaf(np.linspace(-1.4, -.2, 24).reshape(4, 2, 3), device).to(dtype).detach().requires_grad_(True)
    args = ctc_arguments(device)
    fn = lambda: F.ctc_loss(lp, *args, reduction="sum")
    recorded = tp.autograd.grad(fn(), lp, create_graph=True)[0]
    ordinary = tp.autograd.grad(fn(), lp)[0]
    tolerance = 2e-6 if dtype == tp.float32 else 1e-12
    np.testing.assert_allclose(array(recorded), array(ordinary), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("logarithmic", [False, True])
def test_sparse_empty_support(device, logarithmic):
    indices = tp.empty(2, 0, dtype=tp.int64, device=device)
    x = tp.sparse_coo_tensor(indices, tp.empty(0, 2, dtype=tp.float64, device=device),
                            (3, 4, 2), is_coalesced=True).detach().requires_grad_(True)
    go = x.detach().clone().requires_grad_(True)
    kernel = tp._C._sparse_log_softmax_backward_data if logarithmic else tp._C._sparse_softmax_backward_data
    g = tp.autograd.grad(kernel(go, x, 1, x).to_dense().sum(), (go, x))
    for value in g:
        assert value.is_sparse and value._nnz() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_uncoalesced_support(device):
    indices = tp.tensor([[1, 0, 0, 1], [0, 2, 2, 1]], device=device)
    x = tp.sparse_coo_tensor(indices, leaf([.3, .2, .6, .7], device),
                            (2, 3)).detach().requires_grad_(True)
    go = tp.sparse_coo_tensor(indices, leaf([1, 2, 3, 4], device),
                             (2, 3)).detach().requires_grad_(True)
    cotangent = tp.ones(2, 3, dtype=tp.float64, device=device)
    fn = lambda g, o: tp._C._sparse_softmax_backward_data(g, o, 1, o).to_dense()
    actual = tp.autograd.grad(fn(go, x), (go, x), cotangent)
    xc = x.coalesce().detach().requires_grad_(True)
    gc = go.coalesce().detach().requires_grad_(True)
    expected = tp.autograd.grad(fn(gc, xc), (gc, xc), cotangent)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(array(a.to_dense()), array(b.to_dense()), atol=1e-12)
