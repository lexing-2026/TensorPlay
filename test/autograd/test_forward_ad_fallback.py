"""Every differentiable operation answers a tangent.

An operation with a forward formula uses it.  A view applies itself to its
input's tangent; an in-place update takes the tangent of its functional twin;
any other operation takes its forward derivative from its backward (the
vector-Jacobian product, differentiated in the cotangent along the tangent),
replaying the random numbers it drew.  An out= variant refuses the tangent.
None of them drops it.
"""

import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay.autograd import forward_ad as fwAD
from tensorplay.autograd import gradcheck, gradgradcheck


def leaf(shape, seed, low=0.1, high=0.9):
    tp.manual_seed(seed)
    return tp.rand(*shape, dtype=tp.float64) * (high - low) + low


def tangent_of(fn, primals, tangents):
    with fwAD.dual_level():
        duals = [fwAD.make_dual(p, t) for p, t in zip(primals, tangents)]
        out = fn(*duals)
        if isinstance(out, (tuple, list)):
            return [fwAD.unpack_dual(o).tangent for o in out]
        return fwAD.unpack_dual(out).tangent


def central_difference(fn, primals, tangents, eps=1e-6):
    plus = fn(*[p + eps * t for p, t in zip(primals, tangents)])
    minus = fn(*[p - eps * t for p, t in zip(primals, tangents)])
    if isinstance(plus, (tuple, list)):
        return [(a - b) / (2 * eps) for a, b in zip(plus, minus)]
    return (plus - minus) / (2 * eps)


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def _index_put(a, v):
    return a.index_put((tp.tensor([0, 2]),), v * 3)


def _inplace_on_a_clone(a, b):
    c = a.clone()
    c.mul_(b)
    c.add_(b, alpha=2)
    return c


CASES = {
    "acos": (lambda a: a.acos(), [(5,)]),
    "clamp": (lambda a: a.clamp(0.3, 0.6), [(6,)]),
    "addmm": (lambda s, a, b: tp.addmm(s, a, b), [(2, 3), (2, 4), (4, 3)]),
    "cumsum": (lambda a: a.cumsum(0), [(5,)]),
    "dot": (lambda a, b: tp.dot(a, b), [(4,), (4,)]),
    "clone": (lambda a: a.clone(), [(3,)]),
    "conj": (lambda a: a.conj() * a, [(3,)]),
    "atan2": (lambda a, b: tp.atan2(a, b), [(4,), (4,)]),
    "amax": (lambda a: a.amax(1), [(3, 4)]),
    "contiguous": (lambda a: a.t().contiguous(), [(2, 3)]),
    "diagonal": (lambda a: a.diagonal() * 2, [(3, 3)]),
    "narrow": (lambda a: a.narrow(1, 1, 2).exp(), [(3, 4)]),
    "reshape": (lambda a: a.reshape(4, 3).sin(), [(3, 4)]),
    "conv1d": (lambda x, w: F.conv1d(x, w, padding=1), [(1, 2, 5), (3, 2, 3)]),
    "softmax": (lambda a: a.softmax(-1), [(2, 4)]),
    "log_softmax": (lambda a: F.log_softmax(a, -1), [(2, 4)]),
    "layer_norm": (lambda a, w: F.layer_norm(a, (4,), w), [(3, 4), (4,)]),
    "cross_entropy": (lambda a: F.cross_entropy(a, tp.tensor([1, 0, 3])), [(3, 4)]),
    "matmul": (lambda a, b: a @ b, [(2, 3), (3, 2)]),
    "mean": (lambda a: a.mean(), [(3, 2)]),
    "cat": (lambda a, b: tp.cat([a, b * 2]).sin(), [(2,), (3,)]),
    "stack": (lambda a, b: tp.stack([a, b]).exp(), [(3,), (3,)]),
    "index": (lambda a: a[tp.tensor([2, 0, 2])].sin(), [(4, 2)]),
    "index_put": (_index_put, [(4, 2), (2, 2)]),
    "sdpa": (_sdpa, [(1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 4)]),
    "in-place": (_inplace_on_a_clone, [(3,), (3,)]),
    "clamp_min_": (lambda a: a.clone().clamp_min_(0.5), [(6,)]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_tangent_matches_a_central_difference(name):
    fn, shapes = CASES[name]
    primals = [leaf(s, i) for i, s in enumerate(shapes)]
    tangents = [leaf(s, 10 + i, -1.0, 1.0) for i, s in enumerate(shapes)]
    got = tangent_of(fn, primals, tangents)
    want = central_difference(fn, primals, tangents)
    assert got is not None, f"{name} dropped its tangent"
    assert tp.allclose(got, want, atol=1e-6, rtol=1e-5)


def test_each_tensor_output_of_a_tuple_gets_its_tangent():
    a, t = leaf((3, 5), 0), leaf((3, 5), 1, -1.0, 1.0)
    values, indices = tangent_of(lambda x: tp.sort(x, 1), [a], [t])
    want = central_difference(lambda x: tp.sort(x, 1)[0], [a], [t])
    assert tp.allclose(values, want, atol=1e-6)
    assert indices is None


def test_each_element_of_a_list_output_gets_its_tangent():
    a, t = leaf((4, 3), 0), leaf((4, 3), 1, -1.0, 1.0)
    got = tangent_of(lambda x: list(x.exp().unbind(0)), [a], [t])
    want = central_difference(lambda x: list(x.exp().unbind(0)), [a], [t])
    assert len(got) == 4
    for g, w in zip(got, want):
        assert tp.allclose(g, w, atol=1e-6)


def test_a_view_reads_its_inputs_tangent():
    a, t = leaf((3, 3), 0), leaf((3, 3), 1)
    for view in (lambda x: x.diagonal(), lambda x: x.view(9), lambda x: x.unbind(1)[2]):
        assert tp.equal(tangent_of(view, [a], [t]), view(t))


def test_an_update_through_a_view_reaches_the_bases_tangent():
    a, t = leaf((2, 3), 0), leaf((2, 3), 1)
    with fwAD.dual_level():
        base = fwAD.make_dual(a.clone(), t.clone())
        row = base[1]
        row.mul_(3)
        got = fwAD.unpack_dual(base).tangent
    want = t.clone()
    want[1] *= 3
    assert tp.allclose(got, want)


def test_a_geometry_update_moves_the_tangent_with_it():
    a, t = leaf((2, 3), 0), leaf((2, 3), 1)
    with fwAD.dual_level():
        d = fwAD.make_dual(a.clone(), t.clone())
        d.t_()
        got = fwAD.unpack_dual(d).tangent
    assert got.shape == (3, 2)
    assert tp.equal(got, t.t())


@pytest.mark.parametrize("op", ["eq_", "ne_", "ge_", "gt_", "le_", "lt_"])
def test_an_update_by_a_comparison_is_constant(op):
    a, t = leaf((4,), 0), leaf((4,), 1)
    x = a.clone().requires_grad_(True)
    y = x * 2
    getattr(y, op)(1.0)
    y.sum().backward()
    assert tp.equal(x.grad, tp.zeros_like(a))
    with fwAD.dual_level():
        d = fwAD.make_dual(a.clone(), t.clone())
        getattr(d, op)(tp.full_like(a, 0.5))
        assert tp.equal(fwAD.unpack_dual(d).tangent, tp.zeros_like(a))


def test_dropout_carries_the_tangent_through_the_mask_it_drew():
    a, t = leaf((64,), 0), leaf((64,), 1)
    with fwAD.dual_level():
        out = F.dropout(fwAD.make_dual(a, t), 0.5, training=True)
        primal, tangent = fwAD.unpack_dual(out)
    kept = primal != 0
    assert 0 < int(kept.sum()) < 64
    assert tp.allclose(tangent, t * kept * 2)


def test_rrelu_carries_the_tangent_through_the_slopes_it_drew():
    a, t = leaf((32,), 0, -1.0, 1.0), leaf((32,), 1)
    for inplace in (False, True):
        with fwAD.dual_level():
            d = fwAD.make_dual(a.clone(), t.clone())
            out = F.rrelu(d, training=True, inplace=inplace)
            primal, tangent = fwAD.unpack_dual(out)
        slope = tp.where(a >= 0, tp.ones_like(a), primal / a)
        assert tp.allclose(tangent, t * slope)


def test_a_recurrent_layer_carries_the_tangent():
    tp.manual_seed(0)
    lstm = nn.LSTM(3, 4).double()
    x, t = leaf((5, 2, 3), 0), leaf((5, 2, 3), 1, -1.0, 1.0)
    got = tangent_of(lambda i: lstm(i)[0], [x], [t])
    want = central_difference(lambda i: lstm(i)[0], [x], [t])
    assert tp.allclose(got, want, atol=1e-6)


def test_an_out_variant_refuses_a_tangent():
    a, t = leaf((3,), 0), leaf((3,), 1)
    buf = tp.empty(3, dtype=tp.float64)
    with fwAD.dual_level():
        d = fwAD.make_dual(a, t)
        with pytest.raises(NotImplementedError, match="because it is an out= function"):
            tp.sin(d, out=buf)


def test_forward_ad_answers_under_no_grad():
    a, t = leaf((4,), 0), leaf((4,), 1, -1.0, 1.0)
    with tp.no_grad():
        got = tangent_of(lambda x: x.acos(), [a], [t])
    assert tp.allclose(got, -t / (1 - a * a).sqrt())


def test_the_tangent_is_differentiable_in_the_primal():
    # d/dx sum(-t / sqrt(1 - x^2)) = -t x / (1 - x^2)^(3/2)
    x = leaf((4,), 0).requires_grad_(True)
    t = leaf((4,), 1, -1.0, 1.0)
    with fwAD.dual_level():
        tangent = fwAD.unpack_dual(fwAD.make_dual(x, t).acos()).tangent
        (g,) = tp.autograd.grad(tangent.sum(), x)
    xd = x.detach()
    assert tp.allclose(g, -t * xd / (1 - xd * xd) ** 1.5)


def test_functional_jvp_of_an_operation_without_a_formula():
    a, t = leaf((4,), 0), leaf((4,), 1, -1.0, 1.0)
    _, jvp = tp.func.jvp(lambda x: x.acos(), (a,), (t,))
    assert tp.allclose(jvp, -t / (1 - a * a).sqrt())


def test_jacfwd_through_an_operation_without_a_formula():
    a = leaf((4,), 0)
    jac = tp.func.jacfwd(lambda x: x.acos().cumsum(0))(a)
    want = tp.tril(tp.ones(4, 4, dtype=tp.float64)) * (-1 / (1 - a * a).sqrt())
    assert tp.allclose(jac, want)


def test_gradcheck_forward_mode_through_operations_without_formulas():
    x = leaf((3,), 0).requires_grad_(True)
    y = leaf((3,), 1).requires_grad_(True)
    # atan2 of the inputs stays below 1.5, so half of it is inside acos's domain.
    assert gradcheck(lambda a, b: (tp.atan2(a, b) * 0.5).acos() + a.cumsum(0), (x, y),
                     check_forward_ad=True)


def test_forward_over_reverse_through_conjugating_backward_formulas():
    x = leaf((3,), 0).requires_grad_(True)
    assert gradgradcheck(lambda a: a.sin() * a, (x,), check_fwd_over_rev=True)
    assert gradgradcheck(lambda a: a.acos().exp(), (x,), check_fwd_over_rev=True)


def test_complex_tangents_through_conj_and_abs():
    tp.manual_seed(0)
    z = tp.randn(3, dtype=tp.complex128)
    t = tp.randn(3, dtype=tp.complex128)
    got = tangent_of(lambda w: w.conj(), [z], [t])
    assert tp.allclose(got, t.conj())
    got = tangent_of(lambda w: w.abs(), [z], [t])
    want = central_difference(lambda w: w.abs(), [z], [t])
    assert tp.allclose(got, want, atol=1e-6)


def test_a_formula_built_from_answering_operations_gives_a_plain_tangent():
    # polar's formula builds its tangent with complex(), which answers
    # forward AD itself; the tangent it hands on must carry none of its own.
    rho, theta = leaf((4,), 0, 0.5, 1.5), leaf((4,), 1, -0.5, 0.5)
    dr, dtheta = leaf((4,), 2, -1, 1), leaf((4,), 3, -1, 1)
    with fwAD.dual_level():
        out = tp.polar(fwAD.make_dual(rho, dr), fwAD.make_dual(theta, dtheta))
        tangent = fwAD.unpack_dual(out).tangent
        assert fwAD.unpack_dual(tangent).tangent is None
    want = central_difference(tp.polar, [rho, theta], [dr, dtheta])
    assert tp.allclose(tangent, want, atol=1e-6)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="needs CUDA")
def test_dropout_on_cuda_replays_the_mask():
    a = leaf((256,), 0).cuda()
    t = leaf((256,), 1).cuda()
    with fwAD.dual_level():
        out = F.dropout(fwAD.make_dual(a, t), 0.25, training=True)
        primal, tangent = fwAD.unpack_dual(out)
    kept = primal != 0
    assert tp.allclose(tangent, t * kept / 0.75)
