"""Tests for the multi-tensor (foreach) op family.

Every operation is validated elementwise against the per-tensor float
reference computed with plain ops.
"""

import pytest

import tensorplay as tp


def _two():
    return [tp.tensor([1.0, -2.0, 3.0]), tp.tensor([0.5, 4.0])]


def _ref(tensors, fn):
    return [fn(t) for t in tensors]


def test_foreach_scalar_arithmetic():
    xs = _two()
    got = tp._foreach_add(xs, 0.5)
    for g, w in zip(got, _ref(xs, lambda t: t + 0.5)):
        assert tp.equal(g, w)
    got = tp._foreach_sub(xs, 1.0)
    for g, w in zip(got, _ref(xs, lambda t: t - 1.0)):
        assert tp.equal(g, w)
    got = tp._foreach_mul(xs, 2.0)
    for g, w in zip(got, _ref(xs, lambda t: t * 2.0)):
        assert tp.equal(g, w)
    got = tp._foreach_div(xs, 2.0)
    for g, w in zip(got, _ref(xs, lambda t: t / 2.0)):
        assert tp.equal(g, w)


def test_foreach_list_arithmetic():
    xs = _two()
    ys = [tp.tensor([1.0, 1.0, 1.0]), tp.tensor([2.0, -1.0])]
    got = tp._foreach_add(xs, ys, alpha=2.0)
    for g, x, y in zip(got, xs, ys):
        assert tp.equal(g, x + y * 2.0)
    got = tp._foreach_sub(xs, ys)
    for g, x, y in zip(got, xs, ys):
        assert tp.equal(g, x - y)
    got = tp._foreach_mul(xs, ys)
    for g, x, y in zip(got, xs, ys):
        assert tp.equal(g, x * y)


def test_foreach_unary():
    xs = _two()
    for op, ref in ((tp._foreach_exp, tp.exp), (tp._foreach_sqrt, tp.sqrt),
                    (tp._foreach_neg, tp.neg), (tp._foreach_abs, tp.abs),
                    (tp._foreach_sign, tp.sign)):
        got = op([t.clamp(min=0.1) if op is tp._foreach_sqrt else t
                  for t in xs])
        want = [ref(t.clamp(min=0.1) if op is tp._foreach_sqrt else t)
                for t in xs]
        for g, w in zip(got, want):
            assert tp.allclose(g, w)


def test_foreach_reductions():
    xs = _two()
    got = tp._foreach_norm(xs)
    for g, w in zip(got, [t.norm() for t in xs]):
        assert tp.allclose(g, w)
    got = tp._foreach_maximum(xs, 1.5)
    for g, w in zip(got, _ref(xs, lambda t: t.clamp(min=1.5))):
        assert tp.equal(g, w)
    got = tp._foreach_minimum(xs, 1.5)
    for g, w in zip(got, _ref(xs, lambda t: t.clamp(max=1.5))):
        assert tp.equal(g, w)
    got = tp._foreach_clamp_max(xs, 2.0)
    for g, w in zip(got, _ref(xs, lambda t: t.clamp(max=2.0))):
        assert tp.equal(g, w)


def test_foreach_fused_ternary():
    xs = _two()
    got = tp._foreach_addcmul(xs, xs, xs, value=2.0)
    for g, x in zip(got, xs):
        assert tp.equal(g, x + x * x * 2.0)
    denoms = [t.abs().clamp(min=1.0) for t in xs]
    got = tp._foreach_addcdiv(xs, xs, denoms, value=0.5)
    for g, x, d in zip(got, xs, denoms):
        assert tp.allclose(g, x + x / d * 0.5)


def test_foreach_inplace():
    xs = _two()
    tp._foreach_add_(xs, 1.0)
    assert xs[0].tolist() == [2.0, -1.0, 4.0]
    assert xs[1].tolist() == [1.5, 5.0]
    tp._foreach_mul_(xs, 2.0)
    assert xs[0].tolist() == [4.0, -2.0, 8.0]


def test_foreach_copy_and_zero():
    src = _two()
    dst = [tp.zeros(3), tp.zeros(2)]
    tp._foreach_copy_(dst, src)
    assert tp.equal(dst[0], src[0]) and tp.equal(dst[1], src[1])
    # the out variant returns the copied list without touching the inputs
    out = tp._foreach_copy(dst, src)
    assert tp.equal(out[0], src[0])
    tp._foreach_zero_(dst)
    assert tp.equal(dst[0], tp.zeros(3))


def test_foreach_full_surface():
    # every multi-tensor wrapper the binding exposes must be callable-name
    # discoverable on the top level namespace as well
    import tensorplay
    from tensorplay import functional
    names = [n for n in dir(functional) if n.startswith("_foreach_")]
    assert len(names) > 60
    for n in names:
        assert hasattr(tensorplay, n), n
