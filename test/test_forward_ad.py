"""Tests for the forward-mode AD engine (dual tensors + jvp).

JVPs are validated two ways: against the reverse-mode double-backward
trick for ops whose backward is itself differentiable, and against
hand-derived formulas for ops whose backward runs through composite
backward helpers.
"""

import pytest

import tensorplay as tp
import tensorplay.func as func
from tensorplay.autograd import Function
from tensorplay.autograd import forward_ad as fw
from tensorplay.autograd.functional import jacfwd as block_jacfwd
from tensorplay.autograd.functional import jvp as functional_jvp


def assert_close(got, want, tol=1e-5):
    assert got is not None and want is not None
    assert got.shape == want.shape, (got.shape, want.shape)
    assert tp.allclose(got.float(), want.float(), atol=tol, rtol=tol), \
        (got, want)


class TestLevels:
    def test_first_level_is_zero(self):
        assert fw.current_dual_level() == -1
        with fw.dual_level() as lvl:
            assert lvl == 0
            assert fw.current_dual_level() == 0
        assert fw.current_dual_level() == -1

    def test_nesting_is_rejected(self):
        with fw.dual_level():
            with pytest.raises(RuntimeError):
                fw.enter_dual_level()

    def test_exit_without_enter_raises(self):
        with pytest.raises(RuntimeError):
            fw.exit_dual_level()

    def test_exit_wrong_level_raises(self):
        with fw.dual_level():
            with pytest.raises(RuntimeError):
                fw.exit_dual_level(level=3)

    def test_tangents_are_erased_on_exit(self):
        x = tp.randn(3)
        with fw.dual_level():
            dual = fw.make_dual(x, tp.ones(3))
            assert fw.unpack_dual(dual).tangent is not None
        assert fw.unpack_dual(dual, level=0).tangent is None


class TestDual:
    def test_make_unpack_roundtrip(self):
        x = tp.randn(4)
        t = tp.randn(4)
        with fw.dual_level():
            dual = fw.make_dual(x, t)
            assert fw.is_dual_tensor(dual)
            assert not fw.is_dual_tensor(x)
            primal, tangent = fw.unpack_dual(dual)
            assert_close(primal, x)
            assert_close(tangent, t)

    def test_unpack_plain_tensor_gives_none_tangent(self):
        x = tp.randn(4)
        with fw.dual_level():
            primal, tangent = fw.unpack_dual(x)
            assert_close(primal, x)
            assert tangent is None

    def test_make_dual_requires_active_level(self):
        with pytest.raises(RuntimeError):
            fw.make_dual(tp.tensor([1.0]), tp.tensor([1.0]))

    def test_make_dual_twice_at_same_level_raises(self):
        x = tp.randn(3)
        with fw.dual_level():
            dual = fw.make_dual(x, tp.ones(3))
            with pytest.raises(RuntimeError):
                fw.make_dual(dual, tp.ones(3))

    def test_dual_shares_storage_with_primal(self):
        x = tp.randn(3)
        x_before = x.clone()
        with fw.dual_level():
            dual = fw.make_dual(x, tp.ones(3))
            dual += 1.0
            primal, tangent = fw.unpack_dual(dual)
        assert_close(primal, x_before + 1.0)
        assert_close(tangent, tp.ones(3))


# (name, fn, primal shapes).  Inputs require grad so the view read-through
# is armed for the hand-written view methods.
OPS_CASES = [
    ("add", lambda a, b: a + b, ((4,), (4,))),
    ("sub", lambda a, b: a - b, ((4,), (4,))),
    ("mul", lambda a, b: a * b, ((4,), (4,))),
    ("exp", lambda a, b: a.exp(), ((4,), (4,))),
    ("sin", lambda a, b: a.sin(), ((4,), (4,))),
    ("cos", lambda a, b: a.cos(), ((4,), (4,))),
    ("neg", lambda a, b: -a, ((4,), (4,))),
    ("abs", lambda a, b: a.abs(), ((4,), (4,))),
    ("sum", lambda a, b: a.sum(dim=0), ((4,), (4,))),
    ("mm", lambda a, b: a @ b, ((2, 2), (2, 2))),
    ("bmm", lambda a, b: a @ b, ((1, 2, 2), (1, 2, 2))),
    ("mv", lambda a, b: a @ b, ((2, 2), (2,))),
    ("matmul_vecmat", lambda a, b: a @ b, ((4,), (4,))),
    ("transpose", lambda a, b: a.transpose(0, 1), ((2, 2), (2, 2))),
    ("t", lambda a, b: a.t(), ((2, 2), (2, 2))),
    ("permute", lambda a, b: a.permute(0, 2, 1), ((1, 2, 2), (1, 2, 2))),
    ("squeeze", lambda a, b: a.squeeze(0), ((1, 4), (1, 4))),
    ("unsqueeze", lambda a, b: a.unsqueeze(0), ((4,), (4,))),
    ("expand", lambda a, b: a.expand(2, 2), ((2, 1), (2, 1))),
    ("select", lambda a, b: a.select(1, 0), ((2, 2), (2, 2))),
    ("slice", lambda a, b: a.slice(1, 0, 1), ((2, 2), (2, 2))),
]


class TestOpsAgainstReverseMode:
    @pytest.mark.parametrize("name,fn,shapes", OPS_CASES,
                             ids=[c[0] for c in OPS_CASES])
    def test_jvp_matches_reverse_mode(self, name, fn, shapes):
        primals = tuple(
            tp.randn(s, requires_grad=True) for s in shapes)
        for _ in range(2):
            tangents = tuple(tp.randn(s) for s in shapes)
            got_out, got_jvp = func.jvp(fn, primals, tangents)
            want_out, want_jvp = functional_jvp(
                fn, primals, tangents, mode="reversed")
            assert_close(got_out.detach(), want_out.detach())
            assert_close(got_jvp.detach(), want_jvp.detach())


class TestJvpTransform:
    def test_value_and_tangent(self):
        x = tp.randn(4)
        t = tp.randn(4)
        out, jvp_out = func.jvp(lambda a: a * a.exp(), (x,), (t,))
        assert_close(out, x * x.exp())
        assert_close(jvp_out, t * x.exp() + t * x * x.exp())

    # Ops whose backward runs through composite backward helpers: the
    # reverse-mode double trick cannot differentiate those helpers, so the
    # jvp is checked against hand-derived formulas instead.
    def test_elementwise_analytic(self):
        x = tp.randn(4)
        t = tp.randn(4)

        b = tp.randn(4).abs() + 0.5
        _, jv = func.jvp(lambda a: a / b, (x,), (t,))
        assert_close(jv, t / b)

        _, jv = func.jvp(lambda a: (a.abs() + 0.5).log(), (x,), (t,))
        assert_close(jv, t * x.sign() / (x.abs() + 0.5))

        _, jv = func.jvp(lambda a: (a.abs() + 0.5).sqrt(), (x,), (t,))
        assert_close(jv, t * x.sign() / (2 * (x.abs() + 0.5).sqrt()))

        s = tp.sigmoid(x)
        _, jv = func.jvp(lambda a: tp.sigmoid(a), (x,), (t,))
        assert_close(jv, s * (1 - s) * t)

        th = tp.tanh(x)
        _, jv = func.jvp(lambda a: tp.tanh(a), (x,), (t,))
        assert_close(jv, (1 - th * th) * t)

        _, jv = func.jvp(lambda a: tp.relu(a), (x,), (t,))
        mask = tp.where(x > 0, tp.ones(4), tp.zeros(4))
        assert_close(jv, mask * t)

        _, jv = func.jvp(lambda a: a.pow(3.0), (x,), (t,))
        assert_close(jv, 3.0 * x * x * t)

        _, jv = func.jvp(lambda a, c: a / (c.abs() + 0.5),
                         (x, x), (t, tp.zeros(4)))
        assert_close(jv, t / (x.abs() + 0.5))

    def test_has_aux(self):
        x = tp.randn(4)
        out, jvp_out, aux = func.jvp(
            lambda a: (a.sin(), a.sum()), (x,), (tp.ones(4),), has_aux=True)
        assert_close(out, x.sin())
        assert_close(jvp_out, x.cos())
        assert_close(aux, x.sum())

    def test_independent_output_gets_zero_tangent(self):
        x = tp.randn(4)
        out, jvp_out = func.jvp(
            lambda a: (a.sum(), tp.randn(3)), (x,), (tp.ones(4),))
        assert float(jvp_out[1].abs().sum()) == 0.0
        assert float(jvp_out[0]) == 4.0

    def test_strict_rejects_independent_output(self):
        x = tp.randn(4)
        with pytest.raises(RuntimeError, match="independent"):
            func.jvp(
                lambda a: (a.sum(), tp.randn(3)), (x,), (tp.ones(4),),
                strict=True)

    def test_shape_mismatch_raises(self):
        x = tp.randn(4)
        with pytest.raises(RuntimeError, match="shape"):
            func.jvp(lambda a: a, (x,), (tp.ones(3),))

    def test_tangent_of_second_input_flows(self):
        a = tp.randn(4)
        b = tp.randn(4)
        out, jvp_out = func.jvp(
            lambda x, y: x * y, (a, b), (tp.zeros(4), tp.ones(4)))
        assert_close(jvp_out, a)


class TestCustomFunction:
    def test_jvp_hook_runs_when_inputs_are_dual(self):
        class MyMul(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a * b

            @staticmethod
            def backward(ctx, g):
                a, b = ctx.saved_tensors
                return g * b, g * a

            @staticmethod
            def jvp(ctx, ga, gb):
                a, b = ctx.saved_tensors
                return ga * b + a * gb

        x = tp.randn(4)
        y = tp.randn(4)
        dx, dy = tp.randn(4), tp.randn(4)
        with fw.dual_level():
            out = MyMul.apply(fw.make_dual(x, dx), fw.make_dual(y, dy))
            primal, tangent = fw.unpack_dual(out)
        assert_close(primal, x * y)
        assert_close(tangent, dx * y + x * dy)

    def test_without_jvp_hook_tangents_are_dropped(self):
        class Plain(Function):
            @staticmethod
            def forward(ctx, a):
                return a * 2

            @staticmethod
            def backward(ctx, g):
                return g * 2

        x = tp.randn(4)
        with fw.dual_level():
            out = Plain.apply(fw.make_dual(x, tp.ones(4)))
            primal, tangent = fw.unpack_dual(out)
        assert_close(primal, x * 2)
        assert tangent is None

    def test_backward_still_works(self):
        class MyMul(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a * b

            @staticmethod
            def backward(ctx, g):
                a, b = ctx.saved_tensors
                return g * b, g * a

        x = tp.randn(4, requires_grad=True)
        y = tp.randn(4, requires_grad=True)
        MyMul.apply(x, y).sum().backward()
        assert_close(x.grad, y)
        assert_close(y.grad, x)


class TestJacfwd:
    def test_block_jacfwd_diag(self):
        x = tp.randn(4)
        j = block_jacfwd(lambda a: a.sin(), x)
        assert_close(j, tp.diag(x.cos()))

    def test_block_jacfwd_multiple_inputs(self):
        a = tp.randn(3)
        b = tp.randn(3)
        j = block_jacfwd(lambda u, v: u * v, (a, b))
        assert_close(j[0], tp.diag(b))
        assert_close(j[1], tp.diag(a))

    def test_func_jacfwd_matches_jacrev(self):
        x = tp.randn(4)
        m = tp.randn(3, 4)
        f = lambda t: (m @ t).sin()
        rev = func.jacrev(f)(x)
        fwd = func.jacfwd(f)(x)
        assert list(fwd.shape) == [3, 4]
        assert_close(rev.detach(), fwd.detach())


class TestCompose:
    def test_backward_through_dual_graph(self):
        x = tp.randn(4, requires_grad=True)
        with fw.dual_level():
            dual = fw.make_dual(x, tp.zeros(4))
            y = (dual * dual).sum()
        y.backward()
        assert_close(x.grad, 2 * x)

    def test_chained_forward_ops(self):
        x = tp.randn(4)
        t = tp.randn(4)
        out, jvp_out = func.jvp(
            lambda a: (a.exp().sin() + a).tanh(), (x,), (t,))
        want_out = (x.exp().sin() + x).tanh()
        din = t * x.exp() * x.exp().cos() + t
        sech2 = 1.0 - want_out * want_out
        assert_close(out, want_out)
        assert_close(jvp_out, din * sech2)
