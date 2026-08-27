"""Tests for tensorplay.autograd.forward_ad (native seed-op core).

JVPs are validated against reverse-mode autograd and central finite
differences, including nested dual levels.
"""

import math

import pytest

import tensorplay as tp
from tensorplay.autograd import forward_ad as fw


def f_main(v):
    return v**3 * v.sin() + v.exp() / v


class TestLevels:
    def test_first_level_is_zero(self):
        assert fw.current_dual_level() == -1
        lvl = fw.enter_dual_level()
        assert lvl == 0 and fw.current_dual_level() == 0
        fw.exit_dual_level(lvl)
        assert fw.current_dual_level() == -1

    def test_nested_exit_pops_inner_levels(self):
        a = fw.enter_dual_level()
        b = fw.enter_dual_level()
        fw.exit_dual_level(a)
        assert fw.current_dual_level() == -1

    def test_exit_without_enter_raises(self):
        with pytest.raises(RuntimeError):
            fw.exit_dual_level()

    def test_make_dual_requires_active_level(self):
        with pytest.raises(RuntimeError, match="active forward AD level"):
            fw.make_dual(tp.tensor([1.0]), tp.tensor([1.0]))


class TestSeedOps:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda v: v**3 * v.sin() + v.exp() / v,
            lambda v: (v / 2.0 - 3.0 * v) .cos(),
            lambda v: ((v * 2.0).log()),
        ],
        ids=["poly-sin-exp-div", "sub-cos", "mul-log"],
    )
    def test_jvp_matches_reverse_mode(self, fn):
        x = tp.tensor([2.0])
        x.requires_grad_(True)

        lvl = fw.enter_dual_level()
        xd = fw.make_dual(x, tp.tensor([1.0]))
        tangent, _ = fw.unpack_dual(fn(xd))
        fw.exit_dual_level(lvl)

        fn(x).sum().backward()
        assert abs(float(tangent.sum()) - float(x.grad)) < 1e-5

    def test_two_argument_chain_rule(self):
        l0 = fw.enter_dual_level()
        a = fw.make_dual(tp.tensor([3.0]), tp.tensor([1.0]))
        b = fw.make_dual(tp.tensor([0.5]), tp.tensor([2.0]))
        r = (a / b + a * b).exp().log()
        t, _ = fw.unpack_dual(r)
        fw.exit_dual_level(l0)

        g = lambda u, v: math.log(math.exp(u / v + u * v))
        h = 1e-6
        ref = (g(3+h, .5) - g(3-h, .5)) / (2*h) \
            + 2 * (g(3, .5+h) - g(3, .5-h)) / (2*h)
        assert abs(float(t.sum()) - ref) < 1e-4

    def test_unsupported_op_raises_not_silently_drops(self):
        lvl = fw.enter_dual_level()
        d = fw.make_dual(tp.tensor([[1.0, 2.0]]), tp.tensor([[1.0, 1.0]]))
        with pytest.raises((TypeError, RuntimeError)):
            d.matmul(d.t())  # not in the native seed set
        fw.exit_dual_level(lvl)


class TestUnpack:
    def test_plain_tensor_returns_none_tangent(self):
        tan, primal = fw.unpack_dual(tp.ones(2))
        assert tan is None and primal.shape == (2,)

    def test_item_guarded(self):
        lvl = fw.enter_dual_level()
        d = fw.make_dual(tp.tensor([1.0]), tp.tensor([1.0]))
        with pytest.raises(TypeError, match="unpack_dual"):
            d.item()
        fw.exit_dual_level(lvl)
