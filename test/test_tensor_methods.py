"""Spec tests: Tensor method surface batch (new_* factories, dtype
shortcuts, pointwise method forms, operator dunders) vs local torch."""

import pytest
import torch

import tensorplay as tp
from tensorplay._C import DType


def f32(*shape):
    n = 1
    for s in shape:
        n *= s
    return tp.arange(n).to(DType.float32).reshape(list(shape))


class TestNewFactories:
    def test_new_zeros_ones_full(self):
        t = f32(2, 3)
        tt = torch.arange(6.).reshape(2, 3)
        assert tuple(t.new_zeros(2, 3).shape) == (2, 3)
        assert tuple(t.new_zeros([2, 3]).shape) == (2, 3)
        assert t.new_zeros([2, 3]).sum().item() == 0
        assert t.new_ones([2, 3]).tolist() == tt.new_ones([2, 3]).tolist()
        assert t.new_full([2, 2], 7.5).tolist() == \
            tt.new_full([2, 2], 7.5).tolist()

    def test_new_defaults_follow_self(self):
        t = f32(2).half()
        assert t.new_zeros([2]).dtype == DType.float16
        assert t.new_ones([2]).dtype == DType.float16

    def test_new_dtype_override(self):
        t = f32(2)
        assert t.new_zeros([2], dtype=DType.int64).dtype == DType.int64
        assert t.new_full([2], 1, dtype=DType.int64).dtype == DType.int64

    def test_new_empty(self):
        t = f32(2)
        assert tuple(t.new_empty([3, 2]).shape) == (3, 2)
        assert tuple(t.new_empty(4).shape) == (4,)

    def test_new_tensor(self):
        t = f32(2)
        out = t.new_tensor([9.0, 8.0])
        assert out.tolist() == [9.0, 8.0]
        assert out.dtype == DType.float32
        rg = t.new_tensor([1.0], requires_grad=True)
        assert rg.requires_grad is True


class TestDtypeShortcuts:
    @pytest.mark.parametrize("name,dt", [
        ("bool", "bool"), ("byte", "uint8"), ("char", "int8"),
        ("short", "int16"), ("half", "float16"), ("bfloat16", "bfloat16"),
    ])
    def test_shortcuts(self, name, dt):
        t = f32(2)
        assert getattr(t, name)().dtype == getattr(DType, dt)
        tt = torch.arange(2.)
        assert str(getattr(tt, name)().dtype).endswith(dt)


class TestPointwiseMethods:
    def test_fmod_remainder_floordiv_truediv(self):
        a = tp.tensor([-7.0, 7.0])
        b = tp.tensor([2.0, -3.0])
        ra = torch.tensor([-7.0, 7.0])
        rb = torch.tensor([2.0, -3.0])
        assert a.fmod(b).tolist() == torch.fmod(ra, rb).tolist()
        assert a.remainder(b).tolist() == torch.remainder(ra, rb).tolist()
        assert (a // b).tolist() == (ra // rb).tolist()
        assert a.true_divide(b).tolist() == torch.true_divide(ra, rb).tolist()

    def test_int_negative_direction(self):
        a = tp.tensor([-7, 7])
        b = tp.tensor([2, 2])
        ra = torch.tensor([-7, 7])
        rb = torch.tensor([2, 2])
        assert (a // b).tolist() == (ra // rb).tolist()
        assert a.fmod(b).tolist() == torch.fmod(ra, rb).tolist()

    def test_reflected_forms(self):
        x = tp.tensor([7.0, -7.0])
        xx = torch.tensor([7.0, -7.0])
        assert (10.0 // x).tolist() == (10.0 // xx).tolist()
        assert (7.0 % x).tolist() == (7.0 % xx).tolist()

    def test_inplace_mod_and_ifloordiv(self):
        c = tp.tensor([7.0, 9.0])
        cc = torch.tensor([7.0, 9.0])
        c %= tp.tensor([2.0])
        cc %= torch.tensor([2.0])
        assert c.tolist() == cc.tolist()
        d = tp.tensor([-7.0, 7.0])
        dd = torch.tensor([-7.0, 7.0])
        d //= tp.tensor([2.0])
        dd //= torch.tensor([2.0])
        assert d.tolist() == dd.tolist()


class TestBitwiseDunders:
    def setup_method(self):
        self.a = tp.tensor([5, 3])
        self.b = tp.tensor([3, 3])
        self.ta = torch.tensor([5, 3])
        self.tb = torch.tensor([3, 3])

    def test_and_or_xor(self):
        assert (self.a & self.b).tolist() == (self.ta & self.tb).tolist()
        assert (self.a | self.b).tolist() == (self.ta | self.tb).tolist()
        assert (self.a ^ self.b).tolist() == (self.ta ^ self.tb).tolist()

    def test_scalar_left_operands(self):
        assert (5 & self.a).tolist() == (5 & self.ta).tolist()
        assert (5 | self.a).tolist() == (5 | self.ta).tolist()
        assert (5 ^ self.a).tolist() == (5 ^ self.ta).tolist()

    def test_invert_shifts(self):
        assert (~self.a).tolist() == (~self.ta).tolist()
        one = tp.tensor([1, 1])
        tone = torch.tensor([1, 1])
        two = tp.tensor([1, 2])
        ttwo = torch.tensor([1, 2])
        assert (self.a << two).tolist() == (self.ta << ttwo).tolist()
        assert (self.a >> one).tolist() == (self.ta >> tone).tolist()

    def test_reflected_shift(self):
        assert (2 << self.a).tolist() == (2 << self.ta).tolist()

    def test_inplace_bitwise(self):
        d = tp.tensor([7])
        dd = torch.tensor([7])
        d &= tp.tensor([3])
        dd &= torch.tensor([3])
        assert d.tolist() == dd.tolist()
        e = tp.tensor([4, 8])
        ee = torch.tensor([4, 8])
        e <<= tp.tensor([1, 1])
        ee <<= torch.tensor([1, 1])
        assert e.tolist() == ee.tolist()
        f = tp.tensor([6])
        ff = torch.tensor([6])
        f ^= tp.tensor([3])
        ff ^= torch.tensor([3])
        assert f.tolist() == ff.tolist()

    def test_bool_tensor_ops(self):
        m = tp.tensor([True, False])
        tm = torch.tensor([True, False])
        n = tp.tensor([True, True])
        tn = torch.tensor([True, True])
        assert (m & n).tolist() == (tm & tn).tolist()
        assert (m | n).tolist() == (tm | tn).tolist()
        assert (~m).tolist() == (~tm).tolist()


class TestMiscMethods:
    def test_abs_pos(self):
        x = tp.tensor([7.0, -7.0])
        xx = torch.tensor([7.0, -7.0])
        assert abs(x).tolist() == abs(xx).tolist()
        assert (+x).tolist() == (+xx).tolist()

    def test_topk_values_indices(self):
        t = f32(6)
        tt = torch.arange(6.)
        v, i = t.topk(2)
        rv, ri = tt.topk(2)
        assert v.tolist() == rv.tolist()
        assert i.tolist() == ri.tolist()
        v2 = t.topk(2, largest=False)[0]
        rv2 = tt.topk(2, largest=False)[0]
        assert v2.tolist() == rv2.tolist()

    def test_count_nonzero(self):
        t = f32(4) - 2.0
        tt = torch.arange(4.) - 2.0
        assert t.count_nonzero().item() == tt.count_nonzero().item()

    def test_repeat_interleave_method(self):
        t = f32(3)
        tt = torch.arange(3.)
        assert t.repeat_interleave(2).tolist() == \
            tt.repeat_interleave(2).tolist()
        reps = tp.tensor([2, 0, 1])
        assert t.repeat_interleave(reps).tolist() == \
            tt.repeat_interleave(torch.tensor([2, 0, 1])).tolist()

    def test_unique_method(self):
        t = tp.tensor([3.0, 1.0, 3.0, 2.0])
        tt = torch.tensor([3.0, 1.0, 3.0, 2.0])
        assert t.unique().tolist() == tt.unique().tolist()
        got = [x.tolist() for x in t.unique(sorted=True,
                                            return_inverse=True)]
        ref = [x.tolist() for x in tt.unique(sorted=True,
                                             return_inverse=True)]
        assert got == ref

    def test_grad_through_remainder_composite(self):
        g = tp.tensor([-7.0, 7.0]).requires_grad_(True)
        gg = torch.tensor([-7.0, 7.0], requires_grad=True)
        (g.remainder(tp.tensor([2.0])) ** 2).sum().backward()
        (torch.remainder(gg, torch.tensor([2.0])) ** 2).sum().backward()
        assert g.grad.tolist() == gg.grad.tolist()

    def test_bool_truthiness_matches_item(self):
        """nb_bool contract (torch is_nonzero): was always-True before."""
        assert not bool(tp.tensor(False))
        assert not bool(tp.tensor(0.0))
        assert bool(tp.tensor(3.0))
        assert bool(tp.tensor(True))
        with pytest.raises(RuntimeError, match="no values is ambiguous"):
            bool(tp.zeros(0))
        with pytest.raises(RuntimeError, match="more than one value is ambiguous"):
            bool(tp.tensor([True, True]))
