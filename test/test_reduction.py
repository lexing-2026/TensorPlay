import tensorplay as tp

from tensorplay.testing._internal.common_utils import TestCase, run_tests
from tensorplay.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)


class TestReduction(TestCase):
    def test_prod(self, device):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        self.assertEqual(x.prod().item(), 24.0)

        p0 = x.prod(dim=0)
        self.assertEqual(p0[0].item(), 3.0)
        self.assertEqual(p0[1].item(), 8.0)

        p1 = x.prod(dim=1)
        self.assertEqual(p1[0].item(), 2.0)
        self.assertEqual(p1[1].item(), 12.0)

    def test_all_any(self, device):
        x = tp.tensor([[1, 0], [1, 1]], dtype=tp.float32, device=device)
        self.assertTrue(x.any().item())
        self.assertFalse(x.all().item())

        y = tp.tensor([1, 1], dtype=tp.float32, device=device)
        self.assertTrue(y.all().item())

        z = tp.tensor([0, 0], dtype=tp.float32, device=device)
        self.assertFalse(z.any().item())

        x_all0 = x.all(dim=0)
        self.assertTrue(x_all0[0].item())
        self.assertFalse(x_all0[1].item())

    def test_argmax_argmin(self, device):
        x = tp.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], device=device)

        self.assertEqual(x.argmax().item(), 5)
        self.assertEqual(x.argmin().item(), 0)

        am0 = x.argmax(dim=0)
        self.assertEqual(am0[0].item(), 1)
        self.assertEqual(am0[1].item(), 0)
        self.assertEqual(am0[2].item(), 1)

        am1 = x.argmax(dim=1)
        self.assertEqual(am1[0].item(), 1)
        self.assertEqual(am1[1].item(), 2)

    def test_sum_mean(self, device):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        self.assertEqual(x.sum().item(), 10.0)
        self.assertEqual(x.mean().item(), 2.5)

    def test_tensor_equality_via_assert_equal(self, device):
        x = tp.arange(0, 6, device=device).reshape((2, 3))
        expected = tp.tensor([[0, 1, 2], [3, 4, 5]], dtype=tp.int64, device=device)
        self.assertEqual(x, expected)

        # Strided views are compared by value
        self.assertEqual(x.t().contiguous(), expected.t().contiguous())

    def test_logsumexp_reduces_any_set_of_dimensions(self, device):
        tp.manual_seed(0)
        x = tp.randn(2, 3, 4, dtype=tp.float64, device=device, requires_grad=True)
        ref = tp.log(tp.exp(x).sum((0, 2)))
        for dims in ((0, 2), [2, 0], (-1, 0), [0, -1]):
            self.assertTrue(tp.allclose(tp.logsumexp(x, dims), ref))
        self.assertEqual(tuple(x.logsumexp([0, 2], keepdim=True).shape), (1, 3, 1))
        self.assertTrue(tp.allclose(tp.logsumexp(x, ()), tp.log(tp.exp(x).sum())))
        self.assertTrue(tp.allclose(tp.logsumexp(x, [1]), tp.logsumexp(x, 1)))
        # The gradient is the softmax over the reduced entries.
        (g,) = tp.autograd.grad(tp.logsumexp(x, (0, 2)).sum(), x)
        self.assertTrue(tp.allclose(g, tp.exp(x - ref.reshape(1, 3, 1))))
        out = tp.empty(0, dtype=tp.float64, device=device)
        tp.logsumexp(x.detach(), (2, 0), out=out)
        self.assertTrue(tp.allclose(out, ref))
        with self.assertRaisesRegex(RuntimeError, "multiple times"):
            tp.logsumexp(x, (1, -2))

    def test_logsumexp_promotes_integers_and_keeps_infinities(self, device):
        ints = tp.tensor([[0, 1], [2, 3]], device=device)
        got = tp.logsumexp(ints, 1)
        self.assertEqual(got.dtype, tp.get_default_dtype())
        self.assertTrue(tp.allclose(got.cpu(), tp.log(tp.exp(ints.cpu().float()).sum(1))))
        inf = float("inf")
        x = tp.tensor([[inf, inf, 1.0], [-inf, -inf, -inf]], device=device)
        self.assertEqual(tp.logsumexp(x, 1).tolist(), [inf, -inf])

    def test_logsumexp_under_vmap_reduces_each_sample(self, device):
        tp.manual_seed(0)
        x = tp.randn(5, 2, 3, 4, device=device)
        for dims, keepdim in (((0, 2), False), ([-1], True), ((), False)):
            got = tp.func.vmap(lambda t: tp.logsumexp(t, dims, keepdim))(x)
            want = tp.stack([tp.logsumexp(t, dims, keepdim) for t in x])
            self.assertEqual(tuple(got.shape), tuple(want.shape))
            self.assertTrue(tp.allclose(got, want))
        # The batch dimension need not lead.
        moved = tp.func.vmap(lambda t: tp.logsumexp(t, (0,)), in_dims=2)(x)
        self.assertTrue(tp.allclose(moved, tp.logsumexp(x, 0).transpose(0, 1)))

    def test_sum_over_no_listed_dimension_under_vmap_keeps_the_batch(self, device):
        x = tp.arange(24, dtype=tp.float32, device=device).reshape(4, 2, 3)
        got = tp.func.vmap(lambda t: t.sum(dim=[]))(x)
        self.assertEqual(got.tolist(), [t.sum().item() for t in x])
        scalars = tp.arange(4, dtype=tp.float32, device=device)
        self.assertEqual(tp.func.vmap(lambda t: t.sum(dim=[]))(scalars).tolist(),
                         [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(tp.func.vmap(lambda t: tp.logsumexp(t, []))(scalars).tolist(),
                         [0.0, 1.0, 2.0, 3.0])


instantiate_device_type_tests(TestReduction, globals())

if __name__ == "__main__":
    run_tests()
