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


instantiate_device_type_tests(TestReduction, globals())

if __name__ == "__main__":
    run_tests()
