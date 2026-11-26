"""In-place binary ops must actually write when the operands share a shape.

`mul_.Tensor` over packed float32 data used to return the untouched input: the
fast path handed VML the same buffer as an operand and as the destination, and
VML leaves that alias alone. The out-of-place form and every other dtype were
correct, which is why the gap survived. The cases below pin the write, the
aliasing read, the broadcast path and the float64 path that never took the
affected branch.
"""
import unittest

import torch

import tensorplay as tp


class TestInplaceBinary(unittest.TestCase):
    def test_mul_tensor_writes_float32_in_place(self):
        torch.manual_seed(7)
        lhs = torch.randn(4, 5)
        rhs = torch.randn(4, 5)
        want = lhs * rhs
        x = tp.tensor(lhs.numpy())
        y = tp.tensor(rhs.numpy())
        out = tp.ops.tp.mul_.Tensor(x, y)
        self.assertTrue(tp.allclose(out, tp.tensor(want.numpy()), rtol=1e-6, atol=1e-6))
        self.assertTrue(tp.allclose(x, tp.tensor(want.numpy()), rtol=1e-6, atol=1e-6))

    def test_mul_tensor_writes_a_one_dimensional_vector(self):
        x = tp.tensor([1.0, 2.0, 3.0])
        y = tp.tensor([3.0, 4.0, 5.0])
        out = tp.ops.tp.mul_.Tensor(x, y)
        self.assertTrue(tp.allclose(out, tp.tensor([3.0, 8.0, 15.0])))
        self.assertTrue(tp.allclose(x, tp.tensor([3.0, 8.0, 15.0])))

    def test_mul_tensor_reads_the_same_buffer_twice(self):
        z = tp.tensor([2.0, 3.0, 4.0])
        out = tp.ops.tp.mul_.Tensor(z, z)
        self.assertTrue(tp.allclose(out, tp.tensor([4.0, 9.0, 16.0])))

    def test_mul_tensor_still_broadcasts(self):
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tp.tensor([2.0, 3.0])
        out = tp.ops.tp.mul_.Tensor(a, b)
        self.assertTrue(tp.allclose(out, tp.tensor([[2.0, 6.0], [6.0, 12.0]])))

    def test_float64_same_shape_is_unchanged(self):
        x = tp.tensor([1.0, 2.0, 3.0]).to(tp.float64)
        y = tp.tensor([3.0, 4.0, 5.0]).to(tp.float64)
        out = tp.ops.tp.mul_.Tensor(x, y)
        self.assertTrue(tp.allclose(out, tp.tensor([3.0, 8.0, 15.0]).to(tp.float64)))


if __name__ == "__main__":
    unittest.main()
