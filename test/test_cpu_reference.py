"""CPU semantic-reference checks for reduced-precision LLM operators."""

import unittest

import tensorplay as tp


class TestCPUReference(unittest.TestCase):
    def test_mean_preserves_reduced_dtype(self):
        for dtype in (tp.float16, tp.bfloat16, tp.float32):
            x = tp.randn((2, 3, 4), dtype=tp.float32).to(dtype)
            self.assertEqual(x.mean().dtype, dtype)
            self.assertEqual(x.mean([1], False).dtype, dtype)

    def test_sdpa_reduced_dtype_forward_backward(self):
        for dtype in (tp.float16, tp.bfloat16, tp.float32):
            q = tp.randn((1, 2, 4, 8), dtype=tp.float32).to(dtype)
            k = tp.randn((1, 2, 4, 8), dtype=tp.float32).to(dtype)
            v = tp.randn((1, 2, 4, 8), dtype=tp.float32).to(dtype)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            out = tp.scaled_dot_product_attention(q, k, v, is_causal=True, impl=1)
            self.assertEqual(out.dtype, dtype)
            out.mean().backward()

            self.assertEqual(q.grad.dtype, dtype)
            self.assertEqual(k.grad.dtype, dtype)
            self.assertEqual(v.grad.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
