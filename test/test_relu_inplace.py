import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
from tensorplay.nn import functional as F


class TestReluInplace(unittest.TestCase):
    def _run_case(self, device):
        leaf = tp.tensor(
            [-2.0, -0.5, 0.0, 0.5, 2.0],
            dtype=tp.float32,
            device=device,
            requires_grad=True,
        )
        # In-place operations on a non-leaf are the case used by a training
        # model (the activation follows a differentiable layer).
        value = leaf * 2.0
        result = F.relu(value, inplace=True)

        expected_value = tp.tensor(
            [0.0, 0.0, 0.0, 1.0, 4.0], dtype=tp.float32, device=device
        )
        self.assertTrue(tp.allclose(result, expected_value))
        self.assertTrue(tp.allclose(value, expected_value))

        result.sum().backward()
        expected_grad = tp.tensor(
            [0.0, 0.0, 0.0, 2.0, 2.0], dtype=tp.float32, device=device
        )
        self.assertTrue(tp.allclose(leaf.grad, expected_grad))

    def test_cpu_forward_backward(self):
        self._run_case("cpu")

    def test_leaf_is_rejected(self):
        leaf = tp.tensor([-1.0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "leaf.*in-place"):
            F.relu(leaf, inplace=True)

    def test_compiled_functional_relu_preserves_view_input(self):
        def forward(input):
            return F.relu(input.t())

        compiled = tp.compile(forward, fullgraph=True, strict_native=True)
        input = tp.tensor(
            [[-2.0, 0.5], [-0.5, 2.0]],
            dtype=tp.float32,
        )
        original = input.clone()
        result = compiled(input)

        expected = tp.tensor(
            [[0.0, 0.0], [0.5, 2.0]],
            dtype=tp.float32,
        )
        self.assertTrue(tp.allclose(result, expected))
        self.assertTrue(tp.allclose(input, original))

    @unittest.skipUnless(tp.cuda.is_available(), "CUDA not available")
    def test_cuda_forward_backward(self):
        self._run_case("cuda")


if __name__ == "__main__":
    unittest.main()
