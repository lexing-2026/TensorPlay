import unittest
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorplay as tp


pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="requires CUDA build of TensorPlay"
)


def tp_dtype(torch_dtype):
    return {
        torch.float32: tp.float32,
        torch.float16: tp.float16,
        torch.bfloat16: tp.bfloat16,
        torch.int32: tp.int32,
        torch.int64: tp.int64,
    }[torch_dtype]


def make_tp_tensor(torch_tensor):
    return tp.tensor(
        torch_tensor.cpu().tolist(), dtype=tp_dtype(torch_tensor.dtype)
    ).to("cuda")


class TestCUDAEmbedding(unittest.TestCase):
    def test_forward_matches_torch_for_index_types_and_shapes(self):
        for weight_dtype in (torch.float32, torch.float16, torch.int32, torch.int64):
            torch_weight = torch.arange(40, dtype=weight_dtype).reshape(10, 4).cuda()
            torch_indices = torch.tensor([[0, 3], [5, 9]], dtype=torch.int32, device="cuda")
            tp_weight = make_tp_tensor(torch_weight)
            tp_indices = make_tp_tensor(torch_indices)

            torch_out = torch.nn.functional.embedding(torch_indices, torch_weight)
            tp_out = tp.embedding(tp_weight, tp_indices)
            np.testing.assert_array_equal(tp_out.cpu().numpy(), torch_out.cpu().numpy())

    def test_float16_and_bfloat16_forward(self):
        for weight_dtype in (torch.float16, torch.bfloat16):
            torch_weight = torch.randn((17, 37), dtype=weight_dtype, device="cuda")
            torch_indices = torch.tensor([0, 2, 2, 16], dtype=torch.int64, device="cuda")
            tp_weight = make_tp_tensor(torch_weight)
            tp_indices = make_tp_tensor(torch_indices)

            torch_out = torch.nn.functional.embedding(torch_indices, torch_weight)
            tp_out = tp.embedding(tp_weight, tp_indices)
            np.testing.assert_allclose(
                tp_out.to(tp.float32).cpu().numpy(),
                torch_out.to(torch.float32).cpu().numpy(),
                rtol=2e-3,
                atol=2e-3,
            )

    def test_dense_backward_padding_and_frequency_scaling(self):
        torch_indices = torch.tensor([0, 0, 3, 3, 4], dtype=torch.int32, device="cuda")
        torch_grad = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0],
             [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
            device="cuda",
        )
        tp_indices = make_tp_tensor(torch_indices)
        tp_grad = make_tp_tensor(torch_grad)

        for padding_idx in (-1, 3):
            for scale_grad_by_freq in (False, True):
                torch_out = torch.ops.aten.embedding_dense_backward(
                    torch_grad,
                    torch_indices,
                    10,
                    padding_idx,
                    scale_grad_by_freq,
                )
                tp_out = tp.embedding_dense_backward(
                    tp_grad,
                    tp_indices,
                    10,
                    padding_idx,
                    scale_grad_by_freq,
                )
                np.testing.assert_allclose(
                    tp_out.cpu().numpy(), torch_out.cpu().numpy(), rtol=2e-3, atol=2e-3
                )

    def test_functional_negative_padding_is_normalized(self):
        torch_weight = torch.randn((8, 5), device="cuda", requires_grad=True)
        torch_indices = torch.tensor([0, 7, 7], dtype=torch.int64, device="cuda")
        tp_weight = make_tp_tensor(torch_weight.detach()).requires_grad_()
        tp_indices = make_tp_tensor(torch_indices)

        torch.nn.functional.embedding(
            torch_indices, torch_weight, padding_idx=-1
        ).sum().backward()
        tp.nn.functional.embedding(
            tp_indices, tp_weight, padding_idx=-1
        ).sum().backward()
        np.testing.assert_allclose(
            tp_weight.grad.cpu().numpy(), torch_weight.grad.cpu().numpy(), rtol=2e-3, atol=2e-3
        )


if __name__ == "__main__":
    unittest.main()
