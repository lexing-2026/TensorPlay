import unittest
import torch
import numpy as np
import tensorplay
from tensorplay import Tensor
import tensorplay.nn as nn

class TestConvLayers(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu' # Currently only CPU implemented for these backward kernels
        self.dtype = tensorplay.float32
        
    def _compare_tensors(self, tp_tensor, torch_tensor, rtol=1e-4, atol=1e-5):
        self.assertTrue(torch_tensor is not None, "Torch tensor is None")
        self.assertTrue(tp_tensor is not None, "TensorPlay tensor is None")
        tp_np = tp_tensor.detach().numpy()
        torch_np = torch_tensor.detach().numpy()
        np.testing.assert_allclose(tp_np, torch_np, rtol=rtol, atol=atol)

    def test_conv1d(self):
        print("\nTesting Conv1d...")
        N, C_in, L = 2, 4, 10
        C_out, K = 3, 3
        stride, padding = 1, 1
        
        # PyTorch
        t_in = torch.randn(N, C_in, L, requires_grad=True)
        t_conv = torch.nn.Conv1d(C_in, C_out, K, stride=stride, padding=padding, bias=True)
        t_out = t_conv(t_in)
        t_loss = t_out.sum()
        t_loss.backward()
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy(), requires_grad=True)
        tp_conv = nn.Conv1d(C_in, C_out, K, stride=stride, padding=padding, bias=True)
        # Manually copy weights/bias to ensure same initialization
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        tp_conv.bias.data = Tensor(t_conv.bias.detach().numpy())
        
        tp_out = tp_conv(tp_in)
        tp_loss = tp_out.sum()
        tp_loss.backward()
        
        # Compare Forward
        self._compare_tensors(tp_out, t_out)
        
        # Compare Backward
        self._compare_tensors(tp_in.grad, t_in.grad)
        self._compare_tensors(tp_conv.weight.grad, t_conv.weight.grad)
        self._compare_tensors(tp_conv.bias.grad, t_conv.bias.grad)
        print("Conv1d Passed!")

    def test_conv2d(self):
        print("\nTesting Conv2d...")
        N, C_in, H, W = 2, 4, 10, 10
        C_out, K = 3, 3
        stride, padding = 1, 1
        
        # PyTorch
        t_in = torch.randn(N, C_in, H, W, requires_grad=True)
        t_conv = torch.nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding, bias=True)
        t_out = t_conv(t_in)
        t_loss = t_out.sum()
        t_loss.backward()
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy(), requires_grad=True)
        tp_conv = nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding, bias=True)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        tp_conv.bias.data = Tensor(t_conv.bias.detach().numpy())
        
        tp_out = tp_conv(tp_in)
        tp_loss = tp_out.sum()
        tp_loss.backward()
        
        self._compare_tensors(tp_out, t_out)
        self._compare_tensors(tp_in.grad, t_in.grad)
        self._compare_tensors(tp_conv.weight.grad, t_conv.weight.grad)
        self._compare_tensors(tp_conv.bias.grad, t_conv.bias.grad)
        print("Conv2d Passed!")

    def test_conv3d(self):
        print("\nTesting Conv3d...")
        N, C_in, D, H, W = 2, 3, 5, 8, 8
        C_out, K = 4, 3
        stride, padding = 1, 1
        
        # PyTorch
        t_in = torch.randn(N, C_in, D, H, W, requires_grad=True)
        t_conv = torch.nn.Conv3d(C_in, C_out, K, stride=stride, padding=padding, bias=True)
        t_out = t_conv(t_in)
        t_loss = t_out.sum()
        t_loss.backward()
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy(), requires_grad=True)
        tp_conv = nn.Conv3d(C_in, C_out, K, stride=stride, padding=padding, bias=True)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        tp_conv.bias.data = Tensor(t_conv.bias.detach().numpy())
        
        tp_out = tp_conv(tp_in)
        tp_loss = tp_out.sum()
        tp_loss.backward()
        
        self._compare_tensors(tp_out, t_out)
        self._compare_tensors(tp_in.grad, t_in.grad)
        self._compare_tensors(tp_conv.weight.grad, t_conv.weight.grad)
        self._compare_tensors(tp_conv.bias.grad, t_conv.bias.grad)
        print("Conv3d Passed!")

    def test_conv_transpose2d(self):
        print("\nTesting ConvTranspose2d...")
        N, C_in, H, W = 2, 4, 10, 10
        C_out, K = 3, 3
        stride, padding = 2, 1
        output_padding = 1
        
        # PyTorch
        t_in = torch.randn(N, C_in, H, W, requires_grad=True)
        t_conv = torch.nn.ConvTranspose2d(C_in, C_out, K, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        t_out = t_conv(t_in)
        t_loss = t_out.sum()
        t_loss.backward()
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy(), requires_grad=True)
        tp_conv = nn.ConvTranspose2d(C_in, C_out, K, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        tp_conv.bias.data = Tensor(t_conv.bias.detach().numpy())
        
        tp_out = tp_conv(tp_in)
        tp_loss = tp_out.sum()
        tp_loss.backward()
        
        self._compare_tensors(tp_out, t_out)
        self._compare_tensors(tp_in.grad, t_in.grad)
        self._compare_tensors(tp_conv.weight.grad, t_conv.weight.grad)
        self._compare_tensors(tp_conv.bias.grad, t_conv.bias.grad)
        print("ConvTranspose2d Passed!")

    def test_conv_transpose3d(self):
        print("\nTesting ConvTranspose3d...")
        N, C_in, D, H, W = 2, 3, 5, 5, 5
        C_out, K = 2, 3
        stride, padding = 2, 1
        output_padding = 1
        
        # PyTorch
        t_in = torch.randn(N, C_in, D, H, W, requires_grad=True)
        t_conv = torch.nn.ConvTranspose3d(C_in, C_out, K, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        t_out = t_conv(t_in)
        t_loss = t_out.sum()
        t_loss.backward()
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy(), requires_grad=True)
        tp_conv = nn.ConvTranspose3d(C_in, C_out, K, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        tp_conv.bias.data = Tensor(t_conv.bias.detach().numpy())
        
        tp_out = tp_conv(tp_in)
        tp_loss = tp_out.sum()
        tp_loss.backward()
        
        self._compare_tensors(tp_out, t_out)
        self._compare_tensors(tp_in.grad, t_in.grad)
        self._compare_tensors(tp_conv.weight.grad, t_conv.weight.grad)
        self._compare_tensors(tp_conv.bias.grad, t_conv.bias.grad)
        print("ConvTranspose3d Passed!")

    def test_depthwise_conv2d(self):
        print("\nTesting Depthwise Conv2d (groups=in_channels)...")
        N, C_in, H, W = 2, 4, 10, 10
        C_out, K = 4, 3 # C_out must be divisible by groups. Here groups=C_in=4. C_out=4 implies depthwise multiplier=1.
        stride, padding = 1, 1
        groups = C_in
        
        # PyTorch
        t_in = torch.randn(N, C_in, H, W, requires_grad=True)
        t_conv = torch.nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding, groups=groups, bias=True)
        t_out = t_conv(t_in)
        t_loss = t_out.sum()
        t_loss.backward()
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy(), requires_grad=True)
        tp_conv = nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding, groups=groups, bias=True)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        tp_conv.bias.data = Tensor(t_conv.bias.detach().numpy())
        
        tp_out = tp_conv(tp_in)
        tp_loss = tp_out.sum()
        tp_loss.backward()
        
        self._compare_tensors(tp_out, t_out)
        self._compare_tensors(tp_in.grad, t_in.grad)
        self._compare_tensors(tp_conv.weight.grad, t_conv.weight.grad)
        self._compare_tensors(tp_conv.bias.grad, t_conv.bias.grad)
        print("Depthwise Conv2d Passed!")

    def test_conv_transpose2d_output_size(self):
        print("\nTesting ConvTranspose2d with output_size...")
        N, C_in, H, W = 1, 2, 4, 4
        C_out, K = 2, 3
        stride, padding = 2, 1
        
        # PyTorch
        t_in = torch.randn(N, C_in, H, W)
        t_conv = torch.nn.ConvTranspose2d(C_in, C_out, K, stride=stride, padding=padding, bias=False)
        
        # We request 8x8 output (output_padding=1)
        target_output_size = (N, C_out, 8, 8)
        
        t_out = t_conv(t_in, output_size=target_output_size)
        
        # TensorPlay
        tp_in = Tensor(t_in.detach().numpy())
        tp_conv = nn.ConvTranspose2d(C_in, C_out, K, stride=stride, padding=padding, bias=False)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        
        tp_out = tp_conv(tp_in, output_size=target_output_size)
        
        self.assertEqual(tp_out.shape, t_out.shape)
        np.testing.assert_allclose(tp_out.detach().numpy(), t_out.detach().numpy(), rtol=1e-4, atol=1e-5)
        print("ConvTranspose2d output_size Passed!")

    def test_conv_transpose3d_output_size(self):
        print("\nTesting ConvTranspose3d with output_size...")
        N, C_in, D, H, W = 1, 2, 3, 4, 4
        C_out, K = 2, 3
        stride, padding = 2, 1
        
        t_in = torch.randn(N, C_in, D, H, W)
        t_conv = torch.nn.ConvTranspose3d(C_in, C_out, K, stride=stride, padding=padding, bias=False)
        
        target_d = 6
        target_h = 8
        target_w = 8
        
        target_output_size = (N, C_out, target_d, target_h, target_w)
        
        t_out = t_conv(t_in, output_size=target_output_size)
        
        tp_in = Tensor(t_in.detach().numpy())
        tp_conv = nn.ConvTranspose3d(C_in, C_out, K, stride=stride, padding=padding, bias=False)
        tp_conv.weight.data = Tensor(t_conv.weight.detach().numpy())
        
        tp_out = tp_conv(tp_in, output_size=target_output_size)
        
        self.assertEqual(tp_out.shape, t_out.shape)
        np.testing.assert_allclose(tp_out.detach().numpy(), t_out.detach().numpy(), rtol=1e-4, atol=1e-5)
        print("ConvTranspose3d output_size Passed!")

if __name__ == '__main__':
    unittest.main()
