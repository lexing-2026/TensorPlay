"""Extended native quantized-operator coverage.

Covers the operators added to close the CPU/CUDA gap: quantized batch norm
(CPU and CUDA) and the CUDA activation / cat / pooling / convolution families.
CUDA cases are skipped when no GPU is available.
"""

import unittest

import tensorplay as tp


def _q(x, scale, zp, dev):
    return tp.quantize_per_tensor(x.to(dev), scale, zp, tp.qint8)


def _assert_cuda_matches_cpu(self, fn, x, scale=0.1, zp=2, atol=1e-5):
    r_cpu = fn(_q(x, scale, zp, "cpu"))
    r_gpu = fn(_q(x, scale, zp, "cuda"))
    d = tp.dequantize(r_gpu).cpu() - tp.dequantize(r_cpu)
    self.assertLess(float(d.abs().max().item()), atol)


@unittest.skipUnless(tp.cuda.is_available(), "CUDA not available")
class TestCudaQuantizedOps(unittest.TestCase):
    def test_activations_match_cpu(self):
        x = tp.randn(1, 4, 8) * 2
        _assert_cuda_matches_cpu(self, lambda t: tp._C.quantized_relu(t), x)
        _assert_cuda_matches_cpu(self, lambda t: tp._C.quantized_relu6(t), x)
        _assert_cuda_matches_cpu(
            self,
            lambda t: tp._C.quantized_leaky_relu(t, 0.1, 0.15, 3), x)
        _assert_cuda_matches_cpu(
            self, lambda t: tp._C.quantized_elu(t, 0.15, 3), x)
        _assert_cuda_matches_cpu(
            self, lambda t: tp._C.quantized_hardswish(t, 0.15, 3), x)
        _assert_cuda_matches_cpu(
            self, lambda t: tp._C.quantized_hardsigmoid(t, 0.15, 3), x)
        _assert_cuda_matches_cpu(
            self, lambda t: tp._C.quantized_sigmoid(t, 0.15, 3), x)
        _assert_cuda_matches_cpu(
            self, lambda t: tp._C.quantized_tanh(t, 0.15, 3), x)

    def test_cat_matches_cpu(self):
        a = tp.randn(2, 3, 4) * 1.5
        b = tp.randn(2, 5, 4) * 1.5
        r_cpu = tp._C.quantized_cat(
            [_q(a, 0.1, 2, "cpu"), _q(b, 0.1, 2, "cpu")], 1, None, None)
        r_gpu = tp._C.quantized_cat(
            [_q(a, 0.1, 2, "cuda"), _q(b, 0.1, 2, "cuda")], 1, None, None)
        d = tp.dequantize(r_gpu).cpu() - tp.dequantize(r_cpu)
        self.assertLess(float(d.abs().max().item()), 1e-5)

    def test_pool_matches_cpu(self):
        x1 = tp.randn(1, 2, 10) * 2
        _assert_cuda_matches_cpu(
            self,
            lambda t: tp._C.quantized_max_pool1d(
                t, [2], [2], [0], [1], False), x1)
        x2 = tp.randn(1, 2, 6, 6) * 2
        _assert_cuda_matches_cpu(
            self,
            lambda t: tp._C.quantized_max_pool2d(
                t, [2, 2], [2, 2], [0, 0], [1, 1], False), x2)
        x3 = tp.randn(1, 2, 4, 4, 4) * 2
        _assert_cuda_matches_cpu(
            self,
            lambda t: tp._C.quantized_max_pool3d(
                t, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), x3)

    def test_conv_matches_cpu(self):
        xs, ws = 0.1, 0.05
        bias = tp.randn(4)
        x1 = tp.randn(1, 3, 8)
        w1 = tp.randn(4, 3, 3) * 0.5
        r_cpu = tp._C.quantized_conv1d(
            _q(x1, xs, 1, "cpu"), _q(w1, ws, 0, "cpu"), bias.cpu(),
            xs, 1, ws, 0, 0.2, 2, [1], [0], [1], 1)
        r_gpu = tp._C.quantized_conv1d(
            _q(x1, xs, 1, "cuda"), _q(w1, ws, 0, "cuda"), bias.cuda(),
            xs, 1, ws, 0, 0.2, 2, [1], [0], [1], 1)
        self.assertLess(
            float((tp.dequantize(r_gpu).cpu() -
                    tp.dequantize(r_cpu)).abs().max().item()), 1e-5)

        x2 = tp.randn(1, 3, 8, 8)
        w2 = tp.randn(4, 3, 3, 3) * 0.5
        r_cpu = tp._C.quantized_conv2d(
            _q(x2, xs, 1, "cpu"), _q(w2, ws, 0, "cpu"), bias.cpu(),
            xs, 1, ws, 0, 0.2, 2, [1, 1], [0, 0], [1, 1], 1)
        r_gpu = tp._C.quantized_conv2d(
            _q(x2, xs, 1, "cuda"), _q(w2, ws, 0, "cuda"), bias.cuda(),
            xs, 1, ws, 0, 0.2, 2, [1, 1], [0, 0], [1, 1], 1)
        self.assertLess(
            float((tp.dequantize(r_gpu).cpu() -
                    tp.dequantize(r_cpu)).abs().max().item()), 1e-5)

        # depthwise-style grouped convolution
        xg = tp.randn(1, 4, 8, 8)
        wg = tp.randn(4, 2, 3, 3) * 0.5
        r_cpu = tp._C.quantized_conv2d(
            _q(xg, xs, 1, "cpu"), _q(wg, ws, 0, "cpu"), bias.cpu(),
            xs, 1, ws, 0, 0.2, 2, [1, 1], [0, 0], [1, 1], 2)
        r_gpu = tp._C.quantized_conv2d(
            _q(xg, xs, 1, "cuda"), _q(wg, ws, 0, "cuda"), bias.cuda(),
            xs, 1, ws, 0, 0.2, 2, [1, 1], [0, 0], [1, 1], 2)
        self.assertLess(
            float((tp.dequantize(r_gpu).cpu() -
                    tp.dequantize(r_cpu)).abs().max().item()), 1e-5)

    def test_batch_norm_matches_float(self):
        x = tp.randn(1, 3, 4, 4) * 2
        w = tp.randn(3)
        b = tp.randn(3)
        mean = tp.randn(3) * 0.5
        var = tp.rand(3) + 0.5
        for dev in ("cpu", "cuda"):
            qx = _q(x, 0.1, 2, dev)
            r = tp._C.quantized_batch_norm(
                qx, w.to(dev), b.to(dev), mean.to(dev), var.to(dev),
                1e-5, 0.2, 2)
            xf = tp.dequantize(qx)
            m = mean.view(1, 3, 1, 1).to(dev)
            v = var.view(1, 3, 1, 1).to(dev)
            ww = w.view(1, 3, 1, 1).to(dev)
            bb = b.view(1, 3, 1, 1).to(dev)
            ref = (xf - m) / tp.sqrt(v + 1e-5) * ww + bb
            self.assertLess(
                float((tp.dequantize(r) - ref).abs().max().item()), 0.2)


class TestQuantizedBatchNormCpu(unittest.TestCase):
    def test_matches_float_reference(self):
        x = tp.randn(1, 3, 4, 4) * 2
        w = tp.randn(3)
        b = tp.randn(3)
        mean = tp.randn(3) * 0.5
        var = tp.rand(3) + 0.5
        qx = _q(x, 0.1, 2, "cpu")
        r = tp._C.quantized_batch_norm(
            qx, w, b, mean, var, 1e-5, 0.2, 2)
        self.assertTrue(r.is_quantized())
        self.assertAlmostEqual(r.q_scale(), 0.2, places=6)
        xf = tp.dequantize(qx)
        m = mean.view(1, 3, 1, 1)
        v = var.view(1, 3, 1, 1)
        ww = w.view(1, 3, 1, 1)
        bb = b.view(1, 3, 1, 1)
        ref = (xf - m) / tp.sqrt(v + 1e-5) * ww + bb
        ref = tp.quantize_per_tensor(ref, 0.2, 2, tp.qint8).dequantize()
        self.assertLess(
            float((tp.dequantize(r) - ref).abs().max().item()), 0.2)

    def test_requires_qint8(self):
        with self.assertRaises(TypeError):
            tp._C.quantized_batch_norm(
                tp.randn(1, 3, 4, 4), None, None, tp.randn(3),
                tp.rand(3) + 0.5, 1e-5, 0.2, 2)


if __name__ == "__main__":
    unittest.main()
