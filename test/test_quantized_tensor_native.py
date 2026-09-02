"""Native quantized-tensor semantics: dtype surface, quantizer metadata,
and the numeric contract of the affine grid."""

import unittest

import tensorplay as tp


class TestQuantizedDTypeSurface(unittest.TestCase):
    def test_dtypes_exist(self):
        self.assertEqual(tp.qint8.itemsize, 1)
        self.assertEqual(tp.quint8.itemsize, 1)
        self.assertEqual(tp.qint32.itemsize, 4)
        self.assertTrue(tp.qint8.is_quantized)
        self.assertTrue(tp.quint8.is_quantized)
        self.assertTrue(tp.qint32.is_quantized)
        self.assertFalse(tp.int8.is_quantized)
        self.assertFalse(tp.float32.is_quantized)

    def test_plain_int_tensor_not_quantized(self):
        t = tp.tensor([1, 2], dtype=tp.int8)
        self.assertFalse(t.is_quantized())


class TestQuantizerMetadata(unittest.TestCase):
    def test_per_tensor_roundtrip(self):
        x = tp.tensor([-1.0, 0.0, 1.0, 2.0])
        q = tp.quantize_per_tensor(x, 0.1, 10)
        self.assertEqual(q.dtype, tp.qint8)
        self.assertTrue(q.is_quantized())
        self.assertAlmostEqual(q.q_scale(), 0.1)
        self.assertEqual(q.q_zero_point(), 10)
        self.assertEqual(q.qscheme(), 0)
        back = q.dequantize()
        self.assertEqual(back.dtype, tp.float32)
        self.assertTrue(float((back - x).abs().max().item()) < 0.11)

    def test_per_channel_metadata(self):
        x = tp.tensor([[-1.0, 1.0], [2.0, -2.0]])
        q = tp.quantize_per_channel(x, tp.tensor([0.1, 0.2]),
                                    tp.tensor([0, 10]), 0)
        self.assertTrue(q.is_quantized())
        self.assertEqual(q.qscheme(), 1)
        self.assertAlmostEqual(float(q.q_per_channel_scales()[0]), 0.1)
        self.assertEqual(int(q.q_per_channel_zero_points()[1]), 10)
        self.assertEqual(q.q_per_channel_axis(), 0)
        back = q.dequantize()
        self.assertTrue(float((back - x).abs().max().item()) < 0.21)

    def test_qschemes_mismatch_errors(self):
        q = tp.quantize_per_tensor(tp.tensor([1.0]), 0.1, 0)
        with self.assertRaises(RuntimeError):
            q.q_per_channel_scales()
        qc = tp.quantize_per_channel(tp.tensor([[1.0]]), tp.tensor([0.1]),
                                     tp.tensor([0]), 0)
        with self.assertRaises(RuntimeError):
            qc.q_scale()

    def test_int_repr(self):
        q = tp.quantize_per_tensor(tp.tensor([0.0, 1.0]), 0.1, 5)
        codes = q.int_repr()
        self.assertEqual(codes.dtype, tp.int8)
        self.assertFalse(codes.is_quantized())
        self.assertEqual(int(codes[0]), 5)
        self.assertEqual(int(codes[1]), 15)

    def test_make_per_tensor_roundtrip(self):
        codes = tp.tensor([-10, 0, 10], dtype=tp.int8)
        q = tp._C._make_per_tensor_quantized_tensor(codes, 0.2, 3)
        self.assertEqual(q.dtype, tp.qint8)
        self.assertAlmostEqual(q.q_scale(), 0.2)
        self.assertEqual(q.q_zero_point(), 3)
        back = tp._C.int_repr(q)
        self.assertTrue(int((back.to(tp.int32) - codes.to(tp.int32)).abs().max().item()) == 0)


class TestQuantizerPropagation(unittest.TestCase):
    def test_clone_and_views(self):
        q = tp.quantize_per_tensor(tp.tensor([1.5, -2.0]), 0.5, 0)
        self.assertTrue(q.clone().is_quantized())
        v = q.unsqueeze(0).squeeze(0).transpose(0, 0)
        self.assertTrue(v.is_quantized())
        self.assertEqual(v.q_zero_point(), q.q_zero_point())

    def test_to_plain_storage(self):
        q = tp.quantize_per_tensor(tp.tensor([1.5]), 0.5, 0)
        raw = q.to(tp.int8)
        self.assertFalse(raw.is_quantized())
        self.assertEqual(raw.dtype, tp.int8)

    def test_dynamic_quantize_carries_qparams(self):
        q = tp.quantize_per_tensor_dynamic(tp.randn(8, 8), tp.qint8, False)
        self.assertEqual(q.dtype, tp.qint8)
        self.assertTrue(q.is_quantized())
        self.assertGreater(q.q_scale(), 0.0)

    def test_dequantize_passthrough(self):
        x = tp.tensor([1.0, 2.0])
        self.assertEqual(tp.dequantize(x).dtype, tp.float32)
        q = tp.quantize_per_tensor(x, 0.5, 0)
        back = tp.dequantize(q)
        self.assertTrue(float((back - x).abs().max().item()) < 0.26)


class TestQuantizedCompute(unittest.TestCase):
    def test_quantized_add(self):
        a = tp.quantize_per_tensor(tp.tensor([1.0]), 0.1, 0)
        b = tp.quantize_per_tensor(tp.tensor([2.0]), 0.1, 0)
        s = tp.quantized_add(a, b, 0.1, 0, 0.1, 0, 0.1, 0)
        self.assertEqual(s.dtype, tp.qint8)
        self.assertTrue(s.is_quantized())
        self.assertAlmostEqual(float(s.dequantize()[0]), 3.0, places=5)

    def test_quantized_max_pool2d(self):
        q = tp.quantize_per_tensor(
            tp.arange(16.0).reshape(1, 4, 4), 0.5, 0)
        p = tp.quantized_max_pool2d(q, 2)
        self.assertTrue(p.is_quantized())
        self.assertAlmostEqual(p.q_scale(), 0.5)
        ref = tp.nn.functional.max_pool2d(
            tp.arange(16.0).reshape(1, 4, 4), 2)
        self.assertTrue(float((p.dequantize() - ref).abs().max().item()) < 1e-6)


if __name__ == "__main__":
    unittest.main()
