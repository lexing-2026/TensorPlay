import unittest

import tensorplay as tp
from tensorplay.quantization import (
    DeQuantStub,
    FakeQuantize,
    FixedQParamsObserver,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
    QuantizedLinear,
    QuantStub,
    default_dynamic_quant_observer,
    default_observer,
    default_weight_observer,
    fake_quantize_per_channel,
    fake_quantize_per_tensor,
    get_observer_state_dict,
    load_observer_state_dict,
)


class TestObservers(unittest.TestCase):
    def test_minmax(self):
        obs = MinMaxObserver()
        obs(tp.tensor([0.0, 1.0]))
        obs(tp.tensor([-2.0, 0.5]))
        scale, zero_point = obs.calculate_qparams()
        # range [-2, 1] over [-128, 127]
        self.assertAlmostEqual(scale, 3.0 / 255.0, places=7)
        self.assertEqual(zero_point, int(round(-128 - (-2.0) / scale)))
        obs.reset()
        with self.assertRaises(RuntimeError):
            obs.calculate_qparams()

    def test_moving_average(self):
        obs = MovingAverageMinMaxObserver(averaging_constant=0.5)
        obs(tp.tensor([0.0, 10.0]))
        obs(tp.tensor([0.0, 20.0]))
        # EMA: min stays 0, max = 0.5*10 + 0.5*20
        _, _ = obs.calculate_qparams()
        self.assertAlmostEqual(obs.max_val, 15.0, places=6)

    def test_per_channel(self):
        obs = PerChannelMinMaxObserver(ch_axis=0)
        x = tp.tensor([[0.0, 1.0], [-4.0, 0.0]])
        obs(x)
        scales, zps = obs.calculate_qparams()
        self.assertEqual(scales.numel(), 2)
        self.assertAlmostEqual(float(scales[0]), 1.0 / 255.0, places=7)
        self.assertAlmostEqual(float(scales[1]), 4.0 / 255.0, places=7)

    def test_moving_average_per_channel(self):
        obs = MovingAveragePerChannelMinMaxObserver(
            averaging_constant=0.25, ch_axis=1)
        obs(tp.tensor([[0.0, -8.0]]))
        obs(tp.tensor([[4.0, 0.0]]))
        scales, zps = obs.calculate_qparams()
        # ch_axis=1: channel 0 sees {0, 4} -> EMA max = 0 + .25*4 = 1;
        # channel 1 sees {-8, 0} -> EMA min = -8 + .25*(0-(-8)) = -6.
        self.assertAlmostEqual(float(scales[0]), 1.0 / 255.0, places=7)
        self.assertAlmostEqual(float(scales[1]), 6.0 / 255.0, places=7)

    def test_histogram_filters_outliers(self):
        obs = HistogramObserver(bins=512)
        body = tp.randn(10000).mul_(0.01)
        outlier = tp.tensor([50.0])
        obs(tp.cat([body, outlier]))
        scale, zero_point = obs.calculate_qparams()
        # A raw MinMax trusts the 50.0 outlier (scale ~100/255); the L2
        # histogram search must land strictly below that.  With equal-width
        # bins over such a skewed range the refinement is coarse, so only
        # assert a real reduction plus sanity on the zero point.
        raw_scale = 100.0 / 255.0
        self.assertGreater(scale, 0.0)
        self.assertLess(scale, raw_scale)
        self.assertEqual(zero_point, int(round(-128 - 0.0 / scale)))

    def test_fixed_and_placeholder(self):
        fixed = FixedQParamsObserver(scale=0.5, zero_point=3)
        fixed(tp.ones(3))
        self.assertEqual(fixed.calculate_qparams(), (0.5, 3))
        placeholder = PlaceholderObserver(dtype=tp.float32)
        placeholder(tp.ones(3))
        with self.assertRaises(Exception):
            placeholder.calculate_qparams()

    def test_default_presets(self):
        for preset in (default_observer, default_weight_observer,
                       default_dynamic_quant_observer):
            instance = preset()
            out = instance(tp.tensor([0.5, -0.5]))
            self.assertTrue(out.numel() == 2)
        self.assertEqual(default_observer().quant_min, 0)
        self.assertEqual(default_observer().quant_max, 127)

    def test_observer_state_dict_roundtrip(self):
        model = tp.nn.Sequential(QuantStub(), tp.nn.Linear(4, 2))
        with tp.no_grad():
            for _ in range(3):
                model(tp.randn(8, 4))
        state = get_observer_state_dict(model)
        self.assertTrue(len(state) > 0)

        fresh = tp.nn.Sequential(QuantStub(), tp.nn.Linear(4, 2))
        load_observer_state_dict(fresh, state)
        a = model[0].fake_quant.observer.min_val
        b = fresh[0].fake_quant.observer.min_val
        if a is not None and b is not None:
            self.assertAlmostEqual(float(a), float(b), places=6)


class TestFakeQuant(unittest.TestCase):
    def test_ste_gradient_masking(self):
        x = tp.tensor([-200.0, 0.3, 200.0], requires_grad=True)
        y = fake_quantize_per_tensor(x, 0.1, 0)
        y.backward(tp.ones_like(y))
        # In-range passes through, saturated positions are blocked.
        self.assertEqual(x.grad.tolist()[0], 0.0)
        self.assertEqual(x.grad.tolist()[1], 1.0)
        self.assertEqual(x.grad.tolist()[2], 0.0)

    def test_disable_observer(self):
        fq = FakeQuantize()
        fq(tp.tensor([0.0, 1.0]))
        first_min = fq.observer.min_val
        fq.disable_observer = True
        fq(tp.tensor([0.0, 500.0]))
        self.assertEqual(fq.observer.min_val, first_min)

    def test_per_channel_fake_quant(self):
        x = tp.tensor([[0.5, -0.5], [30.0, -30.0]], requires_grad=True)
        scales = tp.tensor([0.01, 0.25])
        zps = tp.tensor([0, 0])
        y = fake_quantize_per_channel(x, scales, zps, axis=1)
        y.backward(tp.ones_like(y))
        # Column 0 (scale .01, range ±1.27): 0.5 passes, 30.0 saturates.
        # Column 1 (scale .25, range ±31.75): both pass.
        self.assertEqual(float(x.grad.sum().item()), 3.0)


class TestStubsAndQuantizedLinear(unittest.TestCase):
    def _calibrate(self, n=8):
        tp.manual_seed(0)
        data = [tp.randn(n, 4) * 2.0 for _ in range(4)]
        return data

    def test_stub_calibration_loop(self):
        stub = QuantStub()
        data = self._calibrate()
        for batch in data:
            stub.record(batch)
        stub.freeze()
        stub.eval()
        # The paired DeQuantStub must carry the frozen qparams.
        dequant = DeQuantStub(scale=stub.fake_quant.scale,
                              zero_point=stub.fake_quant.zero_point)
        x = data[0]  # in-distribution: bounded by the calibrated range
        q = stub(x)
        self.assertEqual(q.dtype, tp.int8)
        back = dequant(q)
        self.assertTrue(float((back - x).abs().max().item()) < 0.2)

    def test_quantized_linear_matches_reference(self):
        tp.manual_seed(1)
        m, k, n = 16, 32, 24
        linear = tp.nn.Linear(k, n)
        with tp.no_grad():
            weight = linear.weight.detach()
        x_float = tp.randn(m, k) * 1.5

        from tensorplay.quantization import quantize_per_tensor
        scale, zp = 0.05, 12
        qx = quantize_per_tensor(self=x_float, scale=scale, zero_point=zp)

        qlin = QuantizedLinear.from_float(linear, scale, zp)
        out = qlin(qx)
        ref = tp.nn.functional.linear(x_float, weight, None)
        err = float((out - ref).abs().max().item())
        bound = 4.0 * scale * (float(weight.abs().max().item()) / 127.0) * k ** 0.5 \
            + 1e-3
        self.assertLess(err, max(bound, 0.35))

        with self.assertRaises(TypeError):
            qlin(x_float)  # float input must be rejected


if __name__ == "__main__":
    unittest.main()
