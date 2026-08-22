import io
import unittest

import tensorplay as tp
from tensorplay.autograd import detect_anomaly, set_detect_anomaly


class NaNBackward(tp.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return tp.tensor([float("nan")], dtype=grad_output.dtype)


class ErrorBackward(tp.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clone()

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("Some error in backward")


class TestAnomalyMode(unittest.TestCase):
    def test_nan_in_backward_raises_when_enabled(self):
        x = tp.tensor([1.0], requires_grad=True)

        # Without anomaly mode the nan silently propagates.
        out = NaNBackward.apply(x).sum()
        out.backward()
        self.assertTrue(x.grad is not None)

        with detect_anomaly():
            y = NaNBackward.apply(x).sum()
            with self.assertRaisesRegex(RuntimeError, "returned nan values"):
                y.backward()

    def test_no_false_positive(self):
        x = tp.tensor([1.0, 2.0], requires_grad=True)
        with detect_anomaly():
            y = (x * x).sum()
            y.backward()
            self.assertEqual(x.grad[0].item(), 2.0)
            self.assertEqual(x.grad[1].item(), 4.0)

    def test_forward_traceback_printed_on_backward_failure(self):
        err = io.StringIO()
        with detect_anomaly():
            x = tp.tensor([1.0], requires_grad=True)
            out = ErrorBackward.apply(x).sum()
            with self.assertRaisesRegex(RuntimeError, "Some error in backward"):
                out.backward()

    def test_state_restored_after_context(self):
        self.assertFalse(tp.autograd.is_anomaly_enabled())
        with detect_anomaly():
            self.assertTrue(tp.autograd.is_anomaly_enabled())
        self.assertFalse(tp.autograd.is_anomaly_enabled())

    def test_set_detect_anomaly_functional_form(self):
        self.assertFalse(tp.autograd.is_anomaly_enabled())
        cm = set_detect_anomaly(True, check_nan=False)
        with cm:
            self.assertTrue(tp.autograd.is_anomaly_enabled())
            self.assertFalse(tp.autograd.is_anomaly_check_nan_enabled())
        self.assertFalse(tp.autograd.is_anomaly_enabled())

    def test_detect_anomaly_warns(self):
        with self.assertWarns(UserWarning):
            with detect_anomaly():
                pass


if __name__ == "__main__":
    unittest.main()
