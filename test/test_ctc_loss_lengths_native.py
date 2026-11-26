"""`_ctc_loss` must take the same length dtypes the reference takes.

The kernel used to view the length buffers as int64 whatever they were, so an
int32 vector was read as garbage and came back as "out of range" while the
reference computes the loss; float vectors were likewise reported as out of
range instead of as non-integral.
"""
import unittest

import torch

import tensorplay as tp


def _ctc_operands():
    torch.manual_seed(2026)
    log_probs = torch.randn(50, 3, 20).log_softmax(2)
    targets = torch.randint(1, 19, (3, 8), dtype=torch.int64)
    return log_probs, targets


class TestCtcLossLengths(unittest.TestCase):
    def test_int32_lengths_match_the_reference(self):
        log_probs, targets = _ctc_operands()
        for dtype in (torch.int64, torch.int32):
            input_lengths = torch.full((3,), 50, dtype=dtype)
            target_lengths = torch.full((3,), 8, dtype=dtype)
            want = torch.ops.aten._ctc_loss(
                log_probs, targets, input_lengths, target_lengths, 0, False
            )[0]
            got = tp.ops.tp._ctc_loss(
                tp.tensor(log_probs.numpy()),
                tp.tensor(targets.numpy()),
                tp.tensor(input_lengths.numpy()),
                tp.tensor(target_lengths.numpy()),
                0,
                False,
            )[0]
            self.assertTrue(tp.allclose(got, tp.tensor(want.numpy()), rtol=1e-4, atol=1e-4))

    def test_float_lengths_are_named_as_non_integral(self):
        log_probs, targets = _ctc_operands()
        input_lengths = torch.full((3,), 50.0, dtype=torch.float32)
        target_lengths = torch.full((3,), 8, dtype=torch.int64)
        with self.assertRaises(RuntimeError) as caught:
            tp.ops.tp._ctc_loss(
                tp.tensor(log_probs.numpy()),
                tp.tensor(targets.numpy()),
                tp.tensor(input_lengths.numpy()),
                tp.tensor(target_lengths.numpy()),
                0,
                False,
            )
        self.assertIn("input_lengths must be integral", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
