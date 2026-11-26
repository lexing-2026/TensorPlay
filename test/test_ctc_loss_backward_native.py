"""`_ctc_loss_backward` must take the one-entry gradient a reduced loss gives.

The kernel read the gradient entry of sequence b from a buffer that holds a
single scalar when the loss was reduced, so the read ran past the storage: the
values were garbage at three sequences and the process died with an access
violation (0xC0000005) at two hundred thousand. A one-entry gradient is now
shared by every sequence, and any other length is rejected instead of read.
"""
import unittest

import torch

import tensorplay as tp


def _operands():
    torch.manual_seed(0)
    log_probs = torch.randn(50, 3, 20).log_softmax(2)
    targets = torch.randint(1, 19, (3, 8), dtype=torch.int64)
    input_lengths = torch.full((3,), 50, dtype=torch.int64)
    target_lengths = torch.full((3,), 8, dtype=torch.int64)
    return log_probs, targets, input_lengths, target_lengths


def _forward(operands):
    log_probs, targets, input_lengths, target_lengths = operands
    return tp.ops.tp._ctc_loss(
        tp.tensor(log_probs.numpy()),
        tp.tensor(targets.numpy()),
        tp.tensor(input_lengths.numpy()),
        tp.tensor(target_lengths.numpy()),
        0,
        False,
    )


class TestCtcLossBackwardGradient(unittest.TestCase):
    def test_one_entry_per_sequence_matches_the_reference(self):
        operands = _operands()
        log_probs, targets, input_lengths, target_lengths = operands
        nll, log_alpha = torch.ops.aten._ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0, False
        )
        want = torch.ops.aten._ctc_loss_backward(
            torch.ones(3), log_probs, targets, input_lengths, target_lengths,
            nll, log_alpha, 0, False,
        )
        nll_tp, log_alpha_tp = _forward(operands)
        got = tp.ops.tp._ctc_loss_backward(
            tp.ones((3,)),
            tp.tensor(log_probs.numpy()),
            tp.tensor(targets.numpy()),
            tp.tensor(input_lengths.numpy()),
            tp.tensor(target_lengths.numpy()),
            nll_tp,
            log_alpha_tp,
            0,
            False,
        )
        self.assertTrue(tp.allclose(got, tp.tensor(want.numpy()), rtol=1e-4, atol=1e-4))

    def test_a_single_scalar_gradient_is_shared(self):
        operands = _operands()
        log_probs, targets, input_lengths, target_lengths = operands
        nll_tp, log_alpha_tp = _forward(operands)
        common = (
            tp.tensor(log_probs.numpy()),
            tp.tensor(targets.numpy()),
            tp.tensor(input_lengths.numpy()),
            tp.tensor(target_lengths.numpy()),
            nll_tp,
            log_alpha_tp,
            0,
            False,
        )
        shared = tp.ops.tp._ctc_loss_backward(tp.tensor(1.0), *common)
        per_item = tp.ops.tp._ctc_loss_backward(tp.ones((3,)), *common)
        self.assertTrue(tp.allclose(shared, per_item, rtol=1e-5, atol=1e-5))

    def test_a_gradient_of_the_wrong_length_is_rejected(self):
        operands = _operands()
        nll_tp, log_alpha_tp = _forward(operands)
        with self.assertRaises(RuntimeError) as caught:
            tp.ops.tp._ctc_loss_backward(
                tp.ones((2,)),
                tp.tensor(operands[0].numpy()),
                tp.tensor(operands[1].numpy()),
                tp.tensor(operands[2].numpy()),
                tp.tensor(operands[3].numpy()),
                nll_tp,
                log_alpha_tp,
                0,
                False,
            )
        self.assertIn("grad_output", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
