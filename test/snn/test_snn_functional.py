"""Functional SNN primitives: spike, surrogate gradients, and the LIF step."""
import math

import pytest

import tensorplay as tp
from tensorplay import snn


def test_heaviside_spike_step():
    x = tp.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    tp.testing.assert_close(snn.heaviside(x), tp.tensor([0, 0, 1, 1, 1], dtype=x.dtype))


def test_surrogate_grad_straight_through():
    x = tp.tensor([-1.0, 0.0, 2.0])
    tp.testing.assert_close(
        snn.surrogate_grad(x, surrogate=0), tp.tensor([1.0, 1.0, 1.0])
    )


def test_surrogate_grad_sigmoid_matches_analytic():
    x = tp.tensor([-2.0, 0.0, 1.5])
    alpha = 4.0
    got = snn.surrogate_grad(x, surrogate=1, alpha=alpha)
    ax = alpha * x
    s = 1.0 / (1.0 + tp.exp(-ax))
    expected = alpha * s * (1.0 - s)
    tp.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_surrogate_grad_fast_sigmoid_matches_analytic():
    x = tp.tensor([-2.0, 0.0, 1.5])
    alpha = 4.0
    got = snn.surrogate_grad(x, surrogate=2, alpha=alpha)
    expected = alpha / (1.0 + alpha * x.abs()) ** 2
    tp.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_surrogate_grad_piecewise_linear_window():
    x = tp.tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
    got = snn.surrogate_grad(x, surrogate=3, alpha=1.0)
    tp.testing.assert_close(got, tp.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))


def test_spike_backward_uses_surrogate_gradient():
    x = tp.tensor([0.3, -0.7], requires_grad=True)
    out = snn.spike(x, snn.SigmoidSurrogate(alpha=2.0))
    out.sum().backward()
    expected = snn.surrogate_grad(x.detach(), surrogate=1, alpha=2.0)
    tp.testing.assert_close(x.grad, expected, atol=1e-6, rtol=1e-5)


def test_lif_step_forward_hard_reset():
    # h = decay * v + x ; spike when h >= threshold ; hard reset to v_reset.
    x = tp.tensor([0.8, 0.2])
    v = tp.tensor([0.5, 0.0])
    spike, v_new = snn.lif_step(
        x, v, decay=0.5, threshold=1.0, v_reset=0.0
    )
    h = 0.5 * v + x  # [1.05, 0.2]
    expected_spike = (h >= 1.0).to(tp.float32)
    expected_v = h * (1.0 - expected_spike)
    tp.testing.assert_close(spike, expected_spike)
    tp.testing.assert_close(v_new, expected_v)


def test_lif_step_forward_soft_reset():
    x = tp.tensor([0.8, 0.2])
    v = tp.tensor([0.5, 0.0])
    spike, v_new = snn.lif_step(
        x, v, decay=0.5, threshold=1.0, v_reset=None
    )
    h = 0.5 * v + x
    expected_spike = (h >= 1.0).to(tp.float32)
    expected_v = h - 1.0 * expected_spike
    tp.testing.assert_close(spike, expected_spike)
    tp.testing.assert_close(v_new, expected_v)


def test_lif_step_backward_matches_surrogate_formula():
    # Verify the analytic backward for the hard-reset path with the sigmoid
    # surrogate, detach_reset=False.
    x = tp.tensor([0.4, -0.3], requires_grad=True)
    v = tp.tensor([0.2, 0.9], requires_grad=True)
    decay = tp.tensor(0.6, requires_grad=True)
    threshold = tp.tensor(1.0, requires_grad=True)
    spike, v_new = snn.lif_step(
        x, v, decay=decay, threshold=threshold, v_reset=0.0,
        surrogate=snn.SigmoidSurrogate(alpha=2.0), detach_reset=False,
    )
    # Drive gradients through both outputs.
    g_spike = tp.tensor([1.0, 2.0])
    g_v = tp.tensor([0.5, -1.0])
    (spike * g_spike + v_new * g_v).sum().backward()

    h = 0.6 * v.detach() + x.detach()
    s = (h >= 1.0).to(tp.float32)
    sg = snn.surrogate_grad(h - 1.0, surrogate=1, alpha=2.0)
    grad_h = ((1.0 - s) + (0.0 - h) * sg) * g_v + sg * g_spike
    tp.testing.assert_close(x.grad, grad_h, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(v.grad, grad_h * 0.6, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(decay.grad, (grad_h * v.detach()).sum(), atol=1e-5, rtol=1e-4)


def test_lif_step_gradients_flow_to_parametric_decay():
    neuron = snn.ParametricLIFNode(decay_init=0.5, decay_shape=(3,))
    x = tp.rand(2, 3)
    out = neuron(x)
    out.sum().backward()
    assert neuron.decay.grad is not None
    assert neuron.decay.grad.shape == (3,)


def test_lif_seq_matches_loop_forward():
    tp.manual_seed(0)
    T, B, N = 4, 2, 3
    x = tp.rand(T, B, N)
    v_init = tp.zeros(B, N)
    decay = tp.tensor(0.7)
    threshold = tp.tensor(1.0)
    surrogate = snn.SigmoidSurrogate(alpha=2.0)

    spikes, v_final = snn.lif_seq(
        x, v_init=v_init, decay=decay, threshold=threshold,
        v_reset=None, surrogate=surrogate,
    )

    # Reference: loop the single-step native op.
    v = v_init
    ref_spikes = []
    for t in range(T):
        spike, v = snn.lif_step(
            x[t], v, decay=decay, threshold=threshold,
            v_reset=None, surrogate=surrogate,
        )
        ref_spikes.append(spike)
    ref_spikes = tp.stack(ref_spikes)

    tp.testing.assert_close(spikes, ref_spikes)
    tp.testing.assert_close(v_final, v)


def test_lif_seq_matches_loop_gradients():
    tp.manual_seed(1)
    T, B, N = 4, 2, 3
    surrogate = snn.SigmoidSurrogate(alpha=2.0)

    # Fused multi-step path.
    x = tp.rand(T, B, N, requires_grad=True)
    decay = tp.tensor(0.7, requires_grad=True)
    threshold = tp.tensor(1.0, requires_grad=True)
    spikes, v_final = snn.lif_seq(
        x, v_init=None, decay=decay, threshold=threshold,
        v_reset=None, surrogate=surrogate, detach_reset=False,
    )
    (spikes.sum() + v_final.sum()).backward()
    fused_grads = (x.grad.clone(), decay.grad.clone(), threshold.grad.clone())

    # Loop reference.
    x2 = x.detach().clone().requires_grad_(True)
    decay2 = decay.detach().clone().requires_grad_(True)
    threshold2 = threshold.detach().clone().requires_grad_(True)
    v = tp.zeros(B, N)
    ref = []
    for t in range(T):
        spike, v = snn.lif_step(
            x2[t], v, decay=decay2, threshold=threshold2,
            v_reset=None, surrogate=surrogate, detach_reset=False,
        )
        ref.append(spike)
    ref = tp.stack(ref)
    (ref.sum() + v.sum()).backward()

    tp.testing.assert_close(fused_grads[0], x2.grad, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(fused_grads[1], decay2.grad, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(fused_grads[2], threshold2.grad, atol=1e-5, rtol=1e-4)


def test_lif_seq_hard_reset_forward():
    T, B = 3, 2
    x = tp.tensor([
        [0.9, 0.9],
        [0.1, 0.1],
        [0.8, 0.8],
    ])
    v_init = tp.zeros(B)
    spikes, v_final = snn.lif_seq(
        x, v_init=v_init, decay=tp.tensor(1.0), threshold=tp.tensor(1.0),
        v_reset=tp.tensor(0.0), surrogate=snn.StraightThroughSurrogate(),
    )
    # decay=1, IF dynamics with hard reset to 0.
    # t0: h = 0.9 -> no spike, v = 0.9
    # t1: h = 1.0 -> spike,    v = 0.0
    # t2: h = 0.8 -> no spike, v = 0.8
    expected = tp.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    tp.testing.assert_close(spikes, expected)
    tp.testing.assert_close(v_final, tp.tensor([0.8, 0.8]))


def test_lif_seq_explicit_v_init():
    x = tp.tensor([[[1.0]], [[0.5]]])  # [T=2, B=1, N=1]
    v_init = tp.tensor([[0.4]])        # [B=1, N=1]
    spikes, v_final = snn.lif_seq(
        x, v_init=v_init, decay=tp.tensor(1.0), threshold=tp.tensor(1.0),
        v_reset=None, surrogate=snn.StraightThroughSurrogate(),
    )
    # t0: h = 1.4 -> spike, soft reset -> v = 0.4
    # t1: h = 0.9 -> no spike
    tp.testing.assert_close(spikes, tp.tensor([[[1.0]], [[0.0]]]))
    tp.testing.assert_close(v_final, tp.tensor([[0.9]]))