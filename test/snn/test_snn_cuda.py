"""Device coverage for the native SNN operators: CUDA forward/backward must
match the CPU implementation for the single-step and multi-step paths and for
the surrogate gradient evaluation."""
import pytest

import tensorplay as tp
from tensorplay import snn

cuda_only = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA not available"
)


@cuda_only
def test_lif_step_cuda_matches_cpu():
    tp.manual_seed(0)
    x = tp.rand(4, 8, device="cuda", requires_grad=True)
    v = tp.rand(4, 8, device="cuda", requires_grad=True)
    decay = tp.tensor(0.6, device="cuda", requires_grad=True)
    threshold = tp.tensor(1.0, device="cuda", requires_grad=True)
    g_spike = tp.rand(4, 8, device="cuda")
    g_v = tp.rand(4, 8, device="cuda")

    spike_cuda, v_new_cuda = snn.lif_step(
        x, v, decay=decay, threshold=threshold, v_reset=0.0,
        surrogate=snn.SigmoidSurrogate(alpha=2.0), detach_reset=False,
    )
    (spike_cuda * g_spike + v_new_cuda * g_v).sum().backward()

    x_cpu = x.detach().cpu().requires_grad_(True)
    v_cpu = v.detach().cpu().requires_grad_(True)
    decay_cpu = decay.detach().cpu().requires_grad_(True)
    threshold_cpu = threshold.detach().cpu().requires_grad_(True)
    spike_cpu, v_new_cpu = snn.lif_step(
        x_cpu, v_cpu, decay=decay_cpu, threshold=threshold_cpu,
        v_reset=0.0, surrogate=snn.SigmoidSurrogate(alpha=2.0),
        detach_reset=False,
    )
    (spike_cpu * g_spike.cpu() + v_new_cpu * g_v.cpu()).sum().backward()

    tp.testing.assert_close(spike_cuda.cpu(), spike_cpu, atol=1e-6, rtol=1e-5)
    tp.testing.assert_close(v_new_cuda.cpu(), v_new_cpu, atol=1e-6, rtol=1e-5)
    tp.testing.assert_close(x.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(v.grad.cpu(), v_cpu.grad, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(
        decay.grad.cpu(), decay_cpu.grad, atol=1e-5, rtol=1e-4
    )
    tp.testing.assert_close(
        threshold.grad.cpu(), threshold_cpu.grad, atol=1e-5, rtol=1e-4
    )


@cuda_only
def test_lif_seq_cuda_matches_cpu():
    tp.manual_seed(1)
    T, B, N = 5, 3, 4
    x = tp.rand(T, B, N, device="cuda", requires_grad=True)
    v_init = tp.rand(B, N, device="cuda", requires_grad=True)
    decay = tp.tensor(0.7, device="cuda", requires_grad=True)
    threshold = tp.tensor(1.0, device="cuda", requires_grad=True)
    g_spikes = tp.rand(T, B, N, device="cuda")
    g_v_final = tp.rand(B, N, device="cuda")

    spikes_cuda, v_final_cuda = snn.lif_seq(
        x, v_init=v_init, decay=decay, threshold=threshold, v_reset=0.0,
        surrogate=snn.FastSigmoidSurrogate(alpha=3.0), detach_reset=False,
    )
    (spikes_cuda * g_spikes + v_final_cuda * g_v_final).sum().backward()

    x_cpu = x.detach().cpu().requires_grad_(True)
    v_init_cpu = v_init.detach().cpu().requires_grad_(True)
    decay_cpu = decay.detach().cpu().requires_grad_(True)
    threshold_cpu = threshold.detach().cpu().requires_grad_(True)
    spikes_cpu, v_final_cpu = snn.lif_seq(
        x_cpu, v_init=v_init_cpu, decay=decay_cpu,
        threshold=threshold_cpu, v_reset=0.0,
        surrogate=snn.FastSigmoidSurrogate(alpha=3.0), detach_reset=False,
    )
    (spikes_cpu * g_spikes.cpu() + v_final_cpu * g_v_final.cpu()).sum().backward()

    tp.testing.assert_close(spikes_cuda.cpu(), spikes_cpu, atol=1e-6, rtol=1e-5)
    tp.testing.assert_close(
        v_final_cuda.cpu(), v_final_cpu, atol=1e-6, rtol=1e-5
    )
    tp.testing.assert_close(x.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-4)
    tp.testing.assert_close(
        v_init.grad.cpu(), v_init_cpu.grad, atol=1e-5, rtol=1e-4
    )
    tp.testing.assert_close(
        decay.grad.cpu(), decay_cpu.grad, atol=1e-5, rtol=1e-4
    )
    tp.testing.assert_close(
        threshold.grad.cpu(), threshold_cpu.grad, atol=1e-5, rtol=1e-4
    )


@cuda_only
def test_surrogate_grad_cuda_matches_cpu():
    x = tp.tensor([-2.0, 0.0, 1.5], device="cuda")
    for surrogate_id in (0, 1, 2, 3):
        got = snn.surrogate_grad(x, surrogate=surrogate_id, alpha=2.0)
        ref = snn.surrogate_grad(
            x.cpu(), surrogate=surrogate_id, alpha=2.0
        )
        tp.testing.assert_close(got.cpu(), ref, atol=1e-6, rtol=1e-5)


@cuda_only
def test_lif_seq_cuda_half_matches_float():
    tp.manual_seed(2)
    x = tp.rand(4, 2, 3, device="cuda", dtype=tp.float16)
    spikes, v_final = snn.lif_seq(
        x, decay=tp.tensor(0.7, device="cuda", dtype=tp.float16),
        threshold=tp.tensor(1.0, device="cuda", dtype=tp.float16),
        v_reset=0.0,
    )
    ref_spikes, ref_v = snn.lif_seq(
        x.to(tp.float32), decay=tp.tensor(0.7, device="cuda"),
        threshold=tp.tensor(1.0, device="cuda"), v_reset=0.0,
    )
    tp.testing.assert_close(
        spikes.to(tp.float32), ref_spikes, atol=1e-2, rtol=1e-2
    )
    tp.testing.assert_close(
        v_final.to(tp.float32), ref_v, atol=1e-2, rtol=1e-2
    )


@cuda_only
def test_lif_step_cuda_soft_reset_and_detach_reset():
    x = tp.rand(4, 8, device="cuda", requires_grad=True)
    v = tp.rand(4, 8, device="cuda", requires_grad=True)
    # Soft reset with detach_reset=True: gradient flows through v only.
    spike, v_new = snn.lif_step(
        x, v, decay=0.5, threshold=1.0, v_reset=None,
        surrogate=snn.StraightThroughSurrogate(), detach_reset=True,
    )
    (spike.sum() + v_new.sum()).backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 8)