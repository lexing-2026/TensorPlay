"""RNN-family CUDA numerics: the device path against the (independently
verified) CPU implementation, plus an end-to-end training smoke test that
exercises the differentiable python path (chunk/split/linear on GPU)."""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp

pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="requires CUDA build of tensorplay"
)


def _tp_tensor(a, dt, device):
    return tp.tensor(np.asarray(a).tolist(), dtype=dt, device=device)


def run_native_case(kind, T, N, feat, H, num_layers, bidir, batch_first,
                    bias, dtype):
    """Returns max |cuda_out - cpu_out| over output/hy[/cy]."""
    tp_dt = {"fp64": tp.float64, "fp32": tp.float32,
             "fp16": tp.float16, "bf16": tp.bfloat16}[dtype]
    rng = np.random.RandomState(7)
    if batch_first:
        x_np = rng.randn(N, T, feat)
    else:
        x_np = rng.randn(T, N, feat)
    dirs = 2 if bidir else 1
    h0_np = rng.randn(num_layers * dirs, N, H)
    c0_np = rng.randn(num_layers * dirs, N, H) if kind == "lstm" else None

    G = 4 * H if kind == "lstm" else (3 * H if kind == "gru" else H)
    # Params generated once in numpy so both devices see identical weights;
    # layer l > 0 consumes dirs * H input features.
    params_np = []
    for l in range(num_layers):
        in_feat = feat if l == 0 else dirs * H
        for _ in range(dirs):
            params_np.append(rng.randn(G, in_feat) * 0.2)
            params_np.append(rng.randn(G, H) * 0.2)
            if bias:
                params_np.append(rng.randn(G) * 0.2)
                params_np.append(rng.randn(G) * 0.2)

    def build(dev):
        x = _tp_tensor(x_np, tp_dt, dev)
        hx = [_tp_tensor(h0_np, tp_dt, dev)]
        params = [_tp_tensor(p, tp_dt, dev) for p in params_np]
        if kind == "lstm":
            hx.append(_tp_tensor(c0_np, tp_dt, dev))
        return x, hx, params

    fn = getattr(tp, kind)
    outs = []
    for dev in ("cpu", "cuda"):
        x, hx, params = build(dev)
        args = (x, hx, params, bias, num_layers, 0.0, False, bidir, batch_first)
        r = fn(*args)
        moved = [t.cpu() for t in r]
        outs.append([np.asarray(t.tolist(), dtype=np.float64) for t in moved])

    errs = [np.abs(a - b).max() for a, b in zip(*outs)]
    return max(errs)


_CASES = list(itertools.product(
    ["lstm", "gru", "rnn_tanh", "rnn_relu"],
    [False, True],   # bidir
    [False, True],   # batch_first
    [1, 2],          # num_layers
    [True],          # bias
    ["fp16", "bf16", "fp32", "fp64"],
))


@pytest.mark.parametrize("kind,bidir,batch_first,num_layers,bias,dtype", _CASES)
def test_rnn_native_device_agreement(kind, bidir, batch_first, num_layers,
                                     bias, dtype):
    err = run_native_case(kind, 6, 3, 4, 5, num_layers, bidir, batch_first,
                          bias, dtype)
    tol = {"fp32": 2e-4, "fp64": 1e-9, "fp16": 1e-2, "bf16": 1e-1}[dtype]
    assert err < tol, f"max abs err {err:.3e} >= tol {tol:.3e}"


def test_training_smoke():
    """Backward through nn.LSTM on cuda: grads must be finite and non-zero;
    one SGD step must reduce loss on a fixed batch."""
    T, N, feat, H = 8, 4, 3, 6
    rng = np.random.RandomState(1)
    x_np = rng.randn(T, N, feat)
    y_np = rng.randn(T, N, H)

    rnn = tp.nn.LSTM(feat, H, num_layers=2, bidirectional=False, device="cuda")
    lin = tp.nn.Linear(H, H, device="cuda")
    params = list(rnn.parameters()) + list(lin.parameters())
    opt = tp.optim.SGD(params, lr=0.01)

    def step_loss():
        opt.zero_grad()
        out, _ = rnn(tp.tensor(x_np.tolist(), dtype=tp.float32, device="cuda"))
        pred = lin(out)
        target = tp.tensor(y_np.tolist(), dtype=tp.float32, device="cuda")
        # NOTE: .mean() is currently non-differentiable (its derivatives.yaml
        # entry fails to load); sum-of-squares scaled by a constant gives the
        # same training signal without hitting that gap.
        diff = pred - target
        k = 1.0 / float(np.asarray(pred.cpu().tolist()).size)
        loss = (diff * diff).sum() * k
        loss.backward()
        return loss

    l0 = step_loss()
    assert all(p.grad is not None for p in params), "some parameters got no gradient"
    grads = [np.asarray(p.grad.cpu().tolist()) for p in params]
    assert all(np.isfinite(g).all() for g in grads), "non-finite gradient"
    assert sum(np.abs(g).sum() for g in grads) > 0, "all-zero gradients"

    opt.step()
    l1 = step_loss()
    assert float(l1.item()) < float(l0.item()), (
        f"loss did not decrease: {float(l0.item())} -> {float(l1.item())}")
