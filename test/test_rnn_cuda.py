"""
verified) CPU implementation plus an end-to-end training smoke test that
exercises the differentiable python path (chunk/split/linear on GPU)."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp


def _tp_tensor(a, dt, device):
    return tp.tensor(np.asarray(a).tolist(), dtype=dt, device=device)


def run_native_case(kind, T, N, feat, H, num_layers, bidir, batch_first,
                    bias, dtype):
    """Returns max |cuda_out - cpu_out| over output/hy[/cy]."""
    tp_dt = {"fp64": tp.float64, "fp32": tp.float32,
             "fp16": tp.float16, "bf16": tp.bfloat16}[dtype]
    np_dt = np.float64 if dtype == "fp64" else np.float32
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


def training_smoke():
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
    missing = [type(m).__name__ + f"[{i}]" for i, p in enumerate(params)
               if p.grad is None for m in [p]]
    gn = [np.abs(np.asarray(p.grad.cpu().tolist())).sum()
          for p in params if p.grad is not None]
    n_with_grad = sum(1 for p in params if p.grad is not None)

    opt.step()
    l1 = step_loss()
    ok = all(np.isfinite(g).all() for g in gn) and sum(gn) > 0 \
        and n_with_grad == len(params) \
        and float(l1.item()) < float(l0.item())
    print(f"training smoke: loss {float(l0.item()):.4f} -> {float(l1.item()):.4f}, "
          f"grads on {n_with_grad}/{len(params)} params "
          f"(missing: {missing}), ok={ok}")
    return ok


def main():
    if not tp.cuda.is_available():
        print("CUDA not available")
        return 1
    import itertools
    failures = 0
    total = 0
    for kind in ["lstm", "gru", "rnn_tanh", "rnn_relu"]:
        for bidir, bf, layers, bias, dtype in itertools.product(
                [False, True], [False, True], [1, 2], [True],
                ["fp16", "bf16", "fp32", "fp64"]):
            total += 1
            case = (kind, 6, 3, 4, 5, layers, bidir, bf, bias, dtype)
            try:
                err = run_native_case(*case)
                tol = {"fp32": 2e-4, "fp64": 1e-9,
                       "fp16": 1e-2, "bf16": 1e-1}[dtype]
                ok = err < tol
            except Exception as e:
                print(f"ERROR {case}: {type(e).__name__}: {e}")
                failures += 1
                continue
            if not ok:
                failures += 1
                print(f"FAIL {case} err={err:.3e}")
    print(f"native cases: {total - failures}/{total} passed")
    smoke_ok = training_smoke()
    return 1 if (failures or not smoke_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
