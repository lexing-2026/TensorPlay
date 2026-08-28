"""Numerical verification of the RNN ops against torch (reference).

Cases: lstm / gru / rnn_tanh x {bidirectional, batch_first, num_layers,
has_biases} x {fp16, bf16, fp32, fp64}.  Weights are copied from a torch
module so both stacks compute the exact same function.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp


def _torch_module(kind):
    return {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU,
            "rnn_tanh": torch.nn.RNN}[kind]


def _params_from_torch(mod, dt):
    params = []
    n_dirs = 2 if mod.bidirectional else 1
    for l in range(mod.num_layers):
        for d in range(n_dirs):
            tag = "_reverse" if d == 1 else ""
            w_ih = getattr(mod, f"weight_ih_l{l}{tag}")
            w_hh = getattr(mod, f"weight_hh_l{l}{tag}")
            # .tolist() (not .numpy()) so reduced dtypes (bf16) work too.
            params.append(tp.tensor(w_ih.detach().tolist(), dtype=dt))
            params.append(tp.tensor(w_hh.detach().tolist(), dtype=dt))
            if mod.bias:
                b_ih = getattr(mod, f"bias_ih_l{l}{tag}")
                b_hh = getattr(mod, f"bias_hh_l{l}{tag}")
                params.append(tp.tensor(b_ih.detach().tolist(), dtype=dt))
                params.append(tp.tensor(b_hh.detach().tolist(), dtype=dt))
    return params


def run_case(kind, T, N, feat, H, num_layers, bidir, batch_first, bias, dtype):
    torch_dt = {"fp64": torch.float64, "fp32": torch.float32,
                "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    tp_dt = {"fp64": tp.float64, "fp32": tp.float32,
             "fp16": tp.float16, "bf16": tp.bfloat16}[dtype]
    rng = np.random.RandomState(0)

    # Array layout follows batch_first exactly as user code would supply it.
    # Non-bf: (T, N, feat); bf: (N, T, feat).  Both stacks get the same array
    # and the same flag, so layouts cancel in the comparison.
    if batch_first:
        x_np = rng.randn(N, T, feat)
        batch = N
    else:
        x_np = rng.randn(T, N, feat)
        batch = N

    tmod = _torch_module(kind)(feat, H, num_layers=num_layers,
                               bidirectional=bidir, batch_first=batch_first,
                               bias=bias).to(torch_dt)
    dirs = 2 if bidir else 1
    h0_np = rng.randn(num_layers * dirs, batch, H)

    hx_t = torch.tensor(h0_np).to(torch_dt)
    with torch.no_grad():
        if kind == "lstm":
            c0_np = rng.randn(num_layers * dirs, batch, H)
            cx_t = torch.tensor(c0_np).to(torch_dt)
            out_t, (hy_t, cy_t) = tmod(torch.tensor(x_np).to(torch_dt), (hx_t, cx_t))
        else:
            c0_np = None
            out_t, hy_t = tmod(torch.tensor(x_np).to(torch_dt), hx_t)

    params = _params_from_torch(tmod, tp_dt)
    x_tp = tp.tensor(x_np.tolist(), dtype=tp_dt)
    hx_tp = [tp.tensor(h0_np.tolist(), dtype=tp_dt)]
    if kind == "lstm":
        hx_tp.append(tp.tensor(c0_np.tolist(), dtype=tp_dt))

    fn = getattr(tp, kind)
    args = (x_tp, hx_tp, params, bias, num_layers, 0.0, False, bidir, batch_first)
    if kind == "lstm":
        out_p, hy_p, cy_p = fn(*args)
    else:
        out_p, hy_p = fn(*args)

    tol = {"fp32": 2e-4, "fp64": 1e-9, "fp16": 1e-2, "bf16": 1e-1}[dtype]
    # .to(float64).numpy(): reduced-dtype torch tensors have no numpy view.
    out_err = np.abs(np.asarray(out_p.tolist(), dtype=np.float64) -
                     out_t.detach().to(torch.float64).numpy()).max()
    hy_err = np.abs(np.asarray(hy_p.tolist(), dtype=np.float64) -
                    hy_t.detach().to(torch.float64).numpy()).max()
    errs = [out_err, hy_err]
    if kind == "lstm":
        cy_err = np.abs(np.asarray(cy_p.tolist(), dtype=np.float64) -
                        cy_t.detach().to(torch.float64).numpy()).max()
        errs.append(cy_err)
    return max(errs), tol


def main():
    failures = 0
    total = 0
    for kind in ["lstm", "gru", "rnn_tanh"]:
        for bidir, bf, layers, bias, dtype in itertools.product(
                [False, True], [False, True], [1, 2], [True],
                ["fp16", "bf16", "fp32", "fp64"]):
            total += 1
            case = (kind, 6, 3, 4, 5, layers, bidir, bf, bias, dtype)
            try:
                err, tol = run_case(*case)
                ok = err < tol
            except Exception as e:
                print(f"ERROR {case}: {type(e).__name__}: {e}")
                failures += 1
                continue
            if not ok:
                failures += 1
            print(f"{'OK  ' if ok else 'FAIL'} {case} err={err:.3e}")
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
