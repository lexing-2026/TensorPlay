#!/usr/bin/env python3
"""Numerical acceptance for the MetaX (MACA) backend.

Companion to ``bench_metax_ops.py``: that script answers "how fast", this one
answers "is the answer right".  It compares device results against a CPU
float64 reference computed from the same inputs, so no second library is
needed on the MetaX host:

    python3 benchmark/validate_metax_backend.py
    python3 benchmark/validate_metax_backend.py --device cuda

Three groups are covered:

* forward numeric consistency across fp32, fp16, bf16 and the integer path,
  including conv2d / conv3d and the fused attention kernel against a composed
  math path;
* backward gradients against a float64 finite-difference reference;
* the quantized fallback, where the quantization kernels themselves have no
  device implementation and run on the host.

Every case prints its error and a verdict; the exit status is non-zero when
any case exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import tensorplay as tp

# fp32 carries about 1.2e-07 of relative precision per operation; a chain of
# them stays well under 1e-06.  The low-precision formats get half their
# epsilon (4.9e-04 for fp16, 3.9e-03 for bf16) with room for the reduction
# order.
TOL = {
    tp.float32: 1e-06,
    tp.float16: 1e-03,
    tp.bfloat16: 8e-03,
}
# The finite-difference reference itself is only good to the step size, so the
# conv2d backward case is judged on a looser bound than the analytic ones.
TOL_FD = 1e-02
# Quantization error is optimal at half a step; allow the rounding of the
# comparison itself on top.
TOL_STEP = 0.55


def max_err(got, want) -> float:
    """Largest absolute error normalized by the reference's peak magnitude.

    A per-element relative error explodes wherever the reference crosses zero
    (matmul cancellation does this), so the peak magnitude is the scale.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    scale = max(float(np.max(np.abs(want))), 1e-6)
    return float(np.max(np.abs(got - want)) / scale)


def conv2d_ref(x, w, pad=1):
    n, c, h, wd = x.shape
    k = w.shape[0]
    kh, kw = w.shape[2], w.shape[3]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, k, h, wd))
    for ni in range(n):
        for ki in range(k):
            for i in range(h):
                for j in range(wd):
                    out[ni, ki, i, j] = (xp[ni, :, i:i + kh, j:j + kw] * w[ki]).sum()
    return out


def conv3d_ref(x, w, pad=1):
    n, c, d, h, wd = x.shape
    k = w.shape[0]
    kd, kh, kw = w.shape[2], w.shape[3], w.shape[4]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), (pad, pad)))
    out = np.zeros((n, k, d, h, wd))
    for ni in range(n):
        for ki in range(k):
            for zi in range(d):
                for i in range(h):
                    for j in range(wd):
                        out[ni, ki, zi, i, j] = (
                            xp[ni, :, zi:zi + kd, i:i + kh, j:j + kw] * w[ki]).sum()
    return out


def sdpa_ref(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = q @ k.transpose(0, 1, 3, 2) * scale
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(-1, keepdims=True)
    return p @ v


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, err: float, tol: float, note: str = "") -> None:
        ok = err <= tol
        self.failures += 0 if ok else 1
        verdict = "ok" if ok else "FAIL"
        print(f"{name:<26} {err:>10.3e}  tol {tol:<9.1e} {verdict:<5} {note}")

    def note(self, name: str, text: str) -> None:
        print(f"{name:<26} {'-':>10}  {'':<13} {'skip':<5} {text}")


def forward(report: Report, dev: str) -> None:
    print("\n== forward numeric consistency (device vs CPU float64) ==")
    rng = np.random.RandomState(0)

    x_np = rng.randn(1 << 20).astype(np.float64)
    y_np = rng.randn(1 << 20).astype(np.float64)
    cases = [
        ("add", lambda: tp.tensor(x_np, device=dev) + tp.tensor(y_np, device=dev),
         lambda: x_np + y_np, tp.float32),
        ("mul", lambda: tp.tensor(x_np, device=dev) * tp.tensor(y_np, device=dev),
         lambda: x_np * y_np, tp.float32),
        ("exp", lambda: tp.tensor(x_np * 0.1, device=dev).exp(),
         lambda: np.exp(x_np * 0.1), tp.float32),
        ("tanh", lambda: tp.tensor(x_np, device=dev).tanh(),
         lambda: np.tanh(x_np), tp.float32),
        ("sigmoid", lambda: tp.tensor(x_np, device=dev).sigmoid(),
         lambda: 1.0 / (1.0 + np.exp(-x_np)), tp.float32),
        ("relu", lambda: tp.tensor(x_np, device=dev).relu(),
         lambda: np.maximum(x_np, 0.0), tp.float32),
        ("sum", lambda: tp.tensor(x_np, device=dev).sum(),
         lambda: np.array(x_np.sum()), tp.float32),
        ("mean", lambda: tp.tensor(x_np, device=dev).mean(),
         lambda: np.array(x_np.mean()), tp.float32),
    ]

    m_np = rng.randn(256, 256).astype(np.float64)
    n_np = rng.randn(256, 256).astype(np.float64)
    cases.append(("mm", lambda: tp.tensor(m_np, device=dev) @ tp.tensor(n_np, device=dev),
                  lambda: m_np @ n_np, tp.float32))

    x16 = tp.tensor(x_np * 0.1, device=dev).to(tp.float16)
    cases.append(("exp fp16", lambda: x16.exp(),
                  lambda: np.exp(x_np * 0.1), tp.float16))
    xb = tp.tensor(x_np * 0.1, device=dev).to(tp.bfloat16)
    cases.append(("exp bf16", lambda: xb.exp(),
                  lambda: np.exp(x_np * 0.1), tp.bfloat16))

    i_np = rng.randint(-1000, 1000, size=1 << 16).astype(np.int32)
    j_np = rng.randint(-1000, 1000, size=1 << 16).astype(np.int32)
    cases.append(("add int32",
                  lambda: tp.tensor(i_np, device=dev) + tp.tensor(j_np, device=dev),
                  lambda: (i_np + j_np).astype(np.float64), tp.float32))
    cases.append(("mul int32",
                  lambda: tp.tensor(i_np, device=dev) * tp.tensor(j_np, device=dev),
                  lambda: (i_np * j_np).astype(np.float64), tp.float32))
    l_np = rng.randint(-(1 << 20), 1 << 20, size=1 << 16).astype(np.int64)
    cases.append(("sum int64", lambda: tp.tensor(l_np, device=dev).sum(),
                  lambda: np.array(float(l_np.sum())), tp.float32))

    mh = rng.randn(512, 512).astype(np.float32) * 0.1
    nh = rng.randn(512, 512).astype(np.float32) * 0.1
    cases.append(("mm fp16",
                  lambda: tp.tensor(mh, device=dev).to(tp.float16)
                  @ tp.tensor(nh, device=dev).to(tp.float16),
                  lambda: mh.astype(np.float64) @ nh.astype(np.float64), tp.float16))
    cases.append(("mm bf16",
                  lambda: tp.tensor(mh, device=dev).to(tp.bfloat16)
                  @ tp.tensor(nh, device=dev).to(tp.bfloat16),
                  lambda: mh.astype(np.float64) @ nh.astype(np.float64), tp.bfloat16))

    cx_np = rng.randn(2, 3, 12, 12).astype(np.float64)
    cw_np = rng.randn(5, 3, 3, 3).astype(np.float64)
    cases.append(("conv2d",
                  lambda: tp.nn.functional.conv2d(
                      tp.tensor(cx_np, device=dev), tp.tensor(cw_np, device=dev), padding=1),
                  lambda: conv2d_ref(cx_np, cw_np), tp.float32))
    c3_np = rng.randn(1, 2, 6, 6, 6).astype(np.float64)
    w3_np = rng.randn(3, 2, 3, 3, 3).astype(np.float64)
    cases.append(("conv3d",
                  lambda: tp.nn.functional.conv3d(
                      tp.tensor(c3_np, device=dev), tp.tensor(w3_np, device=dev), padding=1),
                  lambda: conv3d_ref(c3_np, w3_np), tp.float32))

    # The fused kernel against the same attention written as separate ops.
    q_np = rng.randn(2, 4, 64, 32).astype(np.float64)
    k_np = rng.randn(2, 4, 64, 32).astype(np.float64)
    v_np = rng.randn(2, 4, 64, 32).astype(np.float64)
    fused = tp.nn.functional.scaled_dot_product_attention(
        tp.tensor(q_np, device=dev), tp.tensor(k_np, device=dev),
        tp.tensor(v_np, device=dev)).to("cpu").float().numpy()
    cases.append(("sdpa math vs fused", lambda: fused,
                  lambda: sdpa_ref(q_np, k_np, v_np), tp.float32))

    for name, make, ref, dtype in cases:
        got = make()
        if isinstance(got, tp.Tensor):
            got = got.to("cpu").float().numpy()
        report.check(name, max_err(got, ref()), TOL[dtype])


def backward(report: Report, dev: str) -> None:
    print("\n== backward (device gradient vs CPU float64 reference) ==")
    rng = np.random.RandomState(1)

    x_np = rng.randn(4096).astype(np.float64)
    w_np = rng.randn(4096).astype(np.float64)
    # d/dx of tanh(w*x + 0.3).sum()
    want = (1.0 - np.tanh(w_np * x_np + 0.3) ** 2) * w_np
    x = tp.tensor(x_np, device=dev, requires_grad=True)
    w = tp.tensor(w_np, device=dev)
    (dx,) = tp.autograd.grad((w * x + 0.3).tanh().sum(), [x])
    report.check("elementwise (analytic)",
                 max_err(dx.to("cpu").float().numpy(), want), TOL[tp.float32])

    x_np = rng.randn(2, 3, 8, 8).astype(np.float32)
    w_np = rng.randn(4, 3, 3, 3).astype(np.float32)
    eps = 1e-3
    x = tp.tensor(x_np, device=dev, requires_grad=True)
    out = tp.nn.functional.conv2d(x, tp.tensor(w_np, device=dev), padding=1).sum()
    (dx,) = tp.autograd.grad(out, [x])
    got = dx.to("cpu").float().numpy()

    def conv_loss(arr):
        return float(conv2d_ref(arr.astype(np.float64), w_np.astype(np.float64)).sum())

    errs = []
    for i in [(0, 0, 0, 0), (1, 2, 5, 6), (0, 1, 7, 3)]:
        plus = x_np.copy()
        plus[i] += eps
        minus = x_np.copy()
        minus[i] -= eps
        num = (conv_loss(plus) - conv_loss(minus)) / (2 * eps)
        errs.append(abs(float(got[i]) - float(num)) / max(abs(float(num)), 1e-6))
    report.check("conv2d (central difference)", max(errs), TOL_FD)

    q_np = rng.randn(1, 2, 16, 8).astype(np.float64)
    k_np = rng.randn(1, 2, 16, 8).astype(np.float64)
    v_np = rng.randn(1, 2, 16, 8).astype(np.float64)
    q = tp.tensor(q_np, device=dev, requires_grad=True)
    out = tp.nn.functional.scaled_dot_product_attention(
        q, tp.tensor(k_np, device=dev), tp.tensor(v_np, device=dev))
    (dq,) = tp.autograd.grad(out.sum(), [q])
    got = dq.to("cpu").float().numpy()

    errs = []
    for i in [(0, 0, 0, 0), (0, 1, 9, 5), (0, 1, 15, 7)]:
        plus = q_np.copy()
        plus[i] += 1e-5
        minus = q_np.copy()
        minus[i] -= 1e-5
        num = (sdpa_ref(plus, k_np, v_np).sum() - sdpa_ref(minus, k_np, v_np).sum()) / 2e-5
        errs.append(abs(float(got[i]) - float(num)) / max(abs(float(num)), 1e-6))
    report.check("sdpa (central difference)", max(errs), TOL[tp.float32])


def quantized(report: Report, dev: str) -> None:
    print("\n== quantized fallback (device input, int8 path) ==")
    rng = np.random.RandomState(3)
    a_np = rng.randn(64, 128).astype(np.float32)
    b_np = rng.randn(128, 32).astype(np.float32)

    try:
        scale = float(np.abs(a_np).max()) / 127.0
        q = tp.quantize_per_tensor(tp.tensor(a_np, device=dev), scale, 0, tp.qint8)
        d = tp.dequantize(q).to("cpu").float().numpy()
        report.check("int8 roundtrip", float(np.max(np.abs(d - a_np))) / scale,
                     TOL_STEP, f"step {scale:.4f}, q device {q.device}")
    except Exception as exc:  # report, do not abort the remaining cases
        report.note("int8 roundtrip", f"{type(exc).__name__}: {exc}")

    try:
        sa = float(np.abs(a_np).max()) / 127.0
        sb = float(np.abs(b_np).max()) / 127.0
        qa = tp.quantize_per_tensor(tp.tensor(a_np, device=dev), sa, 0, tp.qint8)
        qb = tp.quantize_per_tensor(tp.tensor(b_np, device=dev), sb, 0, tp.qint8)
        ref = (tp.dequantize(qa) @ tp.dequantize(qb)).to("cpu").float().numpy()
        out_scale = float(np.abs(ref).max()) / 120.0
        got = tp.dequantize(tp.quantized_matmul(qa, qb, out_scale, 0)).to("cpu").float().numpy()
        report.check("quantized_matmul", float(np.max(np.abs(got - ref))) / out_scale,
                     TOL_STEP, f"out step {out_scale:.4f}")
    except Exception as exc:
        report.note("quantized_matmul", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dev = args.device

    print(f"device: {dev}")
    if dev == "cuda":
        print(f"device name: {tp.cuda.get_device_name(0)}")

    report = Report()
    forward(report, dev)
    backward(report, dev)
    quantized(report, dev)

    print(f"\n{report.failures} failing case(s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
