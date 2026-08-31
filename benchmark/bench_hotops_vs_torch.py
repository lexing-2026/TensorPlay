"""Head-to-head eager CPU benchmarks against the reference framework.

Covers the hot operators seen in training/inference workloads:
elementwise unary/binary, reductions, matmul family, normalization,
softmax family, pooling/conv, embedding, gather/scatter-ish indexing.

Each cell reports the best-of-`reps` wall-clock milliseconds.  Both
runtimes are pinned to the same thread count and get the same warmup.

Usage:
  python3 benchmark/bench_hotops_vs_torch.py [--threads N] [--dtype f32]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as tF

import tensorplay as tp
import tensorplay.nn.functional as pF


def timeit(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0


def _np(dtype, *shape, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.standard_normal(shape) * 0.5).astype(dtype)


def build_cases(dtype):
    cases = []  # (name, torch_fn, tp_fn, torch_args, tp_args)

    def add(name, tf, pf, *arrays):
        ta = tuple(torch.from_numpy(a.copy()) for a in arrays)
        pa = tuple(tp.tensor(a.copy()) for a in arrays)
        cases.append((name, tf, pf, ta, pa))

    for n in (1 << 12, 1 << 20, 1 << 23):
        x = _np(dtype, n)
        y = _np(dtype, n, seed=1)
        add(f"add[{n}]", lambda a, b: a + b, lambda a, b: a + b, x, y)
        add(f"mul[{n}]", lambda a, b: a * b, lambda a, b: a * b, x, y)
        add(f"exp[{n}]", lambda a: a.exp(), lambda a: a.exp(), x)
        add(f"gelu[{n}]", lambda a: tF.gelu(a), lambda a: pF.gelu(a), x)
        add(f"gelu_tanh[{n}]", lambda a: tF.gelu(a, approximate="tanh"),
            lambda a: pF.gelu(a, approximate="tanh"), x)
    x = _np(dtype, 1 << 12)
    add("relu[4k]", lambda a: torch.relu(a), lambda a: tp.relu(a), x)
    add("sigmoid[4k]", lambda a: a.sigmoid(), lambda a: a.sigmoid(), x)
    add("tanh[4k]", lambda a: a.tanh(), lambda a: a.tanh(), x)
    add("sqrt[4k]", lambda a: a.sqrt(), lambda a: a.sqrt(), x)
    add("abs[4k]", lambda a: a.abs(), lambda a: a.abs(), x)
    x2 = np.abs(_np(dtype, 1 << 12)) + 0.1
    add("log[4k]", lambda a: a.log(), lambda a: a.log(), x2)

    # reductions
    for shape in ((1 << 20,), (1024, 1024)):
        x = _np(dtype, *shape)
        add(f"sum_all{shape}", lambda a: a.sum(), lambda a: a.sum(), x)
        add(f"mean_all{shape}", lambda a: a.mean(), lambda a: a.mean(), x)
        add(f"max_all{shape}", lambda a: a.max(), lambda a: a.max(), x)
        add(f"norm2{shape}", lambda a: a.norm(), lambda a: a.norm(), x)
    x = _np(dtype, 2048, 2048)
    add("sum_dim1[2048x2048]", lambda a: a.sum(dim=1), lambda a: a.sum(dim=1), x)
    add("sum_dim0[2048x2048]", lambda a: a.sum(dim=0), lambda a: a.sum(dim=0), x)
    add("argmax_dim1[2048x2048]", lambda a: a.argmax(dim=1), lambda a: a.argmax(dim=1), x)
    add("max_dim1[2048x2048]", lambda a: a.max(dim=1).values, lambda a: a.max(dim=1).values, x)
    add("cumsum_dim1[2048x2048]", lambda a: a.cumsum(dim=1), lambda a: a.cumsum(dim=1), x)

    # matmul family
    for (m, k, n_) in ((256, 256, 256), (1024, 1024, 1024), (2048, 512, 2048), (64, 4096, 4096)):
        a = _np(dtype, m, k)
        b = _np(dtype, k, n_)
        add(f"mm[{m}x{k}x{n_}]", lambda x_, y_: x_ @ y_, lambda x_, y_: x_ @ y_, a, b)
    a = _np(dtype, 1024, 1024)
    b = _np(dtype, 1024, 1024)
    c = _np(dtype, 1024, 1024)
    add("addmm[1024]", lambda x_, y_, z_: x_ + y_ @ z_, lambda x_, y_, z_: x_ + y_ @ z_, c, a, b)
    ab = _np(dtype, 32, 128, 128)
    bb = _np(dtype, 32, 128, 128)
    add("bmm[32x128x128]", lambda x_, y_: torch.bmm(x_, y_), lambda x_, y_: tp.bmm(x_, y_), ab, bb)

    # normalization / softmax family
    x = _np(dtype, 1024, 1024)
    g = np.ones(1024, dtype=dtype)
    bias = np.zeros(1024, dtype=dtype)
    g_group = np.ones(64, dtype=dtype)
    bias_group = np.zeros(64, dtype=dtype)
    add("softmax[-1][1024x1024]", lambda a: a.softmax(dim=-1), lambda a: a.softmax(dim=-1), x)
    add("log_softmax[-1][1024x1024]", lambda a: a.log_softmax(dim=-1), lambda a: a.log_softmax(dim=-1), x)
    add("softmax[-1][4096x512]", lambda a: a.softmax(dim=-1), lambda a: a.softmax(dim=-1), _np(dtype, 4096, 512))
    add("softmax[-1][256x256x256]", lambda a: a.softmax(dim=-1), lambda a: a.softmax(dim=-1), _np(dtype, 256, 256, 256))
    add("layer_norm[1024x1024]",
        lambda a: tF.layer_norm(a, (1024,), torch.from_numpy(g), torch.from_numpy(bias)),
        lambda a: pF.layer_norm(a, (1024,), tp.tensor(g), tp.tensor(bias)), x)
    add("group_norm[32x64x56x56]",
        lambda a: tF.group_norm(a, 8, torch.from_numpy(g_group), torch.from_numpy(bias_group)),
        lambda a: pF.group_norm(a, 8, tp.tensor(g_group), tp.tensor(bias_group)),
        _np(dtype, 32, 64, 56, 56))

    # conv (inference)
    xn = _np(dtype, 32, 64, 56, 56)
    w3 = _np(dtype, 128, 64, 3, 3) * 0.05
    b3 = _np(dtype, 128) * 0.0
    add("conv2d_3x3[32x64x56x56]",
        lambda a: tF.conv2d(a, torch.from_numpy(w3), torch.from_numpy(b3), 1, 1),
        lambda a: pF.conv2d(a, tp.tensor(w3), tp.tensor(b3), 1, 1), xn)
    w1 = _np(dtype, 128, 64, 1, 1) * 0.05
    add("conv2d_1x1[32x64x56x56]",
        lambda a: tF.conv2d(a, torch.from_numpy(w1), None, 1, 0),
        lambda a: pF.conv2d(a, tp.tensor(w1), None, 1, 0), xn)
    xn2 = _np(dtype, 8, 3, 224, 224)
    w0 = _np(dtype, 64, 3, 7, 7) * 0.05
    add("conv2d_7x7[8x3x224x224]",
        lambda a: tF.conv2d(a, torch.from_numpy(w0), None, 2, 3),
        lambda a: pF.conv2d(a, tp.tensor(w0), None, 2, 3), xn2)
    add("maxpool2d_2[32x64x56x56]",
        lambda a: tF.max_pool2d(a, 2), lambda a: pF.max_pool2d(a, 2), xn)

    # embedding / indexing
    ids_np = np.random.RandomState(3).randint(0, 32000, 4096).astype(np.int64)
    table = _np(dtype, 32000, 512)
    add("emb[4096x32000x512]",
        lambda a, i: tF.embedding(i, a), lambda a, i: tp.embedding(a, i), table, ids_np)
    x = _np(dtype, 2048, 2048)
    idx = np.random.RandomState(4).randint(0, 2048, 8192).astype(np.int64)
    add("index_select[8k of 2048x2048]",
        lambda a, i: a.index_select(0, i), lambda a, i: a.index_select(0, i), x, idx)
    add("cat_d0[4x2048x2048]",
        lambda a: torch.cat([a, a, a, a], 0), lambda a: tp.cat([a, a, a, a], 0), x)
    add("cat_d1[4x2048x2048]",
        lambda a: torch.cat([a, a, a, a], 1), lambda a: tp.cat([a, a, a, a], 1), x)
    add("transpose_copy[2048x2048]",
        lambda a: a.t().contiguous(), lambda a: a.t().contiguous(), x)
    add("permute_2013_copy[32x64x56x56]",
        lambda a: a.permute(0, 2, 3, 1).contiguous(), lambda a: a.permute(0, 2, 3, 1).contiguous(),
        _np(dtype, 32, 64, 56, 56))

    # misc
    x = _np(dtype, 1 << 20)
    add("where[1M]", lambda a: torch.where(a > 0, a, a * 2), lambda a: tp.where(a > 0, a, a * 2), x)
    add("clamp[1M]", lambda a: a.clamp(-0.5, 0.5), lambda a: a.clamp(-0.5, 0.5), x)
    add("pow2[1M]", lambda a: a ** 2, lambda a: a ** 2, x)
    add("exp[1M]", lambda a: a.exp(), lambda a: a.exp(), x)
    add("dropout[1M]", lambda a: tF.dropout(a, 0.1, True),
        lambda a: pF.dropout(a, 0.1, True), x)
    xk = _np(dtype, 4096, 4096)
    add("topk8[4096x4096]",
        lambda a: a.topk(8, dim=1)[0], lambda a: tp.topk(a, 8, dim=1)[0], xk)
    add("sort[4096x4096]",
        lambda a: a.sort(dim=1)[0], lambda a: a.sort(dim=1)[0], xk)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    args = ap.parse_args()
    dtype = np.float32 if args.dtype == "f32" else np.float64
    nthreads = args.threads or 8
    torch.set_num_threads(nthreads)
    tp.set_num_threads(nthreads)

    print(f"threads={nthreads} dtype={args.dtype} "
          f"torch={torch.__version__} tp={tp.__version__}")
    print(f"{'op':<34}{'tp_ms':>10}{'ref_ms':>10}{'speedup':>9}")
    behind = []
    for name, tf, pf, ta, pa in build_cases(dtype):
        try:
            t_ref = timeit(lambda: tf(*ta), args.reps)
            t_tp = timeit(lambda: pf(*pa), args.reps)
        except Exception as e:
            print(f"{name:<34}  ERROR: {type(e).__name__}: {e}")
            continue
        sp = t_ref / t_tp
        flag = "  <-- BEHIND" if sp < 0.97 else ""
        if sp < 0.97:
            behind.append((name, sp))
        print(f"{name:<34}{t_tp:>10.3f}{t_ref:>10.3f}{sp:>9.2f}{flag}")
    print("\n== SUMMARY ==")
    if behind:
        for n_, s in sorted(behind, key=lambda x: x[1]):
            print(f"  BEHIND {n_}: {s:.2f}x")
    else:
        print("  all ops >= torch (within 3%)")


if __name__ == "__main__":
    main()
