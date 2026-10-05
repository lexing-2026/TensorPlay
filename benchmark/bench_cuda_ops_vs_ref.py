"""Per-kernel CUDA micro-benchmark: this tree's kernels against a second
runtime's, same shapes and dtypes on the same device.

Every case reports best-of-N milliseconds for both sides plus the ratio
(ours / other); ratios above the flag threshold are the candidates for kernel
work.  Each side runs its own kernels, so the ratio is the number to track when
tuning one.

Usage:
    python benchmark/bench_cuda_ops_vs_ref.py [--filter softmax] [--reps 20]
                                              [--threshold 1.2] [--dtype f16,f32]
"""
import argparse
import time

import torch

import tensorplay as tp
import tensorplay.functional as F

DT = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


def T(x):
    """Wrap once, outside every timed region.

    Building the wrapper costs host time that lands between the two CUDA event
    records, so a wrapper created inside the timed callable shows up as kernel
    time -- about 1% on the fastest cases here, which is the same order as the
    gaps this benchmark exists to measure.
    """
    return tp.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))


def O(x):
    return torch.from_dlpack(tp.to_dlpack(x))


def timeit(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


# (标签, 我们的调用, 参考调用)
#
# Every callable binds its tensors through default arguments: the builder
# reuses the same names for each family, so a closure that captured them by
# reference would measure whatever tensor happened to be built last.
def build_cases(dev, dtype):
    g = torch.Generator(device=dev).manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g, device=dev, dtype=dtype)

    C = []
    # ---- softmax 家族（SoftmaxKernels.cu）----
    for shape in [(8192, 512), (4096, 1024), (2048, 2048), (32, 256, 32, 32), (65536, 128),
                  (65536, 768), (16384, 2048), (4, 32, 1024, 1024)]:
        x = r(*shape)
        tx = T(x)
        C.append((f"softmax.dim-1{shape}", lambda tx=tx: F.softmax(tx, -1),
                  lambda x=x: torch.softmax(x, -1)))
        C.append((f"log_softmax.dim-1{shape}", lambda tx=tx: F.log_softmax(tx, -1),
                  lambda x=x: torch.log_softmax(x, -1)))
    x = r(4096, 1024)
    y = torch.rand(4096, 1024, generator=g, device=dev, dtype=dtype)
    tx, ty = T(x), T(y)
    C.append(("softmax.backward",
              lambda tx=tx, ty=ty: F._softmax_backward_data(ty, tx, -1, None),
              lambda x=x, y=y: torch._softmax_backward_data(y, x, -1, dtype)))
    # ---- 归约家族（Reduce*Kernels.cu / ReductionKernels.cu）----
    for shape in [(4096, 4096), (1024, 1024, 64)]:
        x = r(*shape)
        tx = T(x)
        C.append((f"sum.dim-1{shape}", lambda tx=tx: F.sum(tx, [-1]), lambda x=x: torch.sum(x, -1)))
        C.append((f"mean.dim-1{shape}", lambda tx=tx: F.mean(tx, [-1]), lambda x=x: torch.mean(x, -1)))
        C.append((f"amax.dim-1{shape}", lambda tx=tx: F.amax(tx, [-1]), lambda x=x: torch.amax(x, -1)))
        C.append((f"var_mean{shape}", lambda tx=tx: F.var_mean(tx, [-1]),
                  lambda x=x: torch.var_mean(x, -1)))
    # ---- 带索引的维度极值（打包值+下标归一路径）----
    for shape in [(4096, 4096), (1024, 1024, 64)]:
        x = r(*shape)
        tx = T(x)
        C.append((f"max.dim-1{shape}", lambda tx=tx: F.max(tx, -1), lambda x=x: torch.max(x, -1)))
        C.append((f"min.dim-1{shape}", lambda tx=tx: F.min(tx, -1), lambda x=x: torch.min(x, -1)))
        C.append((f"argmax.dim-1{shape}", lambda tx=tx: F.argmax(tx, -1),
                  lambda x=x: torch.argmax(x, -1)))
        C.append((f"argmin.dim-1{shape}", lambda tx=tx: F.argmin(tx, -1),
                  lambda x=x: torch.argmin(x, -1)))
        C.append((f"aminmax.dim-1{shape}", lambda tx=tx: F.aminmax(tx, -1),
                  lambda x=x: torch.aminmax(x, dim=-1)))
    x = r(4096, 4096)
    tx = T(x)
    C.append(("max.dim0", lambda tx=tx: F.max(tx, 0), lambda x=x: torch.max(x, 0)))
    C.append(("argmax.dim0", lambda tx=tx: F.argmax(tx, 0), lambda x=x: torch.argmax(x, 0)))
    # ---- 逐元素家族（ArithmeticKernels.cu / UnaryMathKernels.cu）----
    a, b = r(4096, 4096), r(4096, 1)
    ta, tb = T(a), T(b)
    C.append(("add.broadcast", lambda ta=ta, tb=tb: F.add(ta, tb), lambda a=a, b=b: torch.add(a, b)))
    C.append(("mul.same", lambda ta=ta: F.mul(ta, ta), lambda a=a: torch.mul(a, a)))
    C.append(("div.same", lambda ta=ta: F.div(ta, ta), lambda a=a: torch.div(a, a)))
    C.append(("addcmul", lambda ta=ta: F.addcmul(ta, ta, ta), lambda a=a: torch.addcmul(a, a, a)))
    C.append(("exp", lambda ta=ta: F.exp(ta), lambda a=a: torch.exp(a)))
    C.append(("sigmoid", lambda ta=ta: F.sigmoid(ta), lambda a=a: torch.sigmoid(a)))
    C.append(("erf", lambda ta=ta: F.erf(ta), lambda a=a: torch.erf(a)))
    # ---- 累积极值 / 排序 ----
    x = r(2048, 2048)
    tx = T(x)
    C.append(("cumsum.dim-1", lambda tx=tx: F.cumsum(tx, -1), lambda x=x: torch.cumsum(x, -1)))
    C.append(("cummax.dim-1", lambda tx=tx: F.cummax(tx, -1), lambda x=x: torch.cummax(x, -1)))
    C.append(("cummin.dim-1", lambda tx=tx: F.cummin(tx, -1), lambda x=x: torch.cummin(x, -1)))
    C.append(("cummax.dim0", lambda tx=tx: F.cummax(tx, 0), lambda x=x: torch.cummax(x, 0)))
    C.append(("cummin.dim0", lambda tx=tx: F.cummin(tx, 0), lambda x=x: torch.cummin(x, 0)))
    C.append(("sort.dim-1", lambda tx=tx: F.sort(tx, -1), lambda x=x: torch.sort(x, -1)))
    # topk returns (values, indices); compare the values only.
    C.append(("topk.k64", lambda tx=tx: F.topk(tx, 64, -1)[0],
              lambda x=x: torch.topk(x, 64, -1)[0]))
    # ---- 索引 / embedding ----
    # 524288 lookups of 64 features spread over a varying number of rows: few
    # rows mean long runs of equal indices, many rows mean nearly all distinct.
    for num_weights in (8, 512, 1024, 4096, 65536):
        idx = torch.randint(0, num_weights, (4096, 128), generator=g, device=dev)
        go = r(4096, 128)
        go3 = go.unsqueeze(-1).expand(4096, 128, 64).contiguous()
        tidx, tgo3 = T(idx), T(go3)
        C.append((f"embedding_dense_backward.nw{num_weights}",
                  lambda tidx=tidx, tgo3=tgo3, nw=num_weights:
                      F.embedding_dense_backward(tgo3, tidx, nw, -1, False),
                  lambda idx=idx, go3=go3, nw=num_weights:
                      torch.ops.aten.embedding_dense_backward(go3, idx, nw, -1, False)))
    # ---- 卷积 / 池化 ----
    x = r(32, 64, 56, 56)
    w = r(64, 64, 3, 3)
    tx, tw = T(x), T(w)
    C.append(("conv2d.3x3", lambda tx=tx, tw=tw: F.conv2d(tx, tw, None, [1, 1], [1, 1], [1, 1], 1),
              lambda x=x, w=w: torch.conv2d(x, w, None, 1, 1, 1, 1)))
    x = r(32, 64, 56, 56)
    tx = T(x)
    C.append(("adaptive_avg_pool2d.7",
              lambda tx=tx: F.adaptive_avg_pool2d(tx, [7]),
              lambda x=x: torch.nn.functional.adaptive_avg_pool2d(x, 7)))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.cuda.init()
    print(f"device={torch.cuda.get_device_name(0)} dtype={args.dtype} reps={args.reps}")
    print(f"{'case':44s} {'ours(ms)':>10s} {'ref(ms)':>10s} {'ratio':>7s}")
    slow = []
    for dt in args.dtype.split(","):
        dtype = DT[dt]
        for label, ours, ref in build_cases(args.device, dtype):
            if args.filter and args.filter not in label:
                continue
            try:
                t_ours = timeit(ours, args.reps)
                t_ref = timeit(ref, args.reps)
            except Exception as exc:                      # 未实现的算子直接跳过
                print(f"{label:44s} SKIP ({type(exc).__name__}: {str(exc)[:40]})")
                continue
            r = t_ours / t_ref
            flag = "  <-- slower" if r > args.threshold else ""
            print(f"{label:44s} {t_ours:10.4f} {t_ref:10.4f} {r:7.2f}{flag}")
            if r > args.threshold:
                slow.append((r, label))
    if slow:
        print("\n按差距排序（ours/ref）：")
        for r, label in sorted(slow, reverse=True):
            print(f"  {r:5.2f}  {label}")


if __name__ == "__main__":
    main()
