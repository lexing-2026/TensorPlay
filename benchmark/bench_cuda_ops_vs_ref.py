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


# (标签, 我们的调用, 参考调用, 构造输入)
def build_cases(dev, dtype):
    g = torch.Generator(device=dev).manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g, device=dev, dtype=dtype)

    C = []
    # ---- softmax 家族（SoftmaxKernels.cu）----
    for shape in [(8192, 512), (4096, 1024), (2048, 2048), (32, 256, 32, 32), (65536, 128)]:
        x = r(*shape)
        C.append((f"softmax.dim-1{shape}", lambda x=x: F.softmax(T(x), -1),
                  lambda x=x: torch.softmax(x, -1)))
        C.append((f"log_softmax.dim-1{shape}", lambda x=x: F.log_softmax(T(x), -1),
                  lambda x=x: torch.log_softmax(x, -1)))
    x = r(4096, 1024)
    y = torch.rand(4096, 1024, generator=g, device=dev, dtype=dtype)
    C.append(("softmax.backward",
              lambda y=y, x=x: F._softmax_backward_data(T(y), T(x), -1, None),
              lambda y=y, x=x: torch._softmax_backward_data(y, x, -1, dtype)))
    # ---- 归约家族（Reduce*Kernels.cu / ReductionKernels.cu）----
    for shape in [(4096, 4096), (1024, 1024, 64)]:
        x = r(*shape)
        C.append((f"sum.dim-1{shape}", lambda x=x: F.sum(T(x), [-1]), lambda x=x: torch.sum(x, -1)))
        C.append((f"mean.dim-1{shape}", lambda x=x: F.mean(T(x), [-1]), lambda x=x: torch.mean(x, -1)))
        C.append((f"amax.dim-1{shape}", lambda x=x: F.amax(T(x), [-1]), lambda x=x: torch.amax(x, -1)))
        C.append((f"var_mean{shape}", lambda x=x: F.var_mean(T(x), [-1]),
                  lambda x=x: torch.var_mean(x, -1)))
    # ---- 逐元素家族（ArithmeticKernels.cu / UnaryMathKernels.cu）----
    a, b = r(4096, 4096), r(4096, 1)
    C.append(("add.broadcast", lambda: F.add(T(a), T(b)), lambda: torch.add(a, b)))
    C.append(("mul.same", lambda: F.mul(T(a), T(a)), lambda: torch.mul(a, a)))
    C.append(("div.same", lambda: F.div(T(a), T(a)), lambda: torch.div(a, a)))
    C.append(("addcmul", lambda: F.addcmul(T(a), T(a), T(a)), lambda: torch.addcmul(a, a, a)))
    C.append(("exp", lambda: F.exp(T(a)), lambda: torch.exp(a)))
    C.append(("sigmoid", lambda: F.sigmoid(T(a)), lambda: torch.sigmoid(a)))
    C.append(("erf", lambda: F.erf(T(a)), lambda: torch.erf(a)))
    # ---- 累积极值 / 排序 ----
    x = r(2048, 2048)
    C.append(("cumsum.dim-1", lambda: F.cumsum(T(x), -1), lambda: torch.cumsum(x, -1)))
    C.append(("cummax.dim-1", lambda: F.cummax(T(x), -1), lambda: torch.cummax(x, -1)))
    C.append(("sort.dim-1", lambda: F.sort(T(x), -1), lambda: torch.sort(x, -1)))
    # topk returns (values, indices); compare the values only.
    C.append(("topk.k64", lambda x=x: F.topk(T(x), 64, -1)[0],
              lambda x=x: torch.topk(x, 64, -1)[0]))
    # ---- 索引 / embedding ----
    # 524288 lookups of 64 features spread over a varying number of rows: few
    # rows mean long runs of equal indices, many rows mean nearly all distinct.
    for num_weights in (8, 512, 1024, 4096, 65536):
        idx = torch.randint(0, num_weights, (4096, 128), generator=g, device=dev)
        go = r(4096, 128)
        go3 = go.unsqueeze(-1).expand(4096, 128, 64).contiguous()
        C.append((f"embedding_dense_backward.nw{num_weights}",
                  lambda idx=idx, go3=go3, nw=num_weights:
                      F.embedding_dense_backward(T(go3), T(idx), nw, -1, False),
                  lambda idx=idx, go3=go3, nw=num_weights:
                      torch.ops.aten.embedding_dense_backward(go3, idx, nw, -1, False)))
    # ---- 卷积 / 池化 ----
    x = r(32, 64, 56, 56)
    w = r(64, 64, 3, 3)
    C.append(("conv2d.3x3", lambda: F.conv2d(T(x), T(w), None, [1, 1], [1, 1], [1, 1], 1),
              lambda: torch.conv2d(x, w, None, 1, 1, 1, 1)))
    x = r(32, 64, 56, 56)
    C.append(("adaptive_avg_pool2d.7",
              lambda: F.adaptive_avg_pool2d(T(x), [7]),
              lambda: torch.nn.functional.adaptive_avg_pool2d(x, 7)))
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
