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
    # ---- spatial softmax（inner>1：NHWC 通道维做 softmax，dim=1）----
    for shape in [(8, 64, 512, 512), (2, 512, 32, 32), (4, 1024, 64, 64)]:
        x = r(*shape)
        tx = T(x)
        C.append((f"softmax.dim1.spatial{shape}", lambda tx=tx: F.softmax(tx, 1),
                  lambda x=x: torch.softmax(x, 1)))
        C.append((f"log_softmax.dim1.spatial{shape}",
                  lambda tx=tx: F.log_softmax(tx, 1),
                  lambda x=x: torch.log_softmax(x, 1)))
    # ---- 词表级大行（不整除向量化宽度的典型 vocab）----
    for shape in [(2048, 50257)]:
        x = r(*shape)
        tx = T(x)
        C.append((f"softmax.vocab{shape}", lambda tx=tx: F.softmax(tx, -1),
                  lambda x=x: torch.softmax(x, -1)))
        C.append((f"log_softmax.vocab{shape}",
                  lambda tx=tx: F.log_softmax(tx, -1),
                  lambda x=x: torch.log_softmax(x, -1)))
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
    # ---- 索引家族（IndexingKernels.cu / GatherScatterKernels.cu）----
    # 行收集：embedding 风格（每行 128 元素）与大切片（每行 4096 元素）两种粒度，
    # 以及沿列收集（inner=1 的逐元素路径）。
    x = r(131072, 128)
    idx = torch.randint(0, 131072, (16384,), generator=g, device=dev)
    tx, tidx = T(x), T(idx)
    C.append(("index_select.rows128x16k",
              lambda tx=tx, tidx=tidx: F.index_select(tx, 0, tidx),
              lambda x=x, idx=idx: torch.index_select(x, 0, idx)))
    x = r(8192, 4096)
    idx = torch.randint(0, 8192, (2048,), generator=g, device=dev)
    tx, tidx = T(x), T(idx)
    C.append(("index_select.slice4kx2k",
              lambda tx=tx, tidx=tidx: F.index_select(tx, 0, tidx),
              lambda x=x, idx=idx: torch.index_select(x, 0, idx)))
    x = r(512, 65536)
    idx = torch.randint(0, 65536, (8192,), generator=g, device=dev)
    tx, tidx = T(x), T(idx)
    C.append(("index_select.cols64kx8k",
              lambda tx=tx, tidx=tidx: F.index_select(tx, 1, tidx),
              lambda x=x, idx=idx: torch.index_select(x, 1, idx)))
    idx32 = idx.to(torch.int32)
    tidx32 = T(idx32)
    C.append(("index_select.cols32.8k",
              lambda tx=tx, tidx32=tidx32: F.index_select(tx, 1, tidx32),
              lambda x=x, idx32=idx32: torch.index_select(x, 1, idx32)))
    # gather：最后一维逐元素（inner=1）与内部维（inner=4096）两种。
    x = r(4096, 4096)
    idx = torch.randint(0, 4096, (4096, 4096), generator=g, device=dev)
    tx, tidx = T(x), T(idx)
    C.append(("gather.dim-1", lambda tx=tx, tidx=tidx: F.gather(tx, -1, tidx),
              lambda x=x, idx=idx: torch.gather(x, -1, idx)))
    C.append(("gather.dim0", lambda tx=tx, tidx=tidx: F.gather(tx, 0, tidx),
              lambda x=x, idx=idx: torch.gather(x, 0, idx)))
    x3 = r(64, 256, 512)
    idx3 = torch.randint(0, 256, (64, 64, 512), generator=g, device=dev)
    tx3, tidx3 = T(x3), T(idx3)
    C.append(("gather.dim1.3d", lambda tx3=tx3, tidx3=tidx3: F.gather(tx3, 1, tidx3),
              lambda x3=x3, idx3=idx3: torch.gather(x3, 1, idx3)))
    # scatter：写入（幂等）与累加两种；scatter_add_ 数值随重复调用累积，仅看时间。
    x = r(4096, 4096)
    idx = torch.randint(0, 4096, (4096, 4096), generator=g, device=dev)
    src = r(4096, 4096)
    tx, tidx, tsrc = T(x), T(idx), T(src)
    C.append(("scatter.dim-1", lambda tx=tx, tidx=tidx, tsrc=tsrc: F.scatter(tx, -1, tidx, tsrc),
              lambda x=x, idx=idx, src=src: torch.scatter(x, -1, idx, src)))
    C.append(("scatter_.dim-1", lambda tx=tx, tidx=tidx, tsrc=tsrc: tx.scatter_(-1, tidx, tsrc),
              lambda x=x, idx=idx, src=src: x.scatter_(-1, idx, src)))
    C.append(("scatter_add.dim0", lambda tx=tx, tidx=tidx, tsrc=tsrc: F.scatter_add(tx, 0, tidx, tsrc),
              lambda x=x, idx=idx, src=src: torch.scatter_add(x, 0, idx, src)))
    C.append(("scatter_add_.dim0", lambda tx=tx, tidx=tidx, tsrc=tsrc: tx.scatter_add_(0, tidx, tsrc),
              lambda x=x, idx=idx, src=src: x.scatter_add_(0, idx, src)))
    # index_add_ / index_copy_ / index_fill_：embedding 梯度式的行更新。
    x = r(65536, 128)
    idx = torch.randint(0, 65536, (8192,), generator=g, device=dev)
    src = r(8192, 128)
    tx, tidx, tsrc = T(x), T(idx), T(src)
    C.append(("index_add_.rows128x8k",
              lambda tx=tx, tidx=tidx, tsrc=tsrc: tx.index_add_(0, tidx, tsrc),
              lambda x=x, idx=idx, src=src: x.index_add_(0, idx, src)))
    C.append(("index_copy_.rows128x8k",
              lambda tx=tx, tidx=tidx, tsrc=tsrc: tx.index_copy_(0, tidx, tsrc),
              lambda x=x, idx=idx, src=src: x.index_copy_(0, idx, src)))
    C.append(("index_add.rows128x8k",
              lambda tx=tx, tidx=tidx, tsrc=tsrc: F.index_add(tx, 0, tidx, tsrc),
              lambda x=x, idx=idx, src=src: torch.index_add(x, 0, idx, src)))
    C.append(("index_fill_.rows128x8k",
              lambda tx=tx, tidx=tidx: tx.index_fill_(0, tidx, 0.0),
              lambda x=x, idx=idx: x.index_fill_(0, idx, 0.0)))
    # 高级索引：二维索引对（可能有重复行）的收集与写入。
    x = r(65536, 128)
    i0 = torch.randint(0, 65536, (65536,), generator=g, device=dev)
    i1 = torch.randint(0, 128, (65536,), generator=g, device=dev)
    v = r(65536)
    tx, ti0, ti1, tv = T(x), T(i0), T(i1), T(v)
    C.append(("index.2d.pairs",
              lambda tx=tx, ti0=ti0, ti1=ti1: F.index(tx, [ti0, ti1]),
              lambda x=x, i0=i0, i1=i1: x[i0, i1]))
    C.append(("index_put_.pairs",
              lambda tx=tx, ti0=ti0, ti1=ti1, tv=tv: tx.index_put_((ti0, ti1), tv),
              lambda x=x, i0=i0, i1=i1, v=v: x.index_put_((i0, i1), v)))
    # masked_select / nonzero / count_nonzero：条件收集。
    x = r(4096, 4096)
    x.masked_fill_(x.abs() < 1.0, 0)
    mask = x != 0
    tx, tmask = T(x), T(mask)
    C.append(("masked_select.p50",
              lambda tx=tx, tmask=tmask: F.masked_select(tx, tmask),
              lambda x=x, mask=mask: torch.masked_select(x, mask)))
    C.append(("nonzero.p10", lambda tx=tx: F.nonzero(tx),
              lambda x=x: torch.nonzero(x)))
    C.append(("count_nonzero", lambda tx=tx: F.count_nonzero(tx),
              lambda x=x: torch.count_nonzero(x)))
    C.append(("count_nonzero.dim-1", lambda tx=tx: F.count_nonzero(tx, [-1]),
              lambda x=x: torch.count_nonzero(x, [-1])))
    # take / where / flip / roll：扁平收集与逐元素选择。
    x = r(4194304)
    idx = torch.randint(0, 4194304, (1048576,), generator=g, device=dev)
    tx, tidx = T(x), T(idx)
    C.append(("take.flat4M", lambda tx=tx, tidx=tidx: F.take(tx, tidx),
              lambda x=x, idx=idx: torch.take(x, idx)))
    x = r(4096, 4096)
    a = r(4096, 4096)
    b = r(4096, 4096)
    cond = torch.rand(4096, 4096, generator=g, device=dev) < 0.5
    tx, ta, tb, tcond = T(x), T(a), T(b), T(cond)
    C.append(("where.ternary", lambda tcond=tcond, ta=ta, tb=tb: F.where(tcond, ta, tb),
              lambda cond=cond, a=a, b=b: torch.where(cond, a, b)))
    C.append(("flip.dim1", lambda tx=tx: F.flip(tx, [1]),
              lambda x=x: torch.flip(x, [1])))
    C.append(("roll.dim1", lambda tx=tx: F.roll(tx, 3, [1]),
              lambda x=x: torch.roll(x, 3, [1])))
    # searchsorted / bincount / unique / repeat_interleave：排序与计数系。
    s = torch.sort(torch.rand(1 << 20, generator=g, device=dev, dtype=torch.float32)).values.to(dtype)
    q = torch.rand(1 << 20, generator=g, device=dev, dtype=torch.float32).to(dtype)
    ts, tq = T(s), T(q)
    C.append(("searchsorted.1M", lambda ts=ts, tq=tq: F.searchsorted(ts, tq),
              lambda s=s, q=q: torch.searchsorted(s, q)))
    v1 = torch.randint(0, 50000, (4194304,), generator=g, device=dev)
    tv1 = T(v1)
    C.append(("bincount.4M50k", lambda tv1=tv1: F.bincount(tv1, minlength=50000),
              lambda v1=v1: torch.bincount(v1, minlength=50000)))
    u = torch.randint(0, 200000, (1048576,), generator=g, device=dev)
    tu = T(u)
    C.append(("unique.1M200k", lambda tu=tu: F.unique(tu),
              lambda u=u: torch.unique(u)))
    x = r(4096, 128)
    reps = torch.randint(1, 3, (4096,), generator=g, device=dev)
    tx, treps = T(x), T(reps)
    C.append(("repeat_interleave.tensor",
              lambda tx=tx, treps=treps: F.repeat_interleave(tx, treps, 0),
              lambda x=x, reps=reps: torch.repeat_interleave(x, reps, 0)))
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
    # ---- 多张量 foreach 家族（Foreach*Kernels.cu）----
    # 优化器规模的张量组：尺寸按 5 种典型参数张量循环铺开，列表越长越
    # 能体现单次分组启动相对逐张量启动的差距。
    for count in (120, 400):
        fa = [torch.randn(*[(4096, 4096), (2048, 1024), (1024, 1024),
                            (4096,), (512, 4096)][i % 5],
                          generator=g, device=dev, dtype=dtype)
              for i in range(count)]
        fb = [torch.randn_like(t) for t in fa]
        ta = [T(t) for t in fa]
        tb = [T(t) for t in fb]
        C.append((f"foreach_norm.list{count}",
                  lambda ta=ta: tp._foreach_norm(ta, 2),
                  lambda fa=fa: torch._foreach_norm(fa, 2)))
        C.append((f"foreach_add_.list{count}",
                  lambda ta=ta, tb=tb: tp._foreach_add_(ta, tb, alpha=0.1),
                  lambda fa=fa, fb=fb: torch._foreach_add_(fa, fb, alpha=0.1)))
        C.append((f"foreach_mul_.list{count}",
                  lambda ta=ta: tp._foreach_mul_(ta, 0.9),
                  lambda fa=fa: torch._foreach_mul_(fa, 0.9)))
        C.append((f"foreach_copy_.list{count}",
                  lambda ta=ta, tb=tb: tp._foreach_copy_(ta, tb),
                  lambda fa=fa, fb=fb: torch._foreach_copy_(fa, fb)))
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
