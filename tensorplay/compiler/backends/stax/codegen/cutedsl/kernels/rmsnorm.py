"""RMSNorm forward and backward as single-pass row kernels.

The forward pass reads a row once into registers, reduces the mean square
across the block, and writes the scaled row back.  The backward pass keeps a
per-thread weight gradient in registers while it walks a strided set of rows,
so the whole pass costs three row reads, one row write, and one partial buffer
per block that a caller sums afterwards.
"""

from __future__ import annotations

import tensorplay as tp

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from ._prims import (
    block_sum,
    column_sum,
    compile_cached,
    fake_tensor,
    tail_predicate,
    threads_for_row,
    tiled_copy_for,
    vector_length,
)

DTYPE_MAP = {
    tp.float16: cutlass.Float16,
    tp.bfloat16: cutlass.BFloat16,
    tp.float32: cutlass.Float32,
}


class RmsNormFwd:
    def __init__(self, dtype, n: int, threads: int, vec: int):
        self.dtype = dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads
        self.num_warps = threads // 32

    @cute.jit
    def __call__(self, x, w, out, rstd, eps, stream):
        atom, tiled_copy = tiled_copy_for(x.element_type, self.threads, self.vec)
        self.kernel(x, w, out, rstd, eps, atom, tiled_copy, self.tiler_n).launch(
            grid=[x.shape[0], 1, 1], block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, w, out, rstd, eps, atom, tiled_copy, tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=4)

        w2 = cute.make_tensor(w.iterator, cute.make_layout((1, n), stride=(0, 1)))
        thr = tiled_copy.get_slice(tidx)
        gX = thr.partition_S(cute.local_tile(x, (1, tiler_n), (bidx, 0)))
        gW = thr.partition_S(cute.local_tile(w2, (1, tiler_n), (0, 0)))
        gO = thr.partition_D(cute.local_tile(out, (1, tiler_n), (bidx, 0)))
        gR = cute.local_tile(rstd, (1,), (bidx,))

        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None
        rX = cute.make_rmem_tensor_like(gX)
        rW = cute.make_rmem_tensor_like(gW)
        rO = cute.make_rmem_tensor_like(gO)
        # Positions past the row are neutral for the sum of squares and are
        # never stored, so zero is the right starting value.
        rX.fill(0.0)
        rW.fill(0.0)
        cute.copy(atom, gX, rX, pred=pred)
        cute.copy(atom, gW, rW, pred=pred)

        v = rX.load().to(Float32)
        flat = v.reshape((cute.size(rX),))
        total = block_sum(
            (flat * flat).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
            buf,
            self.num_warps,
        )
        scale = cute.math.rsqrt(total / n + eps)
        y = v * scale * rW.load().to(Float32)
        rO.store(y.to(rO.element_type))
        cute.copy(atom, rO, gO, pred=pred)
        if tidx == 0:
            gR[0] = scale


#: Row elements per thread the backwards pass keeps in registers.  A row that
#: needs more is walked twice instead, which costs two extra row reads but
#: keeps the working set small enough to stay in registers.
RESIDENT_BUDGET = 16


class RmsNormBwd:
    def __init__(self, dtype, n: int, threads: int, vec: int, blocks: int):
        self.dtype = dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.blocks = blocks
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads
        self.num_warps = threads // 32
        self.resident = self.chunks * vec <= RESIDENT_BUDGET

    @cute.jit
    def __call__(self, x, w, dout, rstd, dx, dw_partial, stream):
        atom, tiled_copy = tiled_copy_for(x.element_type, self.threads, self.vec)
        atom32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=self.vec * 32
        )
        self.kernel(
            x, w, dout, rstd, dx, dw_partial, atom, atom32, tiled_copy, self.tiler_n
        ).launch(grid=[self.blocks, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, w, dout, rstd, dx, dw_partial, atom, atom32, tiled_copy,
               tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=4)

        w2 = cute.make_tensor(w.iterator, cute.make_layout((1, n), stride=(0, 1)))
        thr = tiled_copy.get_slice(tidx)
        gW = thr.partition_S(cute.local_tile(w2, (1, tiler_n), (0, 0)))
        gP = thr.partition_D(cute.local_tile(dw_partial, (1, tiler_n), (bidx, 0)))
        rows = x.shape[0]
        proto = thr.partition_S(cute.local_tile(x, (1, tiler_n), (0, 0)))

        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None
        iters = cute.ceil_div(rows - bidx, self.blocks)

        if const_expr(self.resident):
            rX = cute.make_rmem_tensor_like(proto)
            rO = cute.make_rmem_tensor_like(proto)
            rD = cute.make_rmem_tensor_like(proto)
            rW = cute.make_rmem_tensor_like(gW)
            rP = cute.make_rmem_tensor_like(gP)
            rW.fill(0.0)
            cute.copy(atom, gW, rW, pred=pred)
            # A no-op round trip pins the weights in registers across the row loop.
            rW.store((rW.load().to(Float32) + Float32(0.0)).to(rW.element_type))
            rP.fill(0.0)

            for i in cutlass.range(iters):
                row = bidx + i * self.blocks
                gX = thr.partition_S(cute.local_tile(x, (1, tiler_n), (row, 0)))
                gO = thr.partition_S(cute.local_tile(dout, (1, tiler_n), (row, 0)))
                gD = thr.partition_D(cute.local_tile(dx, (1, tiler_n), (row, 0)))
                # Positions past the row are neutral for both accumulations
                # below and are never stored, so zero is the right starting
                # value.
                rX.fill(0.0)
                rO.fill(0.0)
                cute.copy(atom, gX, rX, pred=pred)
                cute.copy(atom, gO, rO, pred=pred)
                inv = rstd[row]
                xv = rX.load().to(Float32)
                dv = rO.load().to(Float32)
                xh = xv * inv
                wdy = dv * rW.load().to(Float32)
                flat_x = xh.reshape((cute.size(rX),))
                flat_d = wdy.reshape((cute.size(rO),))
                c1 = block_sum(
                    (flat_x * flat_d).reduce(
                        cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                    ),
                    buf,
                    self.num_warps,
                ) / n
                out = (wdy - xh * c1) * inv
                rD.store(out.to(rD.element_type))
                cute.copy(atom, rD, gD, pred=pred)
                rP.store(rP.load() + dv * xh)
            cute.copy(atom32, rP, gP, pred=pred)
        else:
            # Wide rows are walked twice instead of held: the first walk folds
            # the row into the reduction and the weight gradient, the second
            # rebuilds each element and writes the input gradient.
            rX = cute.make_rmem_tensor_like(proto[None, None, 0])
            rO = cute.make_rmem_tensor_like(proto[None, None, 0])
            rW = cute.make_rmem_tensor_like(gW[None, None, 0])
            rP = [cute.make_rmem_tensor_like(gP[None, None, k]) for k in range(self.chunks)]
            for k in cutlass.range_constexpr(self.chunks):
                rP[k].fill(0.0)

            for i in cutlass.range(iters):
                row = bidx + i * self.blocks
                gX = thr.partition_S(cute.local_tile(x, (1, tiler_n), (row, 0)))
                gO = thr.partition_S(cute.local_tile(dout, (1, tiler_n), (row, 0)))
                gD = thr.partition_D(cute.local_tile(dx, (1, tiler_n), (row, 0)))
                inv = rstd[row]
                dot = Float32(0.0)
                for k in cutlass.range_constexpr(self.chunks):
                    p = None if pred is None else pred[None, None, k]
                    cute.copy(atom, gX[None, None, k], rX, pred=p)
                    cute.copy(atom, gO[None, None, k], rO, pred=p)
                    cute.copy(atom, gW[None, None, k], rW, pred=p)
                    col = tidx * self.vec + k * self.threads * self.vec
                    if col < n:
                        v = rX.load().to(Float32).reshape((self.vec,))
                        d = rO.load().to(Float32).reshape((self.vec,))
                        wv = rW.load().to(Float32).reshape((self.vec,))
                        dot += (v * d * wv).reduce(
                            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                        )
                        pk = rP[k].load()
                        rP[k].store(pk + (d * v * inv).reshape(pk.shape))
                c1 = block_sum(dot, buf, self.num_warps) * inv / n
                for k in cutlass.range_constexpr(self.chunks):
                    p = None if pred is None else pred[None, None, k]
                    cute.copy(atom, gX[None, None, k], rX, pred=p)
                    cute.copy(atom, gO[None, None, k], rO, pred=p)
                    cute.copy(atom, gW[None, None, k], rW, pred=p)
                    col = tidx * self.vec + k * self.threads * self.vec
                    if col < n:
                        v = rX.load().to(Float32).reshape((self.vec,))
                        d = rO.load().to(Float32).reshape((self.vec,))
                        wv = rW.load().to(Float32).reshape((self.vec,))
                        out = (d * wv - v * inv * c1) * inv
                        rO.store(out.to(rO.element_type))
                        cute.copy(atom, rO, gD[None, None, k], pred=p)
            for k in cutlass.range_constexpr(self.chunks):
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom32, rP[k], gP[None, None, k], pred=p)


_fwd_cache: dict = {}
_bwd_cache: dict = {}
_sm_count: int | None = None


def _multiprocessors() -> int:
    global _sm_count
    if _sm_count is None:
        _sm_count = tp.cuda.get_device_properties("cuda").multi_processor_count
    return _sm_count


def rmsnorm_fwd(x: tp.Tensor, weight: tp.Tensor, eps: float = 1e-6,
                threads: int | None = None):
    """Normalise each row of ``x`` by its root mean square, scaled by ``weight``."""

    m, n = x.shape
    cutlass_dtype = DTYPE_MAP[x.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        threads = threads_for_row(n, cutlass_dtype.width, budget=16)
    out = tp.empty_like(x)
    rstd = tp.empty(m, dtype=tp.float32, device=x.device)

    def build():
        return cute.compile(
            RmsNormFwd(cutlass_dtype, n, threads, vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(cutlass_dtype, (n,), vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(Float32, (cute.sym_int(),), 1),
            Float32(0.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_fwd_cache, (n, x.dtype, threads, vec), build)(
        x, weight, out, rstd, Float32(eps)
    )
    return out, rstd


def rmsnorm_bwd(
    x: tp.Tensor, weight: tp.Tensor, dout: tp.Tensor, rstd: tp.Tensor,
    threads: int | None = None, blocks: int | None = None,
):
    """Gradients of :func:`rmsnorm_fwd` with respect to the input and the weight."""

    m, n = x.shape
    cutlass_dtype = DTYPE_MAP[x.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        # Rows wider than this walk twice instead of holding the row, and a
        # narrower block leaves more registers for the accumulation.
        threads = min(512, threads_for_row(n, cutlass_dtype.width, budget=RESIDENT_BUDGET))
    if blocks is None:
        blocks = min(m, _multiprocessors())
    dx = tp.empty_like(x)
    dw_partial = tp.empty(blocks, n, dtype=tp.float32, device=x.device)
    dw = tp.empty(n, dtype=weight.dtype, device=x.device)

    def build():
        return cute.compile(
            RmsNormBwd(cutlass_dtype, n, threads, vec, blocks),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(cutlass_dtype, (n,), vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(Float32, (cute.sym_int(),), 1),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(Float32, (blocks, n), vec),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_bwd_cache, (n, x.dtype, threads, vec, blocks), build)(
        x, weight, dout, rstd, dx, dw_partial
    )
    column_sum(dw_partial, dw, blocks, n, DTYPE_MAP[weight.dtype], weight.dtype, x.device)
    return dx, dw
