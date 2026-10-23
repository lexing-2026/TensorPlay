"""Softmax forward and backward over the last axis.

One block owns one row.  While the row fits in registers the block reads it
once, reduces the maximum and the exponential sum in place, and writes the
normalised row back; wider rows are staged through shared memory so the row is
still read from global memory only once.
"""

from __future__ import annotations

import tensorplay as tp

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from ._prims import (
    block_max,
    block_sum,
    compile_cached,
    fake_tensor,
    tail_predicate,
    threads_for_row,
    tiled_copy_for,
    vector_length,
)

LOG2E = 1.4426950408889634
DTYPE_MAP = {
    tp.float16: cutlass.Float16,
    tp.bfloat16: cutlass.BFloat16,
    tp.float32: cutlass.Float32,
}
#: Elements per thread a row may hold in registers; above this the row is
#: staged through shared memory instead.
RMEM_BUDGET = 32


class SoftmaxFwd:
    def __init__(self, dtype, n: int, threads: int, vec: int):
        self.dtype = dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads
        self.num_warps = threads // 32
        self.in_rmem = self.chunks * vec <= RMEM_BUDGET

    @cute.jit
    def __call__(self, x, out, stream):
        atom, tiled_copy = tiled_copy_for(x.element_type, self.threads, self.vec)
        self.kernel(x, out, atom, tiled_copy, self.tiler_n).launch(
            grid=[x.shape[0], 1, 1], block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, out, atom, tiled_copy, tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=4)
        thr = tiled_copy.get_slice(tidx)
        gX = thr.partition_S(cute.local_tile(x, (1, tiler_n), (bidx, 0)))
        gO = thr.partition_D(cute.local_tile(out, (1, tiler_n), (bidx, 0)))
        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None

        if const_expr(self.in_rmem):
            rX = cute.make_rmem_tensor_like(gX)
            rO = cute.make_rmem_tensor_like(gO)
            # Positions past the row are never stored.  The sentinel is
            # neutral for both of the reductions below: it loses the maximum
            # and its exponential is zero.
            rX.fill(-3.0e38)
            cute.copy(atom, gX, rX, pred=pred)
            xf = rX.load().to(Float32)
            flat = xf.reshape((cute.size(rX),))
            top = block_max(
                flat.reduce(cute.ReductionOp.MAX, init_val=-3.0e38, reduction_profile=0),
                buf,
                self.num_warps,
            )
            e = cute.math.exp2((xf - top) * LOG2E, fastmath=True)
            total = block_sum(
                e.reshape((cute.size(rX),)).reduce(
                    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                ),
                buf,
                self.num_warps,
            )
            y = (e * (Float32(1.0) / total)).reshape(xf.shape)
            rO.store(y.to(rO.element_type))
            cute.copy(atom, rO, gO, pred=pred)
        else:
            # A row that does not fit in registers is read twice: the first
            # walk folds it into the maximum and the exponential sum, the
            # second rebuilds each element and writes it.
            rK = cute.make_rmem_tensor_like(gX[None, None, 0])
            mx = Float32(-3.0e38)
            sm = Float32(0.0)
            for k in cutlass.range_constexpr(self.chunks):
                col = tidx * self.vec + k * self.threads * self.vec
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom, gX[None, None, k], rK, pred=p)
                if col < n:
                    v = rK.load().to(Float32).reshape((cute.size(rK),))
                    cm = v.reduce(cute.ReductionOp.MAX, init_val=-3.0e38,
                                  reduction_profile=0)
                    mn = cute.arch.fmax(mx, cm)
                    sm = sm * cute.math.exp2((mx - mn) * LOG2E, fastmath=True) + \
                        cute.math.exp2((v - mn) * LOG2E, fastmath=True).reduce(
                            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                        )
                    mx = mn
            top = block_max(mx, buf, self.num_warps)
            sm = sm * cute.math.exp2((mx - top) * LOG2E, fastmath=True)
            inv = Float32(1.0) / block_sum(sm, buf, self.num_warps)
            for k in cutlass.range_constexpr(self.chunks):
                col = tidx * self.vec + k * self.threads * self.vec
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom, gX[None, None, k], rK, pred=p)
                if col < n:
                    v = rK.load().to(Float32).reshape((cute.size(rK),))
                    e = cute.math.exp2((v - top) * LOG2E, fastmath=True) * inv
                    rK.store(e.to(rK.element_type))
                    cute.copy(atom, rK, gO[None, None, k], pred=p)


class SoftmaxBwd:
    def __init__(self, dtype, n: int, threads: int, vec: int):
        self.dtype = dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads
        self.num_warps = threads // 32
        self.in_rmem = self.chunks * vec <= RMEM_BUDGET

    @cute.jit
    def __call__(self, y, dout, dx, stream):
        atom, tiled_copy = tiled_copy_for(y.element_type, self.threads, self.vec)
        self.kernel(y, dout, dx, atom, tiled_copy, self.tiler_n).launch(
            grid=[y.shape[0], 1, 1], block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, y, dout, dx, atom, tiled_copy, tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=4)
        thr = tiled_copy.get_slice(tidx)
        gY = thr.partition_S(cute.local_tile(y, (1, tiler_n), (bidx, 0)))
        gO = thr.partition_S(cute.local_tile(dout, (1, tiler_n), (bidx, 0)))
        gD = thr.partition_D(cute.local_tile(dx, (1, tiler_n), (bidx, 0)))
        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None

        if const_expr(self.in_rmem):
            rY = cute.make_rmem_tensor_like(gY)
            rO = cute.make_rmem_tensor_like(gO)
            # Positions past the row are neutral for the product sum and are
            # never stored, so zero is the right starting value.
            rY.fill(0.0)
            rO.fill(0.0)
            cute.copy(atom, gY, rY, pred=pred)
            cute.copy(atom, gO, rO, pred=pred)
            yv = rY.load().to(Float32)
            dv = rO.load().to(Float32)
            total = block_sum(
                (yv * dv).reshape((cute.size(rY),)).reduce(
                    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                ),
                buf,
                self.num_warps,
            )
            grad = yv * (dv - total)
            rY.store(grad.to(rY.element_type))
            cute.copy(atom, rY, gD, pred=pred)
        else:
            # A row that does not fit in registers is read twice: the first
            # walk folds the product sum, the second rebuilds each element
            # and writes it.
            rK = cute.make_rmem_tensor_like(gY[None, None, 0])
            rO = cute.make_rmem_tensor_like(gO[None, None, 0])
            sm = Float32(0.0)
            for k in cutlass.range_constexpr(self.chunks):
                col = tidx * self.vec + k * self.threads * self.vec
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom, gY[None, None, k], rK, pred=p)
                cute.copy(atom, gO[None, None, k], rO, pred=p)
                if col < n:
                    yv = rK.load().to(Float32).reshape((cute.size(rK),))
                    dv = rO.load().to(Float32).reshape((cute.size(rO),))
                    sm += (yv * dv).reduce(
                        cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                    )
            total = block_sum(sm, buf, self.num_warps)
            for k in cutlass.range_constexpr(self.chunks):
                col = tidx * self.vec + k * self.threads * self.vec
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom, gY[None, None, k], rK, pred=p)
                cute.copy(atom, gO[None, None, k], rO, pred=p)
                if col < n:
                    yv = rK.load().to(Float32)
                    dv = rO.load().to(Float32)
                    grad = yv * (dv - total)
                    rK.store(grad.to(rK.element_type))
                    cute.copy(atom, rK, gD[None, None, k], pred=p)


class SoftmaxBwdUniform:
    """Softmax backward when the incoming gradient is the same everywhere.

    A graph that ends in a reduction hands back one value, and every position
    of the gradient arriving here is that value.  The per-row product sum then
    needs nothing from the incoming gradient but the value itself, so the row
    is read once to fold its sum and once more to be written, and the incoming
    gradient is read as the single element it is.
    """

    def __init__(self, dtype, n: int, threads: int, vec: int):
        self.dtype = dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads
        self.num_warps = threads // 32
        self.in_rmem = self.chunks * vec <= RMEM_BUDGET

    @cute.jit
    def __call__(self, y, g, dx, stream):
        atom, tiled_copy = tiled_copy_for(y.element_type, self.threads, self.vec)
        self.kernel(y, g, dx, atom, tiled_copy, self.tiler_n).launch(
            grid=[y.shape[0], 1, 1], block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, y, g, dx, atom, tiled_copy, tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=4)
        thr = tiled_copy.get_slice(tidx)
        gY = thr.partition_S(cute.local_tile(y, (1, tiler_n), (bidx, 0)))
        gD = thr.partition_D(cute.local_tile(dx, (1, tiler_n), (bidx, 0)))
        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None
        scale = Float32(g[0])

        if const_expr(self.in_rmem):
            rY = cute.make_rmem_tensor_like(gY)
            # Positions past the row are neutral for the sum and are never
            # stored, so zero is the right starting value.
            rY.fill(0.0)
            cute.copy(atom, gY, rY, pred=pred)
            yv = rY.load().to(Float32)
            total = block_sum(
                yv.reshape((cute.size(rY),)).reduce(
                    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                ),
                buf,
                self.num_warps,
            )
            rY.store((yv * (scale - scale * total)).to(rY.element_type))
            cute.copy(atom, rY, gD, pred=pred)
        else:
            # A row that does not fit in registers is read twice: the first
            # walk folds its sum, the second scales it and writes it.
            rK = cute.make_rmem_tensor_like(gY[None, None, 0])
            sm = Float32(0.0)
            for k in cutlass.range_constexpr(self.chunks):
                col = tidx * self.vec + k * self.threads * self.vec
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom, gY[None, None, k], rK, pred=p)
                if col < n:
                    v = rK.load().to(Float32).reshape((cute.size(rK),))
                    sm += v.reduce(
                        cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                    )
            g_eff = scale - scale * block_sum(sm, buf, self.num_warps)
            for k in cutlass.range_constexpr(self.chunks):
                col = tidx * self.vec + k * self.threads * self.vec
                p = None if pred is None else pred[None, None, k]
                cute.copy(atom, gY[None, None, k], rK, pred=p)
                if col < n:
                    v = rK.load().to(Float32)
                    rK.store((v * g_eff).to(rK.element_type))
                    cute.copy(atom, rK, gD[None, None, k], pred=p)


_fwd_cache: dict = {}
_bwd_cache: dict = {}
_bwd_uniform_cache: dict = {}


def softmax_fwd(x: tp.Tensor, threads: int | None = None):
    """Normalised exponentials of ``x`` along its last axis."""

    m, n = x.shape
    cutlass_dtype = DTYPE_MAP[x.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        threads = threads_for_row(n, cutlass_dtype.width)
    out = tp.empty_like(x)

    def build():
        return cute.compile(
            SoftmaxFwd(cutlass_dtype, n, threads, vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_fwd_cache, (n, x.dtype, threads, vec), build)(x, out)
    return out


def softmax_bwd(y: tp.Tensor, dout: tp.Tensor, threads: int | None = None):
    """Gradient of :func:`softmax_fwd` given the forward output ``y``."""

    n = y.shape[1]
    cutlass_dtype = DTYPE_MAP[y.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        threads = threads_for_row(n, cutlass_dtype.width)
    dx = tp.empty_like(y)

    def build():
        return cute.compile(
            SoftmaxBwd(cutlass_dtype, n, threads, vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_bwd_cache, (n, y.dtype, threads, vec), build)(y, dout, dx)
    return dx


def softmax_bwd_uniform(y: tp.Tensor, g: tp.Tensor, threads: int | None = None):
    """Gradient of :func:`softmax_fwd` for a one-element incoming gradient.

    ``g`` holds the single value every position of the incoming gradient has.
    """

    n = y.shape[1]
    cutlass_dtype = DTYPE_MAP[y.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        threads = threads_for_row(n, cutlass_dtype.width)
    dx = tp.empty_like(y)

    def build():
        return cute.compile(
            SoftmaxBwdUniform(cutlass_dtype, n, threads, vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(cutlass_dtype, (1,), 1),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_bwd_uniform_cache, (n, y.dtype, threads, vec), build)(y, g, dx)
    return dx
