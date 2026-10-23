"""Cross entropy over the last axis, forward loss and input gradient.

The forward pass folds the exponential sum over the row in one streaming walk
that keeps a running maximum, so the row is read once and only two small
vectors leave the block: the log-sum-exp and the loss.  The backward pass turns
the log-sum-exp back into probabilities and writes the gradient in place on the
same walk, subtracting one from the target column once the whole row is stored.
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
INDEX_MAP = {
    tp.int32: cutlass.Int32,
    tp.int64: cutlass.Int64,
}


class CrossEntropyFwd:
    def __init__(self, dtype, index_dtype, n: int, threads: int, vec: int):
        self.dtype = dtype
        self.index_dtype = index_dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads
        self.num_warps = threads // 32

    @cute.jit
    def __call__(self, x, target, loss, lse, stream):
        atom, tiled_copy = tiled_copy_for(x.element_type, self.threads, self.vec)
        self.kernel(x, target, loss, lse, atom, tiled_copy, self.tiler_n).launch(
            grid=[x.shape[0], 1, 1], block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, target, loss, lse, atom, tiled_copy, tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=4)
        thr = tiled_copy.get_slice(tidx)
        gX = thr.partition_S(cute.local_tile(x, (1, tiler_n), (bidx, 0)))
        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None
        rK = cute.make_rmem_tensor_like(gX[None, None, 0])

        # Fold the row into (max, scaled sum) one vector at a time; a new
        # maximum rescales the sum accumulated so far.
        mx = Float32(-3.0e38)
        sm = Float32(0.0)
        for k in cutlass.range_constexpr(self.chunks):
            col = tidx * self.vec + k * self.threads * self.vec
            p = None if pred is None else pred[None, None, k]
            cute.copy(atom, gX[None, None, k], rK, pred=p)
            if col < n:
                v = rK.load().to(Float32).reshape((cute.size(rK),))
                cm = v.reduce(cute.ReductionOp.MAX, init_val=-3.0e38, reduction_profile=0)
                mn = cute.arch.fmax(mx, cm)
                sm = sm * cute.math.exp2((mx - mn) * LOG2E, fastmath=True) + cute.math.exp2(
                    (v - mn) * LOG2E, fastmath=True
                ).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
                mx = mn
        top = block_max(mx, buf, self.num_warps)
        sm = sm * cute.math.exp2((mx - top) * LOG2E, fastmath=True)
        total = block_sum(sm, buf, self.num_warps)
        log_denom = top + cute.math.log2(total, fastmath=True) * (1.0 / LOG2E)
        if tidx == 0:
            t = target[bidx]
            lse[bidx] = log_denom
            loss[bidx] = log_denom - x[bidx, t].to(Float32)


class CrossEntropyBwd:
    def __init__(self, dtype, index_dtype, n: int, threads: int, vec: int):
        self.dtype = dtype
        self.index_dtype = index_dtype
        self.n = n
        self.threads = threads
        self.vec = vec
        self.chunks = (n // vec + threads - 1) // threads
        self.tiler_n = vec * self.chunks * threads

    @cute.jit
    def __call__(self, x, target, lse, dloss, grad, stream):
        atom, tiled_copy = tiled_copy_for(x.element_type, self.threads, self.vec)
        self.kernel(x, target, lse, dloss, grad, atom, tiled_copy, self.tiler_n).launch(
            grid=[x.shape[0], 1, 1], block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, target, lse, dloss, grad, atom, tiled_copy,
               tiler_n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n = self.n
        thr = tiled_copy.get_slice(tidx)
        gX = thr.partition_S(cute.local_tile(x, (1, tiler_n), (bidx, 0)))
        gD = thr.partition_D(cute.local_tile(grad, (1, tiler_n), (bidx, 0)))
        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None
        rK = cute.make_rmem_tensor_like(gX[None, None, 0])
        log_denom = lse[bidx]
        scale = dloss[bidx]
        for k in cutlass.range_constexpr(self.chunks):
            p = None if pred is None else pred[None, None, k]
            cute.copy(atom, gX[None, None, k], rK, pred=p)
            v = rK.load().to(Float32).reshape((cute.size(rK),))
            e = cute.math.exp2((v - log_denom) * LOG2E, fastmath=True) * scale
            rK.store(e.to(rK.element_type))
            cute.copy(atom, rK, gD[None, None, k], pred=p)
        cute.arch.barrier()
        if tidx == 0:
            t = target[bidx]
            v0 = grad[bidx, t].to(Float32)
            grad[bidx, t] = (v0 - scale).to(grad.element_type)


_fwd_cache: dict = {}
_bwd_cache: dict = {}


def ce_fwd(x: tp.Tensor, target: tp.Tensor, threads: int | None = None):
    """Per-row cross entropy of ``x`` against ``target``, with log-sum-exp."""

    m, n = x.shape
    cutlass_dtype = DTYPE_MAP[x.dtype]
    index_dtype = INDEX_MAP[target.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        threads = threads_for_row(n, cutlass_dtype.width)
    loss = tp.empty(m, dtype=tp.float32, device=x.device)
    lse = tp.empty(m, dtype=tp.float32, device=x.device)

    def build():
        return cute.compile(
            CrossEntropyFwd(cutlass_dtype, index_dtype, n, threads, vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(index_dtype, (cute.sym_int(),), 1),
            fake_tensor(Float32, (cute.sym_int(),), 1),
            fake_tensor(Float32, (cute.sym_int(),), 1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_fwd_cache, (n, x.dtype, target.dtype, threads, vec), build)(
        x, target, loss, lse
    )
    return loss, lse


def ce_bwd(x: tp.Tensor, target: tp.Tensor, lse: tp.Tensor,
           dloss: tp.Tensor, threads: int | None = None):
    """Gradient of the per-row cross entropy with respect to ``x``."""

    n = x.shape[1]
    cutlass_dtype = DTYPE_MAP[x.dtype]
    index_dtype = INDEX_MAP[target.dtype]
    vec = vector_length(n, cutlass_dtype.width)
    if threads is None:
        threads = threads_for_row(n, cutlass_dtype.width)
    grad = tp.empty_like(x)

    def build():
        return cute.compile(
            CrossEntropyBwd(cutlass_dtype, index_dtype, n, threads, vec),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            fake_tensor(index_dtype, (cute.sym_int(),), 1),
            fake_tensor(Float32, (cute.sym_int(),), 1),
            fake_tensor(Float32, (cute.sym_int(),), 1),
            fake_tensor(cutlass_dtype, (cute.sym_int(), n), vec),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_bwd_cache, (n, x.dtype, target.dtype, threads, vec), build)(
        x, target, lse, dloss, grad
    )
    return grad
