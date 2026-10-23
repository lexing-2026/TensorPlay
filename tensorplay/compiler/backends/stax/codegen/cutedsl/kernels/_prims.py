"""Building blocks shared by the row-wise kernels written in the Python kernel dialect.

A row kernel walks a row in fixed-width vectors, reduces a per-thread partial
across the block, and writes the result back with the same vector width.  The
pieces below are that shape and nothing else: a vectorised tiled copy, the two
block reductions the kernels need, a tail predicate for rows whose length is not
a multiple of the tile, and the compile cache.
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr

LOG2E = 1.4426950408889634


#: Widths the kernels are written for.  A row is read as 128-bit vectors, so the
#: vector length is the largest one that both divides the row and fills a vector.
def vector_length(n: int, element_bits: int) -> int:
    return math.gcd(n, 128 // element_bits)


def threads_for_row(
    n: int, element_bits: int, budget: int = 32, cap: int = 1024, floor: int = 64
) -> int:
    """A block width that leaves at most ``budget`` row elements in each thread.

    A row shorter than the budget still gets one vector per thread where it can,
    so the block never has idle lanes on a narrow row.
    """

    vec = math.gcd(n, 128 // element_bits)
    nvec = n // vec
    need = -(-n // budget)
    threads = floor
    while threads < need and threads < cap:
        threads *= 2
    threads = min(threads, cap, 1 << max(nvec - 1, 0).bit_length())
    return max(threads, 32)


@cute.jit
def block_sum(val, buf, num_warps: cutlass.Constexpr):
    """Sum a per-thread value across the block.

    The reduction is warp shuffles for the first level and one shared-memory
    slot per warp for the second, so no thread waits longer than one barrier.
    """

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    v = cute.arch.warp_reduction(val, operator.add)
    if lane == 0:
        buf[warp] = v
    cute.arch.barrier()
    w = Float32(0.0)
    if lane < num_warps:
        w = buf[lane]
    total = cute.arch.warp_reduction(w, operator.add)
    # The slots are reused by the next reduction, which must not overwrite one
    # before every warp has read it.
    cute.arch.barrier()
    return total


@cute.jit
def block_max(val, buf, num_warps: cutlass.Constexpr):
    """Largest of a per-thread value across the block; NaN wins."""

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    v = cute.arch.warp_reduction(val, cute.arch.fmax)
    if lane == 0:
        buf[warp] = v
    cute.arch.barrier()
    w = Float32(-3.0e38)
    if lane < num_warps:
        w = buf[lane]
    top = cute.arch.warp_reduction(w, cute.arch.fmax)
    # The slots are reused by a later reduction, which must not overwrite one
    # before every warp has read it.
    cute.arch.barrier()
    return top


@cute.jit
def block_sum_once(val, buf, num_warps: cutlass.Constexpr):
    """Sum a per-thread value across the block for a block's only reduction.

    The slots are never read again, so the barrier that would protect them for
    a later reduction is left out.  A kernel that reduces twice must use
    :func:`block_sum` for the first one or give each reduction its own slots.
    """

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    v = cute.arch.warp_reduction(val, operator.add)
    if lane == 0:
        buf[warp] = v
    cute.arch.barrier()
    w = Float32(0.0)
    if lane < num_warps:
        w = buf[lane]
    return cute.arch.warp_reduction(w, operator.add)


@cute.jit
def block_stats(wmax, s, buf_m, buf_s, num_warps: cutlass.Constexpr):
    """Block maximum and the exponential sum taken relative to it.

    ``wmax`` is the calling warp's own maximum and ``s`` is that warp's sum of
    values taken relative to ``wmax``.  Both levels meet in a single
    shared-memory round, so a maximum and a sum cost one barrier between them
    instead of two reductions; the slots are left dirty.
    """

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    wsum = cute.arch.warp_reduction(s, operator.add)
    if lane == 0:
        buf_m[warp] = wmax
        buf_s[warp] = wsum
    cute.arch.barrier()
    mw = Float32(-3.0e38)
    sw = Float32(0.0)
    if lane < num_warps:
        mw = buf_m[lane]
        sw = buf_s[lane]
    top = cute.arch.warp_reduction(mw, cute.arch.fmax)
    total = cute.arch.warp_reduction(
        sw * cute.math.exp2((mw - top) * LOG2E, fastmath=True), operator.add
    )
    return top, total


@cute.jit
def tail_predicate(thr: cute.ThrCopy, tiler_n: cutlass.Constexpr, n: cutlass.Constexpr):
    """Per-vector in-range predicate for a row tile wider than the row.

    A row length is a whole number of vectors, so a vector is either entirely
    inside the row or entirely outside it and one flag per vector is enough.
    """

    coord = cute.local_tile(cute.make_identity_tensor((1, tiler_n)), (1, tiler_n), (0, 0))
    tCoord = thr.partition_S(coord)
    pred = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tCoord, mode=[0, 1]),
                cute.size(tCoord, mode=[1]),
                cute.size(tCoord, mode=[2]),
            ),
            stride=(cute.size(tCoord, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(pred.shape[0]):
        for rest_k in cutlass.range_constexpr(pred.shape[2]):
            pred[rest_v, 0, rest_k] = cute.elem_less(tCoord[(0, rest_v), 0, rest_k][1], n)
    return pred


def fake_tensor(dtype, shape, divisibility: int = 1, leading_dim: int = -1):
    """A tensor of the given signature with symbolic extents, for compilation."""

    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )


def compile_cached(cache: dict, key: Any, build: Callable[[], Any]):
    """Compile once per key, reusing the object file across processes.

    A kernel is specialised on its row length and dtype, so the key is those
    facts: two shapes that agree on them run the same binary.  The object is
    also kept on disk, because generating and assembling a kernel costs a
    tenth of a second where loading the object costs a millisecond.
    """

    compiled = cache.get(key)
    if compiled is not None:
        return compiled
    path = _object_path(build, key)
    if path is not None and path.exists():
        try:
            compiled = _load_object(path)
        except Exception:
            # An object written by another build of the toolchain, or one
            # whose symbols no longer match, is simply rebuilt below.
            compiled = None
        if compiled is not None:
            cache[key] = compiled
            return compiled
    compiled = build()
    cache[key] = compiled
    if path is not None:
        _store_object(compiled, path)
    return compiled


#: Name the object file exports its entry point under.
_EXPORT_NAME = "func"


def _cache_root() -> Path | None:
    """Directory holding compiled objects, or ``None`` when caching is off."""

    if os.environ.get("TP_STAX_CACHE", "1") == "0":
        return None
    override = os.environ.get("TP_STAX_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "tensorplay" / "stax"


@lru_cache(maxsize=1)
def _source_fingerprint() -> str:
    """Identify this build of the kernel sources and of the toolchain."""

    import tvm_ffi

    digest = hashlib.sha256()
    digest.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    digest.update(f"cutlass={cutlass.__version__}".encode())
    digest.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    root = Path(__file__).resolve().parent
    for source in sorted(root.rglob("*.py")):
        content = source.read_bytes()
        digest.update(source.name.encode())
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()[:16]


def _object_path(build: Callable[[], Any], key: Any) -> Path | None:
    root = _cache_root()
    if root is None:
        return None
    fingerprint = _source_fingerprint()
    digest = hashlib.sha256(pickle.dumps((fingerprint, build.__module__, build.__qualname__, key)))
    return root / fingerprint / f"{digest.hexdigest()[:32]}.o"


def _load_object(path: Path):
    return cute.runtime.load_module(str(path), enable_tvm_ffi=True)[_EXPORT_NAME]


def _store_object(compiled, path: Path) -> None:
    """Publish an object file atomically, so a reader never sees it half written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        compiled.export_to_c(object_file_path=str(staged), function_name=_EXPORT_NAME)
        os.replace(staged, path)
    except Exception:
        staged.unlink(missing_ok=True)


def tiled_copy_for(dtype, threads: int, vec: int):
    """A copy that moves 128 bits per thread and iteration."""

    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=vec * dtype.width
    )
    thr_layout = cute.make_ordered_layout((1, threads), order=(1, 0))
    return atom, cute.make_tiled_copy_tv(atom, thr_layout, cute.make_layout((1, vec)))


def stream_atoms(dtype, vec: int):
    """Load and store atoms for a row that is read once and written once.

    The row is never read again, so its lines are marked to leave the cache
    first and the output is streamed past it: the cache keeps whatever the
    block reuses (weights, row scales) instead of two whole rows.
    """

    bits = vec * dtype.width
    load = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        dtype,
        num_bits_per_copy=bits,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
    )
    store = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        dtype,
        num_bits_per_copy=bits,
        store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING,
    )
    return load, store


class ColumnSum:
    """Sum a ``(rows, n)`` buffer down its rows into an ``(n,)`` vector.

    A warp owns a 128-element column tile and walks every row, so the buffer
    is read once and the sum finishes in a single launch: the parallelism is
    spread over the columns, and each row's tile is a single coalesced load.
    """

    def __init__(self, out_dtype, n: int, rows: int, threads: int = 32):
        self.out_dtype = out_dtype
        self.n = n
        self.rows = rows
        self.threads = threads
        self.vec = vector_length(n, 32)
        self.tiler_n = self.vec * threads

    @cute.jit
    def __call__(self, src, dst, stream):
        atom, tiled_copy = tiled_copy_for(Float32, self.threads, self.vec)
        atom_out, tiled_out = tiled_copy_for(self.out_dtype, self.threads, self.vec)
        self.kernel(src, dst, atom, tiled_copy, atom_out, tiled_out, self.tiler_n).launch(
            grid=[cute.ceil_div(self.n, self.tiler_n), 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, src, dst, atom, tiled_copy, atom_out, tiled_out,
               tiler_n: cutlass.Constexpr):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        n = self.n
        thr = tiled_copy.get_slice(tidx)
        thr_out = tiled_out.get_slice(tidx)
        src2 = cute.make_tensor(
            src.iterator, cute.make_layout((self.rows, n), stride=(n, 1))
        )
        dst2 = cute.make_tensor(dst.iterator, cute.make_layout((1, n), stride=(n, 1)))
        proto = thr.partition_S(cute.local_tile(src2, (1, tiler_n), (0, bidx)))
        gD = thr_out.partition_D(cute.local_tile(dst2, (1, tiler_n), (0, bidx)))
        pred = tail_predicate(thr, tiler_n, n) if const_expr(tiler_n != n) else None

        acc = cute.make_rmem_tensor_like(proto)
        part = cute.make_rmem_tensor_like(proto)
        rD = cute.make_rmem_tensor_like(gD)
        acc.fill(0.0)
        for row in cutlass.range(self.rows):
            src_row = thr.partition_S(cute.local_tile(src2, (1, tiler_n), (row, bidx)))
            cute.copy(atom, src_row, part, pred=pred)
            acc.store(acc.load() + part.load())
        rD.store(acc.load().to(rD.element_type))
        cute.copy(atom_out, rD, gD, pred=pred)


_colsum_cache: dict = {}


def _colsum_run(key, src, dst, rows: int, n: int, out_dtype):
    def build():
        return cute.compile(
            ColumnSum(out_dtype, n, rows),
            fake_tensor(Float32, (rows * n,), vector_length(n, 32)),
            fake_tensor(out_dtype, (n,), vector_length(n, 32)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compile_cached(_colsum_cache, key, build)(src, dst)


def column_sum(src, dst, rows: int, n: int, out_dtype, out_key):
    """Write the row sum of ``src`` into ``dst``; ``out_key`` identifies the dtype."""

    _colsum_run((n, rows, out_key), src.view(-1), dst, rows, n, out_dtype)
    return dst
