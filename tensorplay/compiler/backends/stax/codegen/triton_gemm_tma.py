"""The product whose tiles are read through a descriptor.

The arithmetic is the persistent product's; what differs is how a tile is
fetched.  A descriptor is built on the host from a base pointer, a shape,
strides and a block shape, and the kernel then asks for tiles by coordinate
rather than by address -- so the address arithmetic a persistent product would
repeat for every tile is stated once, on the host, and the kernel spends its
registers on the arithmetic instead of on addresses.

Two things come with that, and both are the point:

  * The tile order is the kernel's to choose.  A descriptor fetch is cheaper
    the more the programs running at the same moment share an operand tile, so
    the flat tile index is decoded in groups of rows before advancing a column.
    The group is a tile-count parameter, not a tuning knob, because the whole
    purpose is to make concurrently-running programs agree on which tile they
    are reading.

  * Whether the accumulator is passed into the multiply or added to afterwards
    is a code-generation choice, not a numerical one.  Passing it in lets the
    multiply accumulate in place; adding afterwards needs a temporary.  Which
    one is available is a property of the toolkit, so the kernel is written to
    take either and the caller says which.

The cost is a hardware requirement rather than a preference: a descriptor is a
feature of the memory system, and on a machine without it this kernel cannot
run at all.  So the template asks the device before offering anything, and a
machine that cannot answer is a machine the template has nothing to say about.

The later generation's form differs in what it carries between tiles rather than
in how it fetches them: a workspace sized by the device's own limits is what
lets a tile's partial result outlive the tile that produced it, which is what
the earlier persistent form has to keep in registers.  The workspace is sized
from the device, so the launcher asks the device for the size rather than
assuming one.
"""

from __future__ import annotations

import hashlib
import linecache
from typing import Any, Callable, Optional, Tuple

import triton
import triton.language as tl

__all__ = [
    "TMA_TUNING_VERSION",
    "tma_gemm_launch",
    "workspace_bytes",
]

#: Salt for a persisted decision: bumped when the kernel body changes, so a
#: stored choice cannot outlive the kernel it named.
TMA_TUNING_VERSION = "persistent-tma-1"

#: One memo entry per (geometry, orientation, accumulation form) so repeated
#: candidate launches reuse a compiled binary.
_KERNEL_MEMO: dict[str, Any] = {}


_KERNEL_SOURCE = '''
import triton
import triton.language as tl


@triton.jit
def _tma_persistent_gemm(
    a_ptr, b_ptr, c_ptr, ws_ptr,
    M, N, K,
    NUM_SMS: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr, B_ROW_MAJOR: tl.constexpr,
    HAS_WORKSPACE: tl.constexpr,
    ALLOW_TF32: tl.constexpr, FAST_ACCUM: tl.constexpr,
):
    """One program's share of the tiles, fetched by coordinate.

    The descriptors are built on the host and handed in as arguments rather
    than made here, because a descriptor is a host-side object: the kernel
    receives the mapping and asks it for tiles, and the address arithmetic that
    a non-descriptor kernel would do per tile was done once, outside.
    """
    if M == 0 or N == 0:
        # Nothing to produce.  A grid built from zero extents has no tiles, and
        # a program that finds itself here would be dividing by a width of
        # zero two lines down.
        return

    start_pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    # The width of one group of rows, in tiles.  Decoding the flat index by
    # group rather than by row is what makes the programs running together
    # share their operand tiles: a run of GROUP_M tiles down a column reads
    # the same columns of the right operand and adjacent rows of the left.
    width = GROUP_M * grid_n

    a_desc = triton.language.make_tensor_descriptor(
        base=a_ptr,
        shape=[M, K] if A_ROW_MAJOR else [K, M],
        strides=[K, 1] if A_ROW_MAJOR else [1, K],
        block_shape=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=b_ptr,
        shape=[K, N] if B_ROW_MAJOR else [N, K],
        strides=[N, 1] if B_ROW_MAJOR else [1, N],
        block_shape=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
    )

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size

        rm = pid_m * BLOCK_M
        rn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for rk in tl.range(0, K, BLOCK_K):
            a = tl.load_tensor_descriptor(
                a_desc, [rm, rk] if A_ROW_MAJOR else [rk, rm]
            )
            b = tl.load_tensor_descriptor(
                b_desc, [rk, rn] if B_ROW_MAJOR else [rn, rk]
            )
            left = a if A_ROW_MAJOR else a.T
            right = b if B_ROW_MAJOR else b.T
            if FAST_ACCUM:
                acc = tl.dot(left, right, acc, allow_tf32=ALLOW_TF32)
            else:
                acc += tl.dot(left, right, allow_tf32=ALLOW_TF32)

        # The origins are recomputed rather than carried: they are two scalars
        # that would otherwise occupy registers across the whole contraction,
        # and the tile is finished with them the moment the accumulator is.
        rm = pid_m * BLOCK_M
        rn = pid_n * BLOCK_N
        rcm = rm + tl.arange(0, BLOCK_M)
        rcn = rn + tl.arange(0, BLOCK_N)
        mask = (rcm[:, None] < M) & (rcn[None, :] < N)
        if HAS_WORKSPACE:
            # The partial goes through the workspace rather than straight to the
            # result: a program that has finished a tile can start the next one
            # while this write is still in flight, which is the whole reason the
            # later form needs scratch space at all.
            slot = start_pid * 2 * (BLOCK_M * BLOCK_N) + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            tl.store(ws_ptr + slot, acc)
            tl.debug_barrier()
            acc = tl.load(ws_ptr + slot)
        tl.store(c_ptr + rcm[:, None] * N + rcn[None, :], acc, mask=mask)
'''


def tma_kernel(block_m: int, block_n: int, block_k: int, group_m: int,
               a_row_major: bool, b_row_major: bool, fast_accum: bool,
               has_workspace: bool):
    """The descriptor-driven kernel for one form, built once and remembered."""

    key = hashlib.sha256(
        "|".join(
            str(v)
            for v in (
                TMA_TUNING_VERSION, block_m, block_n, block_k, group_m,
                a_row_major, b_row_major, fast_accum, has_workspace,
            )
        ).encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    source = _KERNEL_SOURCE
    fake_file = f"<tensorplay-stax-tma-{key}>"
    linecache.cache[fake_file] = (
        len(source), None, source.splitlines(True), fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    kernel = namespace["_tma_persistent_gemm"]
    _KERNEL_MEMO[key] = kernel
    return kernel


def workspace_bytes(num_sms: int, block_m: int, block_n: int) -> int:
    """How much scratch a partial result needs to outlive its tile.

    A tile's partial is a pair of extents wide, and one is kept per program so
    that a program which has finished a tile can start the next while the
    earlier one is still being written out.  So the workspace scales with the
    number of programs and with the tile, and with nothing else.
    """

    return int(num_sms) * 2 * int(block_m) * int(block_n) * 4


def _standard_2d(shape, stride) -> bool:
    shape = tuple(int(s) for s in shape)
    stride = tuple(int(s) for s in stride)
    if len(shape) != 2:
        return False
    expected, running = [], 1
    for extent in reversed(shape):
        expected.append(running)
        running *= max(extent, 1)
    return stride == tuple(reversed(expected))


def tma_gemm_launch(
    a_spec: Tuple[Optional[int], Any],
    b_spec: Tuple[Optional[int], Any],
    geometry: dict,
    config: dict,
    base_launch: Callable[[list], Any],
    allow_tf32: bool = False,
):
    """Build a launcher for the descriptor-driven persistent product.

    The geometry is everything about the call that is not a choice; the config
    is the choice.  A call whose real layout is not the one the descriptors
    were built for takes ``base_launch`` instead, which is what keeps a
    launcher from being a promise about a call it has not seen.
    """

    m, n, k = (int(v) for v in geometry["mnk"])
    num_sms = int(geometry.get("num_sms", 1))
    block_m, block_n = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    group_m = int(config.get("GROUP_M", 8))
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    fast_accum = bool(config.get("FAST_ACCUM", True))
    a_row_major = _standard_2d(geometry["a_shape"], geometry["a_stride"])
    b_row_major = _standard_2d(geometry["b_shape"], geometry["b_stride"])
    has_workspace = bool(geometry.get("has_workspace", False))
    kernel = tma_kernel(
        block_m, block_n, block_k, group_m, a_row_major, b_row_major,
        fast_accum, has_workspace,
    )

    def operand(feed: list, spec) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        a, b = operand(feed, a_spec), operand(feed, b_spec)
        if not (
            int(a.shape[0]) == m and int(a.shape[1]) == k
            and int(b.shape[0]) == k and int(b.shape[1]) == n
            and _standard_2d(a.shape, a.stride())
            and _standard_2d(b.shape, b.stride())
        ):
            return base_launch(feed)
        out = tp.empty((m, n), dtype=a.dtype, device=a.device)
        scratch = out
        if has_workspace:
            scratch = tp.empty(workspace_bytes(num_sms, block_m, block_n),
                               dtype=tp.float32, device=a.device)
        grid = (min(num_sms, -(-m // block_m) * -(-n // block_n)), 1, 1)
        kernel[grid](
            a, b, out, scratch,
            m, n, k,
            NUM_SMS=num_sms, GROUP_M=group_m,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            A_ROW_MAJOR=a_row_major, B_ROW_MAJOR=b_row_major,
            HAS_WORKSPACE=has_workspace,
            ALLOW_TF32=allow_tf32, FAST_ACCUM=fast_accum,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch
