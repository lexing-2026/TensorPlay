"""The product whose operands carry their own factors.

Where the factors are applied is the whole difference between the two forms,
and it is not a tuning knob:

  in the main loop   each tile of an operand is scaled as it arrives, so the
                     accumulator only ever holds unscaled products and the
                     factors are paid for once per tile rather than once per
                     element of the result
  in the epilogue    the accumulator is scaled once at the end, which is fewer
                     operations and is wrong whenever a factor would have to be
                     held at the accumulator's width

So a call cannot be offered both without asking which one it meant, and a
template that offered both would have the measurement compare two answers.

The factors come in four shapes, and the shape is named rather than inferred:
one per tensor, one per row, one per 128-by-128 block, and one per
block-that-is-one-row-wide.  The block shapes need the scale block *expanded*
to the tile's shape before it can multiply anything, which is a broadcast
followed by a reshape -- the load is small, the multiply is not, and doing the
expansion once per tile rather than once per element is the point of doing it
here at all.
"""

from __future__ import annotations

import hashlib
import linecache
from typing import Any, Callable, Optional, Tuple

import triton
import triton.language as tl

__all__ = [
    "SCALED_TUNING_VERSION",
    "SCALE_RECIPES",
    "scaled_gemm_launch",
]


def _build(source: str, fake_file: str, symbol: str):
    """Define a kernel from generated source, once.

    A jit-decorated function reads its own source back through the linecache
    when it is compiled, and one that calls another jit function reads the
    callee's source through the module the caller was defined in.  So the text
    is registered under the name it will be compiled as, and a real module is
    registered to own it -- otherwise the inner function resolves against no
    module at all and fails on the first call rather than the first compile.
    """

    import sys
    import types

    linecache.cache[fake_file] = (
        len(source), None, source.splitlines(True), fake_file,
    )
    module = types.ModuleType(fake_file.strip("<>"))
    module.__file__ = fake_file
    module.__dict__["__name__"] = module.__name__
    sys.modules[module.__name__] = module
    exec(compile(source, fake_file, "exec"), module.__dict__)
    return module.__dict__[symbol]


class SCALE_RECIPES:
    """How an operand's factor varies, named rather than guessed at."""

    TENSOR_WISE = 0
    ROW_WISE = 1
    BLOCK_128 = 2
    BLOCK_1xTILE = 3

    @classmethod
    def needs_pointer(cls, recipe: int) -> bool:
        """Whether the recipe is a pointer rather than a scalar already loaded.

        A tensor's factor is a single number, so it is loaded where the kernel
        starts.  Every other shape is a region of factors, so what the caller
        hands over is a pointer and the kernel reads the part its tile needs.
        """

        return recipe != cls.TENSOR_WISE


#: How wide a block-shaped factor is, in each direction.
_BLOCK_FACTOR_WIDTH = 128

#: Salt for a persisted decision: bumped when the kernel body changes.
SCALED_TUNING_VERSION = "scaled-gemm-1"

_KERNEL_MEMO: dict[str, Any] = {}


_KERNEL_SOURCE = '''
import triton
import triton.language as tl

TENSOR_WISE: tl.constexpr = 0
ROW_WISE: tl.constexpr = 1
BLOCK_128: tl.constexpr = 2
BLOCK_1xTILE: tl.constexpr = 3


@triton.jit
def _scale_rows(scale, rows, extent, stride):
    """One factor per row, gathered for the rows this tile covers."""

    return tl.load(scale + rows * stride, mask=rows < extent, other=0.0)


@triton.jit
def _scale_block_128(scale, pid, step, row_blocks, inner_blocks,
                     BLOCK_ROWS: tl.constexpr, BLOCK_INNER: tl.constexpr):
    """A 128-by-128 block of factors, expanded to the tile's shape.

    The load is one factor per 128 in each direction; the multiply is one per
    element.  So the small thing is read once and then broadcast across the
    large thing, which is the only order in which this is worth doing at all.
    """

    row_off = pid * tl.cdiv(BLOCK_ROWS, 128) + tl.arange(0, (BLOCK_ROWS + 127) // 128)
    col_off = step * tl.cdiv(BLOCK_INNER, 128) + tl.arange(0, (BLOCK_INNER + 127) // 128)
    ptrs = scale + row_off[:, None] * inner_blocks + col_off[None, :]
    block = tl.load(
        ptrs,
        mask=(row_off[:, None] < row_blocks) & (col_off[None, :] < inner_blocks),
        other=1.0,
    )
    # A block is 128 wide, so it covers 128 rows of the tile however many rows
    # the tile has: a 256-row tile is two blocks, a 128-row tile is one, and a
    # tile narrower than a block has no such split -- which is why the launcher
    # refuses that pairing rather than rounding it.
    rows_per_block: tl.constexpr = BLOCK_ROWS // ((BLOCK_ROWS + 127) // 128)
    inner_per_block: tl.constexpr = BLOCK_INNER // ((BLOCK_INNER + 127) // 128)
    wide = block[:, :, None, None]
    wide = tl.broadcast_to(
        wide,
        (
            (BLOCK_ROWS + 127) // 128,
            (BLOCK_INNER + 127) // 128,
            rows_per_block,
            inner_per_block,
        ),
    )
    return wide.reshape(
        ((BLOCK_ROWS + 127) // 128) * rows_per_block,
        ((BLOCK_INNER + 127) // 128) * inner_per_block,
    )


@triton.jit
def _scale_block_1xtile(scale, pid, step, row_blocks, inner_blocks,
                        BLOCK_ROWS: tl.constexpr, BLOCK_INNER: tl.constexpr,
                        TILE_INNER: tl.constexpr):
    """A factor per row and per tile-width of the contraction, expanded.

    The same shape of work as the 128-by-128 recipe with one extent collapsed:
    the row direction has one factor per row, and the contraction direction has
    one per tile rather than per 128.
    """

    row_off = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_off = step * tl.cdiv(BLOCK_INNER, TILE_INNER) + tl.arange(
        0, (BLOCK_INNER + TILE_INNER - 1) // TILE_INNER
    )
    ptrs = scale + row_off[:, None] * inner_blocks + col_off[None, :]
    block = tl.load(
        ptrs,
        mask=(row_off[:, None] < row_blocks) & (col_off[None, :] < inner_blocks),
        other=1.0,
    )
    wide = block[:, :, None]
    wide = tl.broadcast_to(
        wide, (BLOCK_ROWS, (BLOCK_INNER + TILE_INNER - 1) // TILE_INNER, TILE_INNER)
    )
    return wide.reshape(
        BLOCK_ROWS, ((BLOCK_INNER + TILE_INNER - 1) // TILE_INNER) * TILE_INNER
    )


@triton.jit
def _scaled_main_loop_gemm(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K,
    RECIPE_A: tl.constexpr, RECIPE_B: tl.constexpr,
    TILE_INNER: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """The factors are applied to each tile as it arrives.

    So the accumulator only ever holds unscaled products: every element of the
    result is multiplied once, by the factors of the tile it came from, and the
    accumulator's width never has to hold a scaled value.
    """
    if M == 0 or N == 0:
        return
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # The group is what makes the programs running together share a factor
    # block, for the same reason the persistent form reorders its tiles.
    width = GROUP_M * tl.cdiv(N, BLOCK_N)
    tile_id = pid_m + pid_n * tl.cdiv(M, BLOCK_M)
    group_id = tile_id // width
    group_rows = min(tl.cdiv(M, BLOCK_M) - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_rows)
    pid_n = (tile_id % width) // group_rows

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row_blocks = tl.cdiv(M, BLOCK_M)
    inner_blocks = tl.cdiv(K, BLOCK_K)
    step = 0
    for rk in tl.range(0, K, BLOCK_K):
        rk_idx = rk // BLOCK_K
        inner = rk_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rm[:, None] * K + inner[None, :],
            mask=(rm[:, None] < M) & (inner[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_ptr + inner[:, None] * N + rn[None, :],
            mask=(inner[:, None] < K) & (rn[None, :] < N), other=0.0,
        )
        # The recipe is dispatched here rather than inside a helper because the
        # branches produce differently shaped things -- a number, a column, a
        # full block -- and a helper that returned any of them would have to
        # agree on one shape for branches only one of which ever runs.
        if RECIPE_A == 0:
            a_scaled = a * tl.load(a_scale_ptr)
        elif RECIPE_A == 1:
            a_scaled = a * _scale_rows(a_scale_ptr, rm, M, 1)[:, None]
        elif RECIPE_A == 2:
            a_scaled = a * _scale_block_128(
                a_scale_ptr, pid_m, rk_idx, row_blocks, inner_blocks,
                BLOCK_M, BLOCK_K,
            )
        else:
            a_scaled = a * _scale_block_1xtile(
                a_scale_ptr, pid_m, rk_idx, row_blocks, inner_blocks,
                BLOCK_M, BLOCK_K, TILE_INNER,
            )
        if RECIPE_B == 0:
            b_scaled = b * tl.load(b_scale_ptr)
        elif RECIPE_B == 1:
            b_scaled = b * _scale_rows(b_scale_ptr, rn, N, 1)[None, :]
        elif RECIPE_B == 2:
            b_scaled = b * _scale_block_128(
                b_scale_ptr, pid_n, rk_idx, tl.cdiv(N, BLOCK_N), inner_blocks,
                BLOCK_N, BLOCK_K,
            )
        else:
            b_scaled = b * _scale_block_1xtile(
                b_scale_ptr, pid_n, rk_idx, tl.cdiv(N, BLOCK_N), inner_blocks,
                BLOCK_N, BLOCK_K, TILE_INNER,
            )
        acc += tl.dot(a_scaled, b_scaled, allow_tf32=ALLOW_TF32)
        step += 1
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _scaled_epilogue_gemm(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K,
    RECIPE_A: tl.constexpr, RECIPE_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """The factors are applied once, to the finished accumulator.

    Fewer operations than scaling each tile, and the only thing that can go
    wrong is a factor that would have had to be held at the accumulator's
    width -- which is why the block-shaped recipes are not offered here.
    """
    if M == 0 or N == 0:
        return
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    width = GROUP_M * tl.cdiv(N, BLOCK_N)
    tile_id = pid_m + pid_n * tl.cdiv(M, BLOCK_M)
    group_id = tile_id // width
    group_rows = min(tl.cdiv(M, BLOCK_M) - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_rows)
    pid_n = (tile_id % width) // group_rows

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for rk in tl.range(0, K, BLOCK_K):
        inner = rk + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rm[:, None] * K + inner[None, :],
            mask=(rm[:, None] < M) & (inner[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_ptr + inner[:, None] * N + rn[None, :],
            mask=(inner[:, None] < K) & (rn[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # One factor per row on each side multiplies into the accumulator from
    # opposite axes, so it is a rank-one product; anything else is one number
    # per tensor and multiplies in as a scalar.
    if RECIPE_A == 1 and RECIPE_B == 1:
        by_row = _scale_rows(a_scale_ptr, rm, M, 1)[:, None]
        by_column = _scale_rows(b_scale_ptr, rn, N, 1)[None, :]
        acc = acc * (by_row * by_column)
    else:
        acc = acc * (tl.load(a_scale_ptr) * tl.load(b_scale_ptr))
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
'''


def scaled_kernel(site: str, recipe_a: int, recipe_b: int, tile_inner: int,
                  block_m: int, block_n: int, block_k: int, group_m: int):
    """The scaled kernel for one site and one pair of recipes, built once."""

    key = hashlib.sha256(
        "|".join(
            str(v)
            for v in (
                SCALED_TUNING_VERSION, site, recipe_a, recipe_b, tile_inner,
                block_m, block_n, block_k, group_m,
            )
        ).encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    kernel = _build(
        _KERNEL_SOURCE,
        f"<tensorplay-stax-scaled-{key}>",
        "_scaled_main_loop_gemm" if site == "main_loop"
        else "_scaled_epilogue_gemm",
    )
    _KERNEL_MEMO[key] = kernel
    return kernel


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


def scaled_gemm_launch(
    a_spec: Tuple[Optional[int], Any],
    b_spec: Tuple[Optional[int], Any],
    a_scale_spec: Tuple[Optional[int], Any],
    b_scale_spec: Tuple[Optional[int], Any],
    geometry: dict,
    config: dict,
    base_launch: Callable[[list], Any],
    allow_tf32: bool = False,
):
    """Build a launcher for the scaled product, for one of the two sites."""

    m, n, k = (int(v) for v in geometry["mnk"])
    site = str(geometry["scale_site"])
    recipe_a = int(geometry["recipe_a"])
    recipe_b = int(geometry["recipe_b"])
    block_m, block_n = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    group_m = int(config.get("GROUP_M", 8))
    tile_inner = int(config.get("TILE_INNER", 32))
    # A block-shaped factor cannot be laid over a tile narrower than the block:
    # the expansion reshapes one factor per block into one per tile lane, and a
    # tile with fewer lanes than the block has no such reshape.  So the recipe
    # and the tile have to agree on that width, and a pair that does not is
    # refused here rather than left to fail inside the kernel.
    for recipe, extent in (
        (recipe_a, (block_m, block_k)),
        (recipe_b, (block_n, block_k)),
    ):
        if recipe == SCALE_RECIPES.BLOCK_128 and any(
            e % _BLOCK_FACTOR_WIDTH for e in extent
        ):
            return base_launch
        if recipe == SCALE_RECIPES.BLOCK_1xTILE and extent[0] % tile_inner:
            return base_launch
    kernel = scaled_kernel(
        site, recipe_a, recipe_b, tile_inner, block_m, block_n, block_k, group_m
    )

    def operand(feed: list, spec) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        a, b = operand(feed, a_spec), operand(feed, b_spec)
        a_scale, b_scale = operand(feed, a_scale_spec), operand(feed, b_scale_spec)
        if not (
            int(a.shape[0]) == m and int(a.shape[1]) == k
            and int(b.shape[0]) == k and int(b.shape[1]) == n
            and _standard_2d(a.shape, a.stride())
            and _standard_2d(b.shape, b.stride())
        ):
            return base_launch(feed)
        out = tp.empty((m, n), dtype=a.dtype, device=a.device)
        grid = (-(-m // block_m), -(-n // block_n), 1)
        args = [a, b, out, a_scale, b_scale, m, n, k]
        if site == "main_loop":
            kernel[grid](
                *args,
                RECIPE_A=recipe_a, RECIPE_B=recipe_b, TILE_INNER=tile_inner,
                ALLOW_TF32=allow_tf32,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
                num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
            )
        else:
            kernel[grid](
                *args,
                RECIPE_A=recipe_a, RECIPE_B=recipe_b, ALLOW_TF32=allow_tf32,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
                num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
            )
        return out

    return launch
