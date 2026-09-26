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
TMA_TUNING_VERSION = "persistent-tma-2"

#: How much room one mapping takes in a workspace, when the toolkit builds it
#: there rather than in the kernel's own registers.  The size is fixed by the
#: mapping's own layout rather than chosen here.
TMA_SIZE = 128


def experimental_descriptor_api() -> bool:
    """Whether the toolkit has only the earlier name for building a mapping.

    A toolkit that has only the earlier one cannot have the mapping made in the
    kernel's registers, so the body that suits it also has to be told which of
    the two it is writing.
    """

    return not hasattr(tl, "make_tensor_descriptor")

#: One memo entry per (geometry, orientation, accumulation form) so repeated
#: candidate launches reuse a compiled binary.
_KERNEL_MEMO: dict[str, Any] = {}


def tma_kernel(block_m: int, block_n: int, block_k: int, group_m: int,
               a_row_major: bool, b_row_major: bool, fast_accum: bool,
               has_workspace: bool, experimental_api: bool):
    """The descriptor-driven kernel for one form, built once and remembered."""

    key = hashlib.sha256(
        "|".join(
            str(v)
            for v in (
                TMA_TUNING_VERSION, block_m, block_n, block_k, group_m,
                a_row_major, b_row_major, fast_accum, has_workspace,
                experimental_api,
            )
        ).encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    # The extents are placeholders: the signature only needs each operand's
    # rank, and the launch passes the call's real extents and strides.  The
    # scratch is declared so that the kernel that uses it has it as a parameter
    # rather than as a global, and its extent and stride are never read -- a
    # slot's offset is computed from the tile, not from the buffer's shape.
    plane = (0, 0)
    from ..templates.select_algorithm import KernelArgs, TritonTemplate

    source = TritonTemplate.from_file(
        "blackwell_ws_persistent_device_tma_mm" if has_workspace
        else "persistent_tma_mm",
        file=("triton_blackwell_ws_persistent_device_tma_mm" if has_workspace
              else "triton_persistent_tma_mm"),
        symbol="_tma_persistent_gemm",
    ).render_with(
        KernelArgs(
            {
                "A": {"shape": plane, "stride": plane},
                "B": {"shape": plane, "stride": plane},
                "WS": {"shape": (0,), "stride": (1,)},
            },
            {"C": {"shape": plane, "stride": plane}},
            {
                "NUM_SMS": 1, "GROUP_M": int(group_m),
                "BLOCK_M": int(block_m), "BLOCK_N": int(block_n),
                "BLOCK_K": int(block_k),
                "A_ROW_MAJOR": bool(a_row_major),
                "B_ROW_MAJOR": bool(b_row_major),
                "HAS_WORKSPACE": bool(has_workspace),
                "ALLOW_TF32": False, "FAST_ACCUM": bool(fast_accum),
                "TMA_SIZE": int(TMA_SIZE), "MAPPING_BASE": 0,
            },
        ),
        TMA_EXPERIMENTAL_API=bool(
            experimental_descriptor_api() and not has_workspace),
    )
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
    experimental = experimental_descriptor_api()
    if has_workspace and experimental:
        # The later form builds its mappings in the kernel's own registers, and
        # a toolkit that has only the earlier call cannot: it would have to
        # build them in the very workspace this form spends on partials.
        return base_launch
    kernel = tma_kernel(
        block_m, block_n, block_k, group_m, a_row_major, b_row_major,
        fast_accum, has_workspace, experimental,
    )
    # The mapping region and the partial region are one buffer, with the mapping
    # after the partials: each is indexed from the origin it is declared with,
    # and a mapping is a fixed number of bytes per program rather than a count
    # of results.
    mapping_base = num_sms * 2 * block_m * block_n if has_workspace else 0
    mapping_elements = -(-num_sms * 2 * TMA_SIZE // 4) if experimental else 0

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
        if has_workspace or experimental:
            scratch = tp.empty(mapping_base + mapping_elements,
                               dtype=tp.float32, device=a.device)
        grid = (min(num_sms, -(-m // block_m) * -(-n // block_n)), 1, 1)
        # The order is the signature's: the operands, then their extents, then
        # their strides.  Everything chosen when the text was written -- the
        # tile, the warp count, the descriptor's shape -- is a constant of the
        # module the kernel is defined in, so there is nothing to pass for it
        # and nothing to keep in step with the text.  The scratch's own extent
        # and stride are one and one because the body computes a slot's offset
        # rather than reading them.
        kernel[grid](
            a, b, scratch, out,
            *(int(v) for v in a.shape), *(int(v) for v in b.shape), 1,
            *(int(v) for v in out.shape),
            *(int(v) for v in a.stride()),
            *(int(v) for v in b.stride()), 1,
            *(int(v) for v in out.stride()),
            NUM_SMS=int(num_sms), GROUP_M=int(group_m),
            BLOCK_M=int(block_m), BLOCK_N=int(block_n), BLOCK_K=int(block_k),
            A_ROW_MAJOR=bool(a_row_major), B_ROW_MAJOR=bool(b_row_major),
            HAS_WORKSPACE=bool(has_workspace), ALLOW_TF32=bool(allow_tf32),
            FAST_ACCUM=bool(fast_accum), TMA_SIZE=int(TMA_SIZE),
            MAPPING_BASE=int(mapping_base), INDEX_DTYPE=tl.int64,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch
