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


#: The two sites: the body each of them runs, and the block extents and other
#: constants that body declares, in the order it declares them.  The launch
#: reads the tail off this list rather than writing its own, because a tail the
#: launch guesses and a signature the render emits are two lists to keep in
#: step, and a kernel that is one argument short is a kernel that cannot run.
_SITES = {
    "main_loop": (
        "main_loop_scaled_mm", "triton_main_loop_scaled_mm",
        "_scaled_main_loop_gemm",
        ("RECIPE_A", "RECIPE_B", "TILE_INNER", "ALLOW_TF32",
         "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "NUM_SMS",
         "USE_TMA_LOAD", "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR", "TMA_SIZE"),
    ),
    "epilogue": (
        "epilogue_scaled_mm", "triton_epilogue_scaled_mm",
        "_scaled_epilogue_gemm",
        ("RECIPE_A", "RECIPE_B", "ALLOW_TF32",
         "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "NUM_SMS",
         "USE_TMA_LOAD", "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR", "TMA_SIZE"),
    ),
}

#: How much room one mapping takes in a workspace, when the toolkit builds it
#: there rather than in the kernel's own registers.  The size is fixed by the
#: mapping's own layout rather than chosen here.
TMA_SIZE = 128

#: How wide a factor is, per recipe: one number, one per row, or one per block
#: of rows and of the contraction.  The rank is what tells a factor of one
#: recipe from a factor of another that happens to hold the same numbers.
_FACTOR_RANK = {
    SCALE_RECIPES.TENSOR_WISE: 0,
    SCALE_RECIPES.ROW_WISE: 1,
    SCALE_RECIPES.BLOCK_128: 2,
    SCALE_RECIPES.BLOCK_1xTILE: 2,
}


def factor_rank(recipe: int) -> int:
    """How many dimensions a factor of this recipe has."""

    if recipe not in _FACTOR_RANK:
        raise AssertionError(f"unknown factor recipe {recipe!r}")
    return _FACTOR_RANK[recipe]


def scaled_kernel(site: str, recipe_a: int, recipe_b: int, tile_inner: int,
                  block_m: int, block_n: int, block_k: int, group_m: int,
                  num_sms: int, fetch: dict):
    """The scaled kernel for one site and one pair of recipes, built once.

    The two sites are two bodies rather than one with a flag, because the work
    is in a different place: one multiplies each tile as it arrives, the other
    multiplies the finished accumulator, and only the second can carry a factor
    per row.  A flag would leave both shapes of work in one function.
    """

    key = hashlib.sha256(
        "|".join(
            str(v)
            for v in (
                SCALED_TUNING_VERSION, site, recipe_a, recipe_b, tile_inner,
                block_m, block_n, block_k, group_m, num_sms,
                *sorted(fetch.items()),
            )
        ).encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    if site not in _SITES:
        raise AssertionError(f"unknown scaling site {site!r}")
    name, file_name, symbol, _tail = _SITES[site]
    # The extents are placeholders: the signature only needs each operand's
    # rank, and the launch passes the call's real extents and strides.  A
    # factor's rank is the one its recipe says it has, which is what tells a
    # factor of one recipe from a factor of another holding the same numbers.
    from ..templates.select_algorithm import KernelArgs, TritonTemplate

    plane = (0, 0)
    factor_a = (0,) * factor_rank(recipe_a)
    factor_b = (0,) * factor_rank(recipe_b)
    inputs = {
        "A": {"shape": plane, "stride": plane},
        "B": {"shape": plane, "stride": plane},
        "SA": {"shape": factor_a, "stride": (1,) * len(factor_a)},
        "SB": {"shape": factor_b, "stride": (1,) * len(factor_b)},
    }
    if fetch["USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR"]:
        # The workspace the mappings are built in is an operand of the body that
        # builds them, so it is declared the way every other operand is.
        inputs["ws"] = {"shape": (num_sms * 2 * TMA_SIZE,), "stride": (1,)}
    source = TritonTemplate.from_file(name, file=file_name, symbol=symbol).render_with(
        KernelArgs(
            inputs,
            {"C": {"shape": plane, "stride": plane}},
            {
                "RECIPE_A": int(recipe_a), "RECIPE_B": int(recipe_b),
                "TILE_INNER": int(tile_inner), "ALLOW_TF32": False,
                "BLOCK_M": int(block_m), "BLOCK_N": int(block_n),
                "BLOCK_K": int(block_k), "GROUP_M": int(group_m),
                "NUM_SMS": int(num_sms), "TMA_SIZE": int(TMA_SIZE),
                "USE_TMA_LOAD": bool(fetch["USE_TMA_LOAD"]),
                "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": bool(
                    fetch["USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR"]),
            } if site == "main_loop" else {
                "RECIPE_A": int(recipe_a), "RECIPE_B": int(recipe_b),
                "ALLOW_TF32": False,
                "BLOCK_M": int(block_m), "BLOCK_N": int(block_n),
                "BLOCK_K": int(block_k), "GROUP_M": int(group_m),
                "NUM_SMS": int(num_sms), "TMA_SIZE": int(TMA_SIZE),
                "USE_TMA_LOAD": bool(fetch["USE_TMA_LOAD"]),
                "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": bool(
                    fetch["USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR"]),
            },
        ),
        # Which of the two a tile is fetched by is decided by rendering rather
        # than by an argument: one body reads by address and the other states a
        # mapping, and a body that read both would carry both.
        USE_TMA_LOAD=bool(fetch["USE_TMA_LOAD"]),
        USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR=bool(
            fetch["USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR"]),
    )
    kernel = _build(source, f"<tensorplay-stax-scaled-{key}>", symbol)
    _KERNEL_MEMO[key] = kernel
    return kernel


def _multiprocessor_count(device) -> int:
    """How many programs to walk the tiles with.

    A tile is walked rather than handed out, so the count of programs is the
    machine's count of multiprocessors: that is how many can be resident at
    once, and walking means the run is longer rather than differently shaped.
    """

    from ..templates.mm_common import num_sms

    return num_sms(device)


def _descriptor_form(a_extents, b_extents, device) -> dict:
    """How this call's tiles may be fetched by descriptor, or by address.

    A descriptor addresses in 32 bits and needs a hardware feature, so it is
    asked about rather than assumed: an operand it cannot name is one it cannot
    describe, and a machine without the feature has no such form to offer.
    Which of the two a descriptor is asked for is not a choice either -- it is
    the layout the operand has.
    """

    from ..templates.mm_common import (
        descriptor_extents_fit, descriptor_offset_fits, device_capability,
    )

    flags = {
        "USE_TMA_LOAD": False,
        "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": False,
    }
    if not (hasattr(tl, "make_tensor_descriptor")
            or hasattr(tl, "_experimental_make_tensor_descriptor")):
        return flags
    major, _minor = device_capability(device)
    if major < 9:
        return flags
    for extents in (a_extents, b_extents):
        rows, inner, block_rows, block_inner = extents
        if not descriptor_extents_fit((rows, inner)):
            return flags
        if not descriptor_offset_fits((rows, inner), (inner, 1)):
            return flags
        if block_rows > rows or block_inner > inner:
            return flags
    flags["USE_TMA_LOAD"] = True
    flags["USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR"] = not hasattr(
        tl, "make_tensor_descriptor"
    )
    return flags


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
        # A factor's rank is what its recipe says it is, and a factor that
        # arrives with a different number of dimensions is a different factor
        # rather than the same numbers: the body would gather it as something
        # it is not.
        if any(
            int(t.dim()) != factor_rank(recipe)
            for t, recipe in ((a_scale, recipe_a), (b_scale, recipe_b))
        ):
            return base_launch(feed)
        # What the tiles may be fetched by, and how many programs walk them, are
        # properties of the machine and the call rather than of the
        # configuration, so they are asked about per call; the body they select
        # is built once and kept.
        num_sms = _multiprocessor_count(a.device)
        fetch = _descriptor_form(
            (m, k, block_m, block_k), (n, k, block_n, block_k), a.device
        )
        kernel = scaled_kernel(
            site, recipe_a, recipe_b, tile_inner, block_m, block_n, block_k,
            group_m, num_sms, fetch,
        )
        out = tp.empty((m, n), dtype=a.dtype, device=a.device)
        # The walk is persistent: a fixed number of programs each take every
        # NUM_SMS'th tile, rather than one program per tile.
        grid = (num_sms, 1, 1)
        # The order is the signature's: the operands, then their extents, then
        # their strides, then the block extents -- the order the body was
        # rendered against, so the two cannot drift apart.
        args = [a, b, a_scale, b_scale]
        if fetch["USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR"]:
            # One mapping per program and per operand, so the workspace is
            # asked for at the size the body indexes it at.
            args.append(tp.empty(
                num_sms * 2 * TMA_SIZE, dtype=tp.uint8, device=a.device))
        args.append(out)
        extents = [int(v) for v in a.shape] + [int(v) for v in b.shape]
        extents += [int(v) for v in a_scale.shape] + [int(v) for v in b_scale.shape]
        extents += [int(v) for v in out.shape]
        strides = [int(v) for v in a.stride()] + [int(v) for v in b.stride()]
        strides += [int(v) for v in a_scale.stride()]
        strides += [int(v) for v in b_scale.stride()]
        strides += [int(v) for v in out.stride()]
        values = {
            "RECIPE_A": recipe_a, "RECIPE_B": recipe_b,
            "TILE_INNER": tile_inner, "ALLOW_TF32": allow_tf32,
            "BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k,
            "GROUP_M": group_m, "NUM_SMS": num_sms, "TMA_SIZE": TMA_SIZE,
            **fetch,
        }
        kernel[grid](
            *args, *extents, *strides,
            **{name: values[name] for name in _SITES[site][3]},
            INDEX_DTYPE=tl.int64,
            num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
        )
        return out

    return launch
