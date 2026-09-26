"""The convolution tile kernel: a product over the input channels, implicit.

A convolution multiplies an input by a weight and adds the results, once per
output position, so it is a product -- but not one whose operands are laid out
as matrices.  The rows of that product are positions, spread across a batch
and a spatial extent, and the weight for a position is a different plane of the
same tensor for every position.  Rather than gather those planes into matrices
first, the product is done where the data already is: a tile of output
positions is read, and for each of them the input positions it needs are
addressed directly, which is what makes the product *implicit*.

The tiling is therefore the product's own:

  rows        the batch and the output's spatial extent, counted together,
              because a tile does not care which image a position came from
  columns     the output channels of one group
  contraction the input channels of one group

Groups are the third axis of the grid rather than a term in the extents,
because each group is an independent product over its own channels: a tile
belongs to one group and never straddles two.

The one-by-one case is not here.  With a kernel of one in each spatial
dimension the input positions a tile needs are the output positions
themselves, the weight has no spatial plane to select, and the product is
plain enough that the product template measures it better.  So a one-by-one
convolution is expressed as a product, elsewhere, and this kernel declines it.
"""

from __future__ import annotations

import hashlib
import linecache
from typing import Any, Callable, Optional, Tuple

import triton
import triton.language as tl

__all__ = ["CONV_TUNING_VERSION", "conv_launch", "conv_rows"]


#: Salt for a persisted decision: bumped when the kernel body changes, so a
#: stored choice cannot outlive the kernel it named.
CONV_TUNING_VERSION = "conv-tile-1"

#: One memo entry per (form, geometry) so repeated candidate launches reuse a
#: compiled binary instead of rebuilding it.
_KERNEL_MEMO: dict[str, Any] = {}


#: The smallest contraction a tiled product can have.  A block-per-lane
#: multiply-accumulate is defined from sixteen lanes up; below that there is no
#: tile to speak of, and the only honest answer is to decline the call.
MIN_CONTRACTION = 16


#: The three bodies, each a file rather than a string: a kernel is long enough
#: that keeping it inline makes the code and the parts that vary hard to tell
#: apart, and a file lets the varying parts be marked as varying.
_CONV_TILE = ("triton_conv_tile", "_conv_tile_kernel")
_DEPTHWISE = ("triton_depthwise_conv", "_depthwise_conv1d_kernel")
_BWD_INPUT = ("triton_conv2d_bwd_input", "_conv2d_bwd_input_kernel")
_BWD_WEIGHT = ("triton_conv2d_bwd_weight", "_conv2d_bwd_weight_kernel")


def _render(which: Tuple[str, str], operands: dict, outputs: dict,
            constexprs: dict, **flags: Any) -> str:
    """One body, rendered against the operands and the block extents.

    The signature is emitted from the names in ``operands`` and ``outputs``, so
    the parameters a body may refer to and the parameters it is given are the
    same list read twice; the launch then passes the real tensors' extents and
    strides in that order.  A decision the body branches on rather than reads --
    how many groups there are, say -- is a rendering question and is passed as
    one, so it is not a parameter the launch has to supply a value for.
    """

    from ..templates.select_algorithm import KernelArgs, TritonTemplate

    name, symbol = which
    return TritonTemplate.from_file(name, symbol=symbol).render_with(
        KernelArgs(operands, outputs, constexprs), **flags
    )


def _build(source: str, fake_file: str, symbol: str):
    """Define a kernel from generated source, once.

    A jit-decorated function reads its own source back through the linecache
    when it is compiled, and a function that calls another jit function reads
    *that* one's source through the module the caller was defined in.  So the
    text is registered under the name it will be compiled as, and a real
    module is registered to own it -- otherwise the inner function resolves
    against no module at all and fails on the first call rather than on the
    first compile, which is a much worse place to find out.
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


def conv_kernel(block_m: int, block_n: int, block_k: int, kernel_h: int,
                kernel_w: int, unroll: bool):
    """The tile kernel for one geometry, built once and remembered.

    The kernel extent is a constexpr rather than an argument because the taps
    are unrolled into the body: a convolution's inner work is a fixed,
    usually small number of taps, and a loop over a count the compiler already
    knows costs more in scheduling than it saves in code size.
    """

    key = hashlib.sha256(
        f"{CONV_TUNING_VERSION}|{block_m}|{block_n}|{block_k}|{kernel_h}|"
        f"{kernel_w}|{int(unroll)}".encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    # The extents are placeholders: the signature only needs each operand's
    # rank, and the launch passes the call's real extents and strides.  What the
    # rank cannot be is anything but four -- a one-dimensional convolution has
    # no second spatial axis for the body to address.
    plane = (0, 0, 0, 0)
    source = _render(
        _CONV_TILE,
        {
            "X": {"shape": plane, "stride": plane},
            "W": {"shape": plane, "stride": plane},
            "B": {"shape": (0,), "stride": (1,)},
        },
        {"O": {"shape": plane, "stride": plane}},
        {
            "KERNEL_H": int(kernel_h), "KERNEL_W": int(kernel_w),
            "STRIDE_H": 1, "STRIDE_W": 1,
            "PADDING_H": 0, "PADDING_W": 0,
            "DILATION_H": 1, "DILATION_W": 1,
            "GROUP_IN_C": 0, "GROUP_OUT_C": 0,
            "HAS_BIAS": False, "ALLOW_TF32": False, "UNROLL": bool(unroll),
            "BLOCK_M": int(block_m), "BLOCK_N": int(block_n),
            "BLOCK_K": int(block_k),
        },
    )
    kernel = _build(source, f"<tensorplay-stax-conv-{key}>", _CONV_TILE[1])
    _KERNEL_MEMO[key] = kernel
    return kernel


def conv_rows(batch: int, out_h: int, out_w: int) -> int:
    """How many output positions there are, which is the product's row count."""

    return int(batch) * int(out_h) * int(out_w)


def _standard_nchw(tensor, rank: int) -> bool:
    """Whether the extents and strides are the ones the kernel bakes in."""

    if tuple(tensor.shape[1:]) and not tensor.is_contiguous():
        return False
    shape = tuple(int(s) for s in tensor.shape)
    stride = tuple(int(s) for s in tensor.stride())
    if len(shape) != rank:
        return False
    expected = []
    running = 1
    for extent in reversed(shape):
        expected.append(running)
        running *= max(extent, 1)
    return stride == tuple(reversed(expected))


def conv_launch(
    x_spec: Tuple[Optional[int], Any],
    w_spec: Tuple[Optional[int], Any],
    bias_spec: Optional[Tuple[Optional[int], Any]],
    geometry: dict,
    config: dict,
    base_launch: Callable[[list], Any],
    allow_tf32: bool = False,
):
    """Build a launcher running one fixed convolution tile.

    The geometry is everything about the call that is not a choice: the
    kernel, stride, padding, dilation, groups and bias.  The config is the
    choice: the tile shape and the launch geometry around it.  A call whose
    real layout is not the one baked here takes ``base_launch`` instead, which
    is what keeps a launcher from being a promise about a call it has not seen.
    """

    # Only the unit-stride form is claimed.  A strided convolution reads a
    # window whose rows are not adjacent, so the tile cannot walk the input as
    # a run of rows; rather than produce numbers that are wrong in a way nobody
    # would notice, the launcher hands those calls to the operator, which is in
    # the candidate list anyway.
    if tuple(int(v) for v in geometry["stride"]) != (1,) * len(geometry["kernel"]):
        return base_launch
    if int(geometry.get("min_contraction", 1)) < MIN_CONTRACTION:
        return base_launch

    block_m = int(config["BLOCK_M"])
    block_n = int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    kernel_h, kernel_w = geometry["kernel"]
    unroll = bool(geometry.get("unroll", True))
    groups = int(geometry["groups"])
    stride_h, stride_w = geometry["stride"]
    pad_h, pad_w = geometry["padding"]
    dil_h, dil_w = geometry["dilation"]
    has_bias = bias_spec is not None
    rank = len(geometry["kernel"]) + 2
    kernel = conv_kernel(block_m, block_n, block_k, kernel_h, kernel_w, unroll)

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        x = operand(feed, x_spec)
        w = operand(feed, w_spec)
        bias = operand(feed, bias_spec) if has_bias else x
        batch, in_c = (int(v) for v in x.shape[:2])
        out_c, in_c_g, *_ = (int(v) for v in w.shape)
        out_h, out_w = geometry["out_size"]
        if not (
            _standard_nchw(x, rank)
            and _standard_nchw(w, rank)
            and (not has_bias or int(bias.shape[0]) == out_c)
            and in_c == in_c_g * groups
            and out_c % groups == 0
        ):
            return base_launch(feed)
        out = tp.empty(
            (batch, out_c, out_h, out_w), dtype=x.dtype, device=x.device
        )
        group_in_c = in_c // groups
        group_out_c = out_c // groups
        rows = conv_rows(batch, out_h, out_w)
        grid = (
            triton.cdiv(rows, block_m),
            triton.cdiv(group_out_c, block_n),
            groups,
        )
        # The order is the signature's: every operand, then every operand's
        # extents, then every operand's strides, then the block extents -- the
        # order the body was rendered against, so the two cannot drift apart.
        # The bias's extent and stride are passed even when the call has no
        # bias, because the operand is declared either way; a stand-in is
        # bound in its place and the flag says it is not read.
        kernel[grid](
            x, w, bias, out,
            *(int(v) for v in x.shape),
            *(int(v) for v in w.shape), int(bias.shape[0]),
            *(int(v) for v in out.shape),
            *(int(v) for v in x.stride()),
            *(int(v) for v in w.stride()), int(bias.stride(0)),
            *(int(v) for v in out.stride()),
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, group_in_c, group_out_c,
            has_bias, allow_tf32, unroll, block_m, block_n, block_k,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch


# ---------------------------------------------------------------------------
# depthwise
# ---------------------------------------------------------------------------

def depthwise_kernel(block_n: int, block_l: int, block_c: int):
    """The depthwise kernel for one tiling, built once and remembered."""

    key = hashlib.sha256(
        f"{CONV_TUNING_VERSION}|dw|{block_n}|{block_l}|{block_c}".encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    # The geometry is a flag rather than an argument for the same reason the
    # taps are: the body has one axis, so the counts are folded in.
    plane = (0, 0, 0)
    source = _render(
        _DEPTHWISE,
        {
            "X": {"shape": plane, "stride": plane},
            "W": {"shape": (0, 0), "stride": (0, 1)},
            "B": {"shape": (0,), "stride": (1,)},
        },
        {"O": {"shape": plane, "stride": plane}},
        {
            "KERNEL": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
            "HAS_BIAS": False,
            "BLOCK_N": int(block_n), "BLOCK_L": int(block_l),
            "BLOCK_C": int(block_c),
        },
    )
    kernel = _build(source, f"<tensorplay-stax-depthwise-{key}>", _DEPTHWISE[1])
    _KERNEL_MEMO[key] = kernel
    return kernel


def depthwise_launch(x_spec, w_spec, bias_spec, geometry: dict, config: dict,
                     base_launch: Callable[[list], Any]):
    """Build a launcher for the depthwise form: channels-last, no contraction."""

    block_n = int(config["BLOCK_N"])
    block_l = int(config["BLOCK_L"])
    block_c = int(config["BLOCK_C"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    kernel_w = int(geometry["kernel"][0])
    stride_w = int(geometry["stride"][0])
    pad_w = int(geometry["padding"][0])
    dil_w = int(geometry["dilation"][0])
    has_bias = bias_spec is not None
    kernel = depthwise_kernel(block_n, block_l, block_c)

    def operand(feed: list, spec) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        x = operand(feed, x_spec)
        w = operand(feed, w_spec)
        bias = operand(feed, bias_spec) if has_bias else x
        n, length, channels = (int(v) for v in x.shape)
        out_length = (length + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
        if not (x.dim() == 3 and w.dim() == 2 and int(w.shape[0]) == channels
                and int(w.shape[1]) == kernel_w):
            return base_launch(feed)
        out = tp.empty((n, out_length, channels), dtype=x.dtype, device=x.device)
        grid = (
            triton.cdiv(n, block_n),
            triton.cdiv(out_length, block_l),
            triton.cdiv(channels, block_c),
        )
        # The order is the signature's, as in the forward: the operands, then
        # their extents, then their strides, then the block extents.  The bias
        # is declared whether or not the call has one, so its extent and stride
        # are passed either way and the flag says it is not read.
        kernel[grid](
            x, w, bias, out,
            n, length, channels,
            *(int(v) for v in w.shape), int(bias.shape[0]),
            n, out_length, channels,
            *(int(v) for v in x.stride()),
            *(int(v) for v in w.stride()), int(bias.stride(0)),
            *(int(v) for v in out.stride()),
            kernel_w, stride_w, pad_w, dil_w, has_bias,
            block_n, block_l, block_c,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch


# ---------------------------------------------------------------------------
# the gradients
# ---------------------------------------------------------------------------

def bwd_kernel(kind: str, block_m: int, block_n: int, block_k: int, groups: int):
    """One gradient kernel for one tiling, built once and remembered.

    The two directions are two bodies rather than one with a flag: they contract
    over different extents, address their operands in opposite orders, and only
    one of them has to invert the forward's window.  A flag would leave both
    shapes of work in one function and pick between them at every step.
    """

    key = hashlib.sha256(
        f"{CONV_TUNING_VERSION}|{kind}|{block_m}|{block_n}|{block_k}"
        f"|{groups}".encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    if kind == "input":
        which = _BWD_INPUT
        operands = {
            "DY": {"shape": (0, 0, 0, 0), "stride": (0, 0, 0, 0)},
            "W": {"shape": (0, 0, 0, 0), "stride": (0, 0, 0, 0)},
        }
        outputs = {"DX": {"shape": (0, 0, 0, 0), "stride": (0, 0, 0, 0)}}
    else:
        which = _BWD_WEIGHT
        operands = {
            "DY": {"shape": (0, 0, 0, 0), "stride": (0, 0, 0, 0)},
            "X": {"shape": (0, 0, 0, 0), "stride": (0, 0, 0, 0)},
        }
        outputs = {"DW": {"shape": (0, 0, 0, 0), "stride": (0, 0, 0, 0)}}
    source = _render(
        which,
        operands,
        outputs,
        {
            "KERNEL_H": 1, "KERNEL_W": 1,
            "STRIDE_H": 1, "STRIDE_W": 1,
            "PADDING_H": 0, "PADDING_W": 0,
            "DILATION_H": 1, "DILATION_W": 1,
            "GROUP_IN_C": 0, "GROUP_OUT_C": 0,
            "BLOCK_M": int(block_m), "BLOCK_N": int(block_n),
            "BLOCK_K": int(block_k),
            "ALLOW_TF32": False,
        },
        GROUPS=int(groups),
    )
    kernel = _build(source, f"<tensorplay-stax-convbwd-{key}>", which[1])
    _KERNEL_MEMO[key] = kernel
    return kernel


def _grad_geometry(meta_geometry: dict, in_size, out_c: int, in_c: int):
    kernel = meta_geometry["kernel"]
    stride = meta_geometry["stride"]
    padding = meta_geometry["padding"]
    dilation = meta_geometry["dilation"]
    groups = meta_geometry["groups"]
    in_h, in_w = int(in_size[0]), int(in_size[1])
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    return {
        "kernel": kernel, "stride": stride, "padding": padding,
        "dilation": dilation, "groups": groups,
        "in_size": (in_h, in_w), "out_size": (out_h, out_w),
        "out_channels": out_c, "in_channels": in_c,
        # Each gradient contracts over the other axis, and a group narrower
        # than a block is not a tile, so the call goes to the operator.
        "min_contraction": min(out_c, in_c) // groups if groups else 0,
    }


def conv_bwd_input_launch(
    dy_spec, w_spec, config, geometry, base_launch, allow_tf32: bool = False
):
    """Build a launcher for the input's gradient."""

    if tuple(int(v) for v in geometry["stride"]) != (1,) * len(geometry["kernel"]):
        return base_launch
    if int(geometry.get("min_contraction", 1)) < MIN_CONTRACTION:
        return base_launch
    block_m, block_n = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    kernel_h, kernel_w = (int(v) for v in geometry["kernel"])
    stride_h, stride_w = (int(v) for v in geometry["stride"])
    pad_h, pad_w = (int(v) for v in geometry["padding"])
    dil_h, dil_w = (int(v) for v in geometry["dilation"])
    groups = int(geometry["groups"])
    in_h, in_w = geometry["in_size"]
    out_c, in_c = geometry["out_channels"], geometry["in_channels"]
    kernel = bwd_kernel("input", block_m, block_n, block_k, groups)

    def operand(feed: list, spec) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        dy, w = operand(feed, dy_spec), operand(feed, w_spec)
        batch = int(dy.shape[0])
        if int(dy.shape[1]) != out_c or int(w.shape[0]) != out_c or int(w.shape[1]) * groups != in_c:
            return base_launch(feed)
        dx = tp.empty((batch, in_c, in_h, in_w), dtype=dy.dtype, device=dy.device)
        group_in_c, group_out_c = in_c // groups, out_c // groups
        grid = (
            triton.cdiv(conv_rows(batch, in_h, in_w), block_m),
            triton.cdiv(group_in_c, block_n),
            # One group is walked by no program axis: the body does not read it.
            1 if groups == 1 else groups,
        )
        # The order is the signature's: the operands, then their extents, then
        # their strides, then the block extents -- the order the body was
        # rendered against, so the two cannot drift apart.
        kernel[grid](
            dy, w, dx,
            *(int(v) for v in dy.shape),
            *(int(v) for v in w.shape),
            *(int(v) for v in dx.shape),
            *(int(v) for v in dy.stride()),
            *(int(v) for v in w.stride()),
            *(int(v) for v in dx.stride()),
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, group_in_c, group_out_c,
            block_m, block_n, block_k, allow_tf32,
            num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
        )
        return dx

    return launch


def conv_bwd_weight_launch(
    dy_spec, x_spec, config, geometry, base_launch, allow_tf32: bool = False
):
    """Build a launcher for the weight's gradient."""

    if tuple(int(v) for v in geometry["stride"]) != (1,) * len(geometry["kernel"]):
        return base_launch
    if int(geometry.get("min_contraction", 1)) < MIN_CONTRACTION:
        return base_launch
    block_m, block_n = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    kernel_h, kernel_w = (int(v) for v in geometry["kernel"])
    stride_h, stride_w = (int(v) for v in geometry["stride"])
    pad_h, pad_w = (int(v) for v in geometry["padding"])
    dil_h, dil_w = (int(v) for v in geometry["dilation"])
    groups = int(geometry["groups"])
    out_c, in_c = geometry["out_channels"], geometry["in_channels"]
    kernel = bwd_kernel("weight", block_m, block_n, block_k, groups)

    def operand(feed: list, spec) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        dy, x = operand(feed, dy_spec), operand(feed, x_spec)
        if int(x.shape[1]) != in_c or int(dy.shape[1]) != out_c:
            return base_launch(feed)
        dw = tp.empty((out_c, in_c // groups, kernel_h, kernel_w), dtype=dy.dtype, device=dy.device)
        group_in_c, group_out_c = in_c // groups, out_c // groups
        grid = (triton.cdiv(group_out_c, block_m), triton.cdiv(group_in_c, block_n),
                1 if groups == 1 else groups)
        # The order is the signature's, as in the input's gradient: the
        # operands, then their extents, then their strides, then the block
        # extents.
        kernel[grid](
            dy, x, dw,
            *(int(v) for v in dy.shape),
            *(int(v) for v in x.shape),
            *(int(v) for v in dw.shape),
            *(int(v) for v in dy.stride()),
            *(int(v) for v in x.stride()),
            *(int(v) for v in dw.stride()),
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, group_in_c, group_out_c,
            block_m, block_n, block_k, allow_tf32,
            num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
        )
        return dw

    return launch
