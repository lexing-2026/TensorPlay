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


_KERNEL_SOURCE = '''
import triton
import triton.language as tl


@triton.jit
def _conv_tile_kernel(
    x_ptr, w_ptr, o_ptr, b_ptr,
    n, c, h, width, out_c, out_h, out_width,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    stride_bn,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    DILATION_H: tl.constexpr, DILATION_W: tl.constexpr,
    GROUPS: tl.constexpr, GROUP_IN_C: tl.constexpr, GROUP_OUT_C: tl.constexpr,
    HAS_BIAS: tl.constexpr, ALLOW_TF32: tl.constexpr, UNROLL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """One tile of output positions against one tile of output channels."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    # The rows are positions: which image, and where in it.  A position that
    # runs past the end of the image is masked off below.
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    spatial = out_h * out_width
    idx_n = rows // spatial
    within = rows % spatial
    idx_y_h = within // out_width
    idx_y_w = within % out_width

    # The columns are output channels, counted within the group.
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_channel = pid_g * GROUP_OUT_C + cols

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # The weight, addressed the other way round: the tile is the contraction
    # against the output channels, and the weight is stored output-major, so
    # the load carries the transpose rather than the product.  It is reloaded
    # per step of the contraction, because the contraction is exactly what
    # steps.
    # The channel index already carries the group, and the weight is stored
    # with all the groups' output channels in one run, so the group must not
    # be added to the base as well.
    w_ptrs = w_ptr + (out_channel * stride_wo)[None, :]
    mask_w = (tl.arange(0, BLOCK_K) < GROUP_IN_C)[:, None] & (cols < GROUP_OUT_C)[None, :]

    # Each group reduces over its own channels, so the contraction is counted
    # from the start of the group rather than from the start of the tensor.
    x_base = x_ptr + idx_n[:, None] * stride_xn + pid_g * GROUP_IN_C * stride_xc
    for k in range(0, tl.cdiv(GROUP_IN_C, BLOCK_K)):
        idx_x_c = k * BLOCK_K + tl.arange(0, BLOCK_K)
        if UNROLL:
            for ky in tl.static_range(KERNEL_H):
                idx_x_h = (
                    idx_y_h - PADDING_H + ky * DILATION_H
                ) * STRIDE_H
                for kx in tl.static_range(KERNEL_W):
                    idx_x_w = (
                        idx_y_w - PADDING_W + kx * DILATION_W
                    ) * STRIDE_W
                    # The tap selects a plane of the weight as well as an
                    # offset into the input: the two are the same tap, and a
                    # kernel that moved one without the other would read every
                    # tap against a single plane.
                    tap = ky * stride_wkh + kx * stride_wkw
                    matrix_w = tl.load(
                        w_ptrs + (idx_x_c * stride_wi)[:, None] + tap,
                        mask=mask_w, other=0.0,
                    )
                    x_ptrs = x_base + (
                        (idx_x_c * stride_xc)[None, :]
                        + (idx_x_h * stride_xh)[:, None]
                        + (idx_x_w * stride_xw)[:, None]
                    )
                    mask_x = (
                        (idx_n < n)[:, None]
                        & (idx_x_h >= 0)[:, None]
                        & (idx_x_h < h)[:, None]
                        & (idx_x_w >= 0)[:, None]
                        & (idx_x_w < width)[:, None]
                        & (idx_x_c < GROUP_IN_C)[None, :]
                    )
                    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
                    acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
        else:
            # The kernel extent folded into the contraction: one lane per tap,
            # so a kernel wider than the contraction block is stepped once.
            for ky in range(KERNEL_H):
                idx_x_h = (idx_y_h - PADDING_H + ky * DILATION_H) * STRIDE_H
                for kx in range(KERNEL_W):
                    idx_x_w = (idx_y_w - PADDING_W + kx * DILATION_W) * STRIDE_W
                    matrix_w = tl.load(
                        w_ptrs + (idx_x_c * stride_wi)[:, None]
                        + ky * stride_wkh + kx * stride_wkw,
                        mask=mask_w, other=0.0,
                    )
                    x_ptrs = x_base + (
                        (idx_x_c * stride_xc)[None, :]
                        + (idx_x_h * stride_xh)[:, None]
                        + (idx_x_w * stride_xw)[:, None]
                    )
                    mask_x = (
                        (idx_n < n)[:, None]
                        & (idx_x_h >= 0)[:, None]
                        & (idx_x_h < h)[:, None]
                        & (idx_x_w >= 0)[:, None]
                        & (idx_x_w < width)[:, None]
                        & (idx_x_c < GROUP_IN_C)[None, :]
                    )
                    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
                    acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)

    if HAS_BIAS:
        acc += tl.load(b_ptr + out_channel * stride_bn, mask=cols < GROUP_OUT_C,
                       other=0.0)[None, :]

    o_ptrs = o_ptr + (
        (idx_n * stride_on)[:, None]
        + (out_channel * stride_oc)[None, :]
        + (idx_y_h * stride_oh)[:, None]
        + (idx_y_w * stride_ow)[:, None]
    )
    mask_o = (
        (idx_n < n)[:, None]
        & (cols < GROUP_OUT_C)[None, :]
        & (idx_y_h < out_h)[:, None]
        & (idx_y_w < out_width)[:, None]
    )
    tl.store(o_ptrs, acc, mask=mask_o)
'''


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
    source = _KERNEL_SOURCE
    fake_file = f"<tensorplay-stax-conv-{key}>"
    # The jit decorator reads the decorated function's source through the
    # linecache, so the text has to be registered under the name the exec
    # below will compile it as.
    linecache.cache[fake_file] = (
        len(source), None, source.splitlines(True), fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    kernel = namespace["_conv_tile_kernel"]
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
        batch, in_c, in_h, in_w = (int(v) for v in x.shape)
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
        kernel[grid](
            x, w, out, bias,
            batch, in_c, in_h, in_w, out_c, out_h, out_w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            1 if not has_bias else int(bias.stride(0)),
            KERNEL_H=kernel_h, KERNEL_W=kernel_w,
            STRIDE_H=stride_h, STRIDE_W=stride_w,
            PADDING_H=pad_h, PADDING_W=pad_w,
            DILATION_H=dil_h, DILATION_W=dil_w,
            GROUPS=groups, GROUP_IN_C=group_in_c, GROUP_OUT_C=group_out_c,
            HAS_BIAS=has_bias, ALLOW_TF32=allow_tf32, UNROLL=unroll,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            EVEN_K=(group_in_c % block_k == 0),
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch


# ---------------------------------------------------------------------------
# depthwise
# ---------------------------------------------------------------------------

_DEPTHWISE_SOURCE = '''
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv1d_kernel(
    x_ptr, w_ptr, o_ptr, b_ptr,
    n, length, channels, out_length,
    stride_xn, stride_xl, stride_xc,
    stride_wn, stride_wc,
    stride_on, stride_ol, stride_oc,
    stride_bn,
    KERNEL: tl.constexpr, STRIDE: tl.constexpr, PADDING: tl.constexpr,
    DILATION: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Each channel of each position, reduced on its own.

    A depthwise convolution is not a product: no output channel is a sum over
    input channels, so there is nothing to contract and nothing for a matrix
    multiply to do.  What it is instead is a multiply-accumulate per channel,
    so the tile is over the three things the work steps over -- images,
    positions along the axis, and channels -- and each lane keeps its own
    accumulator.

    The layout is channels-last because a channel's positions are then
    contiguous, which is what makes the innermost block worth having.
    """
    pid_n = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_c = tl.program_id(2)

    idx_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    lane = tl.arange(0, BLOCK_L)

    acc = tl.zeros([BLOCK_N, BLOCK_L, BLOCK_C], dtype=tl.float32)
    for k in range(0, KERNEL):
        idx_l = (pid_l * BLOCK_L + lane - PADDING + k * DILATION) * STRIDE
        mask_l = (idx_l >= 0) & (idx_l < length)
        mask = (
            (idx_n < n)[:, None, None]
            & (idx_c < channels)[None, None, :]
            & mask_l[None, :, None]
        )
        offset = (
            (idx_n * stride_xn)[:, None, None]
            + (idx_l * stride_xl)[None, :, None]
            + (idx_c * stride_xc)[None, None, :]
        )
        value = tl.load(x_ptr + offset, mask=mask, other=0.0)
        tap = tl.load(
            w_ptr + idx_c * stride_wc + k * stride_wn,
            mask=idx_c < channels, other=0.0,
        )
        acc += value * tap[None, None, :]

    if HAS_BIAS:
        acc += tl.load(b_ptr + idx_c * stride_bn, mask=idx_c < channels, other=0.0)[None, None, :]

    out_l = (pid_l * BLOCK_L + lane) * STRIDE
    out_mask = (
        (idx_n < n)[:, None, None]
        & (idx_c < channels)[None, None, :]
        & (out_l < out_length)[None, :, None]
    )
    tl.store(
        o_ptr
        + (idx_n * stride_on)[:, None, None]
        + (out_l * stride_ol)[None, :, None]
        + (idx_c * stride_oc)[None, None, :],
        acc,
        mask=out_mask,
    )
'''


def depthwise_kernel(block_n: int, block_l: int, block_c: int):
    """The depthwise kernel for one tiling, built once and remembered."""

    key = hashlib.sha256(
        f"{CONV_TUNING_VERSION}|dw|{block_n}|{block_l}|{block_c}".encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    source = _DEPTHWISE_SOURCE
    fake_file = f"<tensorplay-stax-depthwise-{key}>"
    linecache.cache[fake_file] = (
        len(source), None, source.splitlines(True), fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    kernel = namespace["_depthwise_conv1d_kernel"]
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
        kernel[grid](
            x, w, out, bias,
            n, length, channels, out_length,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            1 if not has_bias else int(bias.stride(0)),
            KERNEL=kernel_w, STRIDE=stride_w, PADDING=pad_w, DILATION=dil_w,
            HAS_BIAS=has_bias,
            BLOCK_N=block_n, BLOCK_L=block_l, BLOCK_C=block_c,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch


# ---------------------------------------------------------------------------
# the gradients
# ---------------------------------------------------------------------------

_BWD_SOURCE = '''
import triton
import triton.language as tl


@triton.jit
def _tap_origin(coord, offset, out_extent, stride):
    """Where a tap lands in the output, and whether it lands at all.

    A tap reaches an input position only when the distance back to the origin
    divides by the stride exactly.  The distance is clamped before the
    division rather than after, because a negative dividend and a truncating
    division disagree about the sign, and a kernel that guesses there is wrong
    only along the padded border -- which is exactly the part nobody checks.
    """
    distance = coord + offset
    positive = distance >= 0
    safe = tl.maximum(distance, 0)
    origin = safe // stride
    return origin, positive & (origin < out_extent) & ((safe % stride) == 0)


@triton.jit
def _conv2d_bwd_input_kernel(
    dy_ptr, w_ptr, dx_ptr,
    n, h, width, in_c, out_c, out_h, out_width,
    stride_dyn, stride_dyc, stride_dyh, stride_dyw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_dxn, stride_dxc, stride_dxh, stride_dxw,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    DILATION_H: tl.constexpr, DILATION_W: tl.constexpr,
    GROUPS: tl.constexpr, GROUP_IN_C: tl.constexpr, GROUP_OUT_C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """The input's gradient: the sum of the windows that read each position.

    Rows are input positions and columns are input channels, so this is the
    forward's product with the contraction moved: the forward contracted over
    input channels, and this contracts over output channels instead, once per
    tap.  For a given tap the output position a row was read at is solved for
    rather than stepped to, which is the whole difference between the two
    directions -- one of them walks the window, the other has to invert it.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    spatial = h * width
    idx_n = rows // spatial
    within = rows % spatial
    idx_h = within // width
    idx_w = within % width
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_channel = pid_g * GROUP_IN_C + cols
    mask_col = cols < GROUP_IN_C

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(GROUP_OUT_C, BLOCK_K)):
        lane = tl.arange(0, BLOCK_K)
        out_channel = pid_g * GROUP_OUT_C + k * BLOCK_K + lane
        mask_k = lane < GROUP_OUT_C - k * BLOCK_K
        w_ptrs = (
            w_ptr
            + (lane * stride_wo)[:, None]
            + (in_channel * stride_wi)[None, :]
        )
        mask_w = mask_k[:, None] & mask_col[None, :]
        for ky in tl.static_range(KERNEL_H):
            oh, ok_h = _tap_origin(
                idx_h, PADDING_H - ky * DILATION_H, out_h, STRIDE_H
            )
            for kx in tl.static_range(KERNEL_W):
                ow, ok_w = _tap_origin(
                    idx_w, PADDING_W - kx * DILATION_W, out_width, STRIDE_W
                )
                matrix_w = tl.load(
                    w_ptrs + ky * stride_wkh + kx * stride_wkw,
                    mask=mask_w, other=0.0,
                )
                keep = (idx_n < n) & ok_h & ok_w
                dy_ptrs = dy_ptr + (
                    (idx_n * stride_dyn)[:, None]
                    + (oh * stride_dyh)[:, None]
                    + (ow * stride_dyw)[:, None]
                    + (out_channel * stride_dyc)[None, :]
                )
                matrix_dy = tl.load(
                    dy_ptrs, mask=keep[:, None] & mask_k[None, :], other=0.0
                )
                acc += tl.dot(matrix_dy, matrix_w)
    tl.store(
        dx_ptr + (
            (idx_n * stride_dxn)[:, None]
            + (idx_h * stride_dxh)[:, None]
            + (idx_w * stride_dxw)[:, None]
            + (in_channel * stride_dxc)[None, :]
        ),
        acc,
        mask=(idx_n < n)[:, None] & mask_col[None, :],
    )


@triton.jit
def _conv2d_bwd_weight_kernel(
    dy_ptr, x_ptr, dw_ptr,
    n, h, width, in_c, out_c, out_h, out_width,
    stride_dyn, stride_dyc, stride_dyh, stride_dyw,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_dwon, stride_dwoc, stride_dwoi, stride_dwoh,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    DILATION_H: tl.constexpr, DILATION_W: tl.constexpr,
    GROUPS: tl.constexpr, GROUP_IN_C: tl.constexpr, GROUP_OUT_C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """The weight's gradient: one product per tap, over the output's positions.

    Every tap is a separate product contracting the output's positions, and
    they differ only in which input position each output position read -- which
    is known here without inverting anything, because the forward's window is
    walked forwards here.  So the tap indexes the weight rather than the
    accumulation: one accumulator is reused per tap and stored per tap, because
    a kernel-wide tile per tap would need a register per tap and the number of
    taps is the kernel's area.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    lane_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lane_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_channel = pid_g * GROUP_OUT_C + lane_m
    in_channel = pid_g * GROUP_IN_C + lane_n
    mask_m = lane_m < GROUP_OUT_C
    mask_n = lane_n < GROUP_IN_C
    positions = n * out_h * out_width

    for ky in tl.static_range(KERNEL_H):
        for kx in tl.static_range(KERNEL_W):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(0, tl.cdiv(positions, BLOCK_K)):
                flat = k * BLOCK_K + tl.arange(0, BLOCK_K)
                idx_n = flat // (out_h * out_width)
                rest = flat % (out_h * out_width)
                oh = rest // out_width
                ow = rest % out_width
                idx_h = oh * STRIDE_H - PADDING_H + ky * DILATION_H
                idx_w = ow * STRIDE_W - PADDING_W + kx * DILATION_W
                keep = (
                    (flat < positions)
                    & (idx_n < n)
                    & (idx_h >= 0) & (idx_h < h)
                    & (idx_w >= 0) & (idx_w < width)
                )
                dy_ptrs = dy_ptr + (
                    (idx_n * stride_dyn)[:, None]
                    + (oh * stride_dyh)[:, None]
                    + (ow * stride_dyw)[:, None]
                    + (out_channel * stride_dyc)[None, :]
                )
                matrix_dy = tl.load(
                    dy_ptrs, mask=keep[:, None] & mask_m[None, :], other=0.0
                )
                x_ptrs = x_ptr + (
                    (idx_n * stride_xn)[:, None]
                    + (idx_h * stride_xh)[:, None]
                    + (idx_w * stride_xw)[:, None]
                    + (in_channel * stride_xc)[None, :]
                )
                matrix_x = tl.load(
                    x_ptrs, mask=keep[:, None] & mask_n[None, :], other=0.0
                )
                acc += tl.dot(matrix_dy, tl.trans(matrix_x))
            tl.store(
                dw_ptr + (
                    out_channel * stride_dwon
                    + in_channel * stride_dwoc
                    + ky * stride_dwoi
                    + kx * stride_dwoh
                ),
                acc,
                mask=mask_m & mask_n,
            )
'''


def bwd_kernel(kind: str, block_m: int, block_n: int, block_k: int):
    """One gradient kernel for one tiling, built once and remembered."""

    key = hashlib.sha256(
        f"{CONV_TUNING_VERSION}|{kind}|{block_m}|{block_n}|{block_k}".encode()
    ).hexdigest()[:16]
    cached = _KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    source = _BWD_SOURCE
    fake_file = f"<tensorplay-stax-convbwd-{key}>"
    linecache.cache[fake_file] = (
        len(source), None, source.splitlines(True), fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    kernel = namespace[
        "_conv2d_bwd_input_kernel" if kind == "input" else "_conv2d_bwd_weight_kernel"
    ]
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
    }


def conv_bwd_input_launch(dy_spec, w_spec, config, geometry, base_launch):
    """Build a launcher for the input's gradient."""

    if tuple(int(v) for v in geometry["stride"]) != (1,) * len(geometry["kernel"]):
        return base_launch
    block_m, block_n = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    kernel_h, kernel_w = (int(v) for v in geometry["kernel"])
    stride_h, stride_w = (int(v) for v in geometry["stride"])
    pad_h, pad_w = (int(v) for v in geometry["padding"])
    dil_h, dil_w = (int(v) for v in geometry["dilation"])
    groups = int(geometry["groups"])
    in_h, in_w = geometry["in_size"]
    out_h, out_w = geometry["out_size"]
    out_c, in_c = geometry["out_channels"], geometry["in_channels"]
    kernel = bwd_kernel("input", block_m, block_n, block_k)

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
            groups,
        )
        kernel[grid](
            dy, w, dx,
            batch, in_h, in_w, in_c, out_c, out_h, out_w,
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            KERNEL_H=kernel_h, KERNEL_W=kernel_w,
            STRIDE_H=stride_h, STRIDE_W=stride_w,
            PADDING_H=pad_h, PADDING_W=pad_w,
            DILATION_H=dil_h, DILATION_W=dil_w,
            GROUPS=groups, GROUP_IN_C=group_in_c, GROUP_OUT_C=group_out_c,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
        )
        return dx

    return launch


def conv_bwd_weight_launch(dy_spec, x_spec, config, geometry, base_launch):
    """Build a launcher for the weight's gradient."""

    if tuple(int(v) for v in geometry["stride"]) != (1,) * len(geometry["kernel"]):
        return base_launch
    block_m, block_n = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    kernel_h, kernel_w = (int(v) for v in geometry["kernel"])
    stride_h, stride_w = (int(v) for v in geometry["stride"])
    pad_h, pad_w = (int(v) for v in geometry["padding"])
    dil_h, dil_w = (int(v) for v in geometry["dilation"])
    groups = int(geometry["groups"])
    in_h, in_w = geometry["in_size"]
    out_h, out_w = geometry["out_size"]
    out_c, in_c = geometry["out_channels"], geometry["in_channels"]
    kernel = bwd_kernel("weight", block_m, block_n, block_k)

    def operand(feed: list, spec) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        dy, x = operand(feed, dy_spec), operand(feed, x_spec)
        batch = int(dy.shape[0])
        if int(x.shape[1]) != in_c or int(dy.shape[1]) != out_c:
            return base_launch(feed)
        dw = tp.empty((out_c, in_c // groups, kernel_h, kernel_w), dtype=dy.dtype, device=dy.device)
        group_in_c, group_out_c = in_c // groups, out_c // groups
        grid = (triton.cdiv(group_out_c, block_m), triton.cdiv(group_in_c, block_n), groups)
        kernel[grid](
            dy, x, dw,
            batch, in_h, in_w, in_c, out_c, out_h, out_w,
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dw.stride(0), dw.stride(1), dw.stride(2), dw.stride(3),
            KERNEL_H=kernel_h, KERNEL_W=kernel_w,
            STRIDE_H=stride_h, STRIDE_W=stride_w,
            PADDING_H=pad_h, PADDING_W=pad_w,
            DILATION_H=dil_h, DILATION_W=dil_w,
            GROUPS=groups, GROUP_IN_C=group_in_c, GROUP_OUT_C=group_out_c,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=int(config["num_warps"]), num_stages=int(config["num_stages"]),
        )
        return dw

    return launch
