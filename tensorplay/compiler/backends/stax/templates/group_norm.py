"""A fused backward kernel for group normalization, written from a template.

The backward pass of group normalization has three results -- the input
gradient, the weight gradient and the bias gradient -- that share their
reduction work.  A single kernel per group first reduces the two group sums
over the group's elements, then rewrites the input gradient and accumulates
the per-channel weight and bias gradients atomically, so the whole backward
pass is one launch rather than a reduction and a pointwise call.
"""

from ..heuristics.template.base import SymbolicGridFn
from .select_algorithm import TritonTemplate


@SymbolicGridFn
def group_norm_backward_grid(n, c, h, w, meta, *, cdiv):
    """One program per group: the first axis counts the groups."""

    return (n * meta["GROUPS"], 1, 1)


group_norm_backward_template = TritonTemplate(
    name="group_norm_backward",
    grid=group_norm_backward_grid,
    source=r"""
{{def_kernel("GRAD_OUT", "X", "MEAN", "RSTD", "GAMMA", "DGAMMA", "DBETA")}}
    N = {{size("X", 0)}}
    C = {{size("X", 1)}}
    H = {{size("X", 2)}}
    W = {{size("X", 3)}}
    HW = H * W
    CPG = C // GROUPS
    COUNT = CPG * HW
    GROUP_SIZE = CPG * HW
    INV_COUNT = 1.0 / COUNT

    stride_x0 = {{stride("X", 0)}}
    stride_x1 = {{stride("X", 1)}}
    stride_x2 = {{stride("X", 2)}}
    stride_x3 = {{stride("X", 3)}}
    stride_g0 = {{stride("GRAD_OUT", 0)}}
    stride_g1 = {{stride("GRAD_OUT", 1)}}
    stride_g2 = {{stride("GRAD_OUT", 2)}}
    stride_g3 = {{stride("GRAD_OUT", 3)}}
    stride_o0 = {{stride(None, 0)}}
    stride_o1 = {{stride(None, 1)}}
    stride_o2 = {{stride(None, 2)}}
    stride_o3 = {{stride(None, 3)}}

    ng = tl.program_id(0).to(INDEX_DTYPE)
    n = ng // GROUPS
    g = ng % GROUPS
    c_base = g * CPG
    mean_val = tl.load(MEAN + ng)
    rstd_val = tl.load(RSTD + ng)

    ds = 0.0
    db = 0.0
    for off in range(0, GROUP_SIZE, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < GROUP_SIZE
        c_local = idx // HW
        hw_local = idx % HW
        c = c_base + c_local
        h = hw_local // W
        w = hw_local % W
        x_ptr = X + n * stride_x0 + c * stride_x1 + h * stride_x2 + w * stride_x3
        g_ptr = GRAD_OUT + n * stride_g0 + c * stride_g1 + h * stride_g2 + w * stride_g3
        x_val = tl.load(x_ptr, mask=mask, other=0.0)
        g_val = tl.load(g_ptr, mask=mask, other=0.0)
        gamma_val = tl.load(GAMMA + c, mask=mask, other=0.0)
        y = (x_val - mean_val) * rstd_val
        dy = g_val * gamma_val
        ds += tl.sum(tl.where(mask, dy, 0.0))
        db += tl.sum(tl.where(mask, dy * y, 0.0))

    for off in range(0, GROUP_SIZE, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < GROUP_SIZE
        c_local = idx // HW
        hw_local = idx % HW
        c = c_base + c_local
        h = hw_local // W
        w = hw_local % W
        x_ptr = X + n * stride_x0 + c * stride_x1 + h * stride_x2 + w * stride_x3
        g_ptr = GRAD_OUT + n * stride_g0 + c * stride_g1 + h * stride_g2 + w * stride_g3
        x_val = tl.load(x_ptr, mask=mask, other=0.0)
        g_val = tl.load(g_ptr, mask=mask, other=0.0)
        gamma_val = tl.load(GAMMA + c, mask=mask, other=0.0)
        y = (x_val - mean_val) * rstd_val
        dy = g_val * gamma_val
        dx_val = rstd_val * (dy - (ds + y * db) * INV_COUNT)
        no_store = idx < 0
        {{store_output(("n", "c", "h", "w"), "dx_val", "no_store", val_shape=("BLOCK",), indent_width=8)}}
        tl.store(out_ptr0 + (n * stride_o0 + c * stride_o1 + h * stride_o2 + w * stride_o3), dx_val, mask)
        tl.atomic_add(DGAMMA + c, tl.where(mask, g_val * y, 0.0), mask=mask)
        tl.atomic_add(DBETA + c, tl.where(mask, g_val, 0.0), mask=mask)
""",
)
