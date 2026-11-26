"""Fused cross entropy over a vocabulary-projection logits buffer.

The vocabulary projection is the widest single tensor in a language-model
training step: ``[tokens, vocab]`` logits are as large as everything else put
together once the vocabulary reaches tens of thousands.  The naive loss path
touches that buffer many times:

    log_softmax -> [tokens, vocab]      (second full-size buffer)
    nll_loss                            (a third share for its backward)

and, under autocast, an fp32 upcast on the way in, so a bf16 training step
pays an fp32 read, two fp32 writes, one fp32 read for the loss, plus fp32
reads/writes in the backward.  On a 24 GiB card at 4096 tokens x 65536 classes
that is ~2.5 GiB of transient workspace and ~4.5 GiB of memory traffic for
one loss.

This module answers the same question with the logits buffer as the only
full-size tensor:

* forward reads the logits twice (row max, then ``sum(exp(x - max))``) and
  stores only two tiny per-row vectors, ``max`` and ``sum``;
* backward turns the logits buffer into the gradient in place,
  ``dlogits = (p - onehot) * grad_out`` with ``p = exp(x - max) / sum``.

No probability buffer, no ``log_softmax`` output, no fp32 upcast.  The
reduction runs in fp32 registers, so the result matches the fp32 reference:
``log(sum(exp(x - max))) + max - x[t]`` is exactly what ``log_softmax``
followed by ``nll_loss`` computes, and the per-row sum is accumulated in
fp32 the same way.

The kernels are written in Triton, which TensorPlay already depends on for
the compiler backends, so no rebuild is needed for the fused path.

Scope: this is an *eager-path* optimization.  Inside a captured region
(``tp.compile``) a call falls back to the composed reference, because the graph
replays traced operations and would freeze the two per-row scratch vectors at
their capture-time values.  Compile the trunk and apply the loss eagerly around
it -- which is also how Megatron runs its loss -- and the fused path serves the
step.

Because backward overwrites the logits, the entry points here are destructive:
they are meant for the case where the logits were produced for this loss and
nothing else reads them.  :func:`tensorplay.nn.functional.linear_cross_entropy`
is the safe public face of this: it owns the buffer it fuses.
"""

from __future__ import annotations

import tensorplay as tp
from tensorplay.graph import capturing as _capturing

__all__ = [
    "fused_cross_entropy",
    "fused_linear_cross_entropy",
    "is_eligible",
]


try:  # Triton is a Linux/x86-64 dependency; keep CPU-only installs importable.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


#: Logits dtypes the kernels read.  fp32 is included because a non-autocast
#: fp32 model reaches the same code path.
_LOGIT_DTYPES = (tp.bfloat16, tp.float16, tp.float32)

#: Elements one program walks per step.  Measured flat from 1024 to 8192 on an
#: RTX 3090; 2048 keeps the loop short and the masking simple.
_BLOCK = 2048


def is_eligible(logits, target, ignore_index: int = -100) -> bool:
    """Whether :func:`fused_cross_entropy` can serve these arguments."""
    if triton is None:
        return False
    if not (logits.is_cuda if hasattr(logits, "is_cuda") else False):
        return False
    if logits.dim() != 2 or target.dim() != 1:
        return False
    if logits.shape[0] != target.shape[0]:
        return False
    if logits.dtype not in _LOGIT_DTYPES:
        return False
    if target.dtype not in (tp.int64, tp.int32):
        return False
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        return False
    return True


if triton is not None:

    @triton.jit
    def _ce_fwd_kernel(logits, tgt, max_out, sum_out, V, BLOCK: tl.constexpr,
                       IGNORE: tl.constexpr):
        """Row max, then row ``sum(exp(x - max))``; no full-size writes."""
        row = tl.program_id(0)
        base = row * V
        keep = tl.load(tgt + row) != IGNORE

        peak = tl.full((), -float("inf"), tl.float32)
        for off in tl.range(0, V, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            x = tl.load(logits + base + offs, mask=offs < V,
                        other=-float("inf")).to(tl.float32)
            peak = tl.maximum(peak, tl.max(x))

        total = tl.full((), 0.0, tl.float32)
        for off in tl.range(0, V, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            x = tl.load(logits + base + offs, mask=offs < V,
                        other=-float("inf")).to(tl.float32)
            total += tl.sum(tl.exp(x - peak))

        # Ignored rows keep a neutral (0, 1) pair so the backward scale is 0
        # and no exp() is evaluated against a shifted range.
        tl.store(max_out + row, tl.where(keep, peak, 0.0))
        tl.store(sum_out + row, tl.where(keep, total, 1.0))

    @triton.jit
    def _ce_bwd_kernel(logits, tgt, max_in, sum_in, grad_in, V,
                       BLOCK: tl.constexpr, IGNORE: tl.constexpr,
                       SCALAR: tl.constexpr):
        """``logits <- (softmax(logits) - onehot) * grad_in`` in place."""
        row = tl.program_id(0)
        base = row * V
        target = tl.load(tgt + row)
        keep = target != IGNORE
        peak = tl.load(max_in + row)
        inv = 1.0 / tl.load(sum_in + row)
        g = tl.load(grad_in) if SCALAR else tl.load(grad_in + row)
        scale = tl.where(keep, g, 0.0)

        for off in tl.range(0, V, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            mask = offs < V
            x = tl.load(logits + base + offs, mask=mask, other=0.0).to(tl.float32)
            d = tl.exp(x - peak) * inv * scale
            d = tl.where(offs == target, d - scale, d)
            tl.store(logits + base + offs, d.to(logits.dtype.element_ty),
                     mask=mask)


class _FusedCrossEntropy(tp.autograd.Function):
    """Per-row CE whose backward overwrites the logits with ``dlogits``."""

    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        rows, vocab = logits.shape
        maxes = tp.zeros(rows, device=logits.device, dtype=tp.float32)
        sums = tp.ones(rows, device=logits.device, dtype=tp.float32)
        _ce_fwd_kernel[(rows,)](
            logits, target, maxes, sums, vocab, BLOCK=_BLOCK, IGNORE=ignore_index
        )
        keep = target != ignore_index
        # Ignored rows carry a sentinel that is not a valid class, so gather
        # through an in-range index and discard the value with the mask below.
        # Clamping both ends keeps this independent of the sentinel's sign.
        safe = target.clamp(0, vocab - 1)
        target_logit = logits.gather(1, safe.reshape(-1, 1)).reshape(-1)
        tok_loss = (sums.log() + maxes - target_logit.float()) * keep.to(tp.float32)

        ctx.logits = logits
        ctx.target = target
        ctx.maxes = maxes
        ctx.sums = sums
        ctx.ignore_index = ignore_index
        ctx.block = _BLOCK
        ctx.rows = rows
        ctx.vocab = vocab
        ctx.mark_dirty(logits)
        return tok_loss

    @staticmethod
    def backward(ctx, grad_out):
        grad = grad_out.contiguous()
        scalar = grad.dim() == 0
        _ce_bwd_kernel[(ctx.rows,)](
            ctx.logits, ctx.target, ctx.maxes, ctx.sums, grad, ctx.vocab,
            BLOCK=ctx.block, IGNORE=ctx.ignore_index, SCALAR=scalar,
        )
        dlogits = ctx.logits
        # The logits tensor owns a grad_fn that reaches this context, so keeping
        # the reference past the point of use would leave one full-size buffer
        # per step to be reclaimed by a cycle collector rather than freed.
        ctx.logits = ctx.target = ctx.maxes = ctx.sums = None
        return dlogits, None, None


def fused_cross_entropy(logits, target, ignore_index: int = -100,
                        reduction: str = "mean"):
    """Cross entropy that keeps ``logits`` as the only full-size buffer.

    Equivalent to ``cross_entropy(logits, target, ignore_index=...,
    reduction=...)`` with fp32 loss arithmetic, but *destructive*: the forward
    pass leaves no probability buffer behind and the backward pass replaces the
    contents of ``logits`` with the gradient with respect to the logits.

    Args:
        logits: ``[rows, vocab]`` float tensor on CUDA; overwritten.
        target: ``[rows]`` integer class indices.
        ignore_index: rows whose target equals this value contribute nothing.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        The reduced loss (a 0-d tensor) or the per-row losses for ``"none"``.

    Warning:
        Because backward consumes the buffer, the graph can be traversed only
        once.  ``loss.backward(retain_graph=True)`` followed by a second
        backward, or a second ``autograd.grad`` call on the same graph, would
        differentiate the gradients instead of the logits.  Keep ``logits``
        alive past the step only by cloning it first.
    """
    if _capturing():
        # A captured region replays traced operations, and the two scratch
        # vectors this kernel fills are allocated inside the call, so a graph
        # would keep them at their capture-time values.  Compose the reference
        # instead: identical arithmetic, and every step of it traces.  The
        # fused path is for eager execution; see the module docstring.
        from . import functional as F

        return F.cross_entropy(logits, target, ignore_index=ignore_index,
                              reduction=reduction)
    if not is_eligible(logits, target, ignore_index):
        raise TypeError(
            "fused_cross_entropy requires 2-D CUDA logits of a floating dtype "
            "and a matching 1-D integer target (and an installed Triton)"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unsupported reduction {reduction!r}")
    if not logits.is_contiguous():
        logits = logits.contiguous()
    target = target.contiguous()
    if target.dtype != tp.int64:
        target = target.to(tp.int64)

    tok_loss = _FusedCrossEntropy.apply(logits, target, int(ignore_index))
    if reduction == "none":
        return tok_loss
    if reduction == "sum":
        return tok_loss.sum()
    count = (target != ignore_index).sum().clamp(min=1).to(tok_loss.dtype)
    return tok_loss.sum() / count


def fused_linear_cross_entropy(input, weight, target, bias=None,
                               ignore_index: int = -100, reduction: str = "mean"):
    """``cross_entropy(linear(input, weight, bias), target)`` without the
    logits surviving the step.

    The projection is still a dense matmul; what this saves is everything the
    loss would otherwise keep around it.  Falls back to the composed reference
    when the shapes or dtypes are outside the kernels' reach.
    """
    from . import functional as F

    eligible = (
        triton is not None
        and input.dim() == 2
        and weight.dim() == 2
        and target.dim() == 1
        and input.shape[0] == target.shape[0]
        and input.shape[1] == weight.shape[1]
        and input.shape[0] > 0
        and weight.shape[0] > 0
        and input.dtype in _LOGIT_DTYPES
        and target.dtype in (tp.int64, tp.int32)
        and getattr(input, "is_cuda", False)
    )
    logits = F.linear(input, weight, bias)
    if not eligible:
        return F.cross_entropy(logits, target, ignore_index=ignore_index,
                               reduction=reduction)
    return fused_cross_entropy(logits, target, ignore_index=ignore_index,
                               reduction=reduction)