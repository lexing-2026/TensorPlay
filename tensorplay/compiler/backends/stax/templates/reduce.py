"""Reductions, measured over the forms their kernel can carry.

The kernel is one program for every form; what differs is how many streams it
carries and which of them it stores.  So the form is the whole of the choice, and the
extents around it are not a choice at all.
"""

from __future__ import annotations

from typing import Any

from .bridge import LoopTemplate
from .heuristics import TemplateConfigHeuristics
from .ir import (
    ExternChoiceCaller,
    Layout,
    ReductionKernelInputs,
    TritonChoiceCaller,
    contiguous_stride,
)
from .params import KernelTemplateParams

class ReductionConfigHeuristics(TemplateConfigHeuristics):
    """The candidates for a reduction.

    A reduction's configuration is not a tile shape but a *kind*: a plain sum
    of values, a value stream beside an index stream that yields the position
    rather than the value, a value and an index written out together, or two
    accumulators that produce a mean and a spread in one pass.  Which kinds
    exist for an operation is fixed; which of them apply to a call is not, and
    that is what this decides.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        return isinstance(inputs, ReductionKernelInputs)

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        for kind in kernel_inputs.kinds():
            yield {"choice": "triton", "kind": kind}
class ReductionTemplate(LoopTemplate):
    """Reductions, measured over the forms their kernel can carry.

    The kernel is the same program for every kind; what differs is how many
    streams it carries alongside the values and which of them it stores. So the
    kind is the whole of the choice, and the geometry around it is not a
    choice at all -- it is the reduction's own extents, which the caller
    already knows.
    """

    inputs_class = ReductionKernelInputs

    def __init__(self):
        super().__init__("reduction")
        self.heuristics = ReductionConfigHeuristics()

    def emitter(self) -> str:
        return "reduction:operator+value/index/pair/moments"

    def out_specs(self, meta: dict) -> tuple:
        size = tuple(meta.get("out_size") or ())
        if not size:
            raise NotImplementedError("a reduction without a result shape")
        count = int(meta.get("result_count", 1) or 1)
        device, dtype = meta.get("device"), meta.get("out_dtype")
        return tuple(
            Layout(device, dtype if i == 0 else meta.get("index_dtype", "int64"),
                   size, contiguous_stride(size))
            for i in range(count)
        )

    def probe(self, meta: dict):
        """A deterministic operand set for measuring a reduction's candidates."""

        size = tuple(meta.get("operand_sizes") or ())
        if len(size) != 1 or not size[0]:
            return None
        import tensorplay as tp

        # A ramp rather than noise: a reduction's answer must not depend on
        # which lanes happened to run first, and a ramp makes an ordering
        # mistake visible instead of averaging it away.
        numel = int(size[0])
        return (tp.linspace(-1.0, 1.0, numel, device=meta.get("device")),)

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        kind = params.to_kwargs().get("kind")
        if kind is None:
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_reduction",
                layout=out_specs[0] if out_specs else None,
                description="the operation itself",
                launcher=plain_launch,
            )
        if kind not in tuple(meta.get("kinds") or ()):
            return None
        try:
            launcher = _reduction_launcher(kind, meta, out_specs)
        except NotImplementedError:
            return None
        caller = TritonChoiceCaller(
            name=f"reduction-{kind}",
            layout=out_specs[0] if out_specs else None,
            description=f"{kind} reduction",
            source=repr(sorted((kind, meta.get("out_size"), meta.get("dims")))),
        )
        return caller.bind(launcher)
def _reduction_launcher(kind: str, meta: dict, out_specs: tuple):
    """A launcher for one reduction form, built from the reduction codegen.

    The reduction codegen knows how a form is emitted; what it did not expose
    was a way to be *asked* for one form's launcher.  So the geometry is
    translated here into what that codegen already accepts, and the question
    is answered by trying: a form whose geometry cannot be expressed is refused
    rather than approximated, because a reduction that quietly computes
    something else is worse than one that declines.
    """

    from ..codegen.triton import ReductionSpec

    op = {
        "value": "sum",
        "index": "argmax",
        "pair": "max",
        "moments": "var",
    }.get(kind)
    if op is None:
        raise NotImplementedError(f"no reduction form named {kind!r}")
    dims = tuple(int(d) for d in (meta.get("dims") or ()))
    if op in ("argmax", "max") and not dims:
        # An index or a pair needs an extent to reduce over: with none there is
        # no position to report and nowhere to put the second result.
        raise NotImplementedError("an index or pair reduction without an extent")
    spec = ReductionSpec(op, dims, keepdim=bool(meta.get("keepdim", False)))
    if spec.tracks_indices and not spec.dims:
        raise NotImplementedError("an index reduction without an extent")
    return _bind_reduction_codegen(spec, meta, out_specs)
def _bind_reduction_codegen(spec, meta: dict, out_specs: tuple):
    """The launch closure for one reduction form, over this call's operands."""

    builder = meta.get("reduction_builder")
    if builder is None:
        raise NotImplementedError("no reduction launcher for this region")
    launcher = builder(spec, meta, out_specs)
    if launcher is None:
        raise NotImplementedError("this region's reduction is not in that form")
    return launcher


REDUCTION = ReductionTemplate()
