"""A product over a leading axis, with its own extent carried by the launch.

A batched product is not the plain product with an extra dimension: the batch is
walked by the grid rather than by the tile, the tile is fitted to one item, and
what the grid is built from is therefore a property of the launch rather than of
the tile.
"""

from __future__ import annotations
import logging

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

#: The element types a batched product is decomposed into a plain one for.
#:
#: The decomposition reads the batch as an extra pair of extents on one side and
#: a single item on the other, which lets the plain product's own body do the
#: work.  It costs a copy wherever an operand is not already in that shape, so it
#: is only worth it for the types whose plain body is well fitted.
_BMM_DOT_DECOMPOSE_DTYPES = ("float16", "bfloat16", "float32")

#: The contraction length above which decomposing pays for the copy it needs.
#: Below it the copy is most of the work.
_BMM_DOT_K_DECOMPOSE_THRESHOLD = 32

from typing import Any

import tensorplay as tp
from .triton import CHOICES
from ..op_lowerings import register_lowering
from .select_algorithm import (
    ChoiceCaller,
    ExternChoiceCaller,
    ExternKernelChoice,
    KernelArgs,
    TritonChoiceCaller,
    TritonTemplate,
)
from ..kernel_inputs import KernelInputs, MMKernelInputs
from .params import DictKernelTemplateParams, KernelTemplateParams
from ..ir import Layout
from .triton import CHOICES, dtype_size
import itertools
from .ir import contiguous_stride, next_power_of_2

from .mm_common import mm_grid
from .mm import (
    GemmConfigHeuristics, GemmTemplate, MMKernelInputs, _gemm_source_identity,
    contiguous_stride,
)
from .select_algorithm import ExternKernelChoice, TritonTemplate

def bmm_grid(batch, m, n, meta, *, cdiv):
    """The grid for a batched product's tiles: the same tiles, one axis per batch.

    The batch is a grid axis and nothing else.  It is not a loop, which would
    compute the same numbers one product at a time and make the batch a factor
    in the tile count; and it is not folded into the flat tile index, because
    every matrix in the batch is tiled the same way and the grouping that makes
    neighbouring programs share operand rows means the same thing for each of
    them.
    """

    tiles = cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])
    return (tiles, 1, int(batch))

class BmmConfigHeuristics(GemmConfigHeuristics):

    """The product's candidates, for a call that carries a batch in front.

    The tiles are the product's own, fitted to one matrix's extents rather than
    to all of them: a batch is independent products, so the tile that runs one
    of them well runs the rest of the same way.  The batch therefore changes
    how many times the kernel is launched and not what is launched, which is
    why the fitting below reads the same extents the plain product's does.

    Whether the contraction divides the tile is per configuration, so it is
    answered here rather than in the template: it is a property of this
    configuration's ``BLOCK_K`` and this call's contraction, and a kernel that
    masked the tail when it did not have to would pay for it on every step.

    How many rows of tiles a program walks before advancing a column is stated
    rather than left out, for the same reason: the body reads it, so a
    configuration that did not say what to read would be a kernel this
    template cannot render.
    """

    #: How many rows of tiles a program walks before advancing a column.  It
    #: is a property of the launch, not of the call, so one figure is written
    #: here and every configuration of every call uses it.
    group_m = 8

    def __init__(self, op_name: str = "bmm", device_type: str = "cuda"):
        super().__init__(op_name=op_name, device_type=device_type)

    def should_run(self, inputs: KernelInputs) -> bool:
        return super().should_run(inputs) and len(inputs.batch()) == 1

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        rows, cols, inner = kernel_inputs.mnk_symbolic()
        for config in CHOICES.get_mm_configs(self.device_type)(
            rows, cols, inner, dtype_size=dtype_size(kernel_inputs.dtype(0))
        ):
            yield {
                "choice": "triton",
                "GROUP_M": self.group_m,
                "EVEN_K": int(inner) % int(config.kwargs["BLOCK_K"]) == 0,
                **config.as_kwargs(),
            }

    def get_extra_kwargs(self, kernel_inputs, op_name):
        """The precision the whole call runs at, which is not a choice.

        It belongs to the device and to the switch the caller set, not to the
        tile: two configurations of this call cannot disagree about it, so it
        is stated once for all of them -- and it is stated at all, because a
        kernel that silently chose a precision would put the decision in the
        body where a measurement could not see it.
        """

        from ..codegen.triton_gemm import _matmul_allow_tf32

        return {"allow_tf32": _matmul_allow_tf32()}

class BMMTemplate(TritonTemplate):

    """The product with a batch in front of both matrices and of the result.

    A separate template rather than another configuration of the plain one,
    and for the same reason the plain and the swept products are two: what
    differs is not the tile but the launch.  The batch is a grid axis here, so
    a configuration of this template carries extents the plain product's do
    not -- the operands' own extents and strides, and whether the contraction
    divides the tile -- and a configuration of that template changing would be
    claiming a choice the tile never had a say in.

    The body is source text rather than a decorated function because the batch
    is what varies: which extents there are, and which strides come with them,
    are read out of the operands the kernel is written for, so one body serves
    every shape those operands may take.  What a configuration decides that the
    body cannot is decided here instead, by rendering.
    """

    inputs_class = MMKernelInputs

    #: The block extents the body reads, in the order the signature declares
    #: them after the operands' extents and strides.  The order is this list's,
    #: because the render emits the parameters in it and the launch passes
    #: arguments in it: a second list saying the same thing is a second thing
    #: to keep in step.
    block_names = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "EVEN_K")

    def __init__(self, name: str = "bmm", **kwargs: Any):
        super().__init__(name, **kwargs)
        self.heuristics = BmmConfigHeuristics()

    def out_specs(self, meta: dict) -> tuple:
        """Where this call's result lands: the batch, then the two free extents.

        The result is contiguous because the template allocates it: there is
        nothing else to ask, and a layout that may still change would be a
        question this template has no answer to.
        """

        size = tuple(int(v) for v in (meta.get("out_size") or ()))
        if len(size) != 3:
            raise NotImplementedError("a batched product's result is batched")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                size,
                contiguous_stride(size),
            ),
        )

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates.

        The ramp covers the whole batch, so every matrix in it is distinct and
        a kernel that read one of them for all of them would be caught here
        rather than in a number somebody trusted.
        """

        from ..codegen.triton_gemm import _probe_feed

        layout = self.out_specs(meta)[0]
        sizes = tuple(meta.get("operand_sizes") or ())
        if len(sizes) != 2 or meta.get("out_dtype") is None:
            return None
        (batch, rows, inner), (batch_b, inner_b, cols) = (
            tuple(int(v) for v in shape) for shape in sizes
        )
        if len(layout.size) != 3 or batch != batch_b or inner != inner_b:
            return None
        if (rows, cols) != (layout.size[1], layout.size[2]):
            return None
        return _probe_feed(
            rows, inner, cols, layout.dtype, meta.get("device"),
            bias=False, batch=batch,
        )

    def generate(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                 plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        kwargs = params.to_kwargs()
        layout = out_specs[0] if out_specs else None
        if kwargs.get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_batched_product",
                layout=layout,
                description="the operation itself",
                launcher=plain_launch,
            )
        if layout is None or len(layout.size) != 3:
            return None
        if meta.get("operand_dtype") != "float32":
            return None
        if not meta.get("qualifies", False):
            return None
        shapes = self._kernel_shapes(layout, meta)
        if shapes is None or plain_launch is None:
            return None
        block = tuple((name, kwargs.get(name)) for name in self.block_names)
        if any(value is None for _name, value in block):
            return None

        caller = TritonChoiceCaller(
            name=f"{self.name}-{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}"
            f"x{kwargs['BLOCK_K']}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
            source=self.source,
            src_hash=self.src_hash,
        )
        return caller.bind(
            self.launcher(
                self.kernel_for(block, shapes, bool(kwargs.get("allow_tf32"))),
                block, meta, layout, plain_launch, kwargs,
            )
        )

    def launcher(self, kernel, block, meta, layout, plain_launch, kwargs):
        """The launch this form runs, over the batch as a third grid axis.

        Split out because the forms differ in *which kernel* they run and in
        what they accept as a layout, not in whether they run: the shared form
        reads one left tile for a group of batches, the general one reads it
        per batch, and only the general one will run a left operand that is
        really per batch.
        """

        from ..codegen.triton_gemm import batched_matmul_launch

        return batched_matmul_launch(
            plain_launch,
            meta["operand_specs"],
            layout.size,
            kernel,
            self.grid,
            block,
            num_warps=int(kwargs.get("num_warps", 4)),
            num_stages=int(kwargs.get("num_stages", 3)),
        )

    def _kernel_shapes(self, layout, meta: dict):
        """The three extents the kernel is written against, or ``None``.

        The two matrices and the result have to meet: the result's extents say
        how many rows and columns there are, the matrices say where they come
        from, and a call where the three do not line up is not this kernel's
        call.  Nothing here is inferred -- the extents are the ones the call
        arrived with.
        """

        sizes = tuple(meta.get("operand_sizes") or ())
        if len(sizes) != 2:
            return None
        first = tuple(int(v) for v in sizes[0])
        second = tuple(int(v) for v in sizes[1])
        size = tuple(int(v) for v in layout.size)
        if len(first) != 3 or len(second) != 3:
            return None
        if first[0] != size[0] or second[0] != size[0]:
            return None
        if first[2] != second[1]:
            return None
        if first[1] != size[1] or second[2] != size[2]:
            return None
        return (first, second, size)

    def kernel_for(self, block: tuple, shapes: tuple, allow_tf32: bool):
        """This template's body, rendered for one configuration.

        The signature is emitted from the names the body asks its extents by, so
        the two cannot disagree about which operands there are or in what order
        they come.  The precision is decided by the rendering rather than by an
        argument, because a shortened multiplier is a different kernel rather
        than a different value of one of its parameters.
        """

        args = KernelArgs(
            {
                "A": {"shape": shapes[0], "stride": contiguous_stride(shapes[0])},
                "B": {"shape": shapes[1], "stride": contiguous_stride(shapes[1])},
            },
            {"C": {"shape": shapes[2], "stride": contiguous_stride(shapes[2])}},
            dict(block),
        )
        source = self.render_with(args, allow_tf32=bool(allow_tf32))
        return self.kernel_type.get(
            self.name, source, self.symbol, self.grid, {"block": dict(block)}
        ).build()

BMM = BMMTemplate.from_file(
    "bmm",
    file="triton_bmm",
    grid=bmm_grid,
    symbol="_bmm_kernel",
    cache_codegen_enabled_for_template=True,
)

#: The operation namespace, under a name of this project's own, reached by
#: whatever name this project's operations are registered under.
framework = tp.ops.tp

framework_mm = ExternKernelChoice(None, "framework_mm")

framework_mm_dtype = ExternKernelChoice(None, "framework_mm_dtype")

framework_addmm = ExternKernelChoice(None, "framework_addmm")

framework_int_mm = ExternKernelChoice(None, "framework_int_mm")

framework_sparse_semi_structured_mm = ExternKernelChoice(
    None, "framework_sparse_semi_structured_mm", has_out_variant=False
)

def bmm_shared_a_grid(batch, m, n, meta, *, cdiv):
    """The grid for the shared-left batched product: the same tiles, batched by group.

    The second axis counts groups of batches rather than batches, so one
    program serves ``BLOCK_Q`` of them and loads the left tile they all share
    once.  A batch that does not fill the last group is why the axis is a count
    of groups rather than a count of batches: rounding up is the launcher's
    job and the lanes past the end are dropped at the store.
    """

    tiles = cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])
    return (tiles, cdiv(batch, meta["BLOCK_Q"]), 1)

class BmmSharedAConfigHeuristics(BmmConfigHeuristics):

    """The candidates for the shared-left batched product.

    The tilings are bounded by the register file rather than by the product
    table, because the accumulator is a row-block wide by however many batches
    a group covers: the dot is ``BLOCK_N * BLOCK_Q`` wide, so the two are
    bounded together and neither is free to grow on its own.

    Whether the form is worth measuring at all is a question about the batch
    alone.  With a handful of batches, grouping spends the parallelism it was
    meant to recover -- a group of two leaves half the second axis of the grid
    empty -- so below the threshold the plain batched product's candidates are
    the ones that get measured.  The threshold is a hint rather than a
    requirement: a batch the caller knows and the graph does not is still a
    reason to measure, and refusing those would leave every model with a dynamic
    batch without this form.
    """

    #: The smallest batch worth grouping, as a hint rather than a requirement.
    min_batch = 64

    def should_run(self, inputs: KernelInputs) -> bool:
        return super().should_run(inputs) and inputs.batch_hinted() >= self.min_batch

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        rows, inner = kernel_inputs.mnk_symbolic()[:2]
        for block_m, block_n, block_k, block_q, warps, stages in itertools.product(
            (64, 128), (32, 64), (32, 64), (2, 4, 8), (4, 8), (1, 2, 3)
        ):
            if not 64 <= block_n * block_q <= 256:
                continue
            if block_m * block_n * block_q > 64 * 256:
                continue
            yield {
                "choice": "triton",
                "BLOCK_M": max(next_power_of_2(rows), 16),
                "BLOCK_N": block_n,
                "BLOCK_K": max(min(block_k, next_power_of_2(inner)), 16),
                "BLOCK_Q": block_q,
                "GROUP_M": self.group_m,
                "num_warps": warps,
                "num_stages": stages,
            }

class BmmSharedATemplate(BMMTemplate):

    """The batched product whose left operand is one matrix over the batch.

    A specialisation rather than a configuration of the batched product, and
    the difference is not arithmetic: the plain one re-reads that left tile once
    for every batch, and this one reads it once for a group of them.  Which
    kernel runs is therefore the whole difference, and a configuration of the
    batched product changing it would be claiming a choice the tile never had a
    say in.

    The kernel here peels the contraction's tail off the loop instead of
    guarding every step, so it has no "divides the tile" flag: whether the
    contraction divides the tile is decided once, in the body, rather than
    passed in.
    """

    def __init__(self, name: str = "bmm_shared_a", **kwargs: Any):
        super().__init__(name, **kwargs)
        self.heuristics = BmmSharedAConfigHeuristics()

    block_names = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_Q", "GROUP_M")

    def launcher(self, kernel, block, meta, layout, plain_launch, kwargs):
        """The launch, over a group of batches sharing one left tile.

        The same three-dimensional launch the general batched form runs, with
        the one thing that differs: the second grid axis counts groups of
        batches rather than batches, and the left operand is read as the
        broadcast it is.
        """

        from ..codegen.triton_gemm import shared_a_matmul_launch

        return shared_a_matmul_launch(
            plain_launch,
            meta["operand_specs"],
            layout.size,
            kernel,
            self.grid,
            block,
            num_warps=int(kwargs.get("num_warps", 4)),
            num_stages=int(kwargs.get("num_stages", 3)),
        )

    def kernel_for(self, block: tuple, shapes: tuple, allow_tf32: bool):
        """The body, rendered with the group extent among the block extents."""

        batch, m, n = (int(v) for v in shapes[2])
        _batch_a, rows, inner = (int(v) for v in shapes[0])
        _batch_b, inner_b, cols = (int(v) for v in shapes[1])
        args = KernelArgs(
            {
                "A": {"shape": (batch, rows, inner),
                      "stride": (0, *contiguous_stride((rows, inner)))},
                "B": {"shape": (batch, inner_b, cols),
                      "stride": contiguous_stride((batch, inner_b, cols))},
            },
            {"C": {"shape": (batch, m, n), "stride": contiguous_stride((batch, m, n))}},
            {name: value for name, value in block},
        )
        source = self.render_with(args, allow_tf32=bool(allow_tf32))
        return self.kernel_type.get(
            self.name, source, self.symbol, self.grid, {"block": dict(block)}
        ).build()

BMM_SHARED_A = BmmSharedATemplate.from_file(
    "bmm_shared_a",
    file="triton_bmm_shared_a",
    grid=bmm_shared_a_grid,
    symbol="_bmm_shared_a_kernel",
)

#: The candidates this module offers: the two ways a batched product is walked.
BMM_TEMPLATES = (
    BMM,
    BMM_SHARED_A,
)


#: The two templates this module binds, under the names their operations are
#: declared under.  The module keeps the name and the implementation apart so
#: that a lowering can hold the first without holding the second.
bmm_template = BMM
bmm_shared_a_template = BMM_SHARED_A


def _use_bmm_shared_a(mat1, mat2, layout=None) -> bool:
    """Whether the left operand is one matrix broadcast over the batch.

    Only worth it when there are enough batches to group: with a handful of
    batches the grouping just costs parallelism, because the plain form already
    has as many programs as it has tiles and the shared form has as many as it
    has rows.
    """

    try:
        stride0 = int(mat1.get_stride()[0])
    except Exception:  # noqa: BLE001 - a call that will not say is not shared
        return False
    if stride0 != 0:
        return False
    try:
        batch = int(mat2.get_size()[0])
    except Exception:  # noqa: BLE001
        return False
    return batch >= 64


def _bmm_shared_a_configs(dtype):
    """The shared-left batched product's candidates, and why they stop where they do.

    ``BLOCK_Q`` batches share one tile of the left operand and are laid side by
    side in the multiply, so ``BLOCK_N * BLOCK_Q`` is the real width of that
    multiply and the accumulator is ``BLOCK_M * BLOCK_N * BLOCK_Q`` values.  Both
    are bounded: too wide and the multiply is not worth having, too tall and the
    accumulator stops being in registers, which is the point of holding it there.
    """

    import itertools

    from .mm_common import acc_type

    accumulated = acc_type(dtype)
    for block_m, block_n, block_k, block_q, warps, stages in itertools.product(
        (64, 128), (32, 64), (32, 64), (2, 4, 8), (4, 8), (1, 2, 3)
    ):
        if not 64 <= block_n * block_q <= 256:
            continue
        if block_m * block_n * block_q > 64 * 256:
            continue
        yield {
            "BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k,
            "BLOCK_Q": block_q, "GROUP_M": 8, "ACC_TYPE": accumulated,
            "ALLOW_TF32": False, "num_stages": stages, "num_warps": warps,
        }


@register_lowering(framework.bmm)
def tuned_bmm(mat1, mat2, out_dtype=None, *, layout=None, plain=None):
    """A product over a leading axis, and the candidates to measure it with.

    Which of the two forms it is offered is decided here rather than at the call
    site, because the two differ in what the left operand is -- a matrix per
    batch, or one matrix over all of them -- and that is a fact about the
    operands rather than a choice anybody makes.
    """

    from .mm_common import mm_args

    _m, n, k, out_layout = mm_args(mat1, mat2, layout=layout)
    template = bmm_shared_a_template if _use_bmm_shared_a(
        mat1, mat2, out_layout
    ) else bmm_template
    specs = ((0, None), (1, None))
    sizes = (
        tuple(int(v) for v in mat1.get_size()),
        tuple(int(v) for v in mat2.get_size()),
    )
    meta = {
        "out_size": tuple(int(v) for v in out_layout.get_size()),
        "out_dtype": str(out_layout.get_dtype() if out_dtype is None else out_dtype),
        "device": out_layout.get_device(),
        "operand_specs": specs,
        "operand_sizes": sizes,
        "operand_dtype": str(mat1.get_dtype()),
        "qualifies": True,
        "arg_templates": (),
        "call_method": False,
    }
    return template.configurations(template.out_specs(meta), meta), meta


@register_lowering(framework.baddbmm)
def tuned_baddbmm(mat1, batch1, batch2, beta=1, alpha=1, *, layout=None,
                  plain=None, bias_spec=None):
    """A batched product with a bias added to it, and its candidates.

    The bias is added where the product's store already is, so the two are one
    kernel rather than two; the result is the product's own candidates, with the
    bias riding along in the store.
    """

    from .mm_common import addmm_epilogue, mm_args

    candidates, meta = tuned_bmm(
        batch1, batch2, layout=layout, plain=plain
    )
    if candidates is None:
        return None
    if bias_spec is not None:
        meta = dict(meta, bias_spec=bias_spec)
        meta["epilogue"] = addmm_epilogue(None, alpha, beta)
    return candidates, meta
