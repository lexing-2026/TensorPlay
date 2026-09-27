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
    call_operation,
    ExternKernelChoice,
    KernelArgs,
    TritonChoiceCaller,
    TritonTemplate,
)
from ..heuristics.template.base import SymbolicGridFn
from .mm_common import load_kernel_template
from ..runtime.runtime_utils import get_max_y_grid
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..heuristics.template.params import DictKernelTemplateParams, KernelTemplateParams
from ..ir import (
    ExternKernel,
    FlexibleLayout,
    Layout,
    as_storage_and_layout,
    is_storage_and_layout,
)
from ..loops import V
from .triton import CHOICES, dtype_size
import itertools
from .ir import contiguous_stride, next_power_of_2

from .mm_common import mm_grid
from .mm import (
    MMKernelInputs,
    contiguous_stride,
    framework_addmm,
    framework_mm,
    framework_mm_dtype,
)
from .select_algorithm import ExternKernelChoice, TritonTemplate

@SymbolicGridFn
def bmm_grid(batch, m, n, meta, *, cdiv, max):
    """The grid for a batched product's tiles: the same tiles, one axis per batch.

    The batch is a grid axis and nothing else.  It is not a loop, which would
    compute the same numbers one product at a time and make the batch a factor
    in the tile count; and it is not folded into the flat tile index, because
    every matrix in the batch is tiled the same way and the grouping that makes
    neighbouring programs share operand rows means the same thing for each of
    them.
    """

    tiles = cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])
    # A batch spread over the second axis alone stops being launchable once it
    # outgrows what that axis accepts, so it is spread over the two axes the
    # launch already has and the smaller of the two carries the remainder.  A
    # batch that fits on one axis is unaffected: the other comes out as one.
    max_y_grid = get_max_y_grid()
    grid_z = max(cdiv(batch, max_y_grid), 1)
    grid_y = cdiv(batch, grid_z)
    return (tiles, grid_y, grid_z)



# Each template kernel lives in its own file, named for the dialect it is
# written in, and is loaded by that file's name.  The template's own name is a
# separate fact -- it is what the operation is called and what the rule about
# this kernel is filed under -- so the two are given separately.
bmm_source = load_kernel_template("triton_bmm")

BMM = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=bmm_source,
    cache_codegen_enabled_for_template=True,
)

#: The operation namespace, under a name of this project's own, reached by
#: whatever name this project's operations are registered under.
framework = tp.ops.tp

#: The framework's own batched product, measured against the templates here so
#: that a template is only chosen where it beats what the framework already does.
framework_bmm = ExternKernelChoice(
    tp.ops.tp.bmm, "bmm_out", op_overload=tp.ops.tp.bmm.out,
)

#: The same product asked for a result of a type the inputs do not already
#: have, which is a different call and so is measured on its own.
framework_bmm_dtype = ExternKernelChoice(
    tp.ops.tp.bmm, "bmm_out", name="bmm_dtype",
    op_overload=tp.ops.tp.bmm.out,
)

framework_int_mm = ExternKernelChoice(
    tp.ops.tp._int_mm, "int_mm_out", name="int_mm",
    op_overload=tp.ops.tp._int_mm.out,
)

framework_sparse_semi_structured_mm = ExternKernelChoice(
    tp.ops.tp._sparse_semi_structured_mm,
    "sparse_semi_structured_mm",
    name="sparse_semi_structured_mm",
    has_out_variant=False,
    op_overload=tp.ops.tp._sparse_semi_structured_mm.default,
)

@SymbolicGridFn
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



bmm_shared_a_source = load_kernel_template("triton_bmm_shared_a")

BMM_SHARED_A = TritonTemplate(
    name="bmm_shared_a",
    grid=bmm_shared_a_grid,
    source=bmm_shared_a_source,
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
def tuned_bmm(mat1, mat2, out_dtype=None, *, layout=None):
    """A product over a leading axis, and the candidates to measure it with.

    Which of the two forms it is offered is decided here rather than at the call
    site, because the two differ in what the left operand is -- a matrix per
    batch, or one matrix over all of them -- and that is a fact about the
    operands rather than a choice anybody makes.
    """

    from .mm_common import (
        mm_args,
        use_aten_gemm_kernels,
        use_native_matmul,
        use_triton_template,
    )
    from ..kernel_inputs import MMKernelInputs
    from .select_algorithm import autotune_select_algorithm, get_template_configs

    # A call the framework's own product is better at is not offered a template
    # at all: the measurement would be comparing two answers to a different
    # question, and reporting the faster one as this call's best.
    if use_native_matmul(mat1, mat2):
        return None

    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )
    name = "bmm"
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)
    log.info(
        "Tuned product over a leading axis: batch=%s, m=%s, n=%s, k=%s, left=%s, right=%s, result=%s",
        mat1.get_size()[0], m, n, k,
        mat1.get_dtype(), mat2.get_dtype(), layout,
    )

    framework_handler = framework_bmm
    framework_extra_kwargs = {}
    if out_dtype:
        if mat1.get_device().type != "cuda":
            raise AssertionError("out_dtype is only supported for CUDA")
        framework_handler = framework_bmm_dtype
        framework_extra_kwargs = {"out_dtype": out_dtype}

    templates_to_use: list = []
    kwarg_overrides = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(framework_handler)
        kwarg_overrides[framework_handler.uid] = framework_extra_kwargs
    if use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(bmm_template)
    choices = get_template_configs(
        kernel_inputs, templates_to_use, name, kwarg_overrides=kwarg_overrides,
    )
    if use_triton_template(layout, check_max_autotune=False) and _use_bmm_shared_a(
        mat1, mat2, layout
    ):
        log.info(
            "Shared-left form offered for batch=%s m=%s n=%s k=%s",
            mat2.get_size()[0], m, n, k,
        )
        for config in _bmm_shared_a_configs(mat1.get_dtype()):
            bmm_shared_a_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **config,
            )
    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


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
