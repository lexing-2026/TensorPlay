"""The one thing every rule about a matrix product has in common.

A matrix product is the operation where paying to search pays off most, and also
the one where searching costs the most -- a search over tile shapes is measured by
running the product, and the product is the largest thing in the program.  So the
templates for this family are the ones that only run when the program has said it
wants to spend that, and they share the check for it rather than each repeating
it: a rule that is written out again is a rule that is one edit away from
disagreeing with the others about when it applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import config as tp_config
from ...templates.mm import (
    addmm_contiguous_subgraph_template,
    mm_contiguous_subgraph_template,
    BLACKWELL_WS_PERSISTENT_TMA,
    GEMM,
    GEMM_PERSISTENT,
    GEMM_PERSISTENT_TMA,
)
from ...templates.triton import CHOICES, dtype_size
from ...codegen.triton_gemm import _matmul_allow_tf32
from ...utils import get_num_sms
from ..registry import register_template_heuristic
from .base import TemplateConfigHeuristics

if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs


class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    """A rule for a matrix product that only applies when searching was asked for.

    Whether the program wants to spend time measuring tile shapes is a property
    of the program, not of any one template; asking here rather than at each
    template's call site means a template added to this family later inherits the
    answer instead of having to remember to ask.
    """

    def should_run(self, inputs) -> bool:
        return tp_config.max_autotune or tp_config.max_autotune_gemm


@register_template_heuristic(GEMM.uid, "cuda")
class CudaGemmTemplateConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """The tilings a discrete accelerator is measured with, for a product.

    One rule for all four product templates rather than four copies of the same
    table: they differ in how the tiles are walked, not in which tilings are
    worth trying, and a table written out four times is a table that will be
    four different tables the first time one of them is changed.

    The tilings are grouped by how wide the contraction step is, since that is
    what decides how many times the kernel steps over the contraction -- a
    kernel whose step covers the whole contraction does it once.
    """

    @staticmethod
    def _acc_type(dtype) -> str:
        """The type a sum of this is accumulated in.

        Wider than the operands on purpose whenever they are narrow: a sum of
        many narrow terms loses the small ones if it is kept narrow, and the
        widening is nearly free because the accumulator is not what leaves the
        device. A result already as wide as the accumulator needs no widening,
        so it is kept as it is.
        """

        name = str(dtype).split(".")[-1]
        if name in ("float16", "bfloat16", "float8_e4m3fnuz", "float8_e4m3fn"):
            return "tl.float32"
        return f"tl.{name}"

    #: How many tiles of one axis are walked before moving along the other.
    #: The tiles are reordered so that the ones sharing an input tile are
    #: launched together, which is what keeps that input in the cache while it
    #: is used; how many is worth this is a property of the device, so it is
    #: set here rather than inside the body, which would then have to know it.
    group_m: int = 8

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        rows, cols, inner = kernel_inputs.mnk_symbolic()
        for config in CHOICES.get_mm_configs(kernel_inputs.device_type)(
            rows, cols, inner, dtype_size=dtype_size(kernel_inputs.dtype(0))
        ):
            # What the body is told rather than asked: the type the sum is
            # accumulated in, which is wider than the operands on purpose, and
            # whether the contraction divides by the step, which the body would
            # otherwise test for on every step of the loop.  Both are decided
            # here, where the call is known, rather than in the body, which
            # would then have to be able to answer a question about the call.
            allow_tf32 = _matmul_allow_tf32()
            yield {
                **config.as_kwargs(),
                "ACC_TYPE": "tl.float32",
                "ALLOW_TF32": allow_tf32,
                "EVEN_K": int(inner) % int(config.kwargs["BLOCK_K"]) == 0,
                # Whether the accumulator is carried through the product step or
                # added to afterwards.  It is a choice between two forms of the
                # same step rather than a tuning, so it is asked of the call and
                # arrives with the configuration instead of being read inside
                # the loop where it would be the same answer every time.
                "USE_FAST_ACCUM": True,
                "GROUP_M": self.group_m,
            }


@register_template_heuristic(GEMM_PERSISTENT.uid, "cuda")
@register_template_heuristic(GEMM_PERSISTENT_TMA.uid, "cuda")
@register_template_heuristic(BLACKWELL_WS_PERSISTENT_TMA.uid, "cuda")
class CudaPersistentGemmTemplateConfigHeuristics(
    CudaGemmTemplateConfigHeuristics
):
    """The tilings a discrete accelerator is measured with, for a product
    whose programs stay resident.

    The tilings are the same as for the ordinary product; what differs is how
    many programs there are.  A kernel that keeps a fixed number resident
    launches as many as the device can hold at once and gives each a share of
    the tiles, so the count is part of what is being chosen rather than a
    detail of the launch -- and it is a property of the device, not of the
    call, so it arrives with the configuration.
    """

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs, op_name
        ):
            yield {**template_kwargs, "NUM_SMS": get_num_sms()}


@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, None, op_name="addmm"
)
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics):
    """No contiguous rewrite, and saying so rather than saying nothing.

    Registered for every device so that the question is answered everywhere: a
    device with no answer falls back to the same rule that answers for nothing,
    and a device that means it has none should not be reached by accident.
    """


@register_template_heuristic(mm_contiguous_subgraph_template.uid, "cuda")
class ContiguousMMConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """The contiguous rewrite, where there is something for it to do.

    Offered only when the right operand is not already contiguous and the shape
    is one where reading it as a contiguous one pays: the rewrite copies it into
    that shape so the kernel can address a tile of it directly, and a copy that
    is not repaid by fewer reads is a copy that was not worth making.
    """

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        from ...utils import sympy_product, use_contiguous

        mat1, mat2 = kernel_inputs.mat1mat2()
        if mat2.get_layout().is_contiguous():
            return
        rows, cols, inner = kernel_inputs.mnk_symbolic()
        if not use_contiguous(rows, cols, inner):
            return
        yield {}

