"""Which configurations of a batched product are worth measuring.

A product with a batch in front of both matrices is a separate question from one
without, not another answer to the same question: the batch is a grid axis here,
so a configuration carries extents the un-batched one has no word for, and a
table written for the un-batched one cannot be reused by widening it.  So the
two have tables of their own here rather than one table with a flag.

What a configuration decides that the kernel body cannot is decided by rendering
-- whether the contraction divides the tile, which depends on the tile -- so the
table below carries that as part of each entry rather than leaving it to the
launcher to work out per shape.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_template_heuristic
from ...templates.bmm import BMM, BMM_SHARED_A
from ...templates.select_algorithm import CHOICES
from ...templates.triton import dtype_size
from .base import TemplateConfigHeuristics


class _BatchedProductConfigs(TemplateConfigHeuristics):
    """The tile table for one of the batched templates.

    How many rows of tiles a program walks before it advances a column is a
    property of the launch rather than of the call, so it is stated once here and
    every configuration of every shape uses it.  It is stated rather than left
    out because the kernel body reads it: a configuration that did not say what
    to read would be a kernel this template cannot render.
    """

    #: How many rows of tiles a program walks before advancing a column.
    group_m = 8

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        rows, cols, inner = kernel_inputs.mnk_symbolic()
        device_type = kernel_inputs.device_type
        for config in CHOICES.get_mm_configs(device_type)(
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
        tile: two configurations of this call cannot disagree about it.  It is
        said once for all of them, and it is said at all, because a kernel that
        quietly picked a precision would put the decision in the body where a
        measurement could not see it.
        """

        from ...codegen.triton_gemm import _matmul_allow_tf32

        return {"allow_tf32": _matmul_allow_tf32()}


@register_template_heuristic(BMM.uid, "cuda", op_name="bmm")
class BMMConfigHeuristics(_BatchedProductConfigs):
    """The plain batched template, which is worth offering for a batch of one too.

    A batch of one is a product that happens to carry a leading extent, and the
    un-batched table answers it; offering both is what lets the measurement say
    which of the two the shape actually wanted.
    """

    def should_run(self, inputs) -> bool:
        return super().should_run(inputs) and len(inputs.batch()) == 1


@register_template_heuristic(BMM_SHARED_A.uid, "cuda", op_name="bmm")
class BMMSharedAConfigHeuristics(_BatchedProductConfigs):
    """The template for a left operand that is one matrix read across the batch.

    A different launch rather than a different tile: the batch is not a grid
    axis, so the extents a configuration carries are not the same set as the
    plain template's, and sharing a table between the two would be sharing a
    claim about the launch as well as about the tile.
    """
