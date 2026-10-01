"""What is known about an operation the framework already has a kernel for.

There is exactly one thing to say about such an operation: it can be run, once,
in the way the framework runs it.  There is no search to do and no shape to
choose between, so the rule yields a single configuration with nothing in it.

The point of writing that down as a rule rather than treating these choices
specially at each place they are offered is that they stop being a special case.
A template is measured by asking the same question of a list that may hold
either kind, and a list that had to be walked differently depending on what was
in it would decide the outcome by which kind came first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..registry import register_template_heuristic
from ...templates.bmm import (
    framework_bmm,
    framework_bmm_dtype,
    framework_int_mm,
    framework_mm as framework_bmm_mm,
    framework_mm_dtype as framework_bmm_mm_dtype,
    framework_sparse_semi_structured_mm as framework_bmm_sparse,
)
from ...templates.conv import (
    framework_conv1x1_via_mm,
    framework_convolution,
    framework_convolution_backward,
    framework_dw,
    framework_dx,
)
from ...templates.mm import (
    framework__int_mm,
    framework__sparse_semi_structured_mm,
    framework_addmm,
    framework_bias_addmm,
    framework_fp8_mm,
    framework_mm,
    framework_mm_dtype,
    framework_scaled_mm,
)
from ...templates.mm_grouped import (
    framework__grouped_mm,
    framework__scaled_grouped_mm,
)
from ...templates.mm_plus_mm import framework_mm_plus_mm
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics

if TYPE_CHECKING:
    from collections.abc import Generator


# A rule registered for no device in particular is one that holds wherever
# nothing more specific has been said, which is what an operation the framework
# can always run is: the shape may rule it out, and that is asked of the
# configuration rather than of the registration.
@register_template_heuristic(framework_mm.uid, None)
@register_template_heuristic(framework_mm_dtype.uid, "cuda")
@register_template_heuristic(framework_fp8_mm.uid, None)
@register_template_heuristic(framework__int_mm.uid, None)
@register_template_heuristic(framework__sparse_semi_structured_mm.uid, None)
@register_template_heuristic(framework__grouped_mm.uid, None)
@register_template_heuristic(framework__scaled_grouped_mm.uid, None)
@register_template_heuristic(framework_convolution.uid, None)
@register_template_heuristic(framework_conv1x1_via_mm.uid, None)
@register_template_heuristic(framework_dw.uid, None)
@register_template_heuristic(framework_dx.uid, None)
@register_template_heuristic(framework_convolution_backward.uid, None)
@register_template_heuristic(framework_bmm.uid, None)
@register_template_heuristic(framework_bmm_mm.uid, None)
@register_template_heuristic(framework_bmm_mm_dtype.uid, "cuda")
@register_template_heuristic(framework_int_mm.uid, None)
@register_template_heuristic(framework_bmm_sparse.uid, None)
class TPConfigHeuristics(TemplateConfigHeuristics):
    """One candidate, which is the operation itself.

    Not a rule in the usual sense -- there is no table of configurations to
    choose from -- but shaped like one so that a choice the framework provides
    is offered, measured and possibly chosen by exactly the same path as a
    choice the compiler could write.  A subclass is the way to add something
    real, for an operation whose kernel takes arguments that are not part of
    choosing between alternatives.
    """

    def _get_template_configs_impl(self, kernel_inputs, op_name) -> "Generator[dict, None, None]":
        yield dict()


@register_template_heuristic(framework_addmm.uid, None, op_name="addmm")
@register_template_heuristic(framework_bias_addmm.uid, None, op_name="addmm")
class TPAddMMConfigHeuristics(TPConfigHeuristics):
    """A product with something added, where the coefficients are not a choice.

    The scaling a product carries is fixed by the operation that asked for it, so
    it is not one of the things being chosen between; it is passed on as what the
    kernel is told, which is a different question from which kernel to run.
    """

    def get_extra_kwargs(self, kernel_inputs, op_name) -> dict:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {**kwargs, "alpha": alpha, "beta": beta}


@register_template_heuristic(framework_scaled_mm.uid, None, op_name="_scaled_mm")
@register_template_heuristic(framework_fp8_mm.uid, None, op_name="_scaled_mm")
class TPScaledMMConfigHeuristics(TPConfigHeuristics):
    """A product whose operands carry their own scale, on a device that has it."""


@register_template_heuristic(framework_mm_plus_mm.uid, None, op_name="mm_plus_mm")
class TPMMPlusMMConfigHeuristics(
    TPConfigHeuristics, GemmMaxAutotuneTemplateConfigHeuristics
):
    """Two products added together, worth searching for only when asked.

    Whether the program wants to spend time measuring is not a property of this
    operation, so the question is asked of the shared rule rather than answered
    again here.
    """
