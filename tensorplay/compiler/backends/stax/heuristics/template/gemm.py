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

from ... import config as inductor_config
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
        return inductor_config.max_autotune or inductor_config.max_autotune_gemm
