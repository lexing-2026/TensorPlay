"""Which configurations of one template are worth trying for one call.

The table says what exists and the rule below says which of it fits here.  A tile
wider than the problem it tiles is wasted lanes and a mask paid for on every step, so
each configuration is narrowed to the shape it will actually run against.
"""

from __future__ import annotations

import inspect

import sympy

from ......graph.experimental.sympy_functions import CeilDiv

import functools
from typing import Any, Callable, Iterator, Sequence

from ...op_lowerings import ceildiv
from ...runtime.runtime_utils import next_power_of_2
from .params import DictKernelTemplateParams

class SymbolicGridFn:
    """A grid function whose extents are symbolic rather than concrete.

    A grid is asked for its shape before the shapes it depends on are known --
    a tile that will be measured against several problems has to be able to say
    how many programs it would take for each of them.  So the extents arrive
    as expressions, and the function returns expressions; the caller
    substitutes numbers when it finally launches.

    The decoration is what records that a grid was written this way, so that
    code which needs the distinction can see it without inferring it from the
    body.
    """

    def __init__(self, fn):
        self.fn = fn
        self.symbolic = True
        functools.update_wrapper(self, fn)
        # Which of the rounding and clamping helpers the body asked for, and
        # what each becomes once the extents are known.  A grid is written once
        # and used both ways -- asked for expressions while a template is being
        # offered, and for numbers when it finally launches -- so the body names
        # what it needs and the two forms are supplied from the same signature.
        params = inspect.signature(fn).parameters
        self.kwargs_int = {}
        self.kwargs_sym = {}
        for name, fn_sym, fn_int in (
            ("cdiv", CeilDiv, ceildiv),
            ("min", sympy.Min, min),
            ("max", sympy.Max, max),
        ):
            if name in params:
                self.kwargs_int[name] = fn_int
                self.kwargs_sym[name] = fn_sym

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs, **self.kwargs_int)

    def __get__(self, instance, owner=None):
        return self
class TemplateConfigHeuristics:
    """Which configurations of one template are worth trying for one call.

    Splitting this out is what lets a table be written once and shared: the
    table says what exists, and the rule below says which of it fits here.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        """Whether this heuristic has anything to say about this call."""

        return True

    def get_template_configs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> Iterator[KernelTemplateParams]:
        """The configurations for this call, as parameters."""

        if not self.should_run(kernel_inputs):
            return
        for config_dict in self._get_template_configs_impl(kernel_inputs, op_name):
            yield DictKernelTemplateParams(config_dict)

    def _get_template_configs_impl(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> Iterator[dict[str, Any]]:
        """The configurations for this call, as keyword arguments."""

        return iter(())

    def get_extra_kwargs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> dict[str, Any]:
        """What the template needs whatever configuration it is given.

        These are the parts of the call that do not vary with the tile: the
        geometry of a convolution, the layout a result lands in.  They are
        passed alongside every configuration rather than inside it, so that
        they are not mistaken for part of the choice.
        """

        return {}

    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs:
        """The inputs as this template wants to see them.

        A template that needs a matrix where the call has a batch of them can
        add the axis here, once, instead of at every use.
        """

        return kernel_inputs
