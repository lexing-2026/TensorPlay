"""Two products added together, sharing what they can.

The pair is walked by one set of programs over a table of tiles that covers both,
and the two products only have to be told apart when their tiles disagree about
which block of the contraction they are in.
"""

from __future__ import annotations

import tensorplay as tp
import logging

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

from typing import Any
from .triton import CHOICES
from .select_algorithm import (
    ChoiceCaller,
    ExternChoiceCaller,
    ExternKernelChoice,
    KernelArgs,
    TritonChoiceCaller,
    TritonTemplate,
)
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..ir import Layout
from .triton import CHOICES, dtype_size

from .select_algorithm import TritonTemplate
from .mm import (
    GemmConfigHeuristics,
    GemmTemplate,
    MMKernelInputs,
    _gemm_source_identity,
    contiguous_stride,
    mm_grid,
)
from .mm_common import descriptor_extents_fit, descriptor_offset_fits

class MmPlusMmConfigHeuristics(GemmConfigHeuristics):

    """The candidates for two products measured as one.

    The tiles are fitted to the deeper of the two contractions, because that is
    how many steps each program's loop takes; the shallower one leaves the loop
    early and does not change which tile is the right one.  Whether the
    contraction divides the tile has to hold for both, since one loop guard
    serves both walks and a guard that is right for one and wrong for the other
    is a wrong answer.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        shapes = inputs.shapes
        return (
            len(shapes) == 4
            and all(len(shape) == 2 for shape in shapes)
            and (shapes[0][0], shapes[0][1]) == (shapes[2][0], shapes[2][1])
            and (shapes[1][0], shapes[1][1]) == (shapes[3][0], shapes[3][1])
            and shapes[0][1] == shapes[1][0]
        )

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        shapes = kernel_inputs.shapes
        rows, inner_a = (int(v) for v in shapes[0])
        _inner_b, cols = (int(v) for v in shapes[1])
        inner_c = int(shapes[2][1])
        deeper = max(inner_a, inner_c)
        for config in CHOICES.get_mm_configs(self.device_type)(
            rows, cols, deeper, dtype_size=dtype_size(kernel_inputs.dtype(0))
        ):
            block_k = int(config.kwargs["BLOCK_K"])
            yield {
                "choice": "triton",
                "GROUP_M": 8,
                "EVEN_K": inner_a % block_k == 0 and inner_c % block_k == 0,
                **config.as_kwargs(),
            }

class MmPlusMmTemplate(TritonTemplate):

    """Two products of the same shape, into one result.

    What it is for is a call holding two products whose results are added, and
    which the compiler could not see through: two products of the same extents
    in one region.  Measured as a single choice against two launches of the
    plain product, the form can lose and cannot be wrong -- and the two products
    have to be the same shape, because one tile and one accumulator serve both
    walks, so a pair that differs in any extent is two products and not this.
    """

    inputs_class = MMKernelInputs

    #: The block extents the body reads, in the order the signature declares
    #: them after the operands' extents and strides.
    block_names = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "EVEN_K")

    def __init__(self, name: str = "mm_plus_mm", **kwargs: Any):
        super().__init__(name, **kwargs)
        self.heuristics = MmPlusMmConfigHeuristics()

    def out_specs(self, meta: dict) -> tuple:
        size = tuple(int(v) for v in (meta.get("out_size") or ()))
        if len(size) != 2:
            raise NotImplementedError("two products of one shape have one result shape")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                size,
                contiguous_stride(size),
            ),
        )

    def _kernel_shapes(self, layout, meta: dict):
        """The four operands and the result, or ``None`` when they do not meet.

        The two pairs have to be the same shape, both because one tile and one
        result serve both walks and because the two walks are then the same
        walk.  A pair that differs in any extent is two products and not this
        one, so it is two products' worth of launches -- which is what the
        operator is.
        """

        sizes = tuple(tuple(int(v) for v in shape)
                      for shape in (meta.get("operand_sizes") or ()))
        if len(sizes) != 4 or any(len(shape) != 2 for shape in sizes):
            return None
        if         sizes[0] != sizes[2] or sizes[1] != sizes[3]:
            return None
        m, inner = sizes[0]
        n = sizes[1][1]
        if inner != sizes[1][0]:
            return None
        if (m, n) != tuple(int(v) for v in layout.size):
            return None
        return (sizes[0], sizes[1], sizes[2], sizes[3], tuple(layout.size))

    def generate(self, params, out_specs, meta, plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        kwargs = params.to_kwargs()
        layout = out_specs[0] if out_specs else None
        if kwargs.get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_two_products",
                layout=layout,
                description="the operation itself",
                launcher=plain_launch,
            )
        if layout is None or len(layout.size) != 2:
            return None
        if meta.get("operand_dtype") != "float32" or plain_launch is None:
            return None
        if not meta.get("qualifies", False):
            return None
        shapes = self._kernel_shapes(layout, meta)
        specs = tuple(meta.get("operand_specs") or ())
        if shapes is None or len(specs) != 4:
            return None
        block = tuple((name, kwargs.get(name)) for name in self.block_names)
        if any(value is None for _name, value in block):
            return None
        from ..codegen.triton_gemm import mm_plus_mm_launch

        caller = TritonChoiceCaller(
            name=f"{self.name}-{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}"
            f"x{kwargs['BLOCK_K']}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
            source=self.source,
            src_hash=self.src_hash,
        )
        return caller.bind(
            mm_plus_mm_launch(
                plain_launch,
                specs,
                layout.size,
                self.kernel_for(block, shapes, bool(kwargs.get("allow_tf32"))),
                mm_grid,
                block,
                num_warps=int(kwargs.get("num_warps", 4)),
                num_stages=int(kwargs.get("num_stages", 3)),
            )
        )

    def kernel_for(self, block: tuple, shapes: tuple, allow_tf32: bool):
        """The body, rendered over the two matrices and their result."""

        first, second, third, fourth, size = shapes
        args = KernelArgs(
            {
                name: {"shape": shape, "stride": contiguous_stride(shape)}
                for name, shape in zip(
                    ("A", "B", "C", "D"), (first, second, third, fourth)
                )
            },
            {"E": {"shape": size, "stride": contiguous_stride(size)}},
            dict(block),
        )
        source = self.render_with(args, allow_tf32=bool(allow_tf32))
        return self.kernel_type.get(
            self.name, source, self.symbol, self.grid, {"block": dict(block)}
        ).build()

MM_PLUS_MM = MmPlusMmTemplate.from_file(
    "mm_plus_mm",
    file="triton_mm_plus_mm",
    grid=mm_grid,
    symbol="_mm_plus_mm_kernel",
    cache_codegen_enabled_for_template=True,
)

#: The candidates this module offers.
#: The operation namespace, under a name of this project's own, reached
#: by whatever name this project's operations are registered under.
framework = tp.ops.tp

#: A product added to a product, as one call.  A choice of its own because
#: it is one call to the framework and not two: a caller who asked for
#: the sum of two products did not ask to have them summed afterwards, and
#: a kernel for it can keep the second product in registers.
framework_mm_plus_mm = ExternKernelChoice(None, "mm_plus_mm")

MM_PLUS_MM_TEMPLATES = (
    MM_PLUS_MM,
)


#: The template that computes a sum of two products, under the name the operation
#: is declared under.  The module keeps the name and the implementation apart so
#: that a lowering can hold the first without holding the second.
mm_plus_mm_template = MM_PLUS_MM


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None, plain=None):
    """The sum of two products, or the two products added the ordinary way.

    The two products are walked by one set of programs over a table of tiles that
    covers both, which is worth doing only while they are the same shape: a pair
    that disagrees is two products, and one body walking two different tilings
    would be a body that is right about neither.

    What comes back is the candidates to measure, which is what a lowering holds
    and chooses from -- not a decision, because a decision made here would be a
    decision made before anything was measured.
    """

    from .mm_common import mm_args

    first = mm_args(mat1, mat2, layout=layout)
    second = mm_args(mat3, mat4, layout=layout)
    m1, n1 = first[0], first[1]
    m2, n2 = second[0], second[1]
    out_size = first[3].get_size()
    if (m1, n1) != (m2, n2) or m1 * n1 == 0 or plain is None:
        return None
    meta = {
        "out_size": tuple(int(v) for v in out_size),
        "out_dtype": str(first[3].get_dtype()),
        "device": first[3].get_device(),
        "operand_specs": ((0, None), (1, None), (2, None), (3, None)),
        "operand_sizes": (
            tuple(int(v) for v in mat1.get_size()),
            tuple(int(v) for v in mat2.get_size()),
            tuple(int(v) for v in mat3.get_size()),
            tuple(int(v) for v in mat4.get_size()),
        ),
        "operand_dtype": str(mat1.get_dtype()),
        "qualifies": True,
        "arg_templates": (),
        "call_method": False,
    }
    return MM_PLUS_MM.configurations(MM_PLUS_MM.out_specs(meta), meta), meta
