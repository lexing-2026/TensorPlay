"""Products, measured against the framework's own.

A product is a product whether it arrives as a matrix multiply, an add, a batch, or
a layer's weight, so all of them are one template and one tile space rather than four
templates and four tables.
"""

from __future__ import annotations

from typing import Any, Iterator

from .bridge import LoopTemplate
from .choices import CHOICES
from .heuristics import TemplateConfigHeuristics
from .ir import (
    ExternChoiceCaller,
    KernelInputs,
    Layout,
    MMKernelInputs,
    TritonChoiceCaller,
    contiguous_stride,
)
from .params import KernelTemplateParams

class GemmConfigHeuristics(TemplateConfigHeuristics):
    """The candidates for a product, fitted to the product's size.

    The operator is always the first candidate: it is the floor a measurement
    can never lose against, and having it in the list is what makes the rest
    of the list safe to measure.
    """

    def __init__(self, op_name: str = "mm", device_type: str = "cuda"):
        self.op_name = op_name
        self.device_type = device_type

    def should_run(self, inputs: KernelInputs) -> bool:
        return isinstance(inputs, MMKernelInputs)

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        rows, cols, inner = kernel_inputs.mnk_symbolic()
        generator = CHOICES.get_mm_configs(self.device_type)
        for config in generator(rows, cols, inner):
            yield {"choice": "triton", **config.as_kwargs()}
class GemmTemplate(LoopTemplate):
    """Products, measured against the framework's own.

    Validity belongs to the template: a configuration that does not fit this
    call -- not two-dimensional, not the element type the tiles accumulate in,
    not this device -- is refused here rather than by whoever is asking.
    """

    inputs_class = MMKernelInputs

    def __init__(self):
        super().__init__("gemm")
        self.heuristics = GemmConfigHeuristics()

    def emitter(self) -> str:
        """The tile kernel this template emits, identified by its source.

        The digest covers the kernel body and the tuning version, so a stored
        decision is not reused once either has moved.
        """

        from ..codegen.triton_gemm import GEMM_TUNING_VERSION, _kernel_source_digest

        return f"{GEMM_TUNING_VERSION}:{_kernel_source_digest()}"

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if size is None:
            raise NotImplementedError("a product without a result shape")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates.

        The template owns its result, so it also knows the extents a product
        has and can build the feed that exercises every lane of a candidate
        without being handed the region's real tensors.
        """

        from ..codegen.triton_gemm import _probe_feed

        if meta.get("b_transposed") or len(meta.get("operand_specs", ())) != 2:
            return None
        layout = self.out_specs(meta)[0]
        if len(layout.size) != 2:
            return None
        sizes = meta.get("operand_sizes") or ()
        if len(sizes) != 2:
            return None
        (rows, inner), (inner2, cols) = sizes
        m, n = layout.size
        if rows != m or cols != n or inner != inner2:
            return None
        if meta.get("out_dtype") is None:
            return None
        return _probe_feed(m, inner, n, layout.dtype, meta.get("device"), bias=False)

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        kwargs = params.to_kwargs()
        layout = out_specs[0] if out_specs else None
        if kwargs.get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_product",
                layout=layout,
                description="the operation itself",
                launcher=plain_launch,
            )
        if layout is None or len(layout.size) != 2 or meta.get("transposed"):
            return None
        if meta.get("operand_dtype") != "float32":
            return None
        if not meta.get("qualifies", False):
            return None
        from ..codegen.triton_gemm import tuned_matmul_launch

        caller = ChoiceCaller(
            name=f"gemm-{kwargs.get('BLOCK_M')}x{kwargs.get('BLOCK_N')}"
            f"x{kwargs.get('BLOCK_K')}",
            layout=layout,
            description=repr(kwargs),
        )
        caller.config_patches = {
            key: kwargs[key]
            for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages")
            if key in kwargs
        }
        return caller.bind(
            tuned_matmul_launch(
                None,
                meta.get("probe_feed") or meta.get("feed") or (),
                meta["operand_specs"],
                layout.size,
                bias_spec=meta.get("bias_spec"),
                b_transposed=bool(meta.get("b_transposed", False)),
                config=kwargs,
            )
        )


GEMM = GemmTemplate()
