"""Convolutions, forward and both ways back.

A convolution is a product over the input channels, so it is sized by the product
table; the one-by-one case is a product outright; and a depthwise call is not a
product at all, which is the one form that needs a table of its own.
"""

from __future__ import annotations

from typing import Any

from .bridge import LoopTemplate
from .choices import CHOICES
from .heuristics import TemplateConfigHeuristics
from .ir import (
    ConvKernelInputs,
    ExternChoiceCaller,
    KernelInputs,
    Layout,
    TritonChoiceCaller,
    contiguous_stride,
)
from .mm import GEMM
from .params import KernelTemplateParams
from .select import ExternKernelChoice

def _pair(value, count: int) -> list:
    if isinstance(value, int):
        return [int(value)] * count
    values = [int(v) for v in value]
    if len(values) == 1:
        return values * count
    return values
def _conv_output_size(extent: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (extent + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
class ConvConfigHeuristics(TemplateConfigHeuristics):
    """The candidates for a convolution, fitted to what it convolves.

    A convolution is a product over the input channels, so its tilings are
    fitted to the product the way a product's are: the rows are the output
    positions, the columns the output channels, and the contraction the input
    channels.  The positions are counted across the batch and the whole
    spatial extent together, because a tile does not care where a position
    came from.
    """

    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type

    def should_run(self, inputs: KernelInputs) -> bool:
        return isinstance(inputs, ConvKernelInputs) and not inputs.is_depthwise

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        try:
            rows, cols, inner = kernel_inputs.mnk_symbolic()
        except NotImplementedError:
            return
        for config in CHOICES.get_conv_configs(self.device_type)(rows, cols, inner):
            yield {"choice": "triton", **config.as_kwargs()}

    def get_depthwise_configs_impl(self, kernel_inputs):
        """The depthwise tilings, which are not products and so are not fitted."""

        yield {"choice": "operator"}
        for config in CHOICES.get_depthwise_conv_configs(self.device_type):
            yield {"choice": "triton", **config.as_kwargs()}
class DepthwiseConvTemplate(LoopTemplate):
    """The depthwise convolution, which is not a product and does not use one.

    No output channel is a sum over input channels here, so there is no
    contraction to tile and nothing for a matrix multiply to do.  The work
    steps over images, positions along the axis and channels instead, and each
    lane keeps its own accumulator -- so the three-dimensional tilings are the
    only candidates, and the product table is not consulted at all.
    """

    inputs_class = ConvKernelInputs

    def __init__(self):
        super().__init__("depthwise_conv1d")
        self.heuristics = DepthwiseConvConfigHeuristics()

    def emitter(self) -> str:
        return "depthwise:operator+block_n/l/c"

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("a depthwise convolution without a result")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def geometry_for(self, meta: dict) -> dict:
        kernel = tuple(int(k) for k in meta.get("kernel_size") or ())
        if len(kernel) != 1:
            raise NotImplementedError("a depthwise convolution that is not one-dimensional")
        return {
            "kernel": kernel,
            "stride": tuple(int(v) for v in meta.get("stride") or (1,)),
            "padding": tuple(int(v) for v in meta.get("padding") or (0,)),
            "dilation": tuple(int(v) for v in meta.get("dilation") or (1,)),
            "groups": int(meta.get("groups", 1) or 1),
        }

    def generate_for(self, params, out_specs, meta, plain_launch=None):
        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_depthwise",
                layout=out_specs[0] if out_specs else None,
                description="the operation itself",
                launcher=plain_launch,
            )
        kwargs = params.to_kwargs()
        if "BLOCK_N" not in kwargs or meta.get("transposed"):
            return None
        if len(meta.get("operand_specs") or ()) < 2:
            return None
        try:
            geometry = self.geometry_for(meta)
        except NotImplementedError:
            return None
        from ..codegen.triton_conv import depthwise_launch

        layout = out_specs[0] if out_specs else None
        caller = TritonChoiceCaller(
            name=f"depthwise-{kwargs['BLOCK_N']}x{kwargs['BLOCK_L']}x{kwargs['BLOCK_C']}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
            source=repr(sorted(geometry.items())),
        )
        return caller.bind(
            depthwise_launch(
                meta["operand_specs"][0],
                meta["operand_specs"][1],
                meta.get("bias_spec"),
                geometry,
                kwargs,
                plain_launch,
            )
        )
class DepthwiseConvConfigHeuristics(TemplateConfigHeuristics):
    """The depthwise tilings, which are not fitted because there is nothing
    to fit them to.

    A product's tile is fitted to the problem because a tile wider than the
    problem is wasted lanes.  Here the innermost block is over channels and
    the others over positions, and the table is already small and already
    fixed, so narrowing it per call would buy nothing and hide which tilings
    exist.
    """

    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type

    def should_run(self, inputs: KernelInputs) -> bool:
        return isinstance(inputs, ConvKernelInputs) and inputs.is_depthwise

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        for config in CHOICES.get_depthwise_conv_configs(self.device_type):
            yield {"choice": "triton", **config.as_kwargs()}
class ConvTemplate(LoopTemplate):
    """Convolutions, with the 1x1 case expressed as the product it is.

    A one-by-one convolution with unit stride, no padding and one group is a
    matrix product, so it is measured as one -- the product template's tile
    set applies to it directly, and there is nothing to be gained from a
    second set of candidates for the same arithmetic.

    Wider kernels keep the operator, which is the floor a measurement can never
    lose against, until there is a tiled kernel for them to choose between.
    """

    inputs_class = ConvKernelInputs

    def __init__(self):
        super().__init__("conv")
        self.heuristics = ConvConfigHeuristics()

    def emitter(self) -> str:
        return "conv:operator+product-1x1"

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("a convolution without a result shape")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def is_one_by_one(self, meta: dict) -> bool:
        """Is this the product-shaped case: 1x1 kernel, unit stride, one group?"""

        kernel = tuple(int(k) for k in meta.get("kernel_size") or ())
        if not kernel or any(k != 1 for k in kernel):
            return False
        stride = tuple(int(s) for s in meta.get("stride") or ())
        if any(s != 1 for s in stride):
            return False
        padding = tuple(int(p) for p in meta.get("padding") or ())
        if any(p != 0 for p in padding):
            return False
        if int(meta.get("groups", 1)) != 1:
            return False
        if meta.get("transposed"):
            return False
        dilation = tuple(int(d) for d in meta.get("dilation") or ())
        return all(d == 1 for d in dilation)

    def geometry_for(self, meta: dict) -> dict:
        """Everything about the call that is not a choice.

        The kernel, its stride, padding and dilation, the groups, and the
        result's spatial extents.  These are handed to every configuration
        alike, which is what keeps them from being mistaken for part of the
        choice -- and getting one of them wrong is not something a measurement
        would catch, because a kernel that reads the wrong geometry is fast and
        confidently incorrect.
        """

        kernel = tuple(int(k) for k in meta.get("kernel_size") or ())
        if not kernel:
            raise NotImplementedError("a convolution without a kernel")
        stride = tuple(int(v) for v in meta.get("stride") or (1,) * len(kernel))
        padding = tuple(int(v) for v in meta.get("padding") or (0,) * len(kernel))
        dilation = tuple(int(v) for v in meta.get("dilation") or (1,) * len(kernel))
        size = tuple(int(v) for v in (meta.get("out_size") or ()))
        spatial_out = size[2:]
        return {
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": int(meta.get("groups", 1) or 1),
            "out_size": spatial_out,
            "unroll": True,
        }

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_convolution",
                layout=out_specs[0] if out_specs else None,
                description="the operation itself",
                launcher=plain_launch,
            )
        if self.is_one_by_one(meta):
            # The one-by-one case is a product, and the product template owns
            # that tile space; this template only decides that the case
            # applies.  The product's layouts are permuted to get there, so the
            # caller is told which way round the result lands.
            return self._one_by_one_choice(out_specs, meta, plain_launch)
        return self._tiled_choice(params, out_specs, meta, plain_launch)

    def _one_by_one_choice(self, out_specs, meta, plain_launch):
        """The one-by-one case, measured as the product it is."""

        kernel = tuple(meta.get("kernel_size") or ())
        rank = len(kernel) + 2
        weight = (meta.get("operand_sizes") or ((), ()))[1]
        moved = tuple(range(0, rank - 1)) + (rank - 1,)
        out_size = tuple(int(v) for v in (out_specs[0].size if out_specs else ()))
        gemm_meta = {
            "out_size": out_size,
            "out_dtype": out_specs[0].dtype if out_specs else None,
            "device": out_specs[0].device if out_specs else None,
            "operand_dtype": meta.get("operand_dtype"),
            "operand_sizes": meta.get("operand_sizes"),
            "operand_specs": meta.get("operand_specs"),
            "bias_spec": meta.get("bias_spec"),
            "qualifies": True,
            "probe_feed": meta.get("probe_feed"),
            "feed": meta.get("feed"),
        }
        launcher, _params = GEMM.select(gemm_meta, lambda choice, plain: choice.resolve(plain))
        if launcher is None:
            return None
        caller = ExternChoiceCaller(
            name="conv1x1_via_product",
            layout=out_specs[0] if out_specs else None,
            description=(
                "a one-by-one convolution measured as the product it is; the "
                f"result arrives in the order {moved}"
            ),
            launcher=launcher,
        )
        return caller

    def _tiled_choice(self, params, out_specs, meta, plain_launch):
        """A wider kernel, measured over the tiles that fit it."""

        from ..codegen.triton_conv import conv_launch

        kwargs = params.to_kwargs()
        if "BLOCK_M" not in kwargs:
            return None
        if meta.get("operand_dtype") not in ("float32",):
            return None
        if meta.get("transposed"):
            return None
        if len(meta.get("operand_specs") or ()) < 2:
            return None
        try:
            geometry = self.geometry_for(meta)
        except NotImplementedError:
            return None
        layout = out_specs[0] if out_specs else None
        caller = TritonChoiceCaller(
            name=f"conv2d-{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}x{kwargs['BLOCK_K']}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
            source=repr(sorted(geometry.items())),
        )
        return caller.bind(
            conv_launch(
                meta["operand_specs"][0],
                meta["operand_specs"][1],
                meta.get("bias_spec"),
                geometry,
                kwargs,
                plain_launch,
            )
        )
class _ConvGradientTemplate(LoopTemplate):
    """What a convolution's two gradients share.

    Both are the forward's product with the contraction moved -- the forward
    contracts over input channels, the input's gradient over output channels,
    and the weight's gradient over positions -- so both are sized by the same
    table and differ only in what the extents mean.  The geometry is the
    forward's geometry: a gradient is about the call that produced it, so the
    kernel, stride, padding, dilation and groups are the same numbers.
    """

    inputs_class = ConvKernelInputs

    def __init__(self, name: str, which: str):
        super().__init__(name)
        self.which = which
        self.heuristics = ConvGradientConfigHeuristics(which)

    def emitter(self) -> str:
        return f"conv2d_bwd_{self.which}:operator+tiles"

    def geometry_for(self, meta: dict) -> dict:
        kernel = tuple(int(k) for k in meta.get("kernel_size") or ())
        if len(kernel) != 2:
            raise NotImplementedError("only a two-dimensional gradient is tiled")
        from ..codegen.triton_conv import _grad_geometry

        sizes = tuple(meta.get("operand_sizes") or ())
        if len(sizes) < 2:
            raise NotImplementedError("a gradient without its operands' extents")
        return _grad_geometry(
            {
                "kernel": kernel,
                "stride": tuple(int(v) for v in meta.get("stride") or (1, 1)),
                "padding": tuple(int(v) for v in meta.get("padding") or (0, 0)),
                "dilation": tuple(int(v) for v in meta.get("dilation") or (1, 1)),
                "groups": int(meta.get("groups", 1) or 1),
            },
            sizes[0][-2:],
            sizes[1][0],
            sizes[0][1],
        )

    def _launcher(self, params, meta, plain_launch):
        kwargs = params.to_kwargs()
        if "BLOCK_M" not in kwargs or meta.get("transposed"):
            return None
        if meta.get("operand_dtype") not in ("float32",):
            return None
        try:
            geometry = self.geometry_for(meta)
        except NotImplementedError:
            return None
        from ..codegen.triton_conv import (
            conv_bwd_input_launch,
            conv_bwd_weight_launch,
        )

        specs = tuple(meta.get("operand_specs") or ())
        if len(specs) < 2:
            return None
        if self.which == "input":
            # The gradient and the weight: the forward's own operands.
            launcher = conv_bwd_input_launch(
                specs[0], specs[1], kwargs, geometry, plain_launch
            )
        else:
            # The gradient and the input, in that order, because the weight's
            # gradient needs the forward's *input* rather than its weight.
            launcher = conv_bwd_weight_launch(
                specs[0], specs[1], kwargs, geometry, plain_launch
            )
        return launcher

    def _caller(self, params, out_specs, launcher):
        kwargs = params.to_kwargs()
        return TritonChoiceCaller(
            name=f"{self.name}-{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}x{kwargs['BLOCK_K']}",
            layout=out_specs[0] if out_specs else None,
            description=repr(sorted(kwargs.items())),
            source=f"{self.which}:{sorted(kwargs.items())}",
        ).bind(launcher)
class ConvGradientConfigHeuristics(TemplateConfigHeuristics):
    """The tilings for one gradient, fitted to what that gradient contracts.

    The input's gradient contracts over output channels once per tap, and the
    weight's contracts over the output's positions; either way the table is
    the convolution table, because both are the same product read backwards.
    """

    def __init__(self, which: str, device_type: str = "cuda"):
        self.which = which
        self.device_type = device_type

    def should_run(self, inputs: KernelInputs) -> bool:
        return isinstance(inputs, ConvKernelInputs) and not inputs.is_depthwise

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        try:
            rows, cols, inner = kernel_inputs.mnk_symbolic()
        except NotImplementedError:
            return
        for config in CHOICES.get_conv_configs(self.device_type)(rows, cols, inner):
            yield {"choice": "triton", **config.as_kwargs()}
class ConvBwdInputTemplate(_ConvGradientTemplate):
    """The input's gradient: the sum of the windows that read each position.

    It is the forward's product with the contraction moved to the output
    channels, which means the output position a row was read at has to be
    solved for rather than stepped to.  The solve is the delicate part: a
    distance that does not divide by the stride contributes nothing, and a
    kernel that rounds instead of refusing is wrong only along the padded
    border.
    """

    def __init__(self):
        super().__init__("convolution2d_bwd_input", "input")

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("an input gradient without a result")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def generate_for(self, params, out_specs, meta, plain_launch=None):
        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_convolution_bwd",
                layout=out_specs[0] if out_specs else None,
                description="the operation itself",
                launcher=plain_launch,
            )
        if plain_launch is None:
            return None
        launcher = self._launcher(params, meta, plain_launch)
        return None if launcher is None else self._caller(params, out_specs, launcher)
class ConvBwdWeightTemplate(_ConvGradientTemplate):
    """The weight's gradient: one product per tap, over the output's positions.

    The tap indexes the weight rather than the accumulation, so one
    accumulator serves every tap and each is stored as it is finished: a
    kernel-wide tile per tap would need a register per tap, and the number of
    taps is the kernel's area.
    """

    def __init__(self):
        super().__init__("convolution2d_bwd_weight", "weight")

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("a weight gradient without a result")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def generate_for(self, params, out_specs, meta, plain_launch=None):
        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_convolution_bwd",
                layout=out_specs[0] if out_specs else None,
                description="the operation itself",
                launcher=plain_launch,
            )
        if plain_launch is None:
            return None
        launcher = self._launcher(params, meta, plain_launch)
        return None if launcher is None else self._caller(params, out_specs, launcher)
def conv1x1_via_mm(x, w, *, out=None):
    """A one-by-one convolution, written as the product it is.

    The weight's two trailing extents are one, so squeezing them is a view; the
    contraction is over the channels, which means the input has to be read
    with its channels last and the weight with its output channels first.  The
    result therefore lands in a permuted layout, and un-permuting it is part
    of the same work rather than something the caller does afterwards.
    """

    import tensorplay as tp

    w = w.reshape(w.shape[0], w.shape[1])
    moved = x.permute(0, 2, 3, 1) if x.dim() == 4 else x.permute(0, 2, 1)
    product = tp.matmul(moved, w.permute(1, 0), out=out)
    return product.permute(0, 3, 1, 2) if x.dim() == 4 else product.permute(0, 2, 1)
def _conv_forward_grid(rows, cols, groups, block_m, block_n, cdiv):
    return (cdiv(rows, block_m), cdiv(cols, block_n), groups)
def _conv_bwd_input_grid(rows, cols, groups, block_m, block_n, cdiv):
    return (cdiv(rows, block_m), cdiv(cols, block_n), groups)
def _conv_bwd_weight_grid(groups, block_m, block_n, block_k, cdiv):
    return (cdiv(1, block_m), cdiv(1, block_n), groups)


CONV = ConvTemplate()
DEPTHWISE_CONV = DepthwiseConvTemplate()
CONV_BWD_INPUT = ConvBwdInputTemplate()
CONV_BWD_WEIGHT = ConvBwdWeightTemplate()

framework_convolution = ExternKernelChoice(
    None, "framework_convolution", has_out_variant=False
)
conv1x1_via_product = ExternKernelChoice(conv1x1_via_mm, "conv1x1_via_mm")


CONV_TEMPLATES: dict[str, dict[str, Any]] = {
    "convolution2d": {
        "grid": _conv_forward_grid,
        "rows": "batch times the output's spatial extent",
        "cols": "output channels per group",
        "contraction": "input channels per group",
    },
    "convolution3d": {
        "grid": _conv_forward_grid,
        "rows": "batch times the output's spatial extent",
        "cols": "output channels per group",
        "contraction": "input channels per group",
    },
    "depthwise_conv1d": {
        "grid": None,
        "tiling": "output positions, positions along the axis, channels",
        "not": "a product: each channel is reduced on its own",
    },
    "convolution2d_bwd_input": {
        "grid": _conv_bwd_input_grid,
        "rows": "batch times the input's spatial extent",
        "cols": "input channels per group",
    },
    "convolution2d_bwd_weight": {
        "grid": _conv_bwd_weight_grid,
        "rows": "output channels per group",
        "cols": "input channels per group",
        "contraction": "batch times the input's spatial extent",
    },
}
