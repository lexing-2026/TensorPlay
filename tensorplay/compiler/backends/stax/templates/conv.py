"""Convolutions, forward and both ways back.

A convolution is a product over the input channels, so it is sized by the product
table; the one-by-one case is a product outright; and a depthwise call is not a
product at all, which is the one form that needs a table of its own.
"""

from __future__ import annotations
import logging

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

from typing import Any

import tensorplay as tp

from .triton import CHOICES, dtype_size

from ..codegen.common import KernelTemplate
from ..utils import sympy_product
from ..virtualized import V
from ..heuristics.template.base import SymbolicGridFn, TemplateConfigHeuristics

from .ir import contiguous_stride
from .. import ir
from ..ir import (
    FixedLayout,
    FlexibleLayout,
    Layout,
    convert_shape_to_tp,
)
from ..kernel_inputs import ConvKernelInputs, KernelInputs
from .select_algorithm import call_operation, TritonChoiceCaller

from .. import config
from ..op_lowerings import LOWERINGS, node_val, register, register_lowering, to_dtype
from .mm import GEMM, tuned_addmm, tuned_mm
from .mm_common import load_kernel_template, use_triton_template

from ..heuristics.template.params import DictKernelTemplateParams, KernelTemplateParams

from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)

def _pair(value, count: int) -> list:
    if isinstance(value, int):
        return [int(value)] * count
    values = [int(v) for v in value]
    if len(values) == 1:
        return values * count
    return values

def _conv_output_size(extent: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (extent + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

# ---------------------------------------------------------------------------
# where a convolution's result lands
# ---------------------------------------------------------------------------


class ConvLayoutParams:
    """Everything about a call that decides where its result lands.

    These six are the whole of it.  A result's extent and stride are not
    computed from the input's extents and a formula -- they are whatever the
    operation says, and the operation says it by being asked.  So the set of
    things a caller has to state is exactly this set, and nothing else can
    change the answer.
    """

    __slots__ = ("stride", "padding", "dilation", "transposed", "output_padding",
                 "groups")

    def __init__(self, stride=(), padding=(), dilation=(), transposed=False,
                 output_padding=(), groups=1):
        self.stride = tuple(int(v) for v in stride)
        self.padding = tuple(int(v) for v in padding)
        self.dilation = tuple(int(v) for v in dilation) or (1,) * len(self.stride)
        self.transposed = bool(transposed)
        self.output_padding = tuple(int(v) for v in output_padding)
        self.groups = int(groups)

    @classmethod
    def from_meta(cls, meta: dict) -> "ConvLayoutParams":
        """The parameters this call carried, or the ones a convolution means."""

        kernel = tuple(int(k) for k in meta.get("kernel_size") or (1,))
        width = len(kernel)
        return cls(
            stride=meta.get("stride") or (1,) * width,
            padding=meta.get("padding") or (0,) * width,
            dilation=meta.get("dilation") or (1,) * width,
            transposed=bool(meta.get("transposed", False)),
            output_padding=meta.get("output_padding") or (0,) * width,
            groups=int(meta.get("groups", 1) or 1),
        )


def channels_last_order(rank: int) -> tuple:
    """The order a channels-last tensor is read in.

    Reversed, then the innermost axis moved back out to where the channel axis
    belongs: reversed puts the channel axis last, and the last axis of a
    convolution's spatial extent belongs in the innermost position, so it is
    taken out and put back where the channels are.
    """

    order = list(reversed(range(int(rank))))
    order.insert(1, order.pop(-1))
    return tuple(order)


def conv_layout(x, weight, bias, stride, padding, dilation, transposed,
                output_padding, groups, device=None, dtype=None) -> Layout:
    """Where a convolution's result lands, asked rather than derived.

    The extent and stride of a convolution's result are not a formula over the
    input's extents and the kernel's: they are what the operation produces, and
    a form nobody anticipated -- a dilated transposed convolution with an
    output pad, say -- is exactly where a formula is wrong in a way nothing
    notices.  So the operation is asked, on tensors of the call's own extents,
    and its answer is used as it comes.

    Asking needs real tensors, so the caller passes the ones it has; there is
    no arithmetic fallback, because a fallback here is the failure this exists
    to prevent.
    """

    import tensorplay as tp

    from ..ir import convert_shape_to_tp, ir_node_to_tensor

    # We use guard_int_seq rather than size_hints because the output shape
    # depends on these values — if they ever contained symbols, size_hints
    # would silently substitute a hint that could be wrong, producing an
    # incorrect layout. guard_int_seq will install a proper guard instead.
    guard = V.graph.sizevars.guard_int_seq
    with V.graph.fake_mode:
        output = tp.ops.tp.convolution(
            ir_node_to_tensor(x),
            ir_node_to_tensor(weight),
            ir_node_to_tensor(bias),
            guard(stride),
            guard(padding),
            guard(dilation),
            transposed,
            guard(output_padding),
            groups,
        )
        sizes = convert_shape_to_tp(output.shape)
        out_stride = convert_shape_to_tp(output.stride())  # type: ignore[assignment]

    return FixedLayout(
        device if device is not None else output.device,
        dtype if dtype is not None else output.dtype,
        sizes,
        out_stride,
    )

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
        if _is_one_by_one(kernel_inputs.extra):
            # A one-by-one convolution is a product, so its candidates are the
            # product's -- and the one that expresses it is a choice beside the
            # operator, not a tile of this template's table.  Offering this
            # table's tiles here would be measuring a spatial kernel against a
            # product, which is not a comparison anybody asked for.
            yield {"choice": "one_by_one_product"}
            return
        try:
            rows, cols, inner = kernel_inputs.mnk_symbolic()
        except NotImplementedError:
            return
        for tiling in CHOICES.get_conv_configs(self.device_type)(
            rows, cols, inner, dtype_size=dtype_size(kernel_inputs.dtype(0))
        ):
            yield {"choice": "triton", **tiling.as_kwargs()}

    def get_depthwise_configs_impl(self, kernel_inputs):
        """The depthwise tilings, which are not products and so are not fitted."""

        yield {"choice": "operator"}
        for tiling in CHOICES.get_depthwise_conv_configs(self.device_type):
            yield {"choice": "triton", **tiling.as_kwargs()}

class DepthwiseConvTemplate(KernelTemplate):

    """The depthwise convolution, which is not a product and does not use one.

    No output channel is a sum over input channels here, so there is no
    contraction to tile and nothing for a matrix multiply to do.  The work
    steps over images, positions along the axis and channels instead, and each
    lane keeps its own accumulator -- so the three-dimensional tilings are the
    only candidates, and the product table is not consulted at all.
    """

    inputs_class = ConvKernelInputs

    def __init__(self, name: str = "depthwise_conv1d"):
        super().__init__(name, hash="depthwise:operator+block_n/l/c")
        self.heuristics = DepthwiseConvConfigHeuristics()

    def out_specs(self, meta: dict) -> tuple:
        """Where this call's result lands, asked of the operation.

        Asking needs the operands, and there are two sets of them: the probe
        is a stand-in built for the purpose and the feed is the call's own. The
        probe is preferred, because asking on the call's tensors would need them
        to exist at the moment the result's shape is worked out, and a template
        that needs the region's real tensors to describe the region is a
        template that cannot be asked before the region runs.
        """

        layout = self._asked_layout(meta)
        if layout is not None:
            return (layout,)
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("a convolution without a result")
        return (
            FlexibleLayout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                channels_last_order(len(tuple(size))),
            ),
        )

    def _asked_layout(self, meta: dict):
        """The operation's own answer about its result, or ``None`` if unaskable."""

        feed = tuple(meta.get("probe_feed") or meta.get("feed") or ())
        specs = tuple(meta.get("operand_specs") or ())
        if len(feed) < 2 or len(specs) < 2:
            return None

        def operand(position, literal):
            return feed[position] if position is not None else literal

        bias_spec = meta.get("bias_spec")
        bias = None
        if bias_spec is not None and len(feed) > 3:
            bias = operand(bias_spec[0], bias_spec[1])
        try:
            params = ConvLayoutParams.from_meta(meta)
            return conv_layout(
                operand(*specs[0]), operand(*specs[1]), bias,
                params.stride, params.padding, params.dilation,
                params.transposed, params.output_padding, params.groups,
                device=meta.get("device"), dtype=meta.get("out_dtype"),
            )
        except Exception:  # noqa: BLE001 - an unaskable call falls back
            return None

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

    def generate(self, params, out_specs, meta, plain_launch=None):
        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return call_operation(
                "framework_depthwise",
                plain_launch,
                out_specs[0] if out_specs else None,
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
        for tiling in CHOICES.get_depthwise_conv_configs(self.device_type):
            yield {"choice": "triton", **tiling.as_kwargs()}

class ConvTemplate(KernelTemplate):

    """Convolutions, with the 1x1 case expressed as the product it is.

    A one-by-one convolution with unit stride, no padding and one group is a
    matrix product, so it is measured as one -- the product template's tile
    set applies to it directly, and there is nothing to be gained from a
    second set of candidates for the same arithmetic.

    Wider kernels keep the operator, which is the floor a measurement can never
    lose against, until there is a tiled kernel for them to choose between.
    """

    inputs_class = ConvKernelInputs

    def __init__(self, name: str = "conv"):
        super().__init__(name, hash="conv:operator+product-1x1")
        self.heuristics = ConvConfigHeuristics()

    def out_specs(self, meta: dict) -> tuple:
        """Where this call's result lands, asked of the operation.

        Asking needs the operands, and there are two sets of them: the probe
        is a stand-in built for the purpose and the feed is the call's own. The
        probe is preferred, because asking on the call's tensors would need them
        to exist at the moment the result's shape is worked out, and a template
        that needs the region's real tensors to describe the region is a
        template that cannot be asked before the region runs.
        """

        layout = self._asked_layout(meta)
        if layout is not None:
            return (layout,)
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("a convolution without a result")
        return (
            FlexibleLayout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                channels_last_order(len(tuple(size))),
            ),
        )

    def _asked_layout(self, meta: dict):
        """The operation's own answer about its result, or ``None`` if unaskable."""

        feed = tuple(meta.get("probe_feed") or meta.get("feed") or ())
        specs = tuple(meta.get("operand_specs") or ())
        if len(feed) < 2 or len(specs) < 2:
            return None

        def operand(position, literal):
            return feed[position] if position is not None else literal

        bias_spec = meta.get("bias_spec")
        bias = None
        if bias_spec is not None and len(feed) > 3:
            bias = operand(bias_spec[0], bias_spec[1])
        try:
            params = ConvLayoutParams.from_meta(meta)
            return conv_layout(
                operand(*specs[0]), operand(*specs[1]), bias,
                params.stride, params.padding, params.dilation,
                params.transposed, params.output_padding, params.groups,
                device=meta.get("device"), dtype=meta.get("out_dtype"),
            )
        except Exception:  # noqa: BLE001 - an unaskable call falls back
            return None

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
        sizes = tuple(meta.get("operand_sizes") or ())
        if len(sizes) < 2:
            raise NotImplementedError("a convolution without its operands' extents")
        out_channels = int(sizes[1][0])
        in_channels = int(sizes[0][1])
        groups = int(meta.get("groups", 1) or 1)
        return {
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "out_size": spatial_out,
            # A tile narrower than its block is not a tile, so the call goes to
            # the operator.  The same rule the gradient applies, for the same
            # reason: both contract over a channel axis a group divides.
            "min_contraction": min(out_channels, in_channels) // groups if groups else 0,
            "unroll": True,
        }

    def generate(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return call_operation(
                "framework_convolution",
                plain_launch,
                out_specs[0] if out_specs else None,
            )
        if params.to_kwargs().get("choice") == "one_by_one_product":
            return self._one_by_one_choice(out_specs, meta, plain_launch)
        if self.is_one_by_one(meta):
            # The case applies but this call did not offer it, so there is
            # nothing here to build -- the operator is the floor.
            return None
        return self._tiled_choice(params, out_specs, meta, plain_launch)

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates.

        The only kernel this template offers is the product's, so the operands
        that exercise it are the product's: a ramp of the right extents, built
        from the call's own geometry rather than handed over from the region's
        tensors.
        """

        if not self.is_one_by_one(meta):
            return None
        return GEMM.probe(self._product_meta(meta))

    def _one_by_one_choice(self, out_specs, meta, plain_launch):
        """The one-by-one case, as the choice that expresses it.

        A peer choice beside the operator rather than a nested selection: the
        selection that is already happening measures this against the operator,
        and a template that ran a second one would be measuring against a floor
        that is not this call's.  The kernel does its own reshaping -- squeeze
        the weight's two trailing extents, read the input channels-last, and
        un-permute the result -- because that is the arithmetic, not something
        the caller should have to arrange.
        """

        return call_operation(
            "conv1x1_via_product",
            conv1x1_launch,
            out_specs[0] if out_specs else None,
        )

    def _product_layout(self, gemm_meta: dict, meta: dict):
        """The product's own result, which is this call's result permuted."""

        out_size = gemm_meta["out_size"]
        rank = len(out_size) + 2
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                out_size,
                contiguous_stride(out_size),
            ),
        )

    def _product_order(self, kernel=None) -> tuple:
        """The order the result arrives in when a convolution is a product.

        The contraction is over the channels, so the input is read with them
        last and the weight with its output channels first; the result is
        therefore produced permuted, and un-permuting it belongs to the same
        work rather than to the caller.
        """

        if kernel is None:
            kernel = (1, 1)
        rank = len(tuple(kernel)) + 2
        return tuple(range(0, rank - 1)) + (rank - 1,)

    def _product_meta(self, meta: dict) -> dict:
        """This convolution spelled the way the product template reads it.

        The convolution's own extents are not a product's: a convolution's
        result is positions, channels and a batch, while a product's is two
        extents.  So the batch and the spatial extent are counted together into
        the rows -- because a tile does not care which image a position came
        from -- and what is left is exactly the product the one-by-one case is.
        """

        size = tuple(int(v) for v in (meta.get("out_size") or ()))
        spatial = size[2:] if len(size) > 2 else ()
        batch = size[0] if size else 0
        rows = batch
        for extent in spatial:
            rows *= extent
        sizes = tuple(meta.get("operand_sizes") or ())
        in_channels = int(sizes[0][1]) if sizes else 0
        out_channels = int(size[1]) if len(size) > 1 else 0
        return {
            "out_size": (rows, out_channels),
            "out_dtype": meta.get("out_dtype"),
            "device": meta.get("device"),
            "operand_dtype": meta.get("operand_dtype"),
            "operand_sizes": ((rows, in_channels), (in_channels, out_channels)),
            "operand_specs": meta.get("operand_specs"),
            "bias_spec": meta.get("bias_spec"),
            "qualifies": True,
            "probe_feed": meta.get("probe_feed"),
            "feed": meta.get("feed"),
        }

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

class _ConvGradientTemplate(KernelTemplate):

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
        super().__init__(name, hash=f"conv2d_bwd_{which}:operator+tiles")
        self.which = which
        self.heuristics = ConvGradientConfigHeuristics(which)

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
        for tiling in CHOICES.get_conv_configs(self.device_type)(
            rows, cols, inner, dtype_size=dtype_size(kernel_inputs.dtype(0))
        ):
            yield {"choice": "triton", **tiling.as_kwargs()}

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

    def generate(self, params, out_specs, meta, plain_launch=None):
        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return call_operation(
                "framework_convolution_bwd",
                plain_launch,
                out_specs[0] if out_specs else None,
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

    def generate(self, params, out_specs, meta, plain_launch=None):
        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return call_operation(
                "framework_convolution_bwd",
                plain_launch,
                out_specs[0] if out_specs else None,
            )
        if plain_launch is None:
            return None
        launcher = self._launcher(params, meta, plain_launch)
        return None if launcher is None else self._caller(params, out_specs, launcher)

def _is_one_by_one(meta: dict) -> bool:
    """Is this the product-shaped case: 1x1 kernel, unit stride, one group?"""

    kernel = tuple(int(k) for k in meta.get("kernel_size") or ())
    if not kernel or any(k != 1 for k in kernel):
        return False
    if any(int(v) != 1 for v in (meta.get("stride") or ())):
        return False
    if any(int(v) != 0 for v in (meta.get("padding") or ())):
        return False
    if int(meta.get("groups", 1) or 1) != 1:
        return False
    if meta.get("transposed"):
        return False
    return all(int(v) == 1 for v in (meta.get("dilation") or ()))


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

@SymbolicGridFn
def _conv_forward_grid(rows, cols, groups, block_m, block_n, cdiv):
    return (cdiv(rows, block_m), cdiv(cols, block_n), groups)

@SymbolicGridFn
def _conv_bwd_input_grid(rows, cols, groups, block_m, block_n, cdiv):
    return (cdiv(rows, block_m), cdiv(cols, block_n), groups)

@SymbolicGridFn
def _conv_bwd_weight_grid(groups, block_m, block_n, block_k, cdiv):
    return (cdiv(1, block_m), cdiv(1, block_n), groups)

CONV = ConvTemplate()

DEPTHWISE_CONV = DepthwiseConvTemplate()

CONV_BWD_INPUT = ConvBwdInputTemplate()

CONV_BWD_WEIGHT = ConvBwdWeightTemplate()

framework_convolution = ExternKernelChoice(
    tp.ops.tp.convolution, "convolution",
    has_out_variant=False,
    op_overload=tp.ops.tp.convolution.default,
)

def conv1x1_launch(feed: list):
    """The one-by-one kernel, over the operands a step is handed.

    The kernel takes tensors because that is what a library call takes; a step
    hands over the operands by position, so the two are bridged here rather than
    by teaching the kernel about feeds.
    """

    return conv1x1_via_mm(feed[0], feed[1])


framework_conv1x1_via_mm = ExternKernelChoice(conv1x1_via_mm, "conv1x1_via_mm")

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


#: The operation namespace, under a name of this project's own, reached by
#: whatever name this project's operations are registered under.
framework = tp.ops.tp

#: The templates this module owns, named for what each computes.  A template is
#: an object and a choice is a template with a configuration attached, so these
#: are the templates: what a candidate list is built out of.
#: The two-dimensional call written as a template: the kernel is written from
#: the call's own geometry, so several shapes of it can be measured against one
#: another and the one that runs fastest on this device is the one kept.
conv3d_template = CONV

#: The template for a call that reduces the contracted axis away.
depthwise_conv1d_template = DEPTHWISE_CONV

#: The two gradient templates, one per gradient: they read the call's own
#: operands in different orders and reduce over different axes, so they are two
#: kernels rather than one with a flag.
conv2d_bwd_input_template = CONV_BWD_INPUT
conv2d_bwd_weight_template = CONV_BWD_WEIGHT

#: A call whose kernel is one element wide, computed as a product.  A choice
#: of its own because it is a different computation rather than a different
#: way of the same one, and a measurement against it is measuring that.

#: The primitive namespace, for the same reason and by the same argument.
prims = tp.ops.prims


@SymbolicGridFn
def conv2d_grid(n, c, h, w, meta, *, cdiv):
    """The programs one plane of a call is done by.

    The first axis counts the output's own extent rather than the input's, so a
    program owns a tile of the *result* -- which is what the epilogue writes and
    what decides how much of the input it has to read.
    """

    return (cdiv(n * h * w, meta["BLOCK_M"]), cdiv(c, meta["BLOCK_N"]), meta["GROUPS"])



LOOP_BODY_2D = """
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
"""

conv2d_template = TritonTemplate(
    name="convolution2d",
    grid=conv2d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_H = {{size("X", 2)}}
    IN_W = {{size("X", 3)}}
    OUT_C = {{size(None, 1)}}
    OUT_H = {{size(None, 2)}}
    OUT_W = {{size(None, 3)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xh = {{stride("X", 2)}}
    stride_xw = {{stride("X", 3)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wh = {{stride("W", 2)}}
    stride_ww = {{stride("W", 3)}}

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_2D
    + """
{% endfor %}
{% endfor %}
{% else %}
    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W
        """
    + LOOP_BODY_2D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # A suffix is generated to keep this kernel's name apart.
    {{store_output(("idx_n", "idx_c", "idx_h", "idx_w"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
)

#: The template for a call with a depth to it.  The same template as the
#: two-dimensional one, because what a kernel is given is the kernel's extents
#: and there is nothing about the third of them that changes the kernel's shape.
@SymbolicGridFn
def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
    """The same, for a call with a depth to it."""

    return (
        cdiv(n * d * h * w, meta["BLOCK_M"]),
        cdiv(c, meta["BLOCK_N"]),
        meta["GROUPS"],
    )


@SymbolicGridFn
def depthwise_conv1d_grid(n, c, l, meta, *, cdiv):
    """The programs one depthwise call is done by.

    All three extents are separate axes here rather than one flattened one,
    because a depthwise call reduces the contracted axis and there is nothing to
    flatten it against: the kernel walks the length, so a program that owned a
    tile of a flattened extent would own a tile that is not contiguous in it.
    """

    return (cdiv(n, meta["BLOCK_N"]), cdiv(l, meta["BLOCK_L"]), cdiv(c, meta["BLOCK_C"]))


@SymbolicGridFn
def conv2d_bwd_input_grid(grad_out, x, w, meta, *, cdiv):
    """The programs one input-gradient plane is done by.

    Laid out by the input's own extent rather than the output's, because the
    kernel is written in terms of the value it produces and reading the output
    through a differently shaped index would be a gather.
    """

    *x_size, _, w_in = x.get_size()
    return cdiv(sympy_product(x_size), meta["BLOCK_M"]), cdiv(w_in, meta["BLOCK_N"])


@SymbolicGridFn
def conv2d_bwd_weight_grid(grad_out, x, w, meta, *, cdiv):
    """The programs one weight gradient is done by.

    The reduction is over the batch and the spatial extents, so a program owns a
    tile of the contracted axis itself -- the opposite of the other two, because
    here the value being produced *is* along that axis.
    """

    return cdiv(x.get_size()[0], meta["BLOCK_M"]), cdiv(w.get_size()[0], meta["BLOCK_N"])


def convert_1x1_conv_to_mm(x, weight, bias):
    """A call whose kernel is one element wide, computed as a product instead.

    A kernel one element wide does no work in the contracted axis beyond copying,
    so what is left is a product of the input's channels with the weight's -- and
    a product has more candidates than a call does, which is the whole reason to
    write it as one.  The weight is transposed and the input's channels moved last
    so that the two agree on which axis is contracted.
    """

    rank = len(weight.get_size())
    for _ in range(rank - 2):
        weight = weight.squeeze(dim=-1)
    weight = weight.permute([1, 0])
    x_permute = list(range(rank))
    x_permute.append(x_permute.pop(1))
    x = x.permute(x_permute)
    *sizes, in_chan = x.get_size()
    x = x.reshape([sympy_product(sizes), in_chan])
    if bias is None:
        result = tuned_mm(x, weight)
    else:
        result = tuned_addmm(bias, x, weight)
    result = result.reshape([*sizes, -1])
    result_permute = list(range(rank))
    result_permute.insert(1, result_permute.pop(-1))
    return result.permute(result_permute)


@register_lowering([framework.convolution, framework._convolution])
def _convolution(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32,
):
    """A call with the flags a caller may pass but the lowering does not read.

    Registered so that the flags are accepted rather than refused: a caller that
    names a flag is describing how it would like the call done, and a lowering
    that cannot honour it still computes the call correctly.  So the flags are
    taken and dropped, and what is left is the call.
    """

    return convolution(
        x, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    )


def constrain_conv_to_fx_strides(fx_node, *args, **kwargs):
    """A call's own strides, unless the layouts were inferred and may be chosen.

    When the layouts were inferred the strides are this compiler's to pick, and
    constraining them to what the caller wrote would forbid the very choice the
    inference was made to leave open.
    """

    if fx_node.target is not framework.convolution.default:
        raise AssertionError(
            f"Expected a convolution call, got {fx_node.target}"
        )
    if V.graph.layout_opt:
        return (args, kwargs)
    return args, kwargs


def constrain_conv_bwd_to_fx_strides(fx_node, *args, **kwargs):
    """The same, for a call whose gradients are being asked for."""

    if fx_node.target is not framework.convolution_backward.default:
        raise AssertionError(
            f"Expected a convolution-backward call, got {fx_node.target}"
        )
    if V.graph.layout_opt:
        return (args, kwargs)
    return args, kwargs


def conv_bwd_input_layout(
    grad_out,
    input,
    weight,
    stride: Sequence[int],
    padding: tuple,
    dilation: tuple,
    transposed: bool,
    output_padding: tuple,
    groups: int,
):
    """Where an input gradient lands, asked of the operation rather than guessed.

    The gradient's shape follows from the call's, and following it by hand is a
    formula to keep true.  So the operation is asked, under shapes it can be
    evaluated with, and what it says is what the gradient is -- which is the same
    answer the caller would have got by running it.
    """

    guard = V.graph.sizevars.guard_int_seq
    dx = _gradient_by_asking(
        grad_out, input, weight, stride, padding, dilation, transposed,
        output_padding, groups, (True, False, False),
    )
    sizes = convert_shape_to_tp(dx.size())
    stride_ = convert_shape_to_tp(dx.stride())
    return FixedLayout(
        input.get_device_or_error(), input.get_dtype(), sizes, stride_
    )


def conv_bwd_weight_layout(
    grad_out,
    input,
    weight,
    stride: Sequence[int],
    padding: tuple,
    dilation: tuple,
    transposed: bool,
    output_padding: tuple,
    groups: int,
):
    """Where a weight gradient lands, asked the same way.

    Asked of the same operation rather than of a formula, and taken with the
    weight's own type and device -- a gradient of a weight is that weight's type,
    and a gradient computed in another type would be a different number.
    """

    dw = _gradient_by_asking(
        grad_out, input, weight, stride, padding, dilation, transposed,
        output_padding, groups, (False, True, False),
    )
    sizes = convert_shape_to_tp(dw.size())
    stride_ = convert_shape_to_tp(dw.stride())
    return FixedLayout(
        weight.get_device_or_error(), weight.get_dtype(), sizes, stride_
    )


def _gradient_by_asking(
    grad_out, input, weight, stride, padding, dilation, transposed,
    output_padding, groups, output_mask,
):
    """One gradient of a call, asked of the operation that computes it.

    The shape follows from the call's, and a formula for it is a thing that has
    to be kept true; asking is the same answer the caller would get by running
    the call, which is why the values handed over are all zero -- the shape does
    not depend on them, and zero costs nothing to write.  The ask happens on the
    device the call is for, because there is no way to make a value of an
    unresolved shape here.
    """

    guard = V.graph.sizevars.guard_int_seq
    device = input.get_device_or_error()
    channels_last = channels_last_call(input, len(weight.get_size()) - 2, groups, transposed)
    if channels_last and all(len(node.get_size()) == 4 for node in (grad_out, input, weight)):
        zeros = lambda node: tp.empty(
            [int(dim) for dim in node.get_size()],
            dtype=node.get_dtype(),
            device=device,
            memory_format=tp.channels_last,
        ).zero_()
    else:
        zeros = lambda node: tp.zeros(
            [int(dim) for dim in node.get_size()],
            dtype=node.get_dtype(),
            device=device,
        )
    result = framework.convolution_backward(
        zeros(grad_out), zeros(input), zeros(weight), None,
        guard(stride), guard(padding), guard(dilation), transposed,
        guard(output_padding), groups, output_mask,
    )
    # The backward answers with the full ``(dx, dw, db)`` triple; the caller
    # asked for one gradient, so the element the mask selected is returned.
    if output_mask == (True, False, False):
        return result[0]
    if output_mask == (False, True, False):
        return result[1]
    if output_mask == (False, False, True):
        return result[2]
    return result


def call_framework_dw(
    x_t, go_t, *, w_shape, stride, padding, dilation, transposed,
    output_padding, groups, out=None,
):
    """A weight gradient, computed by handing the call to the framework.

    The weight itself is a placeholder: the operation reads it to learn the
    gradient's shape and does not use its values, and making a real weight to
    throw away would cost an allocation per measurement.  Its memory format is
    taken from the input's, because a gradient is expected in the layout of the
    thing it is a gradient of.  An ``out`` buffer, when one is supplied,
    receives the gradient and is returned instead of a fresh tensor.
    """

    if x_t.is_contiguous(memory_format=tp.channels_last):
        memory_fmt = tp.channels_last
    else:
        memory_fmt = tp.contiguous_format
    dtype = x_t.get_dtype() if hasattr(x_t, "get_dtype") else x_t.dtype
    dummy_weight = tp.empty(
        w_shape, dtype=dtype, device=x_t.device, memory_format=memory_fmt
    )
    result = framework.convolution_backward(
        grad_output=go_t, input=x_t,
        weight=dummy_weight, bias_sizes=None, stride=stride, padding=padding,
        dilation=dilation, transposed=transposed, output_padding=output_padding,
        groups=groups, output_mask=(False, True, False),
    )
    dw = result[1]
    if out is not None:
        out.copy_(dw)
        return out
    return dw


def call_framework_dx(
    go_t, w_t, *, x_shape, stride, padding, dilation, transposed,
    output_padding, groups, out=None,
):
    """An input gradient, computed by handing the call to the framework.

    The input is a placeholder here rather than the weight, for the same reason:
    the operation reads it for the gradient's shape and the values are unused.
    An ``out`` buffer, when one is supplied, receives the gradient and is
    returned instead of a fresh tensor.
    """

    if go_t.is_contiguous(memory_format=tp.channels_last):
        memory_fmt = tp.channels_last
    else:
        memory_fmt = tp.contiguous_format
    dtype = go_t.get_dtype() if hasattr(go_t, "get_dtype") else go_t.dtype
    dummy_input = tp.empty(
        x_shape, dtype=dtype, device=go_t.device, memory_format=memory_fmt
    )
    result = framework.convolution_backward(
        grad_output=go_t, input=dummy_input,
        weight=w_t, bias_sizes=None, stride=stride, padding=padding,
        dilation=dilation, transposed=transposed, output_padding=output_padding,
        groups=groups, output_mask=(True, False, False),
    )
    dx = result[0]
    if out is not None:
        out.copy_(dx)
        return out
    return dx


def pad_listlike(x, size: int):
    """An axis-wise value repeated to as many axes as the call has.

    A caller may name one stride for every axis or one for the call, and a call
    is worked out in axes.  So a single value is taken as naming all of them, and
    a value that already names them all is left alone -- padding a full-length
    value would be a different number of axes than the call has.
    """

    if isinstance(x, int):
        return [x] * size
    if len(x) == 1:
        return type(x)([x[0]]) * size
    return x


def is_ones(items) -> bool:
    """Whether every one of some values is one."""

    return all(x == 1 for x in items)


def is_zeros(items) -> bool:
    """Whether every one of some values is zero."""

    return all(x == 0 for x in items)


def require_stride_order(x, order, allow_padding: bool = False):
    """A value seen with its axes in a stated order, copying it if it is not.

    Asked of the value rather than of the layout, because the caller wants a
    value it can read in that order and not a description of one: a value already
    in the order is itself, and one that is not has to be copied into it, which
    is a cost the caller is choosing to pay.
    """

    return ir.ExternKernel.require_strides(x, order=order, allow_padding=allow_padding)


def channels_last_call(x, ndim: int, groups: int, transposed: bool) -> bool:
    """Whether a call's operands are handed over with their channels last.

    Always, where the region's layouts are chosen as a whole.  Otherwise for a
    reduced-precision, ungrouped two-dimensional call on the GPU: the library
    computes those channels last whatever it is handed, and repacks operands
    in the other order itself -- one pass over each operand before the call
    and one over the result after it.  Asked for here, the order is written by
    whatever produces the operand, which a kernel of the region does anyway,
    and the result arrives channels last for its readers.
    """

    if ndim != 2:
        return False
    if V.graph.layout_opt:
        return True
    dtype = x.get_dtype()
    return (
        config.conv_channels_last_reduced_precision
        and groups == 1
        and not transposed
        and ir.get_device_type(x) == "cuda"
        and (
            dtype in (tp.float16, tp.bfloat16)
            # Single precision runs on the same channels-last tensor-core
            # engines when it may round through TF32.
            or (dtype == tp.float32 and _tf32_allowed())
        )
    )


#: Whether a kernel is written for the reduced-precision path of this build.
#: Read from the framework rather than assumed, because it is a property of how
#: the framework was built and not of this compiler.
def _tf32_allowed() -> bool:
    return bool(tp.backends.cudnn.allow_tf32)


@register("conv2d.default", "conv2d.padding")
def conv2d(
    x,
    weight,
    bias,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
):
    """A two-dimensional call, which is the general call with nothing extended.

    The general form carries a flag for a transposed call and a padding to add
    past the edge; a call that is neither does not have those to say, so it is
    answered by the general call rather than being a second way of computing the
    same thing.  What the call is stays fixed here -- only the way of computing
    it is chosen.  The graph declares the result element type (a half-precision
    call under a mixed-precision graph), and the activation and weight arriving
    here can still be the wider values they came from, so the operands are
    brought to the declared type before a way of computing the call is chosen.
    """

    val = node_val(index=0)
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        if x.get_dtype() != dtype:
            x = to_dtype(x, dtype)
        if weight.get_dtype() != dtype:
            weight = to_dtype(weight, dtype)
        if bias is not None and bias.get_dtype() != dtype:
            bias = to_dtype(bias, dtype)
    return convolution(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        False,
        (0, 0),
        groups,
    )


@register_lowering(framework.convolution)
def convolution(
    x,
    weight,
    bias,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
):
    """A call, answered by whichever of the candidates measures fastest.

    What the call is stays fixed here and only the way of computing it is chosen:
    a call whose kernel is one element wide is a product, a call with a bias is
    the same call without one and then an addition, and everything else is
    measured.  The three special cases are decided before the measurement because
    each of them is a different *computation* rather than a different way of the
    same one, and a measurement would be measuring the wrong thing.
    """

    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)
    if not isinstance(groups, int):
        groups = V.graph.sizevars.guard_int(groups)
    if not isinstance(groups, int):
        raise AssertionError(f"Expected int for groups, got {type(groups)}")
    stride = tuple(V.graph.sizevars.guard_int_seq(stride))
    padding = tuple(V.graph.sizevars.guard_int_seq(padding))
    if transposed:
        dilation = tuple(V.graph.sizevars.guard_int_seq(dilation))
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }
    device_type = ir.get_device_type(x)

    # A call with no batch axis is a batch of one, and is computed as one and then
    # described without the axis: a kernel written for the batched case can read
    # a leading axis of one, and one written for the unbatched case cannot.
    if len(x.get_size()) == len(weight.get_size()) - 1:
        return convolution(
            x.expand([1, *x.get_size()]), weight, bias, **kwargs
        ).squeeze(dim=0)

    out_chan, in_chan, *kernel_shape = V.graph.sizevars.guard_int_seq(
        weight.get_size()
    )

    ndim = len(kernel_shape)
    stride = pad_listlike(stride, ndim)
    padding = pad_listlike(padding, ndim)
    dilation = pad_listlike(dilation, ndim)
    output_padding = pad_listlike(output_padding, ndim)

    def channels_last_conv() -> bool:
        if V.graph.layout_opt and ndim == 2:
            return True
        layout = conv_layout(x, weight, None, **kwargs)
        req_stride_order = ir.get_stride_order(
            V.graph.sizevars.guarding_hints_or_throw(layout.stride)
        )
        return req_stride_order == ir.NHWC_STRIDE_ORDER

    autotuning_gemm = config.max_autotune or config.max_autotune_gemm
    if (
        (config.conv_1x1_as_mm or (autotuning_gemm and channels_last_conv()))
        and is_ones(kernel_shape)
        and is_ones(stride)
        and is_zeros(padding)
        and is_ones(dilation)
        and (not transposed)
        and is_zeros(output_padding)
        and (groups == 1)
        and V.graph.sizevars.statically_known_gt(sympy_product(x.get_size()), 0)
    ):
        return convert_1x1_conv_to_mm(x, weight, bias)

    # A bias is a value per output channel, so it is added after the call rather
    # than folded into it: a kernel that added it would have to read it once per
    # element of the contracted axis, which is the same number every time.
    if bias is not None and device_type != "cpu":
        result = convolution(x, weight, None, **kwargs)
        if V.graph.sizevars.statically_known_equals(result.get_size()[1], 0):
            return result
        return LOWERINGS["add.Tensor"](
            result,
            LOWERINGS["view.default"](bias, [result.get_size()[1]] + ndim * [1]),
        )

    x.realize()
    weight.realize()
    if channels_last_call(x, ndim, groups, transposed):
        V.graph.num_channels_last_conv += 1
        x = ir.ExternKernel.require_channels_last(x)
        weight = ir.ExternKernel.require_channels_last(weight)
        layout = conv_layout(x, weight, None, **kwargs)
    else:
        layout = conv_layout(x, weight, None, **kwargs)
        req_stride_order = ir.get_stride_order(
            V.graph.sizevars.guarding_hints_or_throw(layout.stride)
        )
        x = require_stride_order(x, req_stride_order)
        weight = require_stride_order(weight, req_stride_order)

    ordered_kwargs_for_cpp_kernel = [
        "stride", "padding", "dilation", "transposed", "output_padding", "groups",
    ]
    if bias is None:
        args = [x, weight]
        kwargs["bias"] = None
        ordered_kwargs_for_cpp_kernel.insert(0, "bias")
    else:
        bias = ir.ExternKernel.realize_input(bias)
        if bias is None:
            raise AssertionError("bias must not be None after realize_input")
        args = [x, weight, bias]
        bias.freeze_layout()
        V.graph.sizevars.guard_int_seq(bias.get_size())

    choices = []
    if "EAGER" in config.max_autotune_conv_backends:
        choices = [
            framework_convolution.bind(
                args, layout, ordered_kwargs_for_cpp_kernel, **kwargs
            )
        ]
    if (
        "TRITON" in config.max_autotune_conv_backends
        and use_triton_template(layout)
        and is_ones(dilation)
        and (not transposed)
        and is_zeros(output_padding)
        and V.graph.sizevars.statically_known_equals(
            in_chan * groups, x.get_size()[1]
        )
    ):
        if config.conv_1x1_as_mm and is_ones(kernel_shape) and is_ones(stride) and (
            is_zeros(padding) and groups == 1
        ):
            choices.append(framework_conv1x1_via_mm.bind(args, layout))
        is_depthwise = groups > 1 and in_chan == 1 and (out_chan == groups)
        if is_depthwise and ndim == 1:
            depthwise_configs = CHOICES.get_depthwise_conv_configs(device_type)
            for cfg in depthwise_configs:
                depthwise_conv1d_template.maybe_append_choice(
                    choices,
                    input_nodes=(x, weight),
                    layout=layout,
                    KERNEL_SIZE=kernel_shape[0],
                    CONV_STRIDE=stride[0],
                    PADDING=padding[0],
                    num_stages=cfg.num_stages,
                    num_warps=cfg.num_warps,
                    **cfg.kwargs,
                )
        else:
            conv_configs = CHOICES.get_conv_configs(device_type)
            dtype_size = x.get_dtype().itemsize
            for cfg in conv_configs(
                sympy_product([x.get_size()[0], *x.get_size()[2:]]),
                out_chan,
                in_chan,
                dtype_size=dtype_size,
            ):
                unroll = is_ones(kernel_shape)
                # A kernel that unrolls the whole kernel window has its own
                # register budget, and a configuration with more warps than that
                # budget wants is slower rather than faster.
                num_warps = cfg.num_warps if unroll else min(cfg.num_warps, 4)
                if ndim == 2:
                    conv2d_template.maybe_append_choice(
                        choices,
                        input_nodes=(x, weight),
                        layout=layout,
                        KERNEL_H=kernel_shape[0], KERNEL_W=kernel_shape[1],
                        STRIDE_H=stride[0], STRIDE_W=stride[1],
                        PADDING_H=padding[0], PADDING_W=padding[1],
                        GROUPS=groups, UNROLL=unroll, ALLOW_TF32=_tf32_allowed(),
                        num_stages=cfg.num_stages, num_warps=num_warps, **cfg.kwargs,
                    )
                elif ndim == 3:
                    conv3d_template.maybe_append_choice(
                        choices,
                        input_nodes=(x, weight),
                        layout=layout,
                        KERNEL_D=kernel_shape[0], KERNEL_H=kernel_shape[1],
                        KERNEL_W=kernel_shape[2], STRIDE_D=stride[0],
                        STRIDE_H=stride[1], STRIDE_W=stride[2],
                        PADDING_D=padding[0], PADDING_H=padding[1],
                        PADDING_W=padding[2], GROUPS=groups, UNROLL=unroll,
                        ALLOW_TF32=_tf32_allowed(),
                        num_stages=cfg.num_stages, num_warps=num_warps, **cfg.kwargs,
                    )
    node, _ = autotune_select_algorithm("convolution", choices, args, layout)
    return node


#: A weight gradient, computed by handing the call to the framework.  A choice of
#: its own because the weight is a placeholder in it -- the operation reads the
#: weight for the gradient's shape and never uses its values -- which is a
#: different call from the one that reads a real weight.
framework_dw = ExternKernelChoice(
    call_framework_dw, None, name="dw", has_out_variant=False
)

#: An input gradient, likewise: the input is the placeholder here.
framework_dx = ExternKernelChoice(
    call_framework_dx, None, name="dx", has_out_variant=False
)

#: Both gradients at once, as the operation computes them.  The floor for a call
#: that asked for two gradients, and the only candidate for one whose shape no
#: template here is written for.
framework_convolution_backward = ExternKernelChoice(
    tp.ops.tp.convolution_backward, "convolution_backward",
    has_out_variant=False,
    op_overload=tp.ops.tp.convolution_backward.default,
)


def _conv_bwd_backends(kind: str) -> bool:
    """Whether one of the two gradient's own ways is among the measured ones.

    Asked per gradient and per way because the two gradients are separate calls
    with separate candidates, and a program may measure one of them and not the
    other.
    """

    named = config.max_autotune_conv_backends
    return kind in named


@register_lowering(framework.convolution_backward)
def convolution_backward_lowering(
    grad_out,
    input,
    weight,
    bias_sizes,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    output_mask: Sequence[bool],
):
    """The gradients of a call, each measured on its own.

    Two gradients are two calls with two candidate lists, and they are measured
    apart because a kernel that is good at one is not necessarily good at the
    other: the input gradient reduces over the kernel's window while the weight
    gradient reduces over the batch and the spatial extents.

    A call whose shape no template here is written for is handed to the
    operation whole, before anything is measured -- a measurement needs
    candidates and there are none, and an empty list measured is a failure rather
    than a slow answer.
    """

    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)
    if not isinstance(groups, int):
        groups = V.graph.sizevars.guard_int(groups)
    out_chan, in_chan, *kernel_shape = V.graph.sizevars.guard_int_seq(
        weight.get_size()
    )
    stride = tuple(V.graph.sizevars.guard_int_seq(stride))
    padding = tuple(V.graph.sizevars.guard_int_seq(padding))
    dilation = tuple(V.graph.sizevars.guard_int_seq(dilation))
    input.realize()
    weight.realize()
    grad_out.realize()
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }
    ndim = len(kernel_shape)
    stride = pad_listlike(stride, ndim)
    padding = pad_listlike(padding, ndim)
    dilation = pad_listlike(dilation, ndim)
    output_padding = pad_listlike(output_padding, ndim)
    device_type = ir.get_device_type(input)
    conv_configs = CHOICES.get_conv_configs(device_type)
    dtype_size = input.get_dtype().itemsize

    has_triton_dw_choices = False
    dw = None
    choices_dw = []
    args_w = []
    layout_dw = conv_bwd_weight_layout(grad_out, input, weight, **kwargs)
    channels_last = channels_last_call(input, ndim, groups, transposed)
    if output_mask[1]:
        if channels_last:
            V.graph.num_channels_last_conv += 1
            input = ir.ExternKernel.require_channels_last(input)
            grad_out = ir.ExternKernel.require_channels_last(grad_out)
            layout_dw = conv_bwd_weight_layout(grad_out, input, weight, **kwargs)
        else:
            guard = V.graph.sizevars.guard_int_seq
            stride_order = ir.get_stride_order(guard(layout_dw.stride))
            input = require_stride_order(input, stride_order)
            grad_out = require_stride_order(grad_out, stride_order)
        args_w = [input, grad_out]
        if (
            _conv_bwd_backends("TRITON")
            and use_triton_template(layout_dw)
            and (not transposed)
            and is_zeros(output_padding)
        ):
            for cfg in conv_configs(
                sympy_product([input.get_size()[0], *input.get_size()[2:]]),
                out_chan,
                in_chan,
                dtype_size=dtype_size,
            ):
                if ndim == 2:
                    has_triton_dw_choices = True
                    conv2d_bwd_weight_template.maybe_append_choice(
                        choices_dw,
                        input_nodes=(input, grad_out),
                        layout=layout_dw,
                        KERNEL_H=kernel_shape[0], KERNEL_W=kernel_shape[1],
                        PADDING_H=padding[0], PADDING_W=padding[1],
                        STRIDE_H=stride[0], STRIDE_W=stride[1],
                        DILATION_H=dilation[0], DILATION_W=dilation[1],
                        GROUPS=groups, ALLOW_TF32=_tf32_allowed(),
                        num_stages=cfg.num_stages, num_warps=cfg.num_warps,
                        **cfg.kwargs,
                    )

    has_triton_dx_choices = False
    dx = None
    choices_dx = []
    args_x = []
    layout_dx = conv_bwd_input_layout(grad_out, input, weight, **kwargs)
    if output_mask[0]:
        if channels_last:
            V.graph.num_channels_last_conv += 1
            grad_out = ir.ExternKernel.require_channels_last(grad_out)
            weight = ir.ExternKernel.require_channels_last(weight)
            layout_dx = conv_bwd_input_layout(grad_out, input, weight, **kwargs)
        else:
            guard = V.graph.sizevars.guard_int_seq
            stride_order = ir.get_stride_order(guard(layout_dx.stride))
            grad_out = require_stride_order(grad_out, stride_order)
            weight = require_stride_order(weight, stride_order)
        args_x = [grad_out, weight]
        if (
            _conv_bwd_backends("TRITON")
            and use_triton_template(layout_dx)
            and (not transposed)
            and is_zeros(output_padding)
        ):
            for cfg in conv_configs(
                sympy_product([input.get_size()[0], *input.get_size()[2:]]),
                out_chan,
                in_chan,
                dtype_size=dtype_size,
            ):
                if ndim == 2:
                    has_triton_dx_choices = True
                    conv2d_bwd_input_template.maybe_append_choice(
                        choices_dx,
                        input_nodes=(grad_out, weight),
                        layout=layout_dx,
                        KERNEL_H=kernel_shape[0], KERNEL_W=kernel_shape[1],
                        PADDING_H=padding[0], PADDING_W=padding[1],
                        STRIDE_H=stride[0], STRIDE_W=stride[1],
                        DILATION_H=dilation[0], DILATION_W=dilation[1],
                        GROUPS=groups, ALLOW_TF32=_tf32_allowed(),
                        num_stages=cfg.num_stages, num_warps=cfg.num_warps,
                        **cfg.kwargs,
                    )

    if output_mask[1]:
        if _conv_bwd_backends("EAGER") or not has_triton_dw_choices:
            choices_dw.append(
                framework_dw.bind(
                    input_nodes=args_w,
                    layout=layout_dw,
                    ordered_kwargs_for_cpp_kernel=[
                        "w_shape", "stride", "padding", "dilation", "transposed",
                        "output_padding", "groups",
                    ],
                    w_shape=weight.get_size(), stride=stride, padding=padding,
                    dilation=dilation, transposed=transposed,
                    output_padding=output_padding, groups=groups,
                )
            )
        dw, _ = autotune_select_algorithm(
            "convolution_bwd_weight", choices_dw, args_w, layout_dw
        )
    if output_mask[0]:
        if _conv_bwd_backends("EAGER") or not has_triton_dx_choices:
            choices_dx.append(
                framework_dx.bind(
                    input_nodes=args_x,
                    layout=layout_dx,
                    ordered_kwargs_for_cpp_kernel=[
                        "x_shape", "stride", "padding", "dilation", "transposed",
                        "output_padding", "groups",
                    ],
                    x_shape=input.get_size(), stride=stride, padding=padding,
                    dilation=dilation, transposed=transposed,
                    output_padding=output_padding, groups=groups,
                )
            )
        dx, _ = autotune_select_algorithm(
            "convolution_bwd_input", choices_dx, args_x, layout_dx
        )
    db = None
    if output_mask[2] and bias_sizes is not None:
        # A bias gradient is the gradient summed over everything the bias did not
        # vary along, which is every axis but the channel one.
        db = grad_out.sum(axis=[0] + list(range(2, ndim + 2)))
    return (dx, dw, db)
