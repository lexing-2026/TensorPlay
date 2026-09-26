"""Products, measured against the framework's own.

A product is a product whether it arrives as a matrix multiply, an add, a batch, or
a layer's weight, so all of them are one template and one tile space rather than four
templates and four tables.
"""

from __future__ import annotations
import logging

import tensorplay as tp

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

import functools
import itertools
from typing import Any, Iterator, Sequence

from .triton import CHOICES, dtype_size

from ..codegen.common import KernelTemplate
from .base import SymbolicGridFn, TemplateConfigHeuristics
from ..codegen.subgraph import SubgraphTemplate
from tensorplay.graph.experimental.proxy_tensor import make_graph

from .ir import contiguous_stride, next_power_of_2
from .mm_common import (
    _is_static_problem,
    _use_small_mm_pointwise,
    device_capability,
    mm_args,
    mm_grid,
    use_aten_gemm_kernels,
    use_decompose_k_choice,
    use_native_matmul,
    use_triton_blackwell_tma_template,
    use_triton_template,
    use_triton_tma_template,
)
from ..utils import ceildiv
from ..loops import V
from ..ir import Buffer
from tensorplay.nn.functional import ScalingType
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..ir import Layout
from .params import DictKernelTemplateParams, KernelTemplateParams
from ..op_lowerings import (
    fallback_handler,
    register_lowering,
    select_decomp_table,
)
from .. import config
from ..autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from ..autoheuristic.autoheuristic_utils import (
    AHContext,
    context_add_strides,
    context_add_using_tf32,
)
from .select_algorithm import (
    autotune_select_algorithm,
    get_template_configs,
    realize_inputs,
)
from .select_algorithm import (
    ChoiceCaller,
    ExternChoiceCaller,
    ExternKernelChoice,
    KernelArgs,
    TritonChoiceCaller,
    TritonTemplate,
)





def persistent_mm_grid(m, n, meta, *, cdiv, min):
    """The grid for a product's tiles, swept by a fixed number of programs.

    One program per tile launches more programs than the device can have
    resident, so most of them wait.  A persistent kernel instead launches as
    many programs as there are multiprocessors and gives each a share of the
    tiles to walk, which turns the launcher's scheduling problem into the
    kernel's loop -- and the cap is what makes it persistent, since a grid
    larger than the machine is exactly what it is avoiding.
    """

    return (min(meta["NUM_SMS"], cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])), 1, 1)


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
        # The element's width goes in because the estimate that prunes a tile
        # is in bytes: without it the estimate is unknown, and a tile the
        # device could not stage would be offered and only found out at launch.
        for config in generator(
            rows, cols, inner, dtype_size=dtype_size(kernel_inputs.dtype(0))
        ):
            yield {"choice": "triton", **config.as_kwargs()}


def _gemm_source_identity() -> str:
    """What the tile kernel is, for a stored decision to be invalidated against.

    The digest covers the kernel body and the tuning version, so a decision
    recorded against one is not reused once either has moved.
    """

    from ..codegen.triton_gemm import GEMM_TUNING_VERSION, _kernel_source_digest

    return f"{GEMM_TUNING_VERSION}:{_kernel_source_digest()}"


class GemmTemplate(KernelTemplate):

    """Products, measured against the framework's own.

    Validity belongs to the template: a configuration that does not fit this
    call -- not two-dimensional, not the element type the tiles accumulate in,
    not this device -- is refused here rather than by whoever is asking.
    """

    inputs_class = MMKernelInputs

    def __init__(self, name: str = "gemm", hash: str | None = None):
        super().__init__(name, hash=hash or _gemm_source_identity())
        self.heuristics = GemmConfigHeuristics()

    grid = staticmethod(mm_grid)


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

    def generate(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
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
        caller = ChoiceCaller(
            name=f"{self.name}-{kwargs.get('BLOCK_M')}x{kwargs.get('BLOCK_N')}"
            f"x{kwargs.get('BLOCK_K')}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
        )
        caller.config_patches = {
            key: kwargs[key]
            for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages")
            if key in kwargs
        }
        return caller.bind(self.launcher(kwargs, meta, layout, plain_launch))

    def launcher(self, kwargs: dict, meta: dict, layout, plain_launch):
        """The kernel this template runs for one configuration.

        Split out from generation because the forms differ in *which kernel*
        they run, not in whether they run: the swept form walks the tiles
        itself, the descriptor-driven form fetches them, the scaled forms
        multiply factors somewhere different.  All of them answer the same
        question the same way, so only this differs.
        """

        from ..codegen.triton_gemm import tuned_matmul_launch

        return tuned_matmul_launch(
            plain_launch,
            meta.get("probe_feed") or meta.get("feed") or (),
            meta["operand_specs"],
            layout.size,
            bias_spec=meta.get("bias_spec"),
            b_transposed=bool(meta.get("b_transposed", False)),
            config=kwargs,
        )


















class PersistentGemmTemplate(GemmTemplate):

    """The product again, this time swept by a fixed number of programs.

    Same tile space, same arithmetic, same candidates as the plain product --
    the difference is only in how many programs there are and how the work is
    divided between them.  So it is a separate template rather than another
    configuration of the first one: the geometry it wants is a property of the
    launch, not of the tile, and a configuration that changed the launch's
    shape would be claiming a choice the tile never had a say in.
    """

    grid = staticmethod(persistent_mm_grid)

    def __init__(self, name: str = "mm_persistent", hash: str | None = None):
        super().__init__(
            name, hash=hash or f"persistent:{_gemm_source_identity()}"
        )

    def configurations(self, out_specs, meta):
        """The same candidates, carrying the cap the grid is built from.

        The cap is not a choice -- it is a property of the device -- so it rides
        along with the configuration rather than being one, which is what lets
        the grid be computed from a configuration alone.
        """

        for params in super().configurations(out_specs, meta):
            kwargs = params.to_kwargs()
            kwargs["NUM_SMS"] = num_sms(meta.get("device"))
            yield DictKernelTemplateParams(kwargs)

    def launcher(self, kwargs, meta, layout, plain_launch):
        """The swept kernel, which is the whole difference from the plain one."""

        from ..codegen.triton_gemm import persistent_matmul_launch

        tiles = (
            -(-int(layout.size[0]) // int(kwargs["BLOCK_M"]))
            * -(-int(layout.size[1]) // int(kwargs["BLOCK_N"]))
        )
        num_sms = int(kwargs.get("NUM_SMS", 1))
        return persistent_matmul_launch(
            plain_launch,
            meta["operand_specs"],
            layout.size,
            kwargs,
            num_sms=num_sms,
            group_m=int(kwargs.get("GROUP_M", 8)),
            bias_spec=meta.get("bias_spec"),
            b_transposed=bool(meta.get("b_transposed", False)),
        )

GEMM = GemmTemplate()

GEMM_PERSISTENT = PersistentGemmTemplate()










framework_scaled_mm = ExternKernelChoice(None, "framework_scaled_mm")

framework_fp8_mm = ExternKernelChoice(None, "framework_fp8_mm")

#: The operation namespace, under a name of this project's own.  A product that
#: defers to the framework's own multiply is measured against it, so the
#: namespace it is reached through is part of what this module offers.  It is
#: reached by whatever name this project's operations are registered under, and
#: the overloads are what a product is measured against by name.
framework = tp.ops.tp

#: The primitive namespace: the operations a product is written in terms of before
#: anything decides what it becomes.  A kernel that is measured against a product
#: computed from primitives is being measured against the whole way of doing it,
#: which is why this namespace is named beside the other one.
prims = tp.ops.prims


@functools.cache
def lazy_register_extern_choice(fn):
    """One choice per operation, however many times it is asked for.

    Cached because the same operation is registered from several call sites, and
    two choices for one operation would be two things measured against each other
    rather than one thing measured against the templates.
    """

    return ExternKernelChoice(fn)


def dims_are_int(dims: Sequence[Any]) -> bool:
    """Whether every one of some sizes is a number rather than a symbol.

    Which decides whether a size can be used to choose anything: a size that is
    not known until the program runs cannot pick a tile, and a table keyed by it
    could only be read at run time, which is where the table is not.
    """

    return all(isinstance(dim, int) for dim in dims)


def _is_int8_mat(mat: Buffer) -> bool:
    """Whether a product is between whole-number tiles.

    Which is what makes the integer kernels the only ones that can be right for
    it: they accumulate in a wider type on purpose, and the rest do not.
    """

    return mat.get_dtype() in (tp.int8, tp.uint8)


def _check_addmm_input_metadata(inp: Buffer, mat1: Buffer, mat2: Buffer) -> None:
    """Whether the three of a product-with-a-bias can be added at all.

    Checked before anything is chosen, because a product whose bias is a
    different type from what it multiplies is not a product this kernel can
    compute however well it fits -- and a tile chosen for it would be measured on
    a call that cannot be made.
    """

    tp._check(
        inp.get_dtype() == mat1.get_dtype() and inp.get_dtype() == mat2.get_dtype(),
        lambda: "input dtypes must be the same",
    )
    tp._check(
        inp.get_device() == mat1.get_device() and inp.get_device() == mat2.get_device(),
        lambda: "all inputs must be on the same device",
    )


def check_supported_striding(mat_a: Buffer, mat_b: Buffer) -> None:
    """Whether a product's two tiles are laid out the way a kernel reads them.

    The left tile is read along its last axis and the right along its first, so
    one of them has to run contiguously in that direction; a tile that is not
    contiguous there would be gathered, which is a different kernel and a
    different cost.  An empty tile is allowed, because there is nothing to read
    and a stride is not a fact about nothing.
    """

    def is_row_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[1], 1)

    def is_col_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[0], 1)

    def has_zero_dim(size) -> bool:
        return bool(
            V.graph.sizevars.statically_known_equals(size[0], 0)
            or V.graph.sizevars.statically_known_equals(size[1], 0)
        )

    tp._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )
    tp._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """A product with a bias that is one value rather than a row of them.

    A bias of one value is a different shape from a bias of a row, and the
    framework's own kernel for the two is not the same one.  A single row that
    happens to hold one value repeated is the same computation as one value, so
    it is passed as one value -- which is what makes this the shape the cheaper
    kernel takes, and worth taking whenever the shape is really there.
    """

    if inp.stride(0) == 0 and inp.size(0) != 0 or inp.size(0) == 1:
        return tp.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return tp.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


def decomposeK(a, b, k_splits: int):
    """A product computed as several products over parts of the contracted axis.

    Each part is a product of a slice of each side and the parts are added.  The
    parts are accumulated in the wider type they are made in and the result is
    cast back once at the end, so no part is rounded before it is added -- which
    is the whole reason to split at all.
    """

    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    k_parts = k // k_splits
    B = k_splits
    a_reshaped = tp.permute(a.reshape(m, B, k_parts), (1, 0, 2))
    b_reshaped = b.reshape(B, k_parts, n)
    result = tp.bmm(a_reshaped, b_reshaped, out_dtype=tp.float32)
    reduced_buf = tp.sum(result, 0)
    return reduced_buf.to(a.dtype)


def contiguous_mm(a, b):
    """A product whose right side is made contiguous first.

    A right side that is not contiguous is read the wrong way round by a kernel
    written for a contiguous one, so it is copied.  Which of the two sides is
    worth copying is a question about the layouts, and the left is the one left
    alone because a kernel can read it either way.
    """

    return tp.mm(a, b.contiguous())


def contiguous_addmm(inp, a, b):
    """A product with a bias, whose right side is made contiguous first."""

    return tp.addmm(inp, a, b.contiguous())


class _DescriptorDrivenTemplate(PersistentGemmTemplate):

    """A product whose tiles are read through a descriptor.

    The arithmetic and the tile space are the persistent product's; what
    differs is how a tile is fetched.  A descriptor is built on the host from a
    base pointer, a shape, strides and a block shape, and the kernel then asks
    for tiles by coordinate rather than by address -- so the address arithmetic
    that a persistent product would repeat per tile is stated once, on the
    host, and the kernel spends its registers on the arithmetic instead.

    The cost is a hardware requirement rather than a preference: a descriptor
    is a feature of the memory system, and on a machine without it the kernel
    cannot run at all.  So the template asks the device before it offers
    anything, and a machine that cannot answer is a machine this template has
    nothing to say about.
    """

    minimum_major = 9

    maximum_major: int | None = None

    def __init__(self, name: str, hash_suffix: str = ""):
        super().__init__(
            name, hash=f"{name}:{hash_suffix}:{_gemm_source_identity()}"
        )

    def supports_device(self, device=None) -> bool:
        major, _minor = device_capability(device)
        if major < self.minimum_major:
            return False
        if self.maximum_major is not None and major >= self.maximum_major:
            return False
        return True

    def generate(self, params, out_specs, meta, plain_launch=None):
        """The launcher, or ``None`` when the device cannot run this form.

        The check comes first and it is not a formality: the kernel's tiles are
        fetched through a descriptor, which is a feature of the memory system,
        so on a machine without one there is no kernel to return.  Returning
        the persistent product's launcher instead would be worse than nothing
        -- it would be a kernel that runs and is not the thing that was asked
        for.
        """

        if not self.supports_device(meta.get("device")):
            return None
        if plain_launch is None:
            return None
        kwargs = params.to_kwargs()
        if "BLOCK_M" not in kwargs:
            return None
        from ..codegen.triton_gemm_tma import tma_gemm_launch

        specs = tuple(meta.get("operand_specs") or ())
        sizes = tuple(meta.get("operand_sizes") or ())
        if len(specs) < 2 or len(sizes) < 2:
            return None

        inputs = MMKernelInputs(
            shapes=sizes,
            dtypes=(meta.get("operand_dtype"),) * len(sizes),
            device=meta.get("device"),
            operands=specs,
            out_dtype=meta.get("out_dtype"),
            extra=meta,
        )
        try:
            m, n, k = inputs.mnk()
        except (NotImplementedError, AssertionError):
            return None
        geometry = {
            "mnk": (m, n, k),
            "num_sms": int(kwargs.get("NUM_SMS", 1)),
            "a_shape": sizes[0],
            "a_stride": tuple((meta.get("operand_strides") or ((0,) * len(sizes[0]),))[0]),
            "b_shape": sizes[1],
            "b_stride": tuple((meta.get("operand_strides") or ((0,) * len(sizes[0]), (0,) * len(sizes[1])))[1]),
            "has_workspace": bool(getattr(self, "uses_workspace", False)),
        }
        layout = out_specs[0] if out_specs else None
        return TritonChoiceCaller(
            name=f"{self.name}-{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}x{kwargs['BLOCK_K']}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
            source=f"{self.name}:{sorted(kwargs.items())}",
        ).bind(
            tma_gemm_launch(
                specs[0], specs[1], geometry, kwargs, plain_launch,
                allow_tf32=bool(meta.get("allow_tf32", False)),
            )
        )

class PersistentTmaGemmTemplate(_DescriptorDrivenTemplate):

    """The persistent product, reading its tiles through descriptors."""

    def __init__(self):
        super().__init__("mm_persistent_tma", "descriptor")

class BlackwellWorkspacePersistentTmaTemplate(_DescriptorDrivenTemplate):

    """The persistent descriptor-driven product, as the later generation has it.

    The form differs from its predecessor in what it carries between tiles
    rather than in how it fetches them: a workspace sized by the device's own
    limits is what lets a tile's partial result outlive the tile that produced
    it, which is what the earlier persistent form has to keep in registers.
    The workspace is a property of the device, so the template refuses the
    machines whose limits it does not have.
    """

    #: What distinguishes this form from its predecessor: a tile's partial can
    #: outlive the tile that produced it, which needs scratch space sized by
    #: the device rather than registers.
    uses_workspace = True

    def __init__(self):
        super().__init__("blackwell_ws_persistent_device_tma", "workspace-descriptor")
        self.minimum_major = 10

class ScaledGemmConfigHeuristics(GemmConfigHeuristics):

    """The product's candidates, but only where the call carries factors.

    A scaled product is a different arithmetic rather than a different
    schedule, so it is not a candidate for an unscaled call: offering it there
    would have the measurement compare two answers and report the faster one,
    and the faster one would be whichever ignored the factors.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        return super().should_run(inputs) and bool(
            inputs.extra.get("has_scales")
        )

class ScaledGemmTemplate(PersistentGemmTemplate):

    """A product whose operands carry their own factors.

    Where the factors are applied is the whole difference between the two
    forms, and it is not a tuning knob:

      in the main loop   each tile of an operand is scaled as it arrives, so
                         the accumulator only ever holds unscaled products and
                         the factors are paid for once per tile rather than
                         once per element of the result
      in the epilogue    the accumulator is scaled once at the end, which is
                         fewer operations and is wrong whenever the factors
                         would have to be held at the accumulator's width

    So the site of the scaling is what distinguishes them, and a call cannot be
    offered both without asking which one it meant.
    """

    scale_site = "main_loop"

    def __init__(self, name: str, scale_site: str = "main_loop"):
        super().__init__(
            name, hash=f"{name}:{scale_site}:{_gemm_source_identity()}"
        )
        self.scale_site = scale_site
        self.heuristics = ScaledGemmConfigHeuristics()

    def generate(self, params, out_specs, meta, plain_launch=None):
        if not meta.get("has_scales"):
            return None
        return super().generate(params, out_specs, meta, plain_launch)

    def launcher(self, kwargs, meta, layout, plain_launch):
        """The scaled kernel, for the site this form scales at.

        The factors are operands, not constants: a call that has them names
        where they are, and the kernel reads the part its tile needs.  So the
        two forms differ only in the site, and a call cannot be offered both
        without saying which one it meant.
        """

        from ..codegen.triton_gemm_scaled import scaled_gemm_launch

        scale_specs = tuple(meta.get("scale_specs") or ())
        if len(scale_specs) < 2:
            return plain_launch
        return scaled_gemm_launch(
            meta["operand_specs"][0],
            meta["operand_specs"][1],
            scale_specs[0],
            scale_specs[1],
            {
                "mnk": (int(layout.size[0]), int(layout.size[1]),
                        int((meta.get("operand_sizes") or ((0, 0), (0, 0)))[0][1])),
                "scale_site": self.scale_site,
                "recipe_a": int(meta.get("recipe_a", 0)),
                "recipe_b": int(meta.get("recipe_b", 0)),
            },
            kwargs,
            plain_launch,
        )

class MainLoopScaledGemmTemplate(ScaledGemmTemplate):

    """The scaled product, with each operand's factor applied as it arrives."""

    def __init__(self):
        super().__init__("mm_main_loop_scaling", scale_site="main_loop")

class EpilogueScaledGemmTemplate(ScaledGemmTemplate):

    """The scaled product, with the factors applied once to the accumulator."""

    def __init__(self):
        super().__init__("mm_epilogue_scaling", scale_site="epilogue")

GEMM_PERSISTENT_TMA = PersistentTmaGemmTemplate()

BLACKWELL_WS_PERSISTENT_TMA = BlackwellWorkspacePersistentTmaTemplate()

GEMM_MAIN_LOOP_SCALING = MainLoopScaledGemmTemplate()

GEMM_EPILOGUE_SCALING = EpilogueScaledGemmTemplate()

#: The candidates this module offers.  A product's own module owns the ones that
#: are its way of being done, so a template and the candidates it belongs with
#: are added together rather than in two places.
GEMM_TEMPLATES = (
    GEMM,
    GEMM_PERSISTENT,
    GEMM_PERSISTENT_TMA,
    BLACKWELL_WS_PERSISTENT_TMA,
    GEMM_MAIN_LOOP_SCALING,
    GEMM_EPILOGUE_SCALING,
)


def _is_tensorwise_scaling(sz: Any) -> bool:
    """Whether a scale is one number for the whole tile.

    Which is what makes it a per-tensor scale: a scale that is indexed at all is
    a scale of some part of the tile, and how much of it is a question about the
    axis it indexes rather than about the product.
    """

    return len(sz) == 0 or all(
        V.graph.sizevars.statically_known_equals(d, 1) for d in sz
    )


def _is_rowwise_scaling(sz: Any, transpose: bool) -> bool:
    """Whether a scale is one number per row of the side it scales.

    A row of the left side is a row of the result and a row of the right side is
    a column of it -- so which axis carries the scale depends on which side this
    is, and a scale read against the wrong axis is a scale of the wrong thing.
    """

    idx = 0 if transpose else -1
    return V.graph.sizevars.statically_known_equals(sz[idx], 1)


def _is_blockwise1xTILESIZE_scaling(
    sz: Any, tensor_sz: Any, tile_size: int, transpose: bool
) -> bool:
    """Whether a scale is one number per row of blocks, ``tile_size`` within a row.

    Which is the shape a scale takes when the format can hold a value over only
    that much of the row at a time.  Which side is which depends on the side, for
    the same reason a row's scale does.
    """

    lhs = 1 if transpose else 0
    rhs = 0 if transpose else 1
    return V.graph.sizevars.statically_known_equals(
        sz[lhs], tensor_sz[lhs]
    ) and V.graph.sizevars.statically_known_equals(
        sz[rhs], ceildiv(tensor_sz[rhs], tile_size)
    )


def _is_blockwise128x128_scaling(sz: Any, tensor_sz: Any) -> bool:
    """Whether a scale is one number per 128 by 128 block."""

    return V.graph.sizevars.statically_known_equals(
        sz[0], ceildiv(tensor_sz[0], 128)
    ) and V.graph.sizevars.statically_known_equals(
        sz[1], ceildiv(tensor_sz[1], 128)
    )


def is_desired_scaling(
    t: Any, scale_size: Any, scaling_type: ScalingType, transpose: bool = False
) -> bool:
    """Whether a scale has the shape one kind of scaling means.

    The four shapes are the four the kernels here can read, and they are told
    apart by the scale's own size rather than by anything said about it: two
    scales of the same size and different meanings are one shape, and which is
    meant is the caller's to know and to pass.
    """

    match scaling_type:
        case ScalingType.TensorWise:
            return _is_tensorwise_scaling(scale_size)
        case ScalingType.RowWise:
            return _is_rowwise_scaling(scale_size, transpose)
        case ScalingType.BlockWise1x128:
            return _is_blockwise1xTILESIZE_scaling(
                scale_size, t.get_size(), 128, transpose
            )
        case ScalingType.BlockWise128x128:
            return _is_blockwise128x128_scaling(scale_size, t.get_size())
        case _:
            raise AssertionError(f"Unsupported scaling type {scaling_type}")


def get_tile_size(scale_option: ScalingType) -> int:
    """The tile a kind of scaling is measured in tiles of.

    Both blockwise shapes are 128 in the direction that is blocked and neither is
    128 in the other, so a shape whose tile this cannot name is a shape whose
    tile is unknown -- which is not the same as a shape with no tile.
    """

    match scale_option:
        case ScalingType.BlockWise128x128:
            return 128
        case ScalingType.BlockWise1x128:
            return 128
        case _:
            raise AssertionError(
                f"Unsupported scaling type {scale_option} in get_tile_size"
            )


#: The pairs of scaling the two sides may be read with, in the order they are
#: tried.  A pair is here only if a kernel exists that reads both sides that way,
#: so the order is the order of preference: the first pair is the one whose
#: kernels have been measured longest.
scaling_pairs = [
    (ScalingType.TensorWise, ScalingType.TensorWise),
    (ScalingType.RowWise, ScalingType.RowWise),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise128x128),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise1x128),
    (ScalingType.BlockWise128x128, ScalingType.BlockWise1x128),
]

#: The scalings a product can have applied after it, when nothing was scaled on
#: the way in.  Applied afterwards is what makes them cheap: the scale is read
#: once, on one value, rather than once per tile of the contracted axis.
epilogue_scaling_types = [ScalingType.TensorWise, ScalingType.RowWise]

#: The scalings that have to be applied on the way in, because they change what
#: is accumulated.  A scale applied afterwards would scale an already-scaled sum.
main_loop_scaling_types = [ScalingType.BlockWise1x128, ScalingType.BlockWise128x128]


def get_scaling_options(
    mat_a: Any, mat_b: Any, scale_a_size: Any, scale_b_size: Any
) -> tuple[ScalingType, ScalingType]:
    """Name the scaling both sides have, or refuse to.

    The pairs are tried in order and the first that both sides fit is the answer,
    so a pair one side does not fit is not a weaker answer -- it is not an
    answer.  Refusing rather than guessing is the point: a kernel handed a
    scaling it does not read computes a product nobody asked for.
    """

    for scale_option_a, scale_option_b in scaling_pairs:
        if is_desired_scaling(
            mat_a, scale_a_size, scale_option_a
        ) and is_desired_scaling(mat_b, scale_b_size, scale_option_b, transpose=True):
            return (scale_option_a, scale_option_b)
    raise AssertionError(
        f"Inductor Triton does not support scale_a.shape = {scale_a_size}, "
        f"scale_b.shape = {scale_b_size}"
    )


#: The template a plain product is measured with, named for what it is rather
#: than for the class it happens to be: a caller that wants a plain product
#: should not have to know which class of template that is this week.
mm_template = GEMM

#: The template a product whose tiles are held across the whole launch is
#: measured with.
persistent_mm_template = GEMM_PERSISTENT

#: The template a product whose tiles are read through a descriptor is measured
#: with.
persistent_tma_mm_template = GEMM_PERSISTENT_TMA

#: The template a product that needs a workspace of its own is measured with.
blackwell_ws_persistent_device_tma_mm_template = BLACKWELL_WS_PERSISTENT_TMA


#: A product of whole-number tiles, measured against the framework's own.  The
#: integer kernels accumulate in a wider type on purpose, so this is a product
#: with its own way of being done rather than the same way done faster.
framework_mm = ExternKernelChoice(None, "framework_mm")

#: A product whose result is asked for in a type of its own.
framework_mm_dtype = ExternKernelChoice(None, "framework_mm_dtype")

#: A product with a bias added to it.
framework_addmm = ExternKernelChoice(None, "framework_addmm")

#: A product with a bias that may be one value rather than a row of them, which
#: is why it is a choice of its own: the shape decides which kernel the framework
#: would reach for, and the shape is not known until the call is.
framework_bias_addmm = ExternKernelChoice(bias_addmm, "framework_bias_addmm")

#: A product of whole-number tiles into a whole-number result, which cannot be
#: written as an out-variant because the result's type is fixed by its inputs.
framework__int_mm = ExternKernelChoice(
    None, "framework__int_mm", has_out_variant=False
)

#: A product of two sparse tiles that are mostly not there, whose result is
#: sparse and so has no out-variant to be written into.
framework__sparse_semi_structured_mm = ExternKernelChoice(
    None,
    "framework__sparse_semi_structured_mm",
    has_out_variant=False,
)


@functools.cache
def _is_sm7x_or_older_gpu(index: int | None) -> bool:
    """Whether a device is old enough that a product must be shaped differently.

    Cached because it is asked per product and a device's generation does not
    change while a program runs.  The answer is about the tile and the
    instructions available for it, both of which the device's generation decides.
    """

    props = tp.cuda.get_device_properties(index or 0)
    return props.major <= 7


def get_size_hints(mat1: Buffer, mat2: Buffer, m: Any, n: Any, k: Any):
    """The three sizes of a product, as numbers where they can be.

    A size that is not known until the program runs is asked for again through
    the hints, which are what a table can be keyed by: two products whose sizes
    are only known later are the same product as far as a table is concerned if
    their hints agree, and a kernel chosen on a hint is a kernel chosen for every
    size that shares it.
    """

    if not isinstance(m, int) or not isinstance(k, int):
        m, k = V.graph.sizevars.optimization_hints(mat1.get_size())
    if not isinstance(n, int) or not isinstance(k, int):
        k, n = V.graph.sizevars.optimization_hints(mat2.get_size())
    return (m, n, k)


def get_size_hints_strides(mat1: Buffer, mat2: Buffer):
    """The two tiles' strides, as numbers where they can be.

    Asked for by hint for the same reason the sizes are: a layout that is only
    known later is one layout as far as a table is concerned, and the table is
    what the strides are being read for.
    """

    mat1_stride = mat1.layout.stride
    mat2_stride = mat2.layout.stride
    strides = [mat1_stride, mat2_stride]
    strides_hints = []
    for stride in strides:
        if not isinstance(stride, int):
            stride = V.graph.sizevars.optimization_hints(stride)
        strides_hints.append(stride)
    return (strides_hints[0], strides_hints[1])


#: A product of scaled tiles with a swizzled scale layout has no kernel here that
#: reads the swizzle, so it is handed to the framework whole.  Not added to the
#: general fallback set, because it is reached from the lowering below rather than
#: being the operation's fallback everywhere: the same operation with a layout
#: this tree can read is a kernel, and a set that included this would say
#: otherwise for all of them.
scaled_mm_v2_fallback = fallback_handler(
    framework._scaled_mm_v2.default, add_to_fallback_set=False
)

#: The template a product whose tiles are read through a descriptor and whose
#: scale is applied after the product is measured with.
scaled_mm_device_tma_epilogue_scaling_template = GEMM_EPILOGUE_SCALING

#: The template a product whose tiles are read through a descriptor and whose
#: scale is applied while the product is accumulated is measured with.
scaled_mm_device_tma_main_loop_scaling_template = GEMM_MAIN_LOOP_SCALING


class DecomposeKSugraphTemplate(SubgraphTemplate):
    """A product computed as several products over parts of the contracted axis.

    A separate template rather than a flag on the plain one because the thing it
    computes is the same and the way it is computed is not: a product split along
    the contracted axis can be finished inside a tile, while the same product
    computed whole cannot.  Which of the two is better is a question about the
    sizes, and the answer is what the two templates being measured against each
    other is for.
    """

    def __init__(self):
        super().__init__(name="decompose_k")

    def generate(  # type: ignore[override]
        self,
        input_nodes: list,
        layout: Layout,
        k_split: int,
    ):
        name = f"decompose_k_mm_{k_split}_split"
        description = f"k_split={k_split!r}"
        fn = make_graph(
            functools.partial(decomposeK, k_splits=k_split),
            select_decomp_table(),
        )
        return super().generate(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            make_graph=fn,
            description=description,
        )


decompose_k_subgraph_template = DecomposeKSugraphTemplate()


class ContiguousTemplate(SubgraphTemplate):
    """A product computed with one of its sides made contiguous first.

    A template per function rather than one template with a flag, because the two
    things it is used for -- a plain product and a product with a bias -- differ
    in what they read and in what they have to be given, and a template that
    could do both would have to be told which it was doing every time.
    """

    def __init__(self, name: str, description: str, fn: Any):
        self.name = name
        self.description = description
        self.fn = fn
        super().__init__(name=name)

    def generate(  # type: ignore[override]
        self,
        input_nodes: list,
        layout: Layout,
    ):
        fn = make_graph(self.fn, select_decomp_table())
        return super().generate(
            name=self.name,
            input_nodes=input_nodes,
            layout=layout,
            make_graph=fn,
            description=self.description,
        )


mm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_mm", "contiguous mm", contiguous_mm
)
addmm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_addmm", "contiguous addmm", contiguous_addmm
)



def _realize(value):
    """A value as something outside this compiler has to be handed.

    Made to run along its last axis, because a caller that is not this compiler
    reads what it is given and cannot be told what a stride means.
    """

    return realize_inputs(value)


@register_lowering(framework.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, out_dtype=None, *, layout=None):
    """A product, measured against everything that can compute one.

    What the call becomes is decided by measuring the candidates rather than by a
    rule, because the answer depends on the sizes, the types, the layouts and the
    device, and no rule over those four is right for all of them.  What the call
    *is* is fixed here: two matrices multiplied, optionally into a result of
    another type, and everything after that is a question of how.
    """

    if out_dtype is not None:
        input_dtype = mat1.get_dtype()
        tp._check(
            mat2.get_dtype() == input_dtype,
            lambda: "input dtypes must be the same",
        )
        tp._check(
            out_dtype == input_dtype
            or (
                out_dtype == tp.float32
                and input_dtype in (tp.float16, tp.bfloat16)
            ),
            lambda: (
                "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
            ),
        )

    # A call the framework's own product is better at is not offered a template
    # at all: the measurement would be comparing two answers to a different
    # question, and reporting the faster one as this call's best.
    if use_native_matmul(mat1, mat2):
        return None

    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )

    if out_dtype is None and _use_small_mm_pointwise(m, k, n, layout.device.type):
        return None

    static_shape, is_nonzero = _is_static_problem(layout)
    name = "mm"
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)

    log.info(
        "Tuned product: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    choices: list = []
    aten_handler = framework_mm
    aten_extra_kwargs: dict = {}
    if out_dtype is not None:
        aten_handler = framework_mm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    templates_to_use: list = []
    kwarg_overrides: dict = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        if aten_extra_kwargs:
            kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout, check_max_autotune=True)
    ):
        if use_decompose_k_choice(m, n, k):
            templates_to_use.append(decompose_k_subgraph_template)
        # A template performs poorly for a long contracted axis, so where the
        # split is likely to win the plain template is not offered at all: two
        # candidates that cannot both win is one candidate too many, and the one
        # that cannot win costs a build to find out.
        is_exhaustive = config.max_autotune_gemm_search_space == "EXHAUSTIVE"
        if is_exhaustive or not use_decompose_k_choice(m, n, k, threshold_multiple=2):
            templates_to_use.append(mm_template)
            if use_triton_blackwell_tma_template(
                mat1, mat2, output_layout=layout, add_guards=True
            ):
                templates_to_use.append(
                    blackwell_ws_persistent_device_tma_mm_template
                )
            elif use_triton_tma_template(
                mat1, mat2, output_layout=layout, add_guards=True
            ):
                templates_to_use.append(persistent_tma_mm_template)
            else:
                templates_to_use.append(persistent_mm_template)

        templates_to_use.append(mm_contiguous_subgraph_template)

    choices.extend(
        get_template_configs(
            kernel_inputs,
            templates_to_use,
            "mm",
            kwarg_overrides=kwarg_overrides,
        )
    )

    node, _ = autotune_select_algorithm(
        name, choices, kernel_inputs.nodes(), layout
    )
    return node


@register_lowering(framework._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    """A product of whole-number tiles into a whole-number result.

    Kept apart from the floating-point one because the result's type is fixed by
    the inputs rather than asked for, so there is no out_dtype to settle and a
    template written for a type the result cannot have would be offered a call it
    cannot answer.
    """

    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=tp.int32
    )
    name = "int_mm"
    log.info(
        "Tuned whole-number product: m=%s, n=%s, k=%s, mat1_dtype=%s, "
        "mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )
    static_shape, is_nonzero = _is_static_problem(layout)
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=tp.int32)
    templates_to_use: list = []
    if use_aten_gemm_kernels():
        templates_to_use.append(framework__int_mm)
    if is_nonzero and use_triton_template(layout, enable_int32=True, check_max_autotune=False):
        templates_to_use.append(mm_template)
    choices = get_template_configs(kernel_inputs, templates_to_use, name)
    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


def use_triton_scaling_template(
    scale_option_a: ScalingType, scale_option_b: ScalingType, scaling_types: list
) -> bool:
    """Whether both sides' scaling is of a kind one template can read.

    Asked about both sides together because a template reads the two scales as one
    decision -- where the scale is applied is a property of the pair, not of each
    side -- and a template that read one side's kind and not the other's would be
    reading one scale and scaling by the other.
    """

    return scale_option_a in scaling_types and scale_option_b in scaling_types


@register_lowering(framework._scaled_mm, type_promotion_kind=None)
def tuned_scaled_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum: bool = False,
    layout=None,
):
    """A product of scaled tiles, with the scale read from each side's own shape.

    Which scaling it is comes from the scales' own sizes rather than from
    anything said about them, because two scales of the same size and different
    meanings are one shape and only the shape is knowable here.  Which template
    can read that shape is then a separate question, and the answer decides
    whether the scale is applied as the tiles are read or once at the end -- which
    is a difference in what is accumulated and not in what is computed.
    """

    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)
    scale_a_real, scale_b_real = realize_inputs(scale_a, scale_b)
    input_nodes: list
    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]
    kernel_inputs = MMKernelInputs(
        [mat_a, mat_b, scale_a_real, scale_b_real, bias] if bias else
        [mat_a, mat_b, scale_a_real, scale_b_real],
        out_dtype=out_dtype,
    )

    log.info(
        "Tuned scaled product: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, "
        "output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    templates_to_use: list = []
    kwarg_overrides: dict = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(framework_fp8_mm)
        kwarg_overrides[framework_fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )
    _, is_nonzero = _is_static_problem(layout)

    # The scale's own type is what says which templates were written for it: a
    # scale in the wider type is read by the whole-tile templates, and one in the
    # narrower type by the eight-bit ones.
    if (
        scale_a.dtype == tp.float32
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)
        scale_a_size, scale_b_size = (scale_a_real.shape, scale_b_real.shape)
        scale_option_a, scale_option_b = get_scaling_options(
            mat_a, mat_b, scale_a_size, scale_b_size
        )
        if use_triton_tma_template(
            mat_a, mat_b, output_layout=layout, add_guards=True
        ) and (not bias):
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value
            if use_triton_scaling_template(
                scale_option_a, scale_option_b, epilogue_scaling_types
            ):
                templates_to_use.append(
                    scaled_mm_device_tma_epilogue_scaling_template
                )
                kwarg_overrides[
                    scaled_mm_device_tma_epilogue_scaling_template.uid
                ] = overriders
            elif use_triton_scaling_template(
                scale_option_a, scale_option_b, main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)
                templates_to_use.append(
                    scaled_mm_device_tma_main_loop_scaling_template
                )
                kwarg_overrides[
                    scaled_mm_device_tma_main_loop_scaling_template.uid
                ] = overriders
            else:
                raise AssertionError(
                    "scaling options that are present in both epilogue scaling "
                    "and main loop scaling are not supported"
                )
        if use_triton_blackwell_tma_template(
            mat_a, mat_b, output_layout=layout, add_guards=True
        ) and (not bias):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[
                blackwell_ws_persistent_device_tma_mm_template.uid
            ] = overriders
        if use_triton_scaling_template(
            scale_option_a, scale_option_b, epilogue_scaling_types
        ):
            templates_to_use.append(mm_template)
            kwarg_overrides[mm_template.uid] = overriders

    choices = get_template_configs(
        kernel_inputs, templates_to_use, name, kwarg_overrides=kwarg_overrides
    )
    if scale_a.dtype != tp.float32:
        node, _ = autotune_select_algorithm(name, choices, input_nodes, layout)
        return node
    node, _ = autotune_select_algorithm(
        name, choices, kernel_inputs.nodes(), layout
    )
    return node


@register_lowering(framework.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """A product with a bias, measured against everything that can compute one.

    A bias of zero is the plain product with a subtraction nobody asked for, so a
    call that multiplies the bias by zero is answered as the product it is; and a
    product multiplied by zero is a constant, which no measurement can improve on.
    Both are decided before anything is measured, because a measurement of them
    would be a measurement of a call that was never going to be that shape.
    """

    if beta == 0 and mat1.get_device().type == "cuda":
        _check_addmm_input_metadata(inp, mat1, mat2)
        if alpha == 0:
            _, _, _, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
            return tp.full(
                layout.size, 0, dtype=layout.dtype, device=layout.device
            )
        if layout is not None:
            result = tuned_mm(mat1, mat2, layout=layout)
        else:
            result = tuned_mm(mat1, mat2)
        if alpha != 1:
            result = _scaled(result, alpha)
        return result

    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(
        mat1, mat2, inp, layout=layout
    )
    inp = realize_inputs(inp)
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "addmm"
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    kernel_inputs_aten = MMKernelInputs(
        [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )

    log.info(
        "Tuned product with bias: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, "
        "output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    choices: list = []
    if not is_nonzero or not (config.max_autotune or config.max_autotune_gemm):
        # Nothing to choose between, so the framework's own kernel is used without
        # measuring it: a measurement that could only confirm what there is one
        # of costs the compile it is meant to save.
        choices.extend(
            get_template_configs(kernel_inputs_aten, [framework_addmm], name)
        )
        node, _ = autotune_select_algorithm(
            name, choices, kernel_inputs.nodes(), layout
        )
        return node

    templates_to_use: list = []
    if use_aten_gemm_kernels():
        aten_templates: list = [framework_addmm]
        if inp.get_stride()[0] == 0 and len(inp.get_size()) == 2:
            # A bias that is one value repeated is the same computation as one
            # value, and the framework reaches a different -- and for this shape
            # cheaper -- kernel when told it is one value.
            aten_templates.append(framework_bias_addmm)
        choices.extend(
            get_template_configs(kernel_inputs_aten, aten_templates, name)
        )
    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(mm_template)
        if use_triton_blackwell_tma_template(
            mat1, mat2, output_layout=layout, add_guards=True
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
        elif use_triton_tma_template(
            mat1, mat2, output_layout=layout, add_guards=True
        ):
            templates_to_use.append(persistent_tma_mm_template)
        else:
            templates_to_use.append(persistent_mm_template)
        choices.extend(
            get_template_configs(
                kernel_inputs_aten, [addmm_contiguous_subgraph_template], name
            )
        )
    choices.extend(get_template_configs(kernel_inputs, templates_to_use, name))

    node, _ = autotune_select_algorithm(
        name, choices, kernel_inputs.nodes(), layout
    )
    return node


@register_lowering(framework._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    """A product of two tiles that are mostly not there.

    Only the framework's own kernel is a candidate: a tile this shape is read
    through a description of which of it exists, and no template here reads that
    description.  Offered as a choice anyway so that the call is measured like
    any other and the floor is the one it is measured against.
    """

    mat1, mat1_meta, mat2 = realize_inputs(mat1, mat1_meta, mat2)
    m1, k1 = mat1.get_size()
    m2, _ = mat1_meta.get_size()
    k2, n = mat2.get_size()
    m = V.graph.sizevars.check_equals_and_simplify(m1, m2)
    k = V.graph.sizevars.check_equals_and_simplify(2 * k1, k2)
    if layout is None:
        layout = FixedLayout(
            mat2.get_device(),
            out_dtype if out_dtype else mat2.get_dtype(),
            [m, n],
            [n, 1],
        )
    elif out_dtype is not None:
        raise AssertionError("out_dtype is ignored if layout is specified.")
    choices = (
        [
            framework__sparse_semi_structured_mm.bind(
                (mat1, mat1_meta, mat2), layout, out_dtype=out_dtype
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )
    node, _ = autotune_select_algorithm(
        "sparse_semi_structured_mm", choices, (mat1, mat1_meta, mat2), layout
    )
    return node


def _scaled(value, factor):
    """A value multiplied by a number, as a lowered value."""

    from ..op_lowerings import pointwise
    from ..loops import ops

    return pointwise(value, lambda x: ops.mul(x, float(factor)), value, val=value)


@register_lowering(framework._scaled_mm_v2, type_promotion_kind=None)
def tuned_scaled_mm_v2(
    mat_a,
    mat_b,
    scale_a: list,
    recipe_a: list,
    swizzle_a: list,
    scale_b: list,
    recipe_b: list,
    swizzle_b: list,
    bias=None,
    out_dtype=None,
    contraction_dim=None,
    use_fast_accum: bool = False,
    layout=None,
):
    """A product whose scales say how they are to be read, not only how large.

    The recipe says the *shape* of each scale and the swizzle says how it is laid
    out, so between them they say more than a scale's own size does.  Only one
    combination of those is read by anything here -- a single level of scales, in
    the wider type, not swizzled, of a shape that is blocked along one axis or the
    whole tile -- and a call that says anything else is handed to the operation
    that can do it, because a template offered such a call would read the scale
    the way it reads a simple one and compute a different product.

    The operation is called directly rather than through the measured choice,
    because its own convention for where the second scale goes is not the one the
    measured choice uses, and going through the choice would quietly change what
    the call computes.
    """

    def check_supported_recipe(recipe: list) -> bool:
        disallowed = [ScalingType.BlockWise1x16, ScalingType.BlockWise1x32]
        return all((ScalingType(r) not in disallowed for r in recipe))

    is_single_level_scale = len(scale_a) == 1 and len(scale_b) == 1
    supported_recipe = check_supported_recipe(recipe_a) and check_supported_recipe(
        recipe_b
    )
    if (
        any(s != 0 for s in swizzle_a)
        or any(s != 0 for s in swizzle_b)
        or not supported_recipe
        or not is_single_level_scale
        or scale_a[0].dtype != tp.float32
    ):
        fallback_contraction_dim = [] if contraction_dim is None else contraction_dim
        return scaled_mm_v2_fallback(
            mat_a,
            mat_b,
            scale_a,
            recipe_a,
            swizzle_a,
            scale_b,
            recipe_b,
            swizzle_b,
            bias,
            out_dtype,
            fallback_contraction_dim,
            use_fast_accum,
        )
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)
    scale_a_real, scale_b_real = realize_inputs(scale_a[0], scale_b[0])
    input_nodes: list
    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]
    kernel_inputs = MMKernelInputs(input_nodes, mat1_idx=0, mat2_idx=1, out_dtype=out_dtype)

    log.info(
        "Tuned scaled product with recipes: m=%s, n=%s, k=%s, mat1_dtype=%s, "
        "mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    choices: list = []
    templates_to_use: list = []
    kwarg_overrides: dict = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(framework_fp8_mm)
        kwarg_overrides[framework_fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )
    _, is_nonzero = _is_static_problem(layout)
    if (
        is_single_level_scale
        and supported_recipe
        and (scale_a[0].dtype == tp.float32)
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)
        scale_option_a, scale_option_b = (
            ScalingType(recipe_a[0]),
            ScalingType(recipe_b[0]),
        )
        if use_triton_tma_template(
            mat_a, mat_b, output_layout=layout, add_guards=True
        ) and (not bias):
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value
            if use_triton_scaling_template(
                scale_option_a, scale_option_b, epilogue_scaling_types
            ):
                templates_to_use.append(
                    scaled_mm_device_tma_epilogue_scaling_template
                )
                kwarg_overrides[
                    scaled_mm_device_tma_epilogue_scaling_template.uid
                ] = overriders
            elif use_triton_scaling_template(
                scale_option_a, scale_option_b, main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)
                templates_to_use.append(
                    scaled_mm_device_tma_main_loop_scaling_template
                )
                kwarg_overrides[
                    scaled_mm_device_tma_main_loop_scaling_template.uid
                ] = overriders
            else:
                raise AssertionError(
                    "scaling options that are present in both epilogue scaling "
                    "and main loop scaling are not supported"
                )
        if use_triton_blackwell_tma_template(
            mat_a, mat_b, output_layout=layout, add_guards=True
        ) and (not bias):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[
                blackwell_ws_persistent_device_tma_mm_template.uid
            ] = overriders
        if use_triton_scaling_template(
            scale_option_a, scale_option_b, epilogue_scaling_types
        ):
            templates_to_use.append(mm_template)
            kwarg_overrides[mm_template.uid] = overriders
    choices = get_template_configs(
        kernel_inputs, templates_to_use, name, kwarg_overrides=kwarg_overrides
    )
    node, _ = autotune_select_algorithm(
        name, choices, kernel_inputs.nodes(), layout
    )
    return node


def mm_autoheuristic(
    mat1,
    mat2,
    m,
    n,
    k,
    choices,
    name,
    input_nodes,
    ops,
    precondition,
    top_k: int | None = None,
    always_included=None,
):
    """The candidates a learned rule names for this call, or nothing.

    A rule is a table of what was measured to be fastest on some other machine,
    so what it gives is a *prediction*, and a prediction that cannot be made -- the
    sizes are not known, or the device has no table -- is nothing rather than a
    guess.  Which is why this returns None rather than a default candidate: the
    caller's own answer for no prediction is to measure, and a default here would
    be a measurement skipped.

    ``top_k`` asks for more than one candidate, which is what a caller who wants
    to measure a shortlist rather than a single answer is asking for.
    """

    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None
    mat1_stride, mat2_stride = get_size_hints_strides(mat1, mat2)

    def get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride):
        """What the rule is asked about, and what it may not see.

        The operations are applied after the features rather than instead of
        them, so a rule may ask about a product of two of the features and still
        see the two themselves.
        """

        context = AHContext()
        context.add_feature("m", m)
        context.add_feature("k", k)
        context.add_feature("n", n)
        context.add_feature("mat1_dtype", mat1.layout.dtype, is_categorical=True)
        context.add_feature("mat2_dtype", mat2.layout.dtype, is_categorical=True)
        context_add_strides(context, "mat1", mat1_stride)
        context_add_strides(context, "mat2", mat2_stride)
        context.add_feature(
            "mat1_iscontig", mat1.layout.is_contiguous(), is_categorical=True
        )
        context.add_feature(
            "mat2_iscontig", mat2.layout.is_contiguous(), is_categorical=True
        )
        if name == "mm":
            context_add_using_tf32(context, mat1.layout.dtype)
        return context

    def fallback():
        return None

    context = get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride)
    autoheuristic = AutoHeuristicSelectAlgorithm(
        fallback=fallback,
        choices=choices,
        input_nodes=input_nodes,
        context=context,
        name=name,
        augment_context=ops,
        precondition=precondition,
    )
    if top_k is not None:
        return autoheuristic.get_top_k_choices_caller(
            top_k, always_included=always_included
        )
    return autoheuristic.get_choice_caller()
