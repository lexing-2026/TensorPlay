"""A number of products of the same extents, walked by one fixed set of programs.

Grouping is what makes this one kernel rather than a loop: every group is the
same shape, so one table of tiles fits all of them.  Which extent a group varies
along is not a choice and not something the tile can be told -- it follows from
whether each operand is a matrix per group or one matrix the groups are cut out
of -- so the extents are read off the operands' own ranks.
"""

from __future__ import annotations

import dataclasses

import tensorplay as tp
import logging

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

#: The operation namespace, under a name of this project's own, reached by
#: whatever name this project's operations are registered under.
framework = tp.ops.tp

from dataclasses import asdict
from typing import Any

from ..op_lowerings import register_lowering
from ..codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..ir import Layout
from ..heuristics.template.base import TemplateConfigHeuristics
from .mm_common import (
    check_supported_striding,
    use_aten_gemm_kernels,
    use_triton_template,
)
from .select_algorithm import (
    ChoiceCaller,
    realize_inputs,
    call_operation,
    ExternKernelChoice,
    KernelArgs,
    TritonChoiceCaller,
    TritonTemplate,
    autotune_select_algorithm,
)
from .triton import CHOICES, dtype_size
from ..heuristics.template.cutedsl import get_groupgemm_configs
from ..runtime.triton_compat import tl
from ..utils import get_gpu_shared_memory, get_max_num_sms, get_num_sms, has_free_symbols, use_blackwell_cutedsl_grouped_mm
from ..virtualized import V
from .mm import (
    MMKernelInputs,
    _gemm_source_identity,
    contiguous_stride,
)
from ..utils import counters
from .mm_common import (
    mm_grid,
    descriptor_extents_fit,
    descriptor_form,
    descriptor_offset_fits,
    device_capability,
    load_kernel_template,
    num_sms,
)

#: A grouped product, and the same with a scale, as candidates in their own
#: right: each is a computation the framework does in one call, and a caller
#: who asked for it is asking for that call rather than for a product.
framework__grouped_mm = ExternKernelChoice(
    tp.ops.tp._grouped_mm, "_grouped_mm",
    has_out_variant=False,
    op_overload=tp.ops.tp._grouped_mm.default,
)
framework__scaled_grouped_mm = ExternKernelChoice(
    tp.ops.tp._scaled_grouped_mm, "_scaled_grouped_mm",
    has_out_variant=False,
    op_overload=tp.ops.tp._scaled_grouped_mm.default,
)

def grouped_mm_grid(*args):
    """The grid for the grouped product: as many programs as the device has.

    Every group is the same shape, so the whole run is one flat sequence of
    tiles and the only question a launch has to answer is how many programs to
    start.  Starting one per tile would leave most of them waiting, so the count
    is the machine's and each program walks the run taking every NUM_SMS'th
    tile -- which is why the number of programs has to be a number the kernel
    can see rather than one only the launch knows.
    """

    meta = args[-1]
    return (int(meta["NUM_SMS"]), 1, 1)

def _ramp(rows: int, cols: int, dtype, device) -> Any:
    """A ramp of ``rows`` by ``cols``, kept in a small band of values.

    Shared by the grouped product's probe and the boundaries it cuts with, so
    both are the same values in a different shape -- a boundary that is not a
    row of the operand it cuts is the mistake this makes unlikely.
    """

    import tensorplay as tp

    element = getattr(tp, dtype) if isinstance(dtype, str) else dtype
    flat = tp.arange(int(rows) * int(cols), device=device).to(element)
    return flat * 3.17e-4 - 0.5

def grouped_extents(shapes: tuple, scaled: bool):
    """What a grouped product's operands say about the groups, or ``None``.

    Two things decide it and neither is a choice.  Each operand is either a
    matrix per group (three dimensions) or one matrix the groups are cut out of
    (two, with a vector of boundaries), and which extent a boundary indexes
    follows from that: a boundary over the left operand's rows is a boundary
    over the rows, over the right one's is over the columns, and with both cut
    out it is over the contraction.  The result is batched exactly when the two
    operands agree, because a boundary then indexes the result and not one of
    its extents.

    The group's own extents are what a tile is fitted to, and they are the
    operand's extent divided by the count whenever the extent is the one that
    varies -- the operands are as long as all the groups together in that case,
    and a tile sized for the whole of them would be sized for a problem that
    does not exist.
    """

    sizes = [tuple(int(v) for v in shape) for shape in shapes]
    if len(sizes) < 2 or any(len(shape) not in (2, 3) for shape in sizes[:2]):
        return None
    rest = sizes[2:]
    first, second = sizes[0], sizes[1]
    if len(first) not in (2, 3) or len(second) not in (2, 3):
        return None
    a_is_2d, b_is_2d = len(first) == 2, len(second) == 2
    if scaled:
        # A factor is one value per row, so a matrix that is per group has its
        # factors per group as well -- the group is an axis of the factor, not
        # something the body reconstructs from an index.
        if len(rest) < 2:
            return None
        if len(rest[0]) != (1 if a_is_2d else 2):
            return None
        if len(rest[1]) != (1 if b_is_2d else 2):
            return None
        rest = rest[2:]
    m_total, k_a = first[-2], first[-1]
    k_b, n_total = second[-2], second[-1]
    if k_a != k_b:
        return None
    m_is_varying = a_is_2d and not b_is_2d
    n_is_varying = b_is_2d and not a_is_2d
    k_is_varying = a_is_2d and b_is_2d
    if a_is_2d or b_is_2d:
        # One operand is one matrix the groups are cut out of, so the
        # boundaries are what says how many groups there are -- and the operand
        # that is per group has to agree, because a run of g groups reading a
        # batch of something else is a call nobody described.
        if len(rest) != 1 or len(rest[0]) != 1:
            return None
        groups = int(rest[0][0])
        if not a_is_2d and int(first[0]) != groups:
            return None
        if not b_is_2d and int(second[0]) != groups:
            return None
    else:
        if rest or first[0] != second[0]:
            return None
        groups = int(first[0])
    if groups <= 0:
        return None
    if k_is_varying and k_a % groups:
        return None
    if m_is_varying and m_total % groups:
        return None
    if n_is_varying and n_total % groups:
        return None
    return {
        "A_IS_2D": a_is_2d,
        "B_IS_2D": b_is_2d,
        "M_IS_VARYING": m_is_varying,
        "N_IS_VARYING": n_is_varying,
        "K_IS_VARYING": k_is_varying,
        "groups": groups,
        "m_total": m_total,
        "n_total": n_total,
        "k_total": k_a,
        "m": m_total // groups if m_is_varying else m_total,
        "n": n_total // groups if n_is_varying else n_total,
        "k": k_a // groups if k_is_varying else k_a,
        "shapes": (first, second),
    }

class GroupedMmConfigHeuristics(TemplateConfigHeuristics):

    """The candidates for a number of products of the same extents.

    Which extent the groups vary along is not a choice and not something the
    product table can be told: it follows from whether each operand is one matrix
    per group or one matrix the groups are cut out of, and the four combinations
    are four different problems rather than four tilings of one.  So the extents
    are read off the operands' own ranks, and the tile is fitted to one group's
    extents -- which is what a tile is ever fitted to, whether there is one
    group or a thousand.

    The count of groups is not part of the fit: the groups are walked by a fixed
    number of programs, so a run with more groups is longer rather than
    differently shaped.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        return grouped_extents(inputs.shapes, bool(inputs.extra.get("has_scales"))) is not None

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        extents = grouped_extents(
            kernel_inputs.shapes, bool(kernel_inputs.extra.get("has_scales"))
        )
        if extents is None:
            return
        scaled = bool(kernel_inputs.extra.get("has_scales"))
        for config in CHOICES.get_mm_configs(self.device_type)(
            extents["m"], extents["n"], extents["k"],
            dtype_size=dtype_size(kernel_inputs.dtype(0)),
        ):
            yield {
                "choice": "triton",
                "USE_FAST_ACCUM": scaled,
                "SCALED": scaled,
                **config.as_kwargs(),
            }

    def get_extra_kwargs(self, kernel_inputs, op_name):
        from ..codegen.triton_gemm import _matmul_allow_tf32

        return {"allow_tf32": _matmul_allow_tf32()}

class GroupedMmTemplate(TritonTemplate):

    """A number of products of the same extents, walked by one set of programs.

    Every group is the same shape, so one table of tiles fits all of them and
    the programs are as many as the device has rather than as many as there are
    tiles.  That is the whole difference from running the groups one after
    another, and it is why the count of programs is a number the kernel is
    given: a program has to know its own number to know which tiles are its own.

    An operand is either a matrix per group or one matrix the groups are cut
    out of, and the four combinations are decided by the operands' ranks rather
    than configured -- which extent a group boundary indexes follows from which
    operand the boundary belongs to, so a configuration that chose it would be
    claiming a fact about the call.
    """

    inputs_class = MMKernelInputs

    #: The block extents the body reads, in the order the signature declares
    #: them after the operands' extents and strides.
    block_names = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "NUM_SMS")

    def __init__(self, name: str = "grouped_mm", **kwargs: Any):
        super().__init__(name, **kwargs)
        self.heuristics = GroupedMmConfigHeuristics()

    def extents(self, meta: dict):
        """What the groups are, read off the operands the call arrived with."""

        return grouped_extents(
            tuple(meta.get("operand_sizes") or ()),
            bool(meta.get("has_scales")),
        )

    def layouts(self, meta: dict):
        """The two matrices' extents and strides, or ``None`` if not yet known.

        Which way a matrix is laid out decides how a descriptor built for it has
        to be shaped, and that is a property of the operand rather than of the
        configuration, so it is read here rather than offered as a choice.  The
        strides are the call's own where it stated them and the probe's
        otherwise; either way the launcher checks them again at launch, because a
        body built for one layout and handed another is the one mistake a stride
        cannot make quietly.
        """

        sizes = tuple(tuple(int(v) for v in shape)
                      for shape in (meta.get("operand_sizes") or ()))
        if len(sizes) != 2:
            return None
        stated = tuple(tuple(int(v) for v in stride)
                       for stride in (meta.get("operand_strides") or ()))
        if len(stated) == 2 and all(
            len(one) == len(shape) for one, shape in zip(stated, sizes)
        ):
            return sizes, stated
        feed = tuple(meta.get("probe_feed") or meta.get("feed") or ())
        if len(feed) < 2:
            return None
        try:
            return sizes, tuple(
                tuple(int(v) for v in tensor.stride()) for tensor in feed[:2]
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def descriptor_flags(self, meta: dict) -> dict:
        """How this call's tiles are fetched: by descriptor where it can be.

        The form is a property of the device, the toolkit and the operand, none
        of which the configuration chooses, so it is settled before the
        candidates are enumerated and every configuration of this call shares
        it.  What is left for the configuration is the tile.
        """

        layouts = self.layouts(meta)
        if layouts is None:
            return {
                "USE_TMA_LOAD": False,
                "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": False,
                "A_IS_K_MAJOR": True,
                "B_IS_K_MAJOR": True,
            }
        form = descriptor_form(*layouts)
        if form is None:
            return {
                "USE_TMA_LOAD": False,
                "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": False,
                "A_IS_K_MAJOR": True,
                "B_IS_K_MAJOR": True,
            }
        return form

    def index_width(self, meta: dict) -> str:
        """How wide an index this call can be addressed with.

        A 32-bit index is the cheaper one and is what a body narrows a
        coordinate to when it hands a descriptor an offset, so it is used
        wherever every operand and the result can be named in 32 bits -- which
        is the same question the descriptor form asks of the same operands.
        """

        extents = self.extents(meta)
        layouts = self.layouts(meta)
        if extents is None or layouts is None:
            return "tl.int64"
        sizes, strides = layouts
        everything = list(sizes) + [tuple(int(v) for v in meta.get("out_size") or ())]
        if not all(descriptor_extents_fit(size) for size in everything):
            return "tl.int64"
        if not all(
            descriptor_offset_fits(size, stride)
            for size, stride in zip(sizes, strides)
        ):
            return "tl.int64"
        return "tl.int32"

    def out_specs(self, meta: dict) -> tuple:
        """Where the result lands, which is batched exactly when both operands are.

        A group boundary indexes the result itself when the groups are cut out
        of both operands, and there is no such axis when one of them is per
        group -- so the rank is a fact about the call, not a choice about the
        layout.  When the groups are cut along the contraction, every group has
        the same rows and columns, so the result is the batch of those.
        """

        extents = self.extents(meta)
        if extents is None:
            raise NotImplementedError("a call with no groups to walk")
        if extents["A_IS_2D"] != extents["B_IS_2D"]:
            size = (extents["m_total"], extents["n_total"])
        else:
            size = (extents["groups"], extents["m"], extents["n"])
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                size,
                contiguous_stride(size),
            ),
        )

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates.

        Every group gets its own ramp, so a kernel that read one group's operand
        for all of them measures wrong here rather than in a number somebody
        trusted.  Where the groups are cut out of one matrix, the boundaries cut
        the ramp itself, so a kernel that read past a boundary is caught by the
        same feed rather than by a separate check.
        """

        extents = self.extents(meta)
        if extents is None or meta.get("out_dtype") is None:
            return None
        device = meta.get("device")
        dtype = meta["out_dtype"]
        feed = self._probe_operands(extents, dtype, device)
        if meta.get("has_scales"):
            feed.append(self._probe_factor(extents["A_IS_2D"], extents["m_total"],
                                           extents["groups"], extents["m"],
                                           dtype, device))
            feed.append(self._probe_factor(extents["B_IS_2D"], extents["n_total"],
                                           extents["groups"], extents["n"],
                                           dtype, device))
        if extents["A_IS_2D"] or extents["B_IS_2D"]:
            feed.append(self._probe_boundaries(extents, device))
        return feed

    def _probe_operands(self, extents: dict, dtype, device) -> list:
        """The two matrices, each as long as it really is.

        The ramp is built at the operand's own length and reshaped to its rank,
        so a matrix per group and one matrix holding all of them are the same
        numbers in a different shape -- and a kernel that indexed one as the
        other would still read inside the buffer, which is why the boundaries
        matter as much as the ramp.
        """

        a_is_2d = extents["A_IS_2D"]
        b_is_2d = extents["B_IS_2D"]
        rows, inner, cols = extents["m_total"], extents["k_total"], extents["n_total"]
        if a_is_2d:
            first = _ramp(rows, inner, dtype, device).reshape(rows, inner)
        else:
            first = _ramp(extents["groups"] * rows, inner, dtype, device).reshape(
                extents["groups"], rows, inner
            )
        if b_is_2d:
            second = _ramp(inner, cols, dtype, device).reshape(inner, cols)
        else:
            second = _ramp(extents["groups"] * inner, cols, dtype, device).reshape(
                extents["groups"], inner, cols
            )
        return [first, second]

    def _probe_factor(self, is_2d, total, groups, per_group, dtype, device):
        """One factor per row: over the whole matrix, or per group of it.

        A factor is a value per row, so its extents follow the matrix it scales
        -- all the rows of the one long matrix, or one row's worth for each of
        the groups.  A ramp is not a factor: a factor near one keeps the product
        comparable to the same product unscaled, which is what makes a
        measurement of the two comparable.
        """

        shape = (total,) if is_2d else (groups, per_group)
        flat = _ramp(1, int(total) if is_2d else groups * int(per_group), dtype, device)
        return (flat.reshape(shape) * 0.25 + 0.875).to(dtype)

    def _probe_boundaries(self, extents: dict, device):
        """Group boundaries that cut one matrix into equal parts.

        Every boundary but the last falls on a whole tile's worth of rows, so
        the measurement covers a group whose extents do not divide the tile as
        well as one whose do; the last one is the end of the matrix, because a
        boundary that stopped short would leave the tail ungrouped and a kernel
        that trusted the vector would read past the end.
        """

        import tensorplay as tp

        groups = extents["groups"]
        if extents["K_IS_VARYING"]:
            total = extents["k_total"]
        elif extents["M_IS_VARYING"]:
            total = extents["m_total"]
        else:
            total = extents["n_total"]
        if groups <= 1:
            return _ramp(1, 1, tp.int32, device).to(tp.int32).reshape(1) * total
        step = -(-int(total) // groups)
        ends = [min(step * (index + 1), int(total)) for index in range(groups)]
        return tp.tensor(ends, dtype=tp.int32, device=device)

    def generate(self, params, out_specs, meta, plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        kwargs = params.to_kwargs()
        layout = out_specs[0] if out_specs else None
        if kwargs.get("choice") == "operator":
            if plain_launch is None:
                return None
            return call_operation(
                "framework_grouped_product", plain_launch, layout
            )
        if layout is None or plain_launch is None:
            return None
        if meta.get("operand_dtype") != "float32":
            return None
        if not meta.get("qualifies", False):
            return None
        extents = self.extents(meta)
        specs = tuple(meta.get("operand_specs") or ())
        if extents is None or len(specs) < 2:
            return None
        scaled = bool(meta.get("has_scales"))
        inputs = self.operand_names(extents, scaled)
        if len(specs) < len(inputs):
            return None
        block = tuple((name, kwargs.get(name)) for name in self.block_names)
        if any(value is None for _name, value in block):
            return None
        from ..codegen.triton_gemm import grouped_matmul_launch

        fetch = self.descriptor_flags(meta)
        caller = TritonChoiceCaller(
            name=f"{self.name}-{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}"
            f"x{kwargs['BLOCK_K']}",
            layout=layout,
            description=repr(sorted(kwargs.items())),
            source=self.source,
            src_hash=self.src_hash,
        )
        return caller.bind(
            grouped_matmul_launch(
                plain_launch,
                specs[:len(inputs)],
                layout.size,
                self.kernel_for(block, extents, layout, inputs, scaled,
                                kwargs, fetch, self.index_width(meta)),
                self.grid,
                block,
                fetch,
                num_warps=int(kwargs.get("num_warps", 4)),
                num_stages=int(kwargs.get("num_stages", 3)),
            )
        )

    def operand_names(self, extents: dict, scaled: bool) -> tuple:
        """The operands this rendering is written against, in declaration order.

        The order is the signature's: the two matrices, then the factors if the
        call has them, then the boundaries if the groups are cut out of one of
        the matrices.  The launch reads its arguments off this list, so a form
        that takes more says so here and nowhere else.
        """

        names = ["A", "B"]
        if scaled:
            names.extend(("scale_a", "scale_b"))
        if extents["A_IS_2D"] or extents["B_IS_2D"]:
            names.append("offsets")
        return tuple(names)

    def kernel_for(self, block: tuple, extents: dict, layout, inputs, scaled,
                   kwargs, fetch: dict, index_dtype: str):
        """The body, rendered for one grouping, one fetch form and one tile.

        How a tile is fetched and how wide an index it is fetched with are both
        properties of the call rather than of the tile, so they are decided
        before this and handed in; what the configuration chose is the tile, and
        only the tile.
        """

        shapes = {
            "A": extents["shapes"][0],
            "B": extents["shapes"][1],
        }
        if scaled:
            # A matrix cut out of one long matrix is scaled by the row it is,
            # and a matrix per group is scaled by that group's own rows.
            shapes["scale_a"] = (
                (extents["m_total"],) if extents["A_IS_2D"]
                else (extents["groups"], extents["m"])
            )
            shapes["scale_b"] = (
                (extents["n_total"],) if extents["B_IS_2D"]
                else (extents["groups"], extents["n"])
            )
        if "offsets" in inputs:
            shapes["offsets"] = (extents["groups"],)
        operands = {
            name: {"shape": shapes[name], "stride": contiguous_stride(shapes[name])}
            for name in inputs
        }
        out_size = tuple(int(v) for v in layout.size)
        args = KernelArgs(
            operands,
            {"C": {"shape": out_size, "stride": contiguous_stride(out_size)}},
            dict(block),
        )
        args.index_dtype = index_dtype
        source = self.render_with(
            args,
            operands=list(inputs),
            A_IS_2D=extents["A_IS_2D"],
            B_IS_2D=extents["B_IS_2D"],
            SCALED=scaled,
            USE_FAST_ACCUM=bool(kwargs.get("USE_FAST_ACCUM", False)),
            # The multiply's precision is decided by the rendering rather than
            # by an argument, because a shortened multiplier is a different
            # kernel rather than a different value of one of its parameters.
            precision="tf32" if kwargs.get("allow_tf32", False) else "ieee",
            **fetch,
        )
        return self.kernel_type.get(
            self.name, source, self.symbol, self.grid, {"block": dict(block)}
        ).build()

GROUPED_MM = GroupedMmTemplate.from_file(
    "grouped_mm",
    file="triton_mm_grouped",
    grid=grouped_mm_grid,
    symbol="_grouped_mm_kernel",
)

#: The candidates this module offers.
GROUPED_MM_TEMPLATES = (
    GROUPED_MM,
)


def has_grouped_mm_triton_support() -> bool:
    """Whether this machine can read a tile of a grouped product by descriptor.

    The descriptor is a feature of the memory system rather than a preference,
    and the machine that has it is the machine whose capabilities say so.  A
    toolkit can emit a kernel for a machine that cannot run it, so the question
    is asked of the device.
    """

    import tensorplay as tp

    from .mm_common import device_capability

    try:
        if not tp.cuda.is_available():
            return False
    except Exception:  # noqa: BLE001 - a machine that will not say has no descriptors
        return False
    return device_capability() >= (9, 0)


def has_rocm_fp8_hardware_support() -> bool:
    """Whether this machine's other dialect has the narrow float types.

    The answer is about a dialect this tree does not have, so it is asked only
    where that dialect could be: a machine that is not running it cannot be
    missing its types in any way that matters here.
    """

    return False


def _rocm_gcn_arch() -> str:
    """This machine's other dialect's architecture name, or nothing."""

    return ""


def has_scaled_grouped_mm_triton_support(mat_a, mat_b) -> bool:
    """Whether a grouped product's factors can be carried by a descriptor read.

    A factor narrows an operand to a narrower type, and the read that carries it
    is a read of that narrower type -- so the question is whether the machine's
    descriptors can name it, which for this tree's machine they can.
    """

    if has_rocm_fp8_hardware_support():
        return False
    return True


def can_use_triton_kernel(mat_a, mat_b, offs, bias, scale_result) -> bool:
    """Whether a grouped call is one this body can be the answer to.

    A body reads its tiles by descriptor and writes what it computed, so a call
    that also wants something added to the result, or a factor of its own, is not
    one it can serve: offering it anyway would have the measurement compare an
    answer that is missing part of the request.
    """

    if not has_grouped_mm_triton_support():
        return False
    if bias is not None or scale_result is not None:
        return False
    a_2d = len(mat_a.get_size()) == 2
    b_2d = len(mat_b.get_size()) == 2
    if a_2d or b_2d:
        # one operand is one matrix the groups are cut out of, so the groups have
        # to be said where the cuts fall
        return offs is not None
    return offs is None


def create_offsets(offs_box, m1_is_2d, m2_is_2d, m, n, k, alignment: int = 1):
    """The group boundaries a two-dimensional grouping is cut at.

    A group is a run of the rows, the columns or the contraction, and where one
    ends and the next begins is a choice -- so it is spread evenly across the
    axis, with the interior boundaries rounded to whatever the read wants and the
    last one left at the end so the groups cover the axis exactly.  A grouping
    with no axis to cut -- two matrices given per group -- has no boundaries, and
    says so by returning nothing.
    """

    if m1_is_2d:
        end = k if m2_is_2d else m
    elif m2_is_2d:
        end = n
    else:
        return None
    try:
        groups = int(offs_box.get_size()[0])
    except Exception:  # noqa: BLE001 - boundaries that will not say are not made
        return None
    if groups <= 0:
        return None
    end = int(end)
    step = end // groups
    ends = [step * (index + 1) for index in range(groups - 1)]
    ends = [value - value % int(alignment) for value in ends]
    ends.append(end)
    return tuple(ends)


def grouped_mm_args(mat1, mat2, offs=None, layout=None, out_dtype=None):
    """What a grouped call's operands say, and the boundaries they are cut at."""

    from .mm_common import mm_args

    _m, n, k, out_layout = mm_args(mat1, mat2, layout=layout)
    sizes = (
        tuple(int(v) for v in mat1.get_size()),
        tuple(int(v) for v in mat2.get_size()),
    )
    return [_m, n, k, out_layout, mat1, mat2, offs], sizes


@dataclasses.dataclass
class Config:
    kwargs: dict[str, int]
    num_stages: int
    num_warps: int


_NV_CONFIGS = [
    Config(
        {
            "BLOCK_M": block_size_m,
            "BLOCK_N": block_size_n,
            "BLOCK_K": block_size_k,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [16, 32, 64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
]


def grouped_mm_configs():
    return _NV_CONFIGS


cutedsl_grouped_mm_template = CuteDSLTemplate(
    name="grouped_gemm_cutedsl",
    source=load_kernel_template("cutedsl_mm_grouped"),
)


#: A grouped product read through pointers, as the fused tile reads it.
triton_grouped_mm_template = GROUPED_MM

#: The same, with the scale applied while the product is accumulated.
triton_scaled_grouped_mm_template = GROUPED_MM


def early_config_prune(g, m, dtsize, configs, named_args):
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], config.num_stages)
        if not has_free_symbols((g, m)):
            a_is_2d, b_is_2d = (named_args['A_IS_2D'], named_args['B_IS_2D'])
            m_avg = m // g if a_is_2d and (not b_is_2d) else m
            if m_avg <= 16:
                if BLOCK_M > 32:
                    continue
            elif m_avg <= 32:
                if BLOCK_M > 64:
                    continue
            elif m_avg <= 64:
                if BLOCK_M <= 16:
                    continue
            elif BLOCK_M <= 32:
                continue
        max_shared_memory = get_gpu_shared_memory()
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue
        pruned_configs.append(config)
    return pruned_configs


def _tuned_grouped_mm_common(operator_name: str, algorithm_name: str, extern_kernel_choice: ExternKernelChoice, kernel_template: TritonTemplate, mat_a: TensorBox, mat_b: TensorBox, scale_a: TensorBox | None=None, scale_b: TensorBox | None=None, offs: TensorBox | None=None, bias: TensorBox | None=None, scale_result: TensorBox | None=None, out_dtype: tp.dtype | None=None, use_fast_accum: bool | None=None, layout: Layout | None=None) -> TensorBox:
    if (scale_a is None) != (scale_b is None):
        raise AssertionError('scale_a and scale_b must both be None or both be provided')
    if scale_result is not None and scale_a is None:
        raise AssertionError('scale_result requires scale_a and scale_b')
    m1_size, m2_size, layout, mat_a, mat_b, offs = grouped_mm_args(mat_a, mat_b, offs, layout=layout, out_dtype=out_dtype)
    counters['aten_mm_info'][operator_name] += 1
    log_message = f'Tuned {operator_name}: mat1_shape=%s, mat2_shape=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s'
    log.info(log_message, m1_size, m2_size, mat_a.get_dtype(), mat_b.get_dtype(), layout)
    if scale_a is not None and scale_b is not None:
        check_supported_striding(mat_a, mat_b)
    input_nodes: list[Any] = [mat_a, mat_b]
    if scale_a is not None:
        input_nodes.append(realize_inputs(scale_a))
    if scale_b is not None:
        input_nodes.append(realize_inputs(scale_b))
    if offs is not None:
        input_nodes.append(realize_inputs(offs))
    if use_fast_accum is None:
        aten_choice = extern_kernel_choice.bind(input_nodes, layout, out_dtype=out_dtype)
    else:
        aten_choice = extern_kernel_choice.bind(input_nodes, layout, out_dtype=out_dtype, use_fast_accum=use_fast_accum)
    if use_fast_accum is None:
        use_fast_accum = False
    choices: list[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)
    _, is_nonzero = _is_static_problem(layout)
    if len(m1_size) == 2:
        if len(m2_size) == 2:
            m, k1 = m1_size
            k2, n = m2_size
            g = offs.get_size()[0]
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = (True, True)
        else:
            g1 = offs.layout.size[0]
            m, k1 = m1_size
            g2, k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = (True, False)
    elif len(m2_size) == 2:
        g1 = offs.layout.size[0]
        g2, m, k1 = m1_size
        k2, n = m2_size
        g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
        k = V.graph.sizevars.check_equals(k1, k2)
        a_is_2d, b_is_2d = (False, True)
    else:
        g1, m, k1 = m1_size
        g2, k2, n = m2_size
        g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
        k = V.graph.sizevars.check_equals(k1, k2)
        a_is_2d, b_is_2d = (False, False)
    scaled = scale_a is not None
    if is_nonzero and use_triton_template(layout) and can_use_triton_kernel(mat_a, mat_b, offs, bias, scale_result) and (not scaled or has_scaled_grouped_mm_triton_support(mat_a, mat_b)):
        a_is_k_major = mat_a.get_stride()[-1] == 1
        b_is_k_major = mat_b.get_stride()[-2] == 1
        triton_has_make_tensor_descriptor = hasattr(tl, 'make_tensor_descriptor')
        triton_has_experimental_make_tensor_descriptor = hasattr(tl, '_experimental_make_tensor_descriptor')
        use_tma_load = (triton_has_make_tensor_descriptor or triton_has_experimental_make_tensor_descriptor) and _descriptor_shape_fits_in_int32(mat_a.get_size(), add_guards=True) and _descriptor_shape_fits_in_int32(mat_b.get_size(), add_guards=True) and _tma_descriptor_max_offset_fits_in_int32(mat_a, add_guards=True) and _tma_descriptor_max_offset_fits_in_int32(mat_b, add_guards=True)
        kwargs = {'SCALED': scaled, 'A_IS_2D': a_is_2d, 'B_IS_2D': b_is_2d, 'A_IS_K_MAJOR': a_is_k_major, 'B_IS_K_MAJOR': b_is_k_major, 'USE_FAST_ACCUM': use_fast_accum, 'NUM_SMS': get_num_sms(), 'USE_TMA_LOAD': use_tma_load, 'USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR': triton_has_experimental_make_tensor_descriptor}
        for config in early_config_prune(g, m, mat_a.dtype.itemsize, grouped_mm_configs(), kwargs):
            kernel_template.maybe_append_choice(choices, input_nodes=input_nodes, layout=layout, num_stages=config.num_stages, num_warps=config.num_warps, **kwargs, **config.kwargs)
    if use_blackwell_cutedsl_grouped_mm(mat_a, mat_b, layout, a_is_2d, b_is_2d, offs, bias, scale_result):
        for config in get_groupgemm_configs():
            kwargs = dict(ACC_DTYPE='cutlass.Float32')
            cutedsl_grouped_mm_template.maybe_append_choice(choices, input_nodes=input_nodes, layout=layout, **kwargs, **asdict(config))
    input_gen_fns = {}
    if offs is not None:
        input_offs_idx = 2 if scale_a is None else 4
        alignment = 16 // mat_a.dtype.itemsize
        input_gen_fns[input_offs_idx] = lambda x: create_offsets(x, a_is_2d, b_is_2d, m, n, k, alignment)
    node, _ = autotune_select_algorithm(algorithm_name, choices, input_nodes, layout, input_gen_fns=input_gen_fns)
    return node


@register_lowering(framework._grouped_mm, type_promotion_kind=None)
def tuned_grouped_mm(mat_a: TensorBox, mat_b: TensorBox, offs: TensorBox | None=None, bias: TensorBox | None=None, out_dtype: tp.dtype | None=None, layout: Layout | None=None) -> TensorBox:
    """Auto-tuning for _grouped_mm() operator."""
    return _tuned_grouped_mm_common('framework._grouped_mm.default', 'grouped_mm', framework__grouped_mm, triton_grouped_mm_template, mat_a, mat_b, None, None, offs, bias, None, out_dtype, None, layout)


@register_lowering(framework._scaled_grouped_mm, type_promotion_kind=None)
def tuned_scaled_grouped_mm(mat_a: TensorBox, mat_b: TensorBox, scale_a: TensorBox, scale_b: TensorBox, offs: TensorBox | None=None, bias: TensorBox | None=None, scale_result: TensorBox | None=None, out_dtype: tp.dtype | None=None, use_fast_accum: bool=False, layout: Layout | None=None) -> TensorBox:
    """Auto-tuning for _scaled_grouped_mm() operator."""
    out_dtype = out_dtype or tp.bfloat16
    return _tuned_grouped_mm_common('framework._scaled_grouped_mm.default', 'scaled_grouped_mm', framework__scaled_grouped_mm, triton_scaled_grouped_mm_template, mat_a, mat_b, scale_a, scale_b, offs, bias, scale_result, out_dtype, use_fast_accum, layout)
