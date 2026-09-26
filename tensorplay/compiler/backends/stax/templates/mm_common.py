"""What every product here asks the machine, and asks the call.

A device's capability, how many multiprocessors it has, and whether an operand is
addressable in the 32 bits a descriptor addresses in: none of these is a property
of a product, and a template that asked the question itself would ask it once per
template rather than once.

The other half is the call.  A product is a matrix multiply, a batch of them, a
sum of two, a product with a bias, and they differ in what they are given rather
than in what they do -- so what a call *is* is read off its operands here, once,
and every product module reads the same answer.  A product module that read its
own operands would be a second reading of the same question, and the two would
drift.
"""

from __future__ import annotations
import logging

#: Where this module's messages go.  A measurement that is
#: discarded is worth a line and a discarded one is worth none, so
#: the messages are here rather than printed.
log = logging.getLogger(__name__)

import functools
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import sympy

import tensorplay as tp

from .. import config
from ..ir import Layout
from ..loops import V

__all__ = [
    "acc_type",
    "addmm_epilogue",
    "check_supported_striding",
    "descriptor_extents_fit",
    "descriptor_form",
    "descriptor_offset_fits",
    "device_capability",
    "is_batch_stride_largest_or_zero",
    "load_kernel_template",
    "mm_args",
    "is_gpu",
    "mm_grid",
    "num_sms",
    "persistent_grouped_mm_grid",
    "persistent_mm_grid",
    "scale_mm_epilogue",
    "use_aten_gemm_kernels",
    "use_decompose_k_choice",
    "use_native_matmul",
    "use_template_for_gpu",
    "use_triton_blackwell_tma_template",
    "use_triton_template",
    "use_triton_tma_template",
]

#: Where a kernel body lives when it is a file rather than a string.
_KERNEL_TEMPLATE_DIR = Path(__file__).parent.parent / "codegen" / "templates"


def load_kernel_template(name: str) -> str:
    """A kernel body's text, read from the file it is written in.

    A body is long enough that keeping it inside a string makes the code and the
    parts that vary hard to tell apart, so it is a file and this is how it is
    read.  The name is a file's own name: a body is named for what it computes,
    and a template answers to what the operation is called.
    """

    path = _KERNEL_TEMPLATE_DIR / (name if name.endswith(".py.jinja")
                                  else f"{name}.py.jinja")
    return path.read_text(encoding="utf-8")


#: The operations a product may reduce, and how each one's value is finished.
#:
#: A name here is a name a body carries; one that is not here is a walk this
#: table has no opinion about, which is not the same as a walk it refuses.
FUNCTION_REDUCTION_TYPES = ("sum", "prod", "max", "amin", "amax", "any", "argmax",
                            "argmin")

#: The walks whose closed form is not a plain reduction of the values.
FUNCTION_UNSUPPORTED_REDUCTIONS = ("scan", "sort", "kthvalue", "median")


def num_sms(device=None) -> int:
    """How many multiprocessors the device has, which is the cap on a grid."""

    import tensorplay as tp

    try:
        index = 0 if device is None else int(device.index)
        return int(tp.cuda.get_device_properties(index).multi_processor_count)
    except Exception:  # noqa: BLE001 - a device that will not say is assumed small
        return 1


def device_capability(device=None) -> tuple:
    """What the device can do, as a major/minor pair.

    A template that needs a feature the hardware lacks has to be able to ask,
    and the question is asked of the device rather than assumed from the
    toolkit: a toolkit can emit a kernel for a machine that cannot run it.
    """

    import tensorplay as tp

    try:
        index = 0 if device is None else int(device.index)
        props = tp.cuda.get_device_properties(index)
        return int(props.major), int(props.minor)
    except Exception:  # noqa: BLE001 - a device that will not say cannot be relied on
        return 0, 0


def _is_static_problem(layout) -> tuple:
    """Whether a problem's extents are all numbers, and whether its strides are.

    A product whose extents are all known can be measured once and the answer
    kept; one whose extents are not has to be measured per shape, because a tile
    fitted to one number is not a tile fitted to another.  The strides are asked
    separately because a layout may be settled -- and so no longer changeable --
    while the extents are still symbols.
    """

    if layout is None:
        return False, False
    try:
        static_size = all(
            not hasattr(int(v), "free_symbols") or not int(v).free_symbols
            for v in layout.get_size()
        )
    except Exception:  # noqa: BLE001 - a layout that will not say is not static
        return False, False
    try:
        static_stride = all(
            not hasattr(int(v), "free_symbols") or not int(v).free_symbols
            for v in layout.get_size()
        )
    except Exception:  # noqa: BLE001
        static_stride = False
    return bool(static_size), bool(static_stride)


def mm_grid(m, n, meta, *, cdiv):
    """The grid for a product's tiles: one program per tile, flattened.

    The tiles are numbered rather than addressed in two dimensions, because a
    program that is handed its tile's row and column has to divide the flat
    index to recover them, while a program handed them directly is handed a
    tuple.  Flattening costs the kernel one division and saves the launcher a
    dimension, and the dimension is the part that has to agree with the source.
    """

    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


def persistent_mm_grid(M: int, N: int, meta: dict, *, cdiv, min):
    """The grid for the swept product: as many programs as the device has.

    A grid larger than the machine is a launch whose programs wait, so the count
    of programs is the machine's count and the work is divided between them.  The
    division is the kernel's to make; what the launch owes it is the cap.
    """

    num_sms_ = int(meta.get("NUM_SMS", 1) or 1)
    tile_m = int(meta["BLOCK_M"])
    tile_n = int(meta["BLOCK_N"])
    tiles = cdiv(M, tile_m) * cdiv(N, tile_n)
    return (min(num_sms_, tiles), 1, 1)


def persistent_grouped_mm_grid(*args):
    """The grid for a grouped product's swept form: the machine's, again.

    Every group is the same shape, so the whole run is one flat sequence of
    tiles and the cap on the programs is the same cap.
    """

    from .mm_grouped import grouped_mm_grid

    return grouped_mm_grid(*args)


def acc_type(dtype):
    """The type a product's accumulator is held in, for an operand's type.

    A product accumulates in at least the width of its operands' product, so an
    operand narrower than a float accumulates in a float: the multiply is done in
    the narrow type and the accumulation is not, which is the only place the
    difference shows.
    """

    import tensorplay as tp
    import triton.language as tl

    name = str(dtype)
    if name in ("float16", "bfloat16"):
        return tl.float32
    if "float8" in name:
        return tl.float32
    return getattr(tl, name, tl.float32)


def _extent_of(value, default: int = 0) -> int:
    """One extent as a number, or a default when it is not one yet."""

    try:
        return int(value)
    except Exception:  # noqa: BLE001 - a symbolic extent has no number
        return int(default)


def mm_args(
    mat1,
    mat2,
    *others,
    layout=None,
    out_dtype=None,
    use_4x2_dim: bool = False,
    mat2_transposed: bool = False,
):
    """What a product's two operands say, read the one way.

    The answer every product module wants is the same: the extents, the
    contraction both sides agree on, and the layout the result will have.  A
    product given a transposed right operand reads it as it is stored rather than
    as it is meant, which is the difference between a transposed read and a
    transposed copy -- so the flag is here rather than at each call site.
    """

    m = k1 = k2 = n = 0
    left = list(mat1.get_size()) if hasattr(mat1, "get_size") else []
    right = list(mat2.get_size()) if hasattr(mat2, "get_size") else []
    if len(left) >= 2:
        m, k1 = _extent_of(left[-2]), _extent_of(left[-1])
    if mat2_transposed:
        if len(right) >= 2:
            n, k2 = _extent_of(right[-2]), _extent_of(right[-1])
    else:
        if len(right) >= 2:
            k2, n = _extent_of(right[-2]), _extent_of(right[-1])
    if use_4x2_dim:
        k2 = k2 * 2
    k = k1 if k1 == k2 else min(k1, k2)
    if layout is None:
        from ..ir import Layout, contiguous_strides
        from ..loops import V

        device = mat1.get_device() if hasattr(mat1, "get_device") else None
        dtype = out_dtype or (
            mat1.get_dtype() if hasattr(mat1, "get_dtype") else None
        )
        size = (m, n)
        layout = Layout(device, dtype, size, contiguous_strides(size))
        if V.graph is not None:
            V.graph.sizevars.check_equals_and_simplify(k1, k2)
    elif out_dtype is not None:
        raise AssertionError("out_dtype is ignored if layout is specified.")
    return [m, n, k, layout, mat1, mat2, *others]


def addmm_epilogue(dtype, alpha, beta):
    """The value a product with a bias stores, and the name it is stored under.

    A bias added to a product is not a separate kernel: the value is in the
    accumulator's registers already, so the addition is part of the store the
    product was going to do anyway.  Which value it is depends on the bias and
    on the two scalars, so it is written once here rather than spelled at each
    template that has a bias.
    """

    def epilogue(acc, bias):
        result = acc * alpha + bias * beta
        if dtype is not None:
            result = result.to(dtype)
        return result

    return epilogue


def scale_mm_epilogue():
    """The epilogue a scaled product stores: the accumulator, already scaled.

    A scaled product has already multiplied its factors in by the time the
    accumulator is finished, so there is nothing left to do at the store and the
    epilogue is the identity.  It is named rather than left absent so that a
    caller asking "what does this product do at its store" gets an answer for
    every product rather than for most of them.
    """

    def epilogue(acc):
        return acc

    return epilogue


def use_native_matmul(mat1, mat2) -> bool:
    """Whether this call is better left to the framework's own product.

    Some shapes are better served by whatever the framework already ships, and
    offering a template for them would have the measurement compare two answers
    and report the faster one -- which is the right answer to a different
    question.  So a call this says no to is not offered at all.
    """

    try:
        left = list(mat1.get_size())
        right = list(mat2.get_size())
    except Exception:  # noqa: BLE001 - a call that will not say is left alone
        return False
    if len(left) < 2 or len(right) < 2:
        return False
    m, k = _extent_of(left[-2]), _extent_of(left[-1])
    k2, n = _extent_of(right[-2]), _extent_of(right[-1])
    if k != k2:
        return False
    # A very small product is dominated by the launch rather than by the
    # arithmetic, and the framework's own does not pay for a grid either.
    return m * n * k <= 0


def _use_small_mm_pointwise(
    m,
    k,
    n,
    device_type: str = "cuda",
    statically_known_true=None,
) -> bool:
    """Whether a product is small enough to be worth a pointwise path.

    Below a size, what a product spends is what any kernel spends: the launch
    and the write.  A product that small has more to gain from a cheap body than
    from a wide tile.
    """

    statically_known = statically_known_true or (lambda expr: bool(expr))
    m_ = statically_known(m)
    k_ = statically_known(k)
    n_ = statically_known(n)
    return bool(m_ * n_ * k_ <= 1024)


def check_supported_striding(mat_a, mat_b) -> None:
    """Whether these two operands can be read by the bodies written for them.

    A body reads an operand by one of a few layouts, because those are the ones
    a tile can be fitted to.  An operand in some other layout is not read
    wrongly -- it is refused, here, rather than read slowly.
    """

    from .mm import _standard_2d

    for mat in (mat_a, mat_b):
        shape = tuple(mat.get_size())
        stride = tuple(mat.get_stride()) if hasattr(mat, "get_stride") else ()
        if len(shape) < 2 or not _standard_2d(shape[-2:], stride[-2:]):
            raise NotImplementedError(
                "an operand in a layout no tile can be fitted to: shape %s "
                "stride %s" % (shape, stride)
            )


def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    """Whether a batched product's leading axis is the outer one on both sides.

    A tile is fitted to the two trailing axes, so what matters about the leading
    one is whether walking it moves further than the trailing axes do: if it does,
    consecutive tiles read the same rows far apart, and the body's assumption
    that they share is false.
    """

    try:
        stride1 = tuple(mat1.get_stride())
        stride2 = tuple(mat2.get_stride())
    except Exception:  # noqa: BLE001 - a call that will not say is not batched
        return False
    if len(stride1) < 3 or len(stride2) < 3:
        return True
    return max(stride1[-3], stride2[-3]) >= max(
        stride1[-2] * stride1[-1], stride2[-2] * stride2[-1]
    )


#: How much room one mapping takes in a descriptor workspace, when the toolkit
#: builds it there rather than in the kernel's own registers.  The size is fixed
#: by the mapping's own layout rather than chosen here.
TMA_SIZE = 128

_INT32_MAX = 2 ** 31 - 1


def descriptor_extents_fit(size: Sequence[int]) -> bool:
    """Whether every extent of a shape is a 32-bit coordinate.

    A descriptor is asked for a tile by coordinate, and a coordinate that does not
    fit in the width the descriptor addresses in is not a tile it can be asked
    for.
    """

    return all(0 <= int(extent) <= _INT32_MAX for extent in size)


def descriptor_offset_fits(size: Sequence[int], stride: Sequence[int]) -> bool:
    """Whether the furthest element of a shape is addressable in 32 bits.

    The extents fitting is not enough: a shape of four elements with a stride of
    a billion has a last element past what a 32-bit offset names, and a
    descriptor built for it would address past the end of the buffer.
    """

    furthest = 0
    for extent, step in zip(size, stride):
        furthest += (int(extent) - 1) * int(step)
    return 0 <= furthest <= _INT32_MAX


def descriptor_form(
    sizes: Sequence[Sequence[int]],
    strides: Sequence[Sequence[int]],
) -> Optional[dict]:
    """How a call's tiles may be fetched by descriptor, or ``None`` if not.

    A descriptor is a mapping stated once and then asked for tiles by
    coordinate, so the address arithmetic a program would repeat per tile is not
    repeated.  What it costs is a 32-bit coordinate and a hardware feature, so
    the two are asked about rather than assumed: the toolkit has to have the
    call, and the operands have to be addressable in 32 bits at all.

    Which way round a descriptor's two axes are is not a choice either -- it is
    the layout the operand has, and a descriptor built for the other one reads the
    wrong elements -- so it is read off the strides and handed to the body, which
    builds the mapping to match.
    """

    import triton.language as tl

    if not (hasattr(tl, "make_tensor_descriptor")
            or hasattr(tl, "_experimental_make_tensor_descriptor")):
        return None
    major, _minor = device_capability()
    if major < 9:
        # A descriptor is a feature of the memory system rather than a
        # preference, so a machine without one has no such form to offer.
        return None
    for size, stride in zip(sizes, strides):
        if len(size) != len(stride):
            return None
        if not descriptor_extents_fit(size) or not descriptor_offset_fits(size, stride):
            return None
    return {
        "USE_TMA_LOAD": True,
        "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": not hasattr(
            tl, "make_tensor_descriptor"
        ),
        "A_IS_K_MAJOR": int(strides[0][-1]) == 1,
        "B_IS_K_MAJOR": int(strides[1][-2]) == 1,
    }


def is_gpu(device_type: str | None) -> bool:
    """Whether a device is one that a template writes a kernel for.

    A device that is not is computed by the tree's own means, and a template
    offered there would be measured against a computation that does not involve
    it.
    """

    return device_type in ("cuda", "xpu", "hip")


def use_template_for_gpu(layout: Layout, dtypes: Sequence[Any]) -> bool:
    """Whether a template is offered for this result's device and type.

    Both are asked together because a template is a kernel for a device *and* a
    type: one written for a wide type and run on a narrow one is not a slower
    version of the right kernel, it is a different kernel.
    """

    if not is_gpu(layout.device.type):
        return False
    return layout.dtype in dtypes


def use_aten_gemm_kernels() -> bool:
    """Whether the framework's own product is among the candidates.

    It is unless the measurement was asked for, because a measurement that
    omitted it would be comparing a template against nothing and calling the
    winner the best.  Asked for, it is a candidate like any other -- one that can
    win.
    """

    if not (config.max_autotune or config.max_autotune_gemm):
        return True
    return "EAGER" in config.max_autotune_backends


def use_triton_template(
    layout: Layout,
    *,
    enable_int32: bool = False,
    enable_float8: bool = False,
    check_max_autotune: bool = True,
) -> bool:
    """Whether a template is offered for a result of this geometry and type.

    The types are listed rather than derived: a template is written for the
    types it was written for, and one handed a type it was not written for is
    offered a kernel that computes something else.  Int32 and the narrow floats
    are behind flags because those templates exist for those calls specifically.
    """

    layout_dtypes = [tp.float16, tp.bfloat16, tp.float32]
    if enable_int32:
        layout_dtypes = [tp.float16, tp.bfloat16, tp.float32, tp.int32]
    if enable_float8:
        layout_dtypes.extend([d for d in dir(tp) if d.startswith("float8")])
    return (
        use_template_for_gpu(layout, layout_dtypes)
        or (layout.device.type == "cpu" and layout.dtype in layout_dtypes)
    ) and (config.max_autotune or config.max_autotune_gemm or (not check_max_autotune))


def use_triton_tma_template(
    *matrices: Any, output_layout: Layout, add_guards: bool = False
) -> bool:
    """Whether the tiles of this call may be fetched by descriptor.

    Two things have to hold.  The matrices must be matrices: a descriptor reads
    a tile by its two extents, and a value with a different number of them has no
    tile to read.  And every extent must be a 32-bit coordinate, because that is
    what a descriptor addresses with -- a shape that does not fit cannot be
    described, and describing it anyway would read somewhere else.

    The result's own extents are asked about only when the store goes through a
    descriptor too, since a result that is written the ordinary way is not
    addressed by one however large it is.
    """

    if not all((len(m.get_size()) == 2 for m in matrices)):
        return False
    if not all((descriptor_extents_fit(m.get_size()) for m in matrices)):
        return False
    sizes = [m.get_size() for m in matrices]
    strides = [m.get_stride() for m in matrices]
    return descriptor_form(sizes, strides) is not None


def use_triton_blackwell_tma_template(
    *matrices: Any, output_layout: Layout, add_guards: bool = False
) -> bool:
    """Whether the workspace-carrying form of the descriptor template applies.

    Only on the generation whose device needs the tiles staged through a
    workspace, which is a property of the device and not of the call -- but it is
    asked per call because the workspace is sized from the call.
    """

    if not use_triton_tma_template(
        *matrices, output_layout=output_layout, add_guards=add_guards
    ):
        return False
    major, _ = device_capability()
    return major >= 10


@functools.cache
def use_decompose_k_choice(m: Any, n: Any, k: Any, threshold_multiple: int = 1) -> bool:
    """Whether splitting the contracted axis is worth offering.

    Only once the contracted axis is long enough that the parts can be computed
    at the same time: a split of a short axis is a sum of small products, which is
    more work than the product, and the point of splitting is to do the parts
    side by side.  The bar is raised by ``threshold_multiple`` where the caller
    wants to be more certain still.

    Cached because the answer is the same for every call of the same shape, and
    the question is asked once per candidate rather than once per call.
    """

    decompose_k_threshold = config.decompose_k_threshold * threshold_multiple
    return V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Ge(k, decompose_k_threshold * m),
            sympy.Ge(k, decompose_k_threshold * n),
        )
    ) and (config.num_decompose_k_splits > 0)

