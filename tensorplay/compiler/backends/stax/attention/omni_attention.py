"""Carrying a structured position through arithmetic that thinks in offsets.

A position in a tensor is usually one number: the distance from the start of
the buffer, which is what indexing arithmetic wants because a load is at an
offset.  The emitter used for these kernels wants the opposite -- the position
as one number per dimension, so it can write ``tensor[i, j]`` and be answerable
for strides itself.

Both are wanted at once, because the position is built by arithmetic that only
understands offsets and is finally handed to an emitter that only understands
dimensions.  So the position stays one expression, and the dimensions ride
along inside it: a value that is an expression to everything that touches it,
and a tuple of coordinates to the one place that asks.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import importlib
import importlib.util
import inspect
import math
from itertools import product
from typing import Any

import tensorplay as tp
import sympy

from tensorplay.graph import Node as FxNode
from tensorplay.graph.experimental.sympy_functions import (
    FloorDiv,
    Max,
    Min,
    ModularIndexing,
    Mod,
)

from ..codegen.cutedsl.lane_analysis import (
    classify_lane_expr as _classify_lane_expr,
    decompose_affine_lane_expr,
    lane_group_start,
)
from ..ir import (
    ComputedBuffer,
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    IRNode,
    ReinterpretView,
    StorageBox,
    TensorBox,
)
from .. import config
from ..loops import V, get_fill_order
from tensorplay.utils._pytree import tree_map, tree_map_only
from tensorplay.graph.experimental.sympy_functions import FloorDiv


class HierarchicalIndex(sympy.Function):
    """One position in a tensor, held as one number per dimension.

    Nothing is done to the value it holds.  It is not simplified, not flattened
    and not reordered, because the dimensions are not a number that could be:
    they are a position, and a position that was rearranged would be a
    different one.  So evaluating it produces nothing, which is what tells the
    expression machinery to carry the node as it stands.

    A value like this is meant to be short-lived -- built where a position is
    produced and taken apart where it is consumed -- and only the emitter for
    these kernels reads it, by taking the node's arguments as the coordinates.
    """

    @classmethod
    def eval(cls, *args):
        return None


def _omni_kernel_options_example(kind: str) -> str:
    """A set of options to offer when an option was not understood.

    An example rather than a list of what is allowed, because the allowed set
    is what the error is for and listing it would be a second place to keep
    it up to date.  Backward and forward take different names for the same
    thing -- the backward pass splits its work differently and so has more of
    it to name -- and the two are told apart by the prefix rather than by the
    position, because a kernel is handed both.
    """

    if kind == "backward":
        return (
            "kernel_options={'bwd_BLOCK_M1': 32, 'bwd_BLOCK_N1': 32, "
            "'bwd_BLOCK_M2': 32, 'bwd_BLOCK_N2': 32, "
            "'bwd_num_stages': 1, 'bwd_num_warps': 4}"
        )
    return (
        "kernel_options={'fwd_BLOCK_M': 32, 'fwd_BLOCK_N': 64, "
        "'fwd_num_stages': 1, 'fwd_num_warps': 4}"
    )


def _omni_kernel_tuning_options(kind: str) -> str:
    """Which options a kernel of this kind can be tuned over."""

    if kind == "backward":
        return (
            "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_warps, and "
            "num_stages; use the bwd_ prefix to set backward-only options"
        )
    if kind == "decode":
        return (
            "BLOCK_M, BLOCK_N, num_warps, and num_stages; use the fwd_ "
            "prefix to set decode-only options"
        )
    return (
        "BLOCK_M, BLOCK_N, num_warps, and num_stages; use the fwd_ "
        "prefix to set forward-only options"
    )


# ---------------------------------------------------------------------------
# Reading a captured value several positions at a time
# ---------------------------------------------------------------------------


class LoadKind(enum.Enum):
    """How a captured value behaves when read across a group of positions.

    Three answers because there are three things that can be true.  The
    positions can name places that are next to each other, in which case they
    can be read as one wide read.  They can all name the same place, in which
    case the value is read once and the group does not matter.  Or they can
    name nothing in particular, in which case each is read on its own -- and
    that is the answer that is always available, so it is the one a question
    about this falls back to.
    """

    GATHER = enum.auto()
    LANE_UNIFORM = enum.auto()
    CONTIGUOUS = enum.auto()


@dataclasses.dataclass(frozen=True)
class AuxLoadVecInfo:
    """How one captured value is read across a group of positions.

    The width is carried rather than looked up, because a value that can be
    read wide and a value read one position at a time are different reads and
    a decision made once should not be made again -- and because the two are
    mutually exclusive: a width on a read that is not a wide one would be a
    number that means nothing.
    """

    kind: LoadKind
    vec_size: Any = None

    def __post_init__(self) -> None:
        if self.kind is LoadKind.CONTIGUOUS:
            if self.vec_size is None:
                raise AssertionError("CONTIGUOUS load requires a vec_size")
        else:
            if self.vec_size is not None:
                raise AssertionError(
                    f"non-CONTIGUOUS load must not carry vec_size, got {self.vec_size}"
                )

    @classmethod
    def gather(cls) -> "AuxLoadVecInfo":
        """Each position names a place of its own."""

        return cls(LoadKind.GATHER)

    @classmethod
    def lane_uniform(cls) -> "AuxLoadVecInfo":
        """Every position names the same place, so it is read once."""

        return cls(LoadKind.LANE_UNIFORM)

    @classmethod
    def contiguous(cls, vec_size: int) -> "AuxLoadVecInfo":
        """Consecutive positions name consecutive places, so they read as one."""

        return cls(LoadKind.CONTIGUOUS, vec_size)


@dataclasses.dataclass(frozen=True)
class AuxVecPolicy:
    """The rules one kind of captured value is read under.

    Which positions in a body are the query's and which are the position being
    walked, because a captured value read at either of those means something
    different from one read at any other: read at the walked position it varies
    across the group, and read anywhere else it does not.

    The smallest rank for a wide read is a separate rule because a short mask
    read wide would be a wide read of very few elements -- and a mask that
    short can be packed into a range of positions instead, which is cheaper
    than reading it at all.
    """

    q_idx_placeholder: int
    kv_idx_placeholder: int
    max_vec_size: int
    min_index_rank_for_contiguous_load: int = 1
    non_lane_placeholder_start: int = 0


MASK_MOD_AUX_VEC_POLICY = AuxVecPolicy(
    q_idx_placeholder=2,
    kv_idx_placeholder=3,
    max_vec_size=32,
    min_index_rank_for_contiguous_load=2,
)
SCORE_MOD_AUX_VEC_POLICY = AuxVecPolicy(
    q_idx_placeholder=3,
    kv_idx_placeholder=4,
    max_vec_size=8,
    non_lane_placeholder_start=1,
)


#: How many positions a mask is evaluated at when nothing narrows it.  The same
#: width a packed mask uses, because a mask evaluated that wide can be written
#: as a range of positions -- and one that cannot be packed is one that has to
#: be evaluated a position at a time whatever width the rest of the kernel uses.
DEFAULT_MASK_MOD_VEC_SIZE = 32


@dataclasses.dataclass(frozen=True)
class AuxIndexedTensor:
    """A captured value together with the part of its index already settled.

    The part that is settled is the part that does not depend on which position
    of the group is being read, so it can be worked out once and used for every
    position rather than being worked out again each time.
    """

    buffer: Any
    indices: Any


def make_fx_index_symbols(
    q_idx_node: Any,
    kv_idx_node: Any,
    non_lane_index_nodes: Any = (),
    *,
    kv_expr: Any = None,
) -> Any:
    """Symbols standing for the positions a body is read at.

    One per position rather than the position itself, because the position is
    not known while the body is being analysed -- only which positions there
    are.  The query's position and the walked position are separate symbols
    because a captured value read at one means something different from the
    same value read at the other: read at the walked position it varies across
    the group, and read anywhere else it does not.

    A walked position given as an expression is used as given rather than made
    into a symbol, because a caller that has already worked out where the walk
    is has more to say than a bare name would.
    """

    q_idx = sympy.Symbol("q_idx", integer=True, nonnegative=True)
    kv_idx = sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    index_symbols = {
        node: sympy.Symbol(node.name, integer=True, nonnegative=True)
        for node in non_lane_index_nodes
    }
    index_symbols[q_idx_node] = q_idx
    index_symbols[kv_idx_node] = kv_idx if kv_expr is None else kv_expr
    return q_idx, kv_idx, index_symbols


def select_mask_mod_vec_size(
    *,
    has_mask_mod: bool,
    has_mask_aux_tensors: bool,
    supports_mask_mod_vec: bool,
    graph_module: Any,
    other_buffers: Any,
) -> Any:
    """How many positions a mask is evaluated at.

    Nothing to say if there is no mask or if this kernel cannot evaluate one
    that wide: a width nothing can be read at is not a width.

    A mask that reads no captured value can be evaluated as wide as a packed
    mask is, because that is how wide a mask is written when it is written as a
    range of positions.  A mask that does read captured values is held to
    whatever those reads allow, and a width of one is reported as no width --
    which is not the same thing, and is how the caller is told that reading them
    one at a time is the answer.
    """

    if not has_mask_mod or not supports_mask_mod_vec:
        return None
    if not has_mask_aux_tensors:
        return DEFAULT_MASK_MOD_VEC_SIZE

    vec_size = select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        MASK_MOD_AUX_VEC_POLICY,
    )
    return vec_size if vec_size > 1 else None


def select_score_mod_vec_size(
    *,
    has_score_mod: bool,
    has_aux_tensors: bool,
    is_sm100_or_later: bool,
    graph_module: Any,
    other_buffers: Any,
) -> Any:
    """How many positions a score is applied at.

    Nothing to say when there is no score or nothing captured to read, and a
    width of one -- rather than nothing -- on a device whose wide reads of
    captured values are not written: a score applied a position at a time is
    the answer there, and saying so is different from saying nothing was
    decided.
    """

    if not has_score_mod or not has_aux_tensors:
        return None
    if not is_sm100_or_later:
        return 1
    return select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        SCORE_MOD_AUX_VEC_POLICY,
    )


def fx_aux_index_to_sympy(
    index: Any, index_symbols: Any, node_to_sympy: Any = None
) -> Any:
    """An index expression written in a body, as something positions can be asked of.

    Only the operations these masks are written in are answered, and anything
    else answers with nothing.  That is not a limitation to be worked around:
    whether an expression can be recognised is what decides whether the read it
    belongs to can be done several positions at a time, and an expression
    recognised as something it is not would make a read wide that is not.
    """

    if isinstance(index, (int, sympy.Integer)) and not isinstance(index, bool):
        return sympy.Integer(index)
    if not isinstance(index, FxNode):
        return None
    if index in index_symbols:
        return index_symbols[index]
    if node_to_sympy is not None:
        expr = node_to_sympy(index)
        if expr is not None:
            return expr
    if index.op != "call_function":
        return None

    args = index.args
    target = index.target
    if target is tp.ops.tp.abs.default:
        operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
        return None if operand is None else sympy.Abs(operand)
    if target is tp.ops.tp.neg.default:
        operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
        return None if operand is None else -operand
    if target is tp.ops.tp.clamp.default:
        operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
        if operand is None:
            return None
        lo = args[1] if len(args) > 1 else index.kwargs.get("min")
        hi = args[2] if len(args) > 2 else index.kwargs.get("max")
        for bound, combine in ((lo, Max), (hi, Min)):
            if bound is not None:
                bound_expr = fx_aux_index_to_sympy(
                    bound, index_symbols, node_to_sympy
                )
                if bound_expr is None:
                    return None
                operand = combine(operand, bound_expr)
        return operand

    if len(args) < 2:
        return None
    lhs = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
    rhs = fx_aux_index_to_sympy(args[1], index_symbols, node_to_sympy)
    if lhs is None or rhs is None:
        return None
    if target in (tp.ops.tp.add.Tensor, tp.ops.tp.add.Scalar):
        return V.graph.sizevars.simplify(lhs + rhs)
    if target in (tp.ops.tp.sub.Tensor, tp.ops.tp.sub.Scalar):
        return V.graph.sizevars.simplify(lhs - rhs)
    if target in (tp.ops.tp.mul.Tensor, tp.ops.tp.mul.Scalar):
        return V.graph.sizevars.simplify(lhs * rhs)
    if target is tp.ops.tp.minimum.default:
        return Min(lhs, rhs)
    if target is tp.ops.tp.maximum.default:
        return Max(lhs, rhs)
    if target in (tp.ops.tp.remainder.Tensor, tp.ops.tp.remainder.Scalar):
        return ModularIndexing(lhs, 1, rhs)
    if (
        target is tp.ops.tp.div.Tensor_mode
        and index.kwargs.get("rounding_mode") == "floor"
    ):
        return FloorDiv(lhs, rhs)
    return None


def is_safe_partial_aux_index(
    indices: Any, q_idx_node: Any, kv_idx_node: Any, non_lane_index_nodes: Any
) -> bool:
    """Whether a partly-indexed value can be carried to the next step.

    Carried only while the walked position is not among the indices: once it
    is, the value read so far already depends on which position is being read,
    and what it depends on cannot be carried forward.  A partly-indexed value
    that does not depend on the walked position is the same for every position,
    so it is settled once.
    """

    _, kv_idx, index_symbols = make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    for index in indices:
        expr = fx_aux_index_to_sympy(index, index_symbols)
        if expr is None or kv_idx in expr.free_symbols:
            return False
    return True


def direct_aux_load_vec_size_and_kind(
    indices: Any,
    buffer: Any,
    q_idx_node: Any,
    kv_idx_node: Any,
    non_lane_index_nodes: Any = (),
    max_vec_size: int = 8,
    min_index_rank_for_contiguous_load: int = 1,
) -> Any:
    """How wide one read of a captured value can be, and what kind of read it is.

    The same for every position if no index depends on the walked position.  If
    one does, then only the last axis may depend on it -- a read that varies
    across two axes is two reads, not one wider one -- and that axis has to be
    stored adjacently, and the positions have to be consecutive, and the
    elements have to divide evenly into the width chosen.

    The width is narrowed from the largest on offer until every one of those
    holds, because a width that does not divide the elements evenly would read
    past the end of the value on the last position.
    """

    if not isinstance(indices, (list, tuple)) or not indices:
        return AuxLoadVecInfo.gather()
    if not (max_vec_size >= 2 and max_vec_size.bit_count() == 1):
        raise AssertionError(
            f"max_vec_size must be a power of two >= 2, got {max_vec_size}"
        )

    _, kv_idx, index_symbols = make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    index_exprs = [fx_aux_index_to_sympy(index, index_symbols) for index in indices]
    if any(expr is None for expr in index_exprs):
        return AuxLoadVecInfo.gather()
    if all(kv_idx not in expr.free_symbols for expr in index_exprs):
        return AuxLoadVecInfo.lane_uniform()

    last_expr = index_exprs[-1]
    if kv_idx not in last_expr.free_symbols:
        return AuxLoadVecInfo.gather()
    if len(indices) < min_index_rank_for_contiguous_load:
        return AuxLoadVecInfo.gather()

    prefix_exprs = index_exprs[:-1]
    if any(kv_idx in expr.free_symbols for expr in prefix_exprs):
        return AuxLoadVecInfo.gather()

    sizes = buffer.get_size()
    strides = buffer.get_stride()
    if not V.graph.sizevars.statically_known_equals(strides[-1], 1):
        return AuxLoadVecInfo.gather()

    offset = buffer.get_layout().offset
    vec_size = max_vec_size
    while vec_size >= 2:
        lane_info = _classify_lane_expr(
            last_expr, kv_idx, max_width=vec_size
        )
        if lane_info is not None:
            width, step = lane_info
            if width == vec_size and (
                V.graph.sizevars.statically_known_multiple_of(sizes[-1], vec_size)
                and V.graph.sizevars.statically_known_multiple_of(offset, vec_size)
                and all(
                    V.graph.sizevars.statically_known_multiple_of(stride, vec_size)
                    for stride in strides[:-1]
                )
            ):
                return AuxLoadVecInfo.contiguous(vec_size)
        vec_size //= 2
    return AuxLoadVecInfo.gather()


def select_aux_mod_vec_size(
    graph_module: Any, other_buffers: Any, policy: Any
) -> int:
    """How wide the captured values in a body can be read.

    Follows the indexing from each captured value to where it is finally read,
    carrying along the part of the index that does not depend on the walked
    position, and keeps the narrowest width any of the reads needs.  A read that
    names places of its own is read one position at a time and does not narrow
    the others -- it coexists with a wide read of something else.

    One if nothing could be read wide, which is the honest answer rather than
    the widest on offer: a width nothing is read at would let a kernel be
    written for a read that is not happening.
    """

    if graph_module is None:
        return 1

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    num_fixed_placeholders = policy.kv_idx_placeholder + 1
    if len(placeholders) < num_fixed_placeholders:
        return 1

    aux_indexed_tensors = {
        placeholder: AuxIndexedTensor(buffer, ())
        for placeholder, buffer in zip(
            placeholders[num_fixed_placeholders:], other_buffers
        )
    }
    non_lane_index_nodes = placeholders[
        policy.non_lane_placeholder_start : policy.q_idx_placeholder
    ]
    selected_vec_size = policy.max_vec_size
    found_vectorizable_load = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target is not tp.ops.tp.index.Tensor:
            continue
        buffer_node, indices = node.args
        if buffer_node not in aux_indexed_tensors:
            continue
        indexed_tensor = aux_indexed_tensors[buffer_node]
        if not isinstance(indices, (list, tuple)) or not indices:
            continue
        full_indices = indexed_tensor.indices + tuple(indices)
        rank = len(indexed_tensor.buffer.get_size())
        if len(full_indices) < rank:
            if is_safe_partial_aux_index(
                full_indices,
                placeholders[policy.q_idx_placeholder],
                placeholders[policy.kv_idx_placeholder],
                non_lane_index_nodes,
            ):
                aux_indexed_tensors[node] = AuxIndexedTensor(
                    indexed_tensor.buffer, full_indices
                )
            continue
        if len(full_indices) > rank:
            continue
        aux_load_vec_info = direct_aux_load_vec_size_and_kind(
            full_indices,
            indexed_tensor.buffer,
            placeholders[policy.q_idx_placeholder],
            placeholders[policy.kv_idx_placeholder],
            non_lane_index_nodes=non_lane_index_nodes,
            max_vec_size=policy.max_vec_size,
            min_index_rank_for_contiguous_load=policy.min_index_rank_for_contiguous_load,
        )
        if aux_load_vec_info.kind is LoadKind.LANE_UNIFORM:
            found_vectorizable_load = True
        elif aux_load_vec_info.kind is LoadKind.GATHER:
            pass
        elif aux_load_vec_info.kind is LoadKind.CONTIGUOUS:
            contiguous_vec_size = aux_load_vec_info.vec_size
            if contiguous_vec_size is None:
                raise AssertionError("CONTIGUOUS load must have a vec_size")
            selected_vec_size = min(selected_vec_size, contiguous_vec_size)
            found_vectorizable_load = True

    return selected_vec_size if found_vectorizable_load else 1


@contextlib.contextmanager
def writing_with_omni_indexer() -> Any:
    """Have whatever is written inside name dimensions rather than offsets.

    Around the writing rather than inside it, so that anything written outside is
    written the way every other kernel is -- which is what makes this something
    to apply to the kernels that need it rather than a property of how kernels
    are written here.
    """

    with patch_fixed_layout_indexer_for_cutedsl():
        yield


def generate_omni_flash_choice(template: Any, **kwargs: Any) -> Any:
    """Write one way of the kernel, with positions naming dimensions.

    A failure is passed back rather than raised, because a way of writing a
    kernel that this device cannot run is not an error -- it is one fewer thing
    to measure among several, and the rest may still be the one that wins.
    """

    with patch_fixed_layout_indexer_for_cutedsl():
        return template.generate(**kwargs)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def get_omni_flash_fwd_configs(
    has_score_mod: bool,
    has_aux_tensors: bool,
    device: Any = None,
    score_mod_graph_module: Any = None,
    score_mod_other_buffers: Any = (),
    has_mask_mod: bool = False,
    has_mask_aux_tensors: bool = False,
    mask_mod_graph_module: Any = None,
    mask_mod_other_buffers: Any = (),
    aux_scalar_symbols: Any = (),
) -> Any:
    """The ways the forward kernel may be written, widest mask last.

    Whether a mask can be read several positions at a time is a property of the
    device as well as of the mask: the wide reads it needs are not written on
    every device.  So the device is asked before the width is chosen, rather
    than the width being chosen and then found not to be readable.

    A mask that could be written as ranges of positions is written that way
    whatever else was decided about its width, because that is the width the
    ranges are written at -- the two are the same decision, and asking about
    them separately is how they come to disagree.

    A score that reads nothing captured is not held to any width, and when the
    search is measuring it is offered every width the kernel supports: nothing
    in the graph says which is best, and that is what a measurement is for.
    """

    from ..codegen.cutedsl.aux_scalars import CuteDSLAuxScalarBindings

    cuda_major = None
    if tp.cuda.is_available() and (
        has_mask_mod or (has_score_mod and has_aux_tensors)
    ):
        device_index = None if device is None else device.index
        cuda_major = tp.cuda.get_device_capability(device_index)[0]
    mask_mod_vec_size = select_mask_mod_vec_size(
        has_mask_mod=has_mask_mod,
        has_mask_aux_tensors=has_mask_aux_tensors,
        supports_mask_mod_vec=cuda_major in (10, 11),
        graph_module=mask_mod_graph_module,
        other_buffers=mask_mod_other_buffers,
    )
    score_mod_vec_size = select_score_mod_vec_size(
        has_score_mod=has_score_mod,
        has_aux_tensors=has_aux_tensors,
        is_sm100_or_later=cuda_major is not None and cuda_major >= 10,
        graph_module=score_mod_graph_module,
        other_buffers=score_mod_other_buffers,
    )
    mask_mod_packed_intervals = None
    if has_mask_mod and cuda_major in (10, 11) and mask_mod_graph_module is not None:
        mask_mod_packed_intervals = select_packed_mask_intervals(
            mask_mod_graph_module,
            mask_mod_other_buffers,
            CuteDSLAuxScalarBindings(tuple(aux_scalar_symbols)).symbol_codes(),
        )
    if mask_mod_packed_intervals is not None:
        mask_mod_vec_size = DEFAULT_MASK_MOD_VEC_SIZE

    if (
        has_score_mod
        and score_mod_vec_size is None
        and config.max_autotune
    ):
        # Nothing captured held the score's width, and a captured number is the
        # same for every position -- so every width the kernel supports is
        # allowed, and which is best is what the search is for.
        score_mod_vec_sizes = (1, 2, 4, 8, 16, 32, 64, 128)
    else:
        score_mod_vec_sizes = (score_mod_vec_size,)
    configs = [
        OmniFlashConfig(
            score_mod_vec_size=v,
            mask_mod_vec_size=mask_mod_vec_size,
            mask_mod_packed_intervals=mask_mod_packed_intervals,
        )
        for v in score_mod_vec_sizes
    ]
    max_configs = config.test_configs.max_omni_configs
    if max_configs is not None and len(configs) > max_configs:
        configs = configs[:max_configs]
    return configs


def _get_omni_flash_bwd_configs() -> Any:
    """The backward kernel has only the one way of being written.

    Not measured, because there is nothing to choose: a score that is more than
    itself is not yet accounted for in the backward pass, so the score is the
    score and the kernel has one shape.
    """

    return [OmniFlashConfig()]


# ---------------------------------------------------------------------------
# Whether the kernel can be written at all
# ---------------------------------------------------------------------------


FLASH_ATTENTION_INSTALL_MESSAGE = (
    "Install a compatible Flash Attention package, for example "
    '`pip install --pre flash-attn-4` (`pip install --pre "flash-attn-4[cu13]"` '
    "for CUDA 13), and see https://pypi.org/project/flash-attn-4/ "
    "for PyPI packaging details."
)


def _flash_attention_unavailable_message() -> str:
    return (
        "CUTE flash attention library is not available. "
        f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
    )


@functools.lru_cache(maxsize=1)
def ensure_flash_available() -> bool:
    """Whether the attention library this kernel is written against is here.

    Asked once and remembered, because it is a fact about what is installed and
    does not change while the program runs.  Asked by looking rather than by
    importing, so that a missing library is a missing library rather than an
    error from something that was never there.
    """

    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def flash_supports_aux_scalars() -> bool:
    """Whether the installed library can be handed a number beside the values.

    Asked by looking at what the entry points accept rather than by trying one,
    because trying one would produce values rather than an answer, and the values
    would then be the ones the answer was about.
    """

    try:
        interface = importlib.import_module("flash_attn.cute.interface")
    except ImportError:
        return False
    return (
        "aux_scalars" in inspect.signature(interface._flash_attn_fwd).parameters
        and "aux_scalars" in inspect.signature(interface._flash_attn_bwd).parameters
    )


# ---------------------------------------------------------------------------
# How a position becomes a place in memory
# ---------------------------------------------------------------------------


def _hierarchical_indexer_cute(
    size: Any, stride: Any = None, offset: Any = None
) -> Any:
    """Turn a position into the dimensions it names, rather than into an offset.

    Everywhere else a position is one number: the distance from the start of the
    buffer, because a read is at an offset and the offset is what the hardware
    wants.  This kernel wants the opposite -- one number per dimension, so that
    it can be written as ``tensor[i, j]`` and be answerable for strides itself.

    A single dimension is passed through as itself, because there is nothing to
    keep together: one number is already one number.  No position at all is
    zero, which is what reading a value with no dimensions means.
    """

    if offset is None:
        offset = sympy.Integer(0)

    def indexer(indices: Any) -> Any:
        if offset != sympy.Integer(0):
            raise AssertionError("Offset not supported for hierarchical indexing")
        if len(indices) != len(size):
            raise AssertionError(
                f"Rank mismatch: got {len(indices)} indices for tensor of rank {len(size)}"
            )
        if not indices:
            return sympy.Integer(0)
        if len(indices) == 1:
            return indices[0]
        return HierarchicalIndex(*indices)

    return indexer


@contextlib.contextmanager
def patch_fixed_layout_indexer_for_cutedsl() -> Any:
    """Make positions name dimensions for as long as a kernel is being written.

    The layout knows how to turn a position into an offset and is written to do
    that, because that is what a read is everywhere else.  This kernel wants the
    dimensions instead, and it wants them only while it is being written -- so
    the layout is changed for that time and changed back, rather than taught a
    second way of doing something every other caller would then have to know
    about.

    These kernels read and compute but do not store, so the values whose layout
    this changes are only ever read, and how a read is addressed is not part of
    what is stored.
    """

    original_make_indexer = FixedLayout.make_indexer

    def cutedsl_make_indexer(self: Any) -> Any:
        return _hierarchical_indexer_cute(self.size, self.stride, self.offset)

    FixedLayout.make_indexer = cutedsl_make_indexer
    try:
        yield
    finally:
        FixedLayout.make_indexer = original_make_indexer


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OmniFlashConfig:
    """One way of writing the kernel, among the ways worth measuring.

    How many elements one pass over the kernel handles is a choice about what
    the hardware does with them rather than about what the kernel computes, and
    the two are answered separately because they are: the width a mask is
    evaluated at is a property of the mask, and the width a score is applied at
    is a property of the score.
    """

    #: How many elements one pass applies the score to.  Left unset to take the
    #: kernel's own width, which is the right answer when nothing has been
    #: measured to say otherwise.
    score_mod_vec_size: Any = None
    #: How many consecutive lanes one pass evaluates the mask over.  The same
    #: width as the mask is read at, so that a mask written as a range of lanes
    #: is one value rather than a comparison per lane.
    mask_mod_vec_size: Any = None
    #: The ranges the mask keeps, worked out ahead of the kernel rather than in
    #: it.  Only usable for a mask that is a range; unset leaves the kernel to
    #: work it out per lane.
    mask_mod_packed_intervals: Any = None


def collect_aux_scalar_symbols(*buffer_groups: Any) -> tuple:
    """The symbols every captured value is written in terms of, in one order.

    Taken across all of a graph's captures rather than per capture, and in a
    name order rather than an arrival order, because these are the arguments a
    kernel is called with: a kernel's signature has to be the same whichever
    order the values arrived in, or the same graph captured twice would produce
    two kernels.
    """

    symbols: dict = {}
    for buffers in buffer_groups:
        for buffer in buffers:
            if isinstance(buffer, sympy.Expr):
                for symbol in sorted(buffer.free_symbols, key=lambda s: s.name):
                    symbols.setdefault(symbol, None)
    return tuple(symbols)


# ---------------------------------------------------------------------------
# What a graph is made of
# ---------------------------------------------------------------------------


def input_buffers_require_grads(graph_module: Any, num_score_mod_placeholders: int) -> bool:
    """Whether any input beyond the score's own placeholders needs a gradient.

    The placeholders the score was given are the score's business; what matters
    here is whether the values the kernel was called with are ones a backward
    pass has to be able to reach.
    """

    inputs = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if len(inputs) <= num_score_mod_placeholders:
        return False

    def requires_grad(n: Any) -> bool:
        tensor_meta = n.meta.get("tensor_meta")
        return tensor_meta.requires_grad if tensor_meta is not None else False

    return any(requires_grad(n) for n in inputs[num_score_mod_placeholders:])


def is_trivial_score_graph(graph_module: Any) -> bool:
    """Whether the score is just the score, passed through.

    Which is the case the backward pass can be written for: a score that is
    more than itself changes the gradient in a way the backward pass does not
    yet account for, and recognising it here is what keeps that from being
    discovered as a wrong gradient rather than as an unsupported score.
    """

    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]
    return output_val == placeholders[0]


def is_trivial_mask_graph(graph_module: Any) -> bool:
    """Whether the mask is the one that keeps everything.

    A mask that keeps everything is not a mask, and recognising that is what
    lets the kernel take the path that has no mask in it at all.
    """

    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]
    return len(placeholders) == 4 and output_val.target is tp.ops.tp.full.default


def has_unsupported_cpu_scalar_tensor_captures(
    score_mod_other_buffers: Any, mask_mod_other_buffers: Any
) -> bool:
    """Whether a captured value is a lone number held on the host.

    A kernel is handed numbers, not the values that happen to contain one, so a
    capture that is a single value on the host has to be turned into a number
    before it can be one -- and until it has been, the kernel cannot be told
    what it would have been handed.
    """

    from ...ir import TensorBox

    for buf in list(score_mod_other_buffers) + list(mask_mod_other_buffers):
        if isinstance(buf, TensorBox):
            device = buf.get_device()
            size = buf.get_size()
            if device is not None and getattr(device, "type", None) == "cpu" and len(size) == 0:
                return True
    return False


# ---------------------------------------------------------------------------
# Shapes these kernels assume
# ---------------------------------------------------------------------------


def is_power_of_2(n: Any) -> bool:
    """Whether a number is a power of two.

    A power of two is one bit set, so a number that is one has exactly one bit
    that is not in the number below it.  Zero is not one however it is
    written, and is the one value where the test would otherwise say yes.
    """

    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_two(n: Any) -> Any:
    """The smallest power of two at least as large.

    A kernel that processes a number of elements at a time wants that number to
    be one it can halve, because halving is how a range of work is split.  A
    length that is not a power of two is rounded up rather than down, because
    rounding down would leave positions with nobody to process them.
    """

    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


def set_head_dim_values(
    kernel_options: Any, qk_head_dim: Any, v_head_dim: Any, graph_sizevars: Any
) -> None:
    """Record the two head sizes a kernel is written for, and the rounded forms.

    Two sizes rather than one, because the size the scores are computed at and
    the size the values are combined at are not always the same, and a kernel
    written for one of them cannot be used for the other.  Recorded rather than
    read from the graph because the kernel is written before the graph is read
    and has to agree with it.

    The rounded forms are recorded alongside: a kernel works in whole groups of
    a power of two, and a size that is not one is padded up to the next, which
    is why the padding is the kernel's business and not the caller's.
    """

    qk_head_dim_static = graph_sizevars.guard_int(qk_head_dim)
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim_static)
    kernel_options.setdefault(
        "QK_HEAD_DIM_ROUNDED", next_power_of_two(qk_head_dim_static)
    )

    v_head_dim_static = graph_sizevars.guard_int(v_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim_static)
    kernel_options.setdefault(
        "V_HEAD_DIM_ROUNDED", next_power_of_two(v_head_dim_static)
    )

    # Whether either size is already a whole number of groups.  Recorded
    # because a kernel can take a shorter path when both are, and a kernel
    # that has to pad is a different kernel from one that does not -- so
    # whether it does is part of what the kernel is.
    kernel_options.setdefault(
        "SAFE_HEAD_DIM",
        is_power_of_2(qk_head_dim_static) and is_power_of_2(v_head_dim_static),
    )


def can_skip_boundary_checks(seq_len: Any, sparse_block_size: Any) -> bool:
    """Whether an axis divides into whole tiles, so no tile runs off its end.

    Asked before a configuration is chosen, and therefore against the largest
    tile a candidate might use rather than against whichever one is chosen:
    a kernel that skips the check must skip it for every shape it will see, not
    for the one that happens to be measured first.
    """

    return V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Eq(Mod(seq_len, 128), 0),
            sympy.Or(
                sympy.Eq(Mod(seq_len, sparse_block_size), 0),
                sympy.Ge(sparse_block_size, seq_len),
            ),
        )
    )


def is_tensor_ir_node(node: Any) -> bool:
    """Whether a node is a value rather than a number.

    The two are told apart here because a value and a number are both things a
    graph holds, and a list of them has to be taken apart before it can be
    passed on -- which is only possible if which is which is known.
    """

    return isinstance(node, IRNode) and node.has_tensor_output()


def contiguous_last_dim(x: Any) -> Any:
    """A value whose innermost axis has no gaps between its elements.

    Asked for by a kernel that reads along that axis one element after another,
    which is only the same thing as walking positions when the elements are
    adjacent.  Reordered rather than copied, because a copy here would be a
    copy of the whole value to fix a property of one axis of it.
    """

    strides = x.maybe_get_stride()
    if strides and strides[-1] != 1:
        contiguous_stride_order = list(reversed(range(len(x.get_size()))))
        return ExternKernel.require_stride_order(x, contiguous_stride_order)
    return x


def maybe_realize(args: Any) -> Any:
    """Write down every value in a list that has not been written down yet.

    Taken one at a time and asked of each, because what a kernel is handed is
    a list of things of different kinds -- values, and numbers that stand for
    shapes -- and only some of them are things that can be written down.
    """

    from ..op_lowerings import realize_inputs

    return tree_map(
        lambda x: (
            realize_inputs(x) if x is not None and not isinstance(x, sympy.Expr) else x
        ),
        args,
    )


def freeze_irnodes(tree: Any) -> Any:
    """Stop every value in a tree from being written anywhere else.

    A kernel is handed values and decides for itself where they live, so a
    value whose layout could still change after it was handed would be a value
    the kernel and the graph disagree about.  A value that cannot be frozen is
    one that is not a value -- a number standing for a shape -- and is left
    alone rather than refused.
    """

    if tree is None:
        return None

    def _freeze(node: Any) -> Any:
        try:
            node.freeze_layout()
        except (NotImplementedError, AttributeError):
            pass
        return node

    return tree_map_only(IRNode, _freeze, tree)


def create_placeholder(
    name: str, dtype: Any, device: Any, size: Any = None
) -> Any:
    """A value the kernel is handed that nothing has produced yet.

    A kernel is written against a signature, and the values in it are what the
    kernel reads; the ones nothing has produced are the arguments, and they
    are given a shape here so that the kernel can be written before anything
    has been computed.
    """

    input_buffer = InputBuffer(
        name=name,
        layout=FixedLayout(
            device,
            dtype,
            size if size else [],
            FlexibleLayout.contiguous_strides(size) if size else [],
        ),
    )
    return TensorBox.create(input_buffer)


def construct_strides(sizes: Any, fill_order: Any) -> Any:
    """The strides a shape has when its axes are filled in a given order.

    Filled innermost-first because that is what makes the result dense: the
    axis filled first has the smallest stride, and every axis after it moves
    further.  The order is given rather than inferred because a kernel needs
    the layout its own reads assume, and which one that is is a property of
    how it reads rather than of the value it reads.
    """

    if len(sizes) != len(fill_order):
        raise AssertionError("Length of sizes must match the length of the fill order")
    strides = [0] * len(sizes)
    current_stride: Any = 1
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride = current_stride * sizes[dim]
    return strides


def infer_dense_strides(size: Any, orig_strides: Any) -> Any:
    """Dense strides that keep the layout the value already had.

    A value is read in the order its layout says, and reordering the axes would
    make every read wrong -- so the layout is kept and only the gaps are
    removed.  The innermost axis is made adjacent whatever the layout said,
    because these kernels read along it one element after another, and that is
    only walking positions when the elements are next to each other.
    """

    fill_order = get_fill_order(orig_strides, V.graph.sizevars.shape_env)
    strides = construct_strides(size, fill_order)

    if strides[-1] != 1:
        last_dim = len(size) - 1
        fill_order = list(fill_order)
        fill_order.remove(last_dim)
        fill_order = [last_dim] + fill_order
        strides = construct_strides(size, fill_order)

    return strides


def get_fwd_subgraph_outputs(subgraph_buffer: Any, mask_graph_buffer: Any) -> Any:
    """What the forward pass produces: the score's outputs, then the mask's.

    In that order because the mask is applied to what the score produced, and a
    caller that took them the other way round would be applying a mask to
    something that has not been computed.
    """

    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, (list, tuple)) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer if isinstance(mask_graph_buffer, (list, tuple)) else [mask_graph_buffer]
    )
    return [*subgraph_buffer, *mask_graph_buffer]


def create_indices_fake(x: Any) -> Any:
    """A stand-in for an index, for measuring a kernel with.

    Every position named, rather than the first one or none: an index that
    named only some positions would make the kernel look cheaper than it is,
    because the work of following an index is not the same as the work of
    reading a position.
    """

    size = V.graph.sizevars.optimization_hints(x.get_size())
    indices = tp.arange(0, size[-1], dtype=x.get_dtype(), device=x.get_device())
    indices = indices.expand(size).contiguous()
    return indices


def create_num_blocks_fake_generator(sparse_indices: Any) -> Any:
    """A stand-in for a count of blocks, for measuring a kernel with.

    A count has to be one the kernel would really do that much work for, or the
    measurement is of a different kernel than the one that will run.  A count
    of no blocks would measure a kernel that reads nothing; a count of every
    block would measure a kernel that takes far longer than any real one would,
    for no better answer.  So a count in between: enough that reading ahead
    would help if it were going to, few enough that measuring is quick.
    """

    def create_num_blocks_fake(x: Any) -> Any:
        num_blocks_for_autotuning = V.graph.sizevars.optimization_hint(
            sparse_indices.shape[-1]
        )
        size = V.graph.sizevars.optimization_hints(x.get_size())
        return tp.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_num_blocks_fake


def zeros_and_scatter_lowering(shape: Any, indices: Any, values: Any) -> Any:
    """A value of zeros with a value added into it at a position, for gradients.

    The gradient a captured value contributes is a sum over every position that
    was read, and a sum is not a walk -- several positions may be the same one,
    and each has to be added to what the others put there.  So the addition is
    an atomic one: two positions being the same has to be decided by the
    hardware as they happen, because which of them gets there first is not
    something that can be known beforehand.

    Accumulated in single precision and converted after, because the order the
    additions happen in is not fixed and a narrow accumulator would make the
    answer depend on that order as well as on the values.
    """

    from ..ir import ComputedBuffer, MutationLayoutSHOULDREMOVE, Scatter
    from ..op_lowerings import (
        _full,
        check_and_broadcast_indices,
        index_output_size_and_inner_fn,
        to_dtype,
    )

    # Always accumulate into fp32 then cast
    grad = _full(0, values.get_device(), tp.float32, shape)
    if not isinstance(grad, TensorBox):
        grad = TensorBox.create(grad)
    grad.realize()
    x_size = grad.get_size()
    values = to_dtype(values, grad.get_dtype())
    device = grad.get_device()
    if device is None:
        raise AssertionError("device must not be None")
    if not indices:
        if shape:
            raise AssertionError(
                "zeros_and_scatter with no indices only supports scalar outputs"
            )
        expected_vals_size = values.get_size()

        def inner_fn(index: Any) -> Any:
            return []

    else:
        indices_loaders = [i.make_loader() if i is not None else None for i in indices]
        indices, tensor_indices = check_and_broadcast_indices(
            indices, grad.get_device()
        )
        tensor_size = list(indices[tensor_indices[0]].get_size())
        indexed_size = [x_size[i] for i in range(len(indices))]

        expected_vals_size, inner_fn = index_output_size_and_inner_fn(
            x_size,
            indices,
            tensor_indices,
            tensor_size,
            indices_loaders,
            indexed_size,
            None,
            check=True,
        )
        values = lower_expand(values, expected_vals_size)
    scatter = Scatter(
        device=device,
        dtype=grad.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,
        output_indexer=inner_fn,
        scatter_mode="atomic_add",
    )

    buffer = ComputedBuffer(
        name=grad.data.data.name,
        layout=MutationLayoutSHOULDREMOVE(grad),
        data=scatter,
    )
    return buffer


def build_subgraph_module_buffer(args: Any, graph_module: Any) -> Any:
    """What a captured body produces, as something a kernel can hold.

    The body is run as a graph of its own rather than as part of the outer one,
    because a kernel is written against a body and not against the graph around
    it: what the body produces has to be something the kernel can be handed,
    which is not what a node in the outer graph is.

    The one operation the body is allowed to change something with is the one
    that adds a gradient into a position, and it is given its own lowering
    here -- so the body is lowered with that answer available rather than being
    refused for using it.
    """

    from ..ir import ComputedBuffer, FlexibleLayout, StorageBox
    from ..subgraph_lowering import PointwiseSubgraphLowering
    from tensorplay.utils._ordered_set import OrderedSet

    from . import zeros_and_scatter_lowering

    # This one we gotta keep lazy
    allowed = OrderedSet([tp.ops.omni.zeros_and_scatter.default])
    pw_subgraph = PointwiseSubgraphLowering(
        graph_module,
        root_graph_lowering=V.graph,
        allowed_mutations=allowed,
        additional_lowerings={
            tp.ops.omni.zeros_and_scatter.default: zeros_and_scatter_lowering
        },
    )
    with V.set_graph_handler(pw_subgraph):
        pw_subgraph.run(*args)

    def convert_output_node_to_buffer(output_buffer: Any) -> Any:
        if output_buffer is None:
            return None
        if isinstance(output_buffer, ComputedBuffer):
            return output_buffer
        if not isinstance(output_buffer, TensorBox):
            raise AssertionError(
                f"The output node for the attention subgraph must be a TensorBox, "
                f"but got: {type(output_buffer)}"
            )
        if not isinstance(output_buffer.data, StorageBox):
            raise AssertionError(
                f"The output node for the attention subgraph must be a StorageBox, "
                f"but got: {type(output_buffer.data)}"
            )
        device = output_buffer.data.get_device()
        if device is None:
            raise AssertionError("device must not be None for output buffer")
        subgraph_buffer = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=output_buffer.data.get_dtype(),
                size=output_buffer.data.get_size(),
            ),
            data=output_buffer.data.data,
        )
        return subgraph_buffer

    return tree_map(convert_output_node_to_buffer, pw_subgraph.graph_outputs)


def build_subgraph_buffer(args: Any, subgraph: Any) -> Any:
    """The same, for a body that is already a graph of its own."""

    return build_subgraph_module_buffer(args, subgraph.graph_module)


def realize_captures_for_cutedsl(buffers: Any) -> Any:
    """Write down the values a kernel was handed that are not already written.

    A kernel is handed physical values, so a captured value that was a
    computation has to be somewhere before it can be handed over.  A value that
    was already a graph input is left as it is: copying it would be a copy of
    the whole value to satisfy a property it already has.

    A captured view is the case that needs a name of its own.  Two views of the
    same value are two arguments with different shapes, offsets and strides, and
    a kernel told about one of them and given the other would read the wrong
    bytes.  So each is given a name, and the name stands for the view rather
    than for what it is a view of.
    """

    from ..ir import (
        ExternKernel,
        FixedLayout,
        InputBuffer,
        ReinterpretView,
        StorageBox,
    )

    view_captures: dict = {}

    def _add_alignment_check_for_input(input_buffer: Any) -> None:
        # A captured value can be read several at a time, so it has to be
        # aligned whichever way of reading it is used -- and it was not a direct
        # argument of the kernel when the arguments that need checking were
        # chosen, so it is added here rather than there.
        name = input_buffer.get_name()
        graph_input_names = list(getattr(V.graph, "graph_input_names", ()) or ())
        if name in graph_input_names:
            idx = graph_input_names.index(name)
            inputs_to_check = list(V.graph.inputs_to_check or ())
            if idx not in inputs_to_check:
                V.graph.inputs_to_check = [*inputs_to_check, idx]

    def _realize(x: Any) -> Any:
        if x is None or isinstance(x, sympy.Expr):
            return x
        realized = ExternKernel.realize_input(x)
        if isinstance(realized, StorageBox) and realized.is_input_buffer():
            realized = realized.data
        if isinstance(realized, ReinterpretView):
            layout = realized.get_layout()
            capture_index = len(V.graph._cutedsl_capture_nodes) + len(view_captures)
            name = f"cutedsl_capture{capture_index}"
            view_captures[name] = realized
            # Each captured view gets a name of its own, so two views of the
            # same value do not collapse into one argument.
            return InputBuffer(
                name=name,
                layout=FixedLayout(
                    layout.device,
                    layout.dtype,
                    layout.size,
                    layout.stride,
                    is_pinned=layout.is_pinned,
                ),
            )
        if isinstance(realized, InputBuffer):
            _add_alignment_check_for_input(realized)
            return realized
        return ExternKernel.copy_input(realized)

    buffers = tree_map(_realize, buffers)
    freeze_irnodes(buffers)

    for buf in (tree_map_only(IRNode, lambda x: x, buffers) if buffers else []):
        if isinstance(buf, IRNode) and (name := buf.maybe_get_name()):
            V.graph._cutedsl_capture_nodes[name] = buf
    # The views are kept as they were, because the call site reads them through
    # the view rather than through the value.
    V.graph._cutedsl_capture_nodes.update(view_captures)

    return buffers
