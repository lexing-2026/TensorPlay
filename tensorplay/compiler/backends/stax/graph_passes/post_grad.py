from typing import Any

import tensorplay as tp
from tensorplay.graph import Graph, Node
from tensorplay.graph.experimental.symbolic_shapes import statically_known_true, sym_eq
from tensorplay.graph.experimental.sympy_functions import OrderedSet
from tensorplay._decomp import register_decomposition
from tensorplay.primitives.common import is_integer_dtype

from .. import config
from ..fx_utils import get_fake_args_kwargs, get_node_storage
from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    register_lowering_pattern as _register_lowering_pattern,
)


tp_ops = tp.ops.tp
prims = tp.ops.prims

#: The pattern tables the passes draw from, applied in order: a match found by
#: an earlier table is taken, so a table written later has the first say on
#: which of two ways of writing the same computation is used.
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def register_lowering_pattern(
    pattern,
    extra_check=lambda match: True,
    pass_number=1,
    *,
    output_metadata_ignores_input_storage: bool = True,
    output_metadata_is_input: int | str | None = None,
    output_metadata_fn=None,
):
    """Register a replacement of one way of writing a computation by another."""

    return _register_lowering_pattern(
        pattern,
        extra_check,
        pass_dict=pass_patterns[pass_number],
        output_metadata_ignores_input_storage=output_metadata_ignores_input_storage,
        output_metadata_is_input=output_metadata_is_input,
        output_metadata_fn=output_metadata_fn,
    )


def same_meta(node1: Node, node2: Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and isinstance(val1, tp.Tensor)
        and isinstance(val2, tp.Tensor)
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != tp.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
        # Check conjugate and negative bits - a clone that resolves these is not a no-op
        and val1.is_conj() == val2.is_conj()
        and val1.is_neg() == val2.is_neg()
    )




noop_registry: dict[Any, Any] = {}


def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        register_decomposition(targets, registry=noop_registry)(
            (cond, nop_arg)  # type: ignore[arg-type]
        )
        return cond

    return register_fun


def _needs_spmd_graph_preservation() -> bool:
    """Check if SPMD graph preservation is needed for distributed overlap."""
    return (
        config.aten_distributed_optimizations.enable_overlap_scheduling
        or config.reorder_for_compute_comm_overlap
    )


@register_noop_decomp(tp_ops.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if _needs_spmd_graph_preservation():
        # Keep no-op slices so all ranks produce identical FX graphs (SPMD)
        # with matching op counts and runtime estimations.
        return False
    if start is None or end is None:
        return False

    slice_dim_size = self.shape[dim]
    if (
        statically_known_true(sym_eq(start, 0))
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_dim_size)
        )
        and statically_known_true(sym_eq(step, 1))
    ):
        return True
    return False


@register_noop_decomp(tp_ops.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if start is None:
        start = 0
    if end is None:
        end = 2**63 - 1
    slice_scatter_dim_size = self.shape[dim]
    if (
        self.shape == src.shape
        and start == 0
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_scatter_dim_size)
        )
        and step == 1
    ):
        return True
    return False


@register_noop_decomp(tp_ops.repeat)
def repeat_noop(self, repeats):
    return all(r == 1 for r in repeats)


@register_noop_decomp(tp_ops.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    if _needs_spmd_graph_preservation():
        # Keep no-op pads so all ranks produce identical FX graphs (SPMD)
        # with matching op counts and runtime estimations.
        return False
    return all(p == 0 for p in padding)


# A change of type and a change of device are not registered here.  Both are
# plain functions here rather than dispatched operations, so a node naming one
# is a call to that function and not a call that a decomposition stands in
# front of -- there is nothing wrapped around it to take off, and a
# registration would have nothing to attach to.


@register_noop_decomp([tp_ops.ceil, tp_ops.floor, tp_ops.round, tp_ops.trunc])
def int_noop(x):
    return is_integer_dtype(x.dtype)


@register_noop_decomp([tp_ops.pow])
def pow_noop(a, b):
    return isinstance(b, int) and b == 1


@register_noop_decomp([tp_ops.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    return len(inputs) == 1


@register_noop_decomp(tp_ops.view.default)
def view_default_noop(arg, size):
    return statically_known_true(sym_eq(arg.shape, tuple(size)))


@register_noop_decomp(tp_ops.view.dtype)
def view_dtype_noop(arg, dtype):
    return arg.dtype == dtype


# Note, we also always have a check for identical metadata, which is why these
# are safe
@register_noop_decomp([tp_ops.copy], nop_arg=1)
@register_noop_decomp([tp_ops.alias, tp_ops.clone])
def true_noop(*args, **kwargs):
    return True


def remove_noop_ops(graph: Graph):
    """
    Removes both operations that are essentially tp_ops.clone and operations that are essentially tp_ops.alias from the graph.
    """
    inputs = OrderedSet[Node]()
    input_storages = OrderedSet[int | None]()
    output_storages = OrderedSet[int | None]()

    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    output_node = next(iter(reversed(graph.nodes)))
    if output_node.op != "output":
        raise AssertionError(f"expected output node, got {output_node.op}")
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # nested subgraphs can have singleton outputs
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, Node):
            output_storages.add(get_node_storage(out))

    for node in graph.nodes:
        if node.target in noop_registry:
            cond, src_index = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, Node):
                continue

            if node.target is tp_ops.copy.default:
                dst = node.args[0]
                if (
                    isinstance(dst, Node)
                    and dst.op == "call_function"
                    and dst.kwargs.get("pin_memory") is True
                ):
                    continue

            # Don't introduce new aliasing between inputs and outputs.
            # See fx_passes/README.md for a discussion of why this is
            # necessary.
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue

            # Even if input and outputs are expected to alias,
            # don't make "node is src" True
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)

