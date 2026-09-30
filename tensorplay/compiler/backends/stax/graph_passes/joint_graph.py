# mypy: allow-untyped-defs
import functools
import itertools
import logging
import operator
import typing
from collections import Counter
from collections.abc import Sequence
from typing import Any

import tensorplay as tp
from tensorplay.utils import _pytree as pytree
from tensorplay.graph import GraphModule, Node
from tensorplay.graph import map_arg
from tensorplay.utils._dispatch import _disable_current_modes
from tensorplay.graph.experimental.symbolic_shapes import (
    statically_known_true,
    sym_eq,
)
from tensorplay.graph.experimental.sympy_functions import OrderedSet

from .. import config
from ..custom_graph_pass import get_custom_graph_passes
from ..pattern_matcher import (
    Arg,
    CallFunction,
    init_once_fakemode,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternMatcherPass as PatternMatcherPassBase,
    register_graph_pattern,
    stable_topological_sort,
)
from ..constant_folding import ConstantFolder
from .dedupe_symint_uses import _SymHashingDict
from ..utils import counters, get_gpu_type
from ..runtime.storage_ref import StorageWeakRef
from ..virtualized import V
from .decompose_mem_bound_mm import check_device
from .replace_random import replace_random_passes


PatternMatcherPass = functools.partial(
    PatternMatcherPassBase, subsystem="joint_graph_passes"
)

log = logging.getLogger(__name__)

#: The key under which a graph records that a matmul standing in for a grouped
#: matmul is to be left as it is.  Nothing here sets it, so the check that
#: reads it is the answer "no" the key not being there already gives.
_PRESERVE_FLEX_GEMM_GEMM_OP = "preserve_flex_gemm_gemm_op"
early_patterns = PatternMatcherPass()
patterns = PatternMatcherPass()
tp_ops = tp.ops.tp
prims = tp.ops.prims

pass_patterns = [
    patterns,
    PatternMatcherPass(),
]


def _is_lossless_fp_widening_cast(
    src_dtype: tp.dtype, dst_dtype: tp.dtype
) -> bool:
    if src_dtype == dst_dtype:
        return True

    if not (src_dtype.is_floating_point and dst_dtype.is_floating_point):
        return False

    src_info = tp.finfo(src_dtype)
    dst_info = tp.finfo(dst_dtype)

    # A floating-point cast is only pointless if the first conversion cannot
    # discard precision or range from the source values.
    return (
        dst_info.eps <= src_info.eps
        and dst_info.max >= src_info.max
        and dst_info.tiny <= src_info.tiny
    )


@init_once_fakemode
def lazy_init(input_device: tp.device | None = None):
    from .fuse_attention import _sfdp_init
    from .misc_patterns import _misc_patterns_init
    from .pad_mm import _pad_mm_init

    _pad_mm_init(input_device)
    _sfdp_init(input_device)
    _misc_patterns_init(input_device)


def remove_no_ops(
    gm: GraphModule,
    zeros: OrderedSet[Node],
    ones: OrderedSet[Node],
):
    """
    Removes operations that are essentially no-ops e.g. (+ 0, - 0, * 1, / 1)
    """
    with _disable_current_modes():
        graph = gm.graph

        def fake_tensors_eq(t1, t2, fields=("shape", "dtype", "device")):
            if any(not isinstance(t, tp.Tensor) for t in (t1, t2)):
                return False
            for field in fields:
                v1 = getattr(t1, field)
                v2 = getattr(t2, field)
                if field == "shape":
                    # Shapes may contain unbacked SymInts; tuple `!=` would
                    # force a guard. Conservatively treat unknown as "not equal".
                    if not V.graph.sizevars.guard_or_false(sym_eq(v1, v2)):
                        return False
                elif v1 != v2:
                    return False
            return True

        def is_mutated(n):
            """Check if a node is mutated by any in-place operation."""
            for user in n.users:
                if user.op != "call_function" or not hasattr(user.target, "_schema"):
                    continue
                for i, arg in enumerate(user.args):
                    if arg is n:
                        schema_arg = user.target._schema.arguments[i]
                        if schema_arg.alias_info and schema_arg.alias_info.is_write:
                            return True
            return False

        def isScalarValue(arg):
            return isinstance(arg, (int, float))

        def replace_no_op(node, replace_input_index):
            replacement = node.args[replace_input_index]

            # A call can carry a plain value where its description says it
            # carries only tensors, so whether every argument is a node is
            # asked rather than assumed.  TODO - decompose/type promote to
            # avoid this
            if not all(isinstance(arg, Node) for arg in node.args):
                if all(isScalarValue(arg) for arg in node.args) or not isinstance(
                    replacement, Node
                ):
                    return

            # Don't replace if the replacement value is mutated in-place.
            # The original node acts as an implicit copy; removing it would
            # cause users to observe the post-mutation value instead.
            if is_mutated(replacement):
                return

            if not fake_tensors_eq(node.meta["val"], replacement.meta["val"]):
                if fake_tensors_eq(
                    node.meta["val"],
                    replacement.meta["val"],
                    ("shape", "device"),
                ):
                    with graph.inserting_after(node):
                        replacement = graph.call_function(
                            prims.convert_element_type,
                            args=(replacement, node.meta["val"].dtype),
                        )
                else:
                    return

            node.replace_all_uses_with(replacement)
            replacement.meta.update(node.meta)
            graph.erase_node(node)

        for node in graph.find_nodes(op="call_function", target=tp_ops.add.Tensor):
            if len(node.args) == 2:
                if (
                    not any(
                        e in zeros or (isScalarValue(e) and e == 0) for e in node.args
                    )
                    or node.kwargs.get("alpha", 1) != 1
                ):
                    continue

                replace_index = (
                    1
                    if node.args[0] in zeros
                    or (isScalarValue(node.args[0]) and node.args[0] == 0)
                    else 0
                )
                replacement = node.args[replace_index]
                if isinstance(replacement, Node):
                    val = replacement.meta.get("val")
                    if isinstance(val, tp.Tensor) and val.is_conj():
                        continue
                replace_no_op(node, replace_index)

        for node in graph.find_nodes(op="call_function", target=tp_ops.sub.Tensor):
            if len(node.args) == 2:
                if (
                    not (
                        node.args[1] in zeros
                        or (isScalarValue(node.args[1]) and node.args[1] == 0)
                    )
                    or node.kwargs.get("alpha", 1) != 1
                ):
                    continue

                replace_no_op(node, 0)

        for node in graph.find_nodes(op="call_function", target=tp_ops.mul.Tensor):
            if len(node.args) == 2:
                if not any(
                    e in ones or (isScalarValue(e) and e == 1) for e in node.args
                ):
                    continue

                replace_input_index = (
                    1
                    if node.args[0] in ones
                    or (isScalarValue(node.args[0]) and node.args[0] == 1)
                    else 0
                )
                replace_no_op(node, replace_input_index)

        for node in graph.find_nodes(op="call_function", target=tp_ops.div.Tensor):
            if len(node.args) == 2 and (
                node.args[1] in ones
                or (isScalarValue(node.args[1]) and node.args[1] == 1)
            ):
                replace_no_op(node, 0)

        # meta tensors returned from the graph have no data and can be replaced with empty_strided
        for output_node in graph.find_nodes(op="output"):
            had_meta_return = False

            def visit(n):
                nonlocal had_meta_return
                val = n.meta.get("val")
                if isinstance(val, tp.Tensor) and val.device.type == "meta":
                    with graph.inserting_before(output_node):
                        # size/stride may be symbolic under dynamic shapes;
                        # materialize them so we never pass raw SymInts as args.
                        # Use materialize_symints (roots backed sizes on input
                        # placeholders) rather than create_size_node(n, d), which
                        # would query `n` and pin it alive, blocking the
                        # eliminate_dead_code() that removes this meta tensor.
                        size = graph.materialize_symints(val.size())
                        stride = graph.materialize_symints(val.stride())
                        n.replace_all_uses_with(
                            graph.call_function(
                                tp_ops.empty_strided.default,
                                args=(size, stride),
                                kwargs={"dtype": val.dtype, "device": val.device},
                            )
                        )
                    had_meta_return = True

            map_arg(output_node.args, visit)
            if had_meta_return:
                graph.eliminate_dead_code()


def remove_redundant_views(gm: GraphModule):
    """
    Removes redundant views by reusing existing ones.
    """
    with _disable_current_modes():
        # A dictionary mapping a tensor to all aliased views.
        views: dict[Node, dict[tp.dtype, Node]] = {}
        graph = gm.graph

        for node in graph.find_nodes(
            op="call_function", target=tp_ops.view.dtype
        ):
            src = node.args[0]
            to_type = node.args[1]
            existing_views = views.get(src)
            is_needed = True

            if existing_views:
                # Replace the view with an existing view if available.
                alias = existing_views.get(to_type)
                if alias:
                    is_needed = False
                    node.replace_all_uses_with(alias)
                    alias.meta.update(node.meta)
                    graph.erase_node(node)
            else:
                from_type = src.meta["val"].dtype
                existing_views = {from_type: src}
                views[src] = existing_views

            if is_needed:
                # Save the new alias but do not replace existing one.
                existing_views.setdefault(to_type, node)
                views[node] = existing_views

        # Clean up unused views.
        while True:
            unused_views = [alias for alias in views if not alias.users]
            if len(unused_views) == 0:
                break
            for unused in unused_views:
                views.pop(unused)
                if unused.op == "placeholder":
                    # Placeholders are graph inputs; erasing one would silently
                    # shrink the compiled function's arity while callers keep
                    # passing the original argument count.
                    continue
                graph.erase_node(unused)


class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a uniform value
    with a tensor constructor call: tp_ops.full([shape], value, ...)
    """

    def __init__(self, gm, skip_constructors=False) -> None:
        super().__init__(gm, skip_constructors)
        self.node_storages_ptrs: dict[Node, int] = {}
        self.constant_data_ptrs: dict[Node, StorageWeakRef] = {}
        # we may constant fold a tensor which in the graph has a sym size
        # see: [constant folding refining of symints]
        self.node_replacements_shapes: dict[Node, list[int]] = {}

        # initialize symint -> node mapping so that we can
        # use symint nodes in full constructors
        self.symint_nodes = _SymHashingDict()
        for n in self.module.graph.nodes:  # type: ignore[union-attr]
            if "val" in n.meta and isinstance(n.meta["val"], tp.SymInt):
                if n.meta["val"] not in self.symint_nodes:
                    self.symint_nodes[n.meta["val"]] = n

        self.view_op_packets = [
            tp_ops.squeeze,
            tp_ops.unsqueeze,
            tp_ops.alias,
            tp_ops.view,
            tp_ops.slice,
            tp_ops.t,
            prims.broadcast_in_dim,
            tp_ops.expand,
            tp_ops.as_strided,
            tp_ops.permute,
        ]

        self.indexing_op_packets = OrderedSet(
            [
                tp_ops.slice,
            ]
        )

        self._add_peephole_patterns()

    def _add_peephole_patterns(self) -> None:
        """
        Add peephole patterns for nodes where we can infer constant value even if some inputs
        of the node are unknown.
        """
        for op in itertools.chain(
            self.module.graph.find_nodes(  # type: ignore[operator, union-attr]
                op="call_function", target=tp_ops.mul.Tensor
            ),
            self.module.graph.find_nodes(  # type: ignore[operator, union-attr]
                op="call_function", target=tp_ops.mul.Scalar
            ),
        ):
            tensor_val = op.meta.get("val", None)
            if not isinstance(tensor_val, tp.Tensor):
                continue

            def is_zero_int(arg: object) -> bool:
                return isinstance(arg, int) and arg == 0

            if not any(is_zero_int(a) for a in op.args):
                continue

            # x * 0 is only uniformly 0 for integer/bool dtypes. For floating
            # point (and complex) dtypes nan * 0 == nan and (+/-inf) * 0 == nan,
            # so folding x * 0 -> 0 would incorrectly drop NaN/Inf when x is not
            # known to be finite.
            if tensor_val.dtype.is_floating_point or tensor_val.dtype.is_complex:
                continue

            t = tp.full(
                [1],  # shape
                0,  # value
                dtype=tensor_val.dtype,
                device=tensor_val.device,
                pin_memory=False,
            )
            self.add_node_replacement(op, t)

    def _support_dynamic_shape(self):
        return True

    def insertable_tensor_check(self, t: tp.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: Node, tensor: tp.Tensor) -> None:
        self.node_replacements[node] = tensor.flatten()[0].item()
        self.node_replacements_shapes[node] = node.meta["val"].shape
        self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())

    def insert_placerholder_values(self, env: dict[Node, Any]) -> None:
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            if "val" in n.meta and isinstance(n.meta["val"], tp.SymInt):
                env[n] = n.meta["val"]
            else:
                env[n] = self.unknown_value

    def _deduce_value(self, node: Node):
        # deduce value for full-like nodes
        # 1. for constructors, substitute value is a tensor of size [1]
        # 2. for view ops/indexing, substitute value is the same as the input
        # 3. for pointwise ops, run node to get the substitute value
        # 4. deal with some special ops
        # otherwise, stop deduce value and return unknown value

        # TODO: cat, more indexing
        # TODO - do on cpu to avoid syncs

        # single-elem attrs
        if node.op == "get_attr" or (
            node.op == "call_function"
            and node.target is tp_ops.lift_fresh_copy.default
        ):
            out = super(ConstantFolder, self).run_node(node)
            if isinstance(out, tp.Tensor) and out.numel() == 1:
                return out

        # handle device_put op
        if node.target == prims.device_put.default:
            return super(ConstantFolder, self).run_node(node)

        # constructors ops
        if (
            node.op == "call_function"
            and node.target is tp_ops.full.default
            and len(node.args) == 2
        ):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            value = args[1]
            # Don't specialize symbolic value.
            if not isinstance(value, (tp.SymInt, tp.SymFloat, tp.SymBool)):
                new_args = [[1], value]
                return tp_ops.full.default(*new_args, **node.kwargs)

        # handle before view ops because this changes value
        if node.target is tp_ops.view.dtype:
            (input_tensor, output_dtype), kwargs = self.fetch_args_kwargs_from_env(node)
            # view.dtype with different element sizes changes element count
            # (e.g., complex64 [1+0j] viewed as float32 becomes [1.0, 0.0]),
            # making uniform values non-uniform. Also crashes on 0-d tensors.
            if input_tensor.element_size() != output_dtype.itemsize:
                return self.unknown_value
            return super(ConstantFolder, self).run_node(node)

        # view ops, return input tensor, the first argument
        if hasattr(node.target, "overloadpacket") and (
            node.target.overloadpacket in self.view_op_packets
            or node.target.overloadpacket in self.indexing_op_packets
        ):
            if not isinstance(node.args[0], Node):
                raise AssertionError(f"expected fx.Node, got {type(node.args[0])}")
            return self.env[node.args[0]]

        # we don't want to return unknown value for symints so that we can
        # still constant fold through their use in constructors or views
        # if we see them in a pointwise node (e.g., tensor * symint)
        # we will bail
        if "val" in node.meta and isinstance(node.meta["val"], tp.SymInt):
            return node.meta["val"]

        # pointwise ops
        if isinstance(node.target, tp._ops.OpOverload) and (
            "pointwise" in node.target.tags
            or node.target is tp_ops.scalar_tensor.default
        ):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

            if any(isinstance(inp, tp.SymInt) for inp in flattened_inputs):
                return self.unknown_value

            # we run the ops with dim 1, so remove memory_format to avoid error
            kwargs = dict(kwargs)
            kwargs.pop("memory_format", None)

            return node.target(*args, **kwargs)

        return self.unknown_value


def _has_self_referential_shape(
    shapes: list[int | Node], node: Node
) -> bool:
    """
    Check if any shape in `shapes` depends on `node`.

    This is used to detect cycles when constant_fold_uniform_value creates a
    replacement full() node whose shape includes a sym_size computed from the
    original tensor being replaced.

    Checks direct args only - shape nodes typically come from sym_size(tensor, dim)
    where tensor is a direct arg.
    """
    for shape_node in shapes:
        if isinstance(shape_node, Node):
            if node in shape_node.args:
                return True
    return False


def constant_fold_uniform_value(gm: GraphModule):
    """Runs constant folding and replaces constants which can be constructed with a single `full` call. Calls into remove_no_ops."""
    with _disable_current_modes():
        tp_ops = tp.ops.tp

        # Constant folding can leak memory, especially with repeated compilation, so we are only going to
        # remove constants which can be replaced with a constructor.
        cf = UniformValueConstantFolder(gm)
        cf.run()

        node_replacements = cf.node_replacements

        # note: [constant folding refining of symints]
        # constant folding will partially evaluate a graph such that values which have dependencies which
        # are entirely known at compile time may also become compile time constants. in some cases,
        # this will include symints which we had not yet previously deduced are guaranteed a
        # constant value and is then deduced in constant folding. an example is:
        # unbacked_symint_eq_11 = tp.full((), 11).item()
        # tp.full((unbacked_symint_eq_11,), 0)
        node_replacements_shapes = cf.node_replacements_shapes

        graph = gm.graph

        zeros = OrderedSet[Any]()
        ones = OrderedSet[Any]()

        # Got failures in `test_is_set_to_cuda` if we change aliasing on constants,
        # so just constant-ify if a Tensor is unaliased
        constant_data_ptr_count: typing.Counter[StorageWeakRef] = Counter()

        for node in cf.node_replacements:
            constant_data_ptr_count[cf.constant_data_ptrs[node]] += 1

        for node, value in node_replacements.items():
            # we don't have a functional way right now of instantiating a non-contiguous tensor with full/zeros/ones right now
            # hasn't shown up to be important yet
            if "val" not in node.meta:
                # This can only happen in the ahead-of-time export
                continue

            fake_tensor = node.meta["val"]
            if not fake_tensor.is_contiguous(memory_format=tp.contiguous_format):
                continue

            # TODO - not sure about lossy uint->python value->uint conversions
            if fake_tensor.dtype in (
                tp.uint8,
                tp.uint16,
                tp.uint32,
                tp.uint64,
            ):
                continue

            if constant_data_ptr_count[cf.constant_data_ptrs[node]] > 1:
                continue

            with graph.inserting_after(node):
                # the conversion from tensor and back to value can be lossy, just use the original full ctor value
                if (
                    node.op == "call_function"
                    and node.target is tp_ops.full.default
                    and len(node.args) == 2
                ):
                    value = node.args[1]

                # refines symints, see [constant folding refining of symints] above
                for runtime_size, compile_time_size in zip(
                    node_replacements_shapes[node], fake_tensor.shape
                ):
                    tp._check(runtime_size == compile_time_size)

                # replace SymInt as Node before creating a new full node
                # e.g. (1, s0) -> (1, arg0_1)
                node_shape = node_replacements_shapes[node]
                if not all(
                    not isinstance(s, tp.SymInt) or s in cf.symint_nodes
                    for s in node_shape
                ):
                    continue

                shapes = [
                    cf.symint_nodes[s] if isinstance(s, tp.SymInt) else s
                    for s in node_replacements_shapes[node]
                ]

                # Check if any shape depends on a symint that was computed from
                # the node being replaced - this would create a cycle
                if _has_self_referential_shape(shapes, node):
                    continue

                # zeros and ones just get traced into full, so we insert those
                new_node = graph.call_function(
                    tp_ops.full.default,
                    args=(shapes, value),
                    kwargs={
                        "dtype": fake_tensor.dtype,
                        "layout": tp.strided,
                        "device": fake_tensor.device,
                        "pin_memory": node.kwargs.get("pin_memory", False),
                    },
                )

                new_node.meta.update(node.meta)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)

                if value == 0:
                    zeros.add(new_node)
                elif value == 1:
                    ones.add(new_node)

        remove_no_ops(gm, zeros, ones)
        remove_redundant_views(gm)


def canonicalize_quant_mapping(gm: GraphModule):
    """


    tp.ops.higher_order.invoke_quant_packed(repeated_subgraph0, 'quant_invoke_0_0', (arg0_1, arg1_1));
    ->
    tp.ops.higher_order.invoke_quant(repeated_subgraph0, arg0_1, arg1_1, scheme = 'nf4');
    """
    graph = gm.graph
    invoke_quant_invocations = graph.find_nodes(
        op="call_function", target=tp.ops.higher_order.invoke_quant_packed
    )
    for invoke_quant in invoke_quant_invocations:
        kwargs = dict(invoke_quant.kwargs)

        quant_options_node = kwargs.pop("quant_options", None)
        if quant_options_node is not None:
            if not isinstance(quant_options_node, Node):
                raise AssertionError(
                    f"expected fx.Node, got {type(quant_options_node)}"
                )
            quant_options = tp._higher_order_ops.InvokeQuant(
                *invoke_quant.kwargs["quant_options"].args,
                **invoke_quant.kwargs["quant_options"].kwargs,
            )
        else:
            quant_options = tp._higher_order_ops.InvokeQuant()

        subgraph, *args = invoke_quant.args
        with gm.graph.inserting_before(invoke_quant):
            invoke_quant_replacement = graph.call_function(
                tp._higher_order_ops.invoke_quant,
                (subgraph, *args),
                # pyrefly: ignore [bad-argument-type]
                kwargs,
            )
            invoke_quant_replacement.meta.update(subgraph.meta)
            invoke_quant_replacement.meta["quant_options"] = quant_options

            invoke_quant.replace_all_uses_with(invoke_quant_replacement)
            graph.erase_node(invoke_quant)

            if quant_options_node and len(quant_options_node.users) == 0:
                graph.erase_node(quant_options_node)

            first_user = next(iter(invoke_quant_replacement.users))

            if (
                len(invoke_quant_replacement.users) == 1
                and len(subgraph.users) == 1
                and first_user.target is operator.getitem
                and first_user.args[1] == 0
            ):
                subgraph_graph = getattr(gm, subgraph.target)
                output_node = subgraph_graph.output_node
                if not (
                    isinstance(output_node.args[0], (list, tuple))
                    and len(output_node.args[0]) == 1
                ):
                    raise AssertionError(
                        "expected subgraph output to be a single-element list or tuple"
                    )

                unpacked_output = output_node.args[0][0]
                # pyrefly: ignore [bad-argument-type]
                output_node.args = (unpacked_output,)
                if "val" in output_node.meta:
                    output_node.meta["val"] = output_node.meta["val"][0]
                subgraph_graph.recompile()

                invoke_quant_replacement.meta.update(first_user.meta)
                first_user.replace_all_uses_with(invoke_quant_replacement)
                graph.erase_node(first_user)


def canonicalize_tp_ir_passes(gm: GraphModule):
    """
    Canonicalization passes that will run immediately after aot autograd
    tracing. Thsis must be run before all other graph passes.
    """
    canonicalize_quant_mapping(gm)


def joint_graph_passes(
    graph: GraphModule,
    input_device: tp.device | None = None,
):
    """
    Run FX transformations on the joint forwards+backwards graph.
    """
    GraphTransformObserver = functools.partial(
        GraphTransformObserver,
        subsystem="joint_graph_passes",
    )

    lazy_init(input_device)
    count = 0

    # must occur before other passes
    canonicalize_tp_ir_passes(graph)

    for joint_custom_pre_pass in get_custom_graph_passes(config.joint_custom_pre_pass):
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            joint_custom_pre_pass
        )
        count += 1

    from .post_grad import remove_noop_ops

    GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

    if config.joint_graph_constant_folding:
        GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
            constant_fold_uniform_value
        )

    if config.pattern_matcher:
        count += early_patterns.apply(graph.graph)

    # Make sure AutoChunker happens before pad_mm so we don't need
    # to handle padding when searching for chunking patterns.
    if config.auto_chunker.enable:
        from .auto_chunker import CantChunk, chunk

        try:
            graph = chunk(graph)
        except CantChunk:
            auto_chunker_log = tp.getArtifactLogger(
                __name__, "auto_chunker"
            )
            auto_chunker_log.debug("AutoChunker fail.", exc_info=True)

    if config.pattern_matcher:
        for i, patterns in enumerate(pass_patterns):
            maybe_count = GraphTransformObserver(
                graph, f"pass_pattern_{i}"
            ).apply_graph_pass(patterns.apply)
            count += maybe_count if maybe_count is not None else 0

    if not config.fallback_random:
        # not trying into the bisector because decomps may have already affected rng reproducibility
        # we'll instead explicitly turn off the config
        count += replace_random_passes(graph)

    for joint_custom_post_pass in get_custom_graph_passes(
        config.joint_custom_post_pass
    ):
        GraphTransformObserver(graph, "joint_custom_post_pass").apply_graph_pass(
            joint_custom_post_pass
        )
        count += 1

    if count:
        stable_topological_sort(graph.graph)
        graph.graph.lint()
        graph.recompile()
    return graph


@register_graph_pattern(
    CallFunction(
        prims.iota,
        KeywordArg("length"),
        start=KeywordArg("start"),
        step=KeywordArg("step"),
        dtype=KeywordArg("dtype"),
        device=KeywordArg("device"),
        requires_grad=KeywordArg("requires_grad"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
)
def fix_iota_device(match: Match, length, start, step, dtype, device, requires_grad):
    """
    Eager supports:

        tp_ops.index(cuda_tensor, tp.arange(..., device="cpu"))

    But this results in an implicit host-device-copy and breaks cudagraphs.
    Rewrite the arange to use CUDA.
    """
    (node,) = match.nodes
    user_devices = OrderedSet[tp.device]()
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target in (tp_ops.index.Tensor, tp_ops.index_put.default)
            and hasattr(user.meta.get("val"), "device")
        ):
            user_devices.add(user.meta["val"].device)  # type: ignore[union-attr]
        else:
            return  # bail out

    if len(user_devices) == 1 and "val" in node.meta:
        (user_device,) = user_devices
        if device.type != user_device.type:
            repl = match.graph.call_function(
                prims.iota,
                (length,),
                {
                    "start": start,
                    "step": step,
                    "dtype": dtype,
                    "device": user_device,
                    "requires_grad": requires_grad,
                },
            )
            repl.meta.update(node.meta)
            repl.meta["val"] = repl.meta["val"].to(user_device)
            node.replace_all_uses_with(repl)
            match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        prims.convert_element_type,
        CallFunction(
            prims.convert_element_type,
            KeywordArg("arg"),
            KeywordArg("dtype1"),
        ),
        KeywordArg("dtype2"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
)
def pointless_convert(match: Match, arg, dtype1: tp.dtype, dtype2: tp.dtype):
    """Remove chain of dtype conversions often created by AMP"""
    graph = match.graph
    node = match.output_node()
    allowed = tp.float16, tp.bfloat16, tp.float32, tp.float64
    arg_val = arg.meta.get("val", None)
    if not isinstance(arg_val, tp.Tensor):
        return

    if arg_val.dtype in allowed and dtype1 in allowed and dtype2 in allowed:
        # A narrower intermediate can be worth materializing before upcasting.
        if dtype1.itemsize < dtype2.itemsize:
            return
        if config.emulate_precision_casts and not _is_lossless_fp_widening_cast(
            arg_val.dtype, dtype1
        ):
            return
        repl = graph.call_function(
            prims.convert_element_type, (arg, dtype2)
        )
        repl.meta.update(node.meta)
        node.replace_all_uses_with(repl)
        match.erase_nodes()


def definitely_equal(
    old_sizes: Sequence[tp.SymInt | int],
    new_sizes: Sequence[tp.SymInt | Node | int],
) -> bool:
    """
    Leverage guard_or_true/false to compare if two lists of int/symint are equal.
    Useful to compare sizes, strides etc.

    Can handle -1 in new_sizes which happens in the size arguments of a
    view op. old_sizes is supposed to be the tensor shape and should not
    contain -1.

    new_sizes can contains fx.Node when dynamic shape is enabled. In that
    case new_sizes[i].meta['val'] contains the real tp.SymInt.
    """

    num_neg1 = 0

    if len(old_sizes) != len(new_sizes):
        return False

    for lhs_item, rhs_item in zip(old_sizes, new_sizes):
        if isinstance(rhs_item, Node):
            rhs_item = rhs_item.meta["val"]

        if not isinstance(lhs_item, (int, tp.SymInt)):
            raise AssertionError(type(lhs_item))
        if not isinstance(rhs_item, (int, tp.SymInt)):
            raise AssertionError(type(rhs_item))

        # It still makes sense to call guard_or_true/false since lhs_item
        # rhs_item are tp.SymInt rather than sympy expressions when
        # dynamic shape is enabled.
        if V.graph.sizevars.guard_or_false(lhs_item == rhs_item):
            continue

        if V.graph.sizevars.guard_or_true(rhs_item != -1):
            return False

        num_neg1 += 1

        if num_neg1 > 1:
            return False
    return True


@register_graph_pattern(
    CallFunction(tp_ops.view.default, KeywordArg("arg"), KeywordArg("size")),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=early_patterns,
)
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""
    node = match.output_node()
    arg_size = list(node.args[0].meta["val"].shape)  # type: ignore[union-attr]
    if definitely_equal(arg_size, size):
        node.replace_all_uses_with(node.args[0])  # type: ignore[arg-type]
        match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        tp_ops.view.default,
        CallFunction(tp_ops.view.default, KeywordArg("arg"), KeywordArg("size1")),
        KeywordArg("size2"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=early_patterns,
)
def pointless_view_pair(match: Match, arg, size1, size2):
    """
    Remove a pair of views that are pointless.
    """
    node = match.output_node()
    arg_size = list(arg.meta["val"].shape)
    if definitely_equal(arg_size, size2):
        node.replace_all_uses_with(arg)
        match.erase_nodes()
        counters["tp"]["removed_pointless_view_pair"] += 1


@register_graph_pattern(
    CallFunction(
        tp_ops.permute.default,
        CallFunction(tp_ops.permute.default, KeywordArg("arg"), KeywordArg("perm1")),
        KeywordArg("perm2"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=early_patterns,
)
def pointless_permute_pair(match: Match, arg, perm1, perm2):
    rank = len(perm1)
    if len(perm2) != rank:
        raise AssertionError(f"expected len(perm2) == {rank}, got {len(perm2)}")

    for i in range(rank):
        if perm1[perm2[i]] != i:
            return  # bail out
    node = match.output_node()
    node.replace_all_uses_with(arg)
    match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        tp_ops.bmm,
        Arg(),
        Arg(),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
)
def bmm_to_mm(match: Match, mat1: Node, mat2: Node):
    """Convert bmm to mm when batch size is 1"""
    # See Note [Preserving FlexGEMM body GEMMs].
    if match.output_node().meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP):
        return

    def repl(a, b):
        return tp.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)

    if (
        check_device(mat1.meta["val"], mat2.meta["val"], get_gpu_type())
        and statically_known_true(mat1.meta["val"].shape[0] == 1)
        and statically_known_true(mat2.meta["val"].shape[0] == 1)
    ):
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])


# When softmax is used with temperature or other scaling, we get the pattern
#
#   scale(x) - scale(x).amax(dim, keepdim=True)
#
# which is expected to be at most zero, but we may end up with numerical
# discrepancies # between the recomputed values of scale(x) inside and out
# of the reduction, # depending on compiler optimizations, e.g. use of fma
# instructions.
#
# Here we replace it with the mathematically equivalent,
#
#   scale(x - x.amax(dim, keepdim=True))
#
# which is more stable as we only compute the scaling once.
#
# NOTE: This pattern must come after fused attention matching!


def _partial_softmax_pattern(linear_func, reverse=False, to_dtype=False):
    # Allow matching inp * other and other * input
    if reverse:
        scaled = CallFunction(
            linear_func, KeywordArg("other"), KeywordArg("inp"), _users=MULTIPLE
        )
    else:
        scaled = CallFunction(
            linear_func, KeywordArg("inp"), KeywordArg("other"), _users=MULTIPLE
        )
    if to_dtype:
        scaled = CallFunction(
            prims.convert_element_type, scaled, KeywordArg("dtype"), _users=MULTIPLE
        )
    amax = CallFunction(
        tp_ops.amax.default, scaled, KeywordArg("dim"), KeywordArg("keepdim")
    )
    return CallFunction(tp_ops.sub.Tensor, scaled, amax)


def _preserve_scaled_softmax_nonfinite_semantics(scaled, stable, dim, keepdim):
    # This pattern also matches the raw scaled - amax(scaled) expression.  Only
    # use the stable form when it preserves that expression's nonfinite behavior.
    original = scaled - tp.amax(scaled, dim=dim, keepdim=keepdim)
    finite_scaled = tp.all(tp.isfinite(scaled), dim=dim, keepdim=True)
    return tp.where(finite_scaled, stable, original)


def _other_is_broadcasted_in_dim(match):
    # Check that the scaling factor is constant across the reduction dim,
    # so scaling doesn't change which index corresponds to the maximum value
    other = match.kwargs["other"]
    if isinstance(other, (int, float)):
        return True

    inp = match.kwargs["inp"]
    if not all(isinstance(x, Node) for x in (inp, other)):
        return False

    inp_example = inp.meta["val"]
    other_example = other.meta["val"]
    if isinstance(other_example, (tp.SymInt, tp.SymFloat)):
        return True

    if not all(isinstance(x, tp.Tensor) for x in (inp_example, other_example)):
        return False

    inp_ndim = inp_example.ndim
    other_shape = other_example.shape
    if inp_ndim < len(other_shape):
        return False

    # Pad other_shape to the same ndim as inp
    other_shape = [1] * (inp_ndim - len(other_shape)) + list(other_shape)

    dim = match.kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)

    if any(d >= len(other_shape) for d in dim):
        return False

    return all(statically_known_true(other_shape[d] == 1) for d in dim)


def mul_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    def repl(inp, other):
        scaled = inp * other
        if dtype is not None:
            scaled = scaled.to(dtype)
            inp = inp.to(dtype)

        sign: int | float | tp.Tensor
        if isinstance(other, (int, float, tp.SymInt, tp.SymFloat)):
            sign = 1 if other >= 0 else -1
        else:
            one = tp.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = tp.where(other >= 0, one, -one)

        inp = inp * sign
        max_ = tp.amax(inp, dim=dim, keepdim=keepdim)

        stable = (inp - max_) * (sign * other)
        return _preserve_scaled_softmax_nonfinite_semantics(
            scaled, stable, dim, keepdim
        )

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, other])


for reverse, to_dtype in itertools.product((False, True), repeat=2):
    register_graph_pattern(
        _partial_softmax_pattern(tp_ops.mul.Tensor, reverse=reverse, to_dtype=to_dtype),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(mul_softmax_pattern)


def div_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    def repl(inp, other):
        scaled = inp / other
        if dtype is not None:
            scaled = scaled.to(dtype)
            inp = inp.to(dtype)

        sign: int | float | tp.Tensor
        if isinstance(other, (int, float, tp.SymInt, tp.SymFloat)):
            sign = 1 if other >= 0 else -1
        else:
            one = tp.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = tp.where(other >= 0, one, -one)

        inp = inp * sign
        max_ = tp.amax(inp, dim=dim, keepdim=keepdim)

        stable = (inp - max_) / (sign * other)
        return _preserve_scaled_softmax_nonfinite_semantics(
            scaled, stable, dim, keepdim
        )

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, other])


for to_dtype in (False, True):
    register_graph_pattern(
        _partial_softmax_pattern(tp_ops.div.Tensor, to_dtype=to_dtype),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(div_softmax_pattern)


def scatter_upon_const_tensor_extra_check(m):
    if not config.optimize_scatter_upon_const_tensor:
        return False
    full_shape = m.kwargs["shape"]
    selector = m.kwargs["selector"]
    dim = m.kwargs["dim"]
    if dim < 0:
        dim += len(full_shape)

    selector_ft = selector.meta["val"]
    if selector_ft.dim() != len(full_shape):
        raise AssertionError(
            f"expected selector_ft.dim() == {len(full_shape)}, got {selector_ft.dim()}"
        )

    for idx, select_sz, full_sz in zip(
        itertools.count(), selector_ft.shape, full_shape
    ):
        if idx == dim:
            continue

        # TODO: the pattern can be updated to support the case that index tensor
        # is shorter. But that will need a more complex condition expression
        # especially for multi-dimensional tensors.
        # Skip it for now.
        if isinstance(full_sz, Node):
            full_sz = full_sz.meta["val"]
        if select_sz < full_sz:
            return False

    # Actually we can support small size larger than 1. It would be a bit
    # tedious. E.g., we load all the index values (not many) and compare
    # them with the position in tensor to decide what value to return.
    return selector_ft.size(dim) == 1


@register_graph_pattern(
    CallFunction(
        tp_ops.scatter.value,
        CallFunction(
            tp_ops.full,
            KeywordArg("shape"),
            KeywordArg("background_val"),
            dtype=KeywordArg("dtype"),
        ),
        KeywordArg("dim"),
        KeywordArg("selector"),
        KeywordArg("val"),  # scalar value
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=patterns,
    extra_check=scatter_upon_const_tensor_extra_check,
)
def scatter_upon_const_tensor(
    match: Match, shape, background_val, dtype, dim, selector, val
):
    """
    Match the pattern of full+scatter into a pointwise operation in joint graph.

    TODO: Right now the scatter value must be a scalar. But we could support it
    when it is a tensor as well.
    """
    from .. import metrics

    # pyrefly: ignore  # bad-assignment
    metrics.num_matches_for_scatter_upon_const_tensor += 1

    # Create a replacement that uses tp.where for the pointwise operation
    def repl_fn(shape, background_val, dim, selector, val):
        # Create a tensor of indices for the scatter dimension
        length = shape[dim]
        indices = tp.arange(length, device=selector.device, dtype=tp.int64)

        # Reshape indices to have size 'length' at dim, then broadcast
        view_shape = [1] * len(shape)
        view_shape[dim] = length
        indices_view = indices.view(*view_shape)

        # Broadcast selector to match full tensor shape
        selector_expanded = selector.expand(shape)

        # Create a mask for where to scatter
        mask = selector_expanded == indices_view

        # Use tp.where to implement the scatter pointwise operation.
        # When val and background_val are Python scalars, tp.where promotes
        # the result to the default floating dtype (float32), which loses the
        # const tensor's dtype. Cast back to dtype so the rewrite preserves
        # tp_ops.scatter.value semantics (the result has self's dtype, with the
        # scalar value cast to it).
        return tp.where(mask, val, background_val).to(dtype)

    # replace the scatter operation with pointwise equivalent
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl_fn, [shape, background_val, dim, selector, val])
