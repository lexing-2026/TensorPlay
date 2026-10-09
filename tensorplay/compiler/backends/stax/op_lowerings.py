"""Operator lowerings from a dispatch-level graph to the loop IR.

Each lowering receives the graph node and its already-lowered arguments
(``TensorBox`` for tensors, plain Python values otherwise) and returns the
lowered result.  Element types and shapes always come from the traced value
recorded on the node, so every lowering stores exactly the type the captured
program produced; arithmetic itself runs at float32 (float64 when involved)
and only an explicit conversion rounds in between.

Normalization layers decompose into a welford reduction plus pointwise
work, and their gradients into per-channel sums plus pointwise work, so the
surrounding elementwise code fuses into the same kernels.  Anything without
a lowering runs as a library call on realized inputs.
"""

from __future__ import annotations

import dataclasses
import functools
from collections import defaultdict
import itertools
import logging
import math
import operator
import warnings
from numbers import Number
from collections.abc import Sequence
from typing import Any, Callable

import sympy

import tensorplay as tp

from . import config
from .utils import (
    is_dynamic,
    is_gpu,
    is_triton_fp8_dtype_supported,
    is_view,
    parallel_num_threads,
    register_op_dtype_propagation_rules,
)

from tensorplay.primitives.common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_integer_dtype,
)
from .dtype_propagation import promoted_dtype_of_values
from tensorplay.graph import Node
from tensorplay.utils._pytree import arg_tree_leaves, tree_leaves, tree_map

from . import ir
from .codegen.common import BackendFeature
from .ir import (
    BaseView,
    Buffer,
    Constant,
    IndexingConstant,
    DeviceCopy,
    DynamicScalar,
    MutationLayoutSHOULDREMOVE,
    ExpandView,
    FixedLayout,
    FallbackKernel,
    IRNode,
    PermuteView,
    Pointwise,
    Reduction,
    ReinterpretView,
    resolve_unbacked_bindings,
    SliceView,
    SqueezeView,
    MutableBox,
    StorageBox,
    TensorBox,
    View,
    has_free_unbacked_symbols,
    ops_wrapper,
    validate_ir,
)
# Divisions and index arithmetic as symbolic expressions rather than as
# arithmetic on numbers not yet known: what a boundary or a position
# works out to is a function of numbers the program has not produced
# yet, and a shape written before then is a guess.
from tensorplay.graph.experimental.sympy_functions import (
    CeilDiv,
    FloorDiv,
    Max,
    Min,
    Mod,
    ModularIndexing,
)
from .loops import (
    V,
    as_index,
    contiguous_strides,
    dtype_name,
    modular_indexing,
    ops,
    prod,
)

log = logging.getLogger(__name__)

tp_ops = tp.ops.tp
prims = tp.ops.prims

#: A change of element type.  A conversion is what a traced graph holds rather
#: than the operation it was asked for, so this is the one a node's target is
#: compared against.
_CONVERT_ELEMENT_TYPE = tp.ops.tp.to.dtype

#: The dtypes a value in the e8m0 scale format is read as, being the dtypes
#: that format is a multiple of.
_FLOAT8_E8M0FNU_TO_FLOAT_DTYPES = (
    tp.float32,
    tp.float64,
    tp.float16,
    tp.bfloat16,
)


def _warn_complex_not_supported():
    warnings.warn(
        "This compiler does not support code generation for complex operators. "
        "Performance may be worse than eager."
    )


# There are some types (CPU) which we accept as input but not as
# output.
def unsupported_input_tensor(t: tp.Tensor, node=None):
    "Do not support reading or writing to this tensor"
    if t.is_complex():
        # Complex views are supported with IR ComplexView
        _warn_complex_not_supported()
        return True

    if t.is_meta:
        return True

    if t.is_sparse:
        return True

    if not is_triton_fp8_dtype_supported(t.dtype, t.device):
        from .codegen.triton_utils import (
            use_uint8_triton_storage_for_cuda_float8_e4m3fn,
        )

        if not use_uint8_triton_storage_for_cuda_float8_e4m3fn(
            t.dtype, device=t.device
        ):
            return True

        # uint8 storage reinterprets fp8 bytes: allow bitcast, views, memory
        # movement, and dequant (convert out of fp8)
        if not node:
            return True
        return not (
            isinstance(node.target, tp.ops.OpOverload)
            and node.target
            in (
                tp_ops.view.dtype,
                tp_ops.cat.default,
                tp_ops.clone.default,
                tp_ops._scaled_mm.default,
                tp_ops._scaled_mm_v2.default,
                _CONVERT_ELEMENT_TYPE,
            )
            or (isinstance(node.target, tp.ops.OpOverload) and is_view(node.target))
        )

    if t.dtype == tp.float8_e8m0fnu:
        if not node:
            return True

        # Allow bitcasts, views, memory movement, and supported conversions,
        # but not arithmetic.
        if not isinstance(node.target, tp.ops.OpOverload):
            return True
        if node.target in (
            tp_ops.view.dtype,
            tp_ops.cat.default,
            tp_ops.clone.default,
            tp_ops._scaled_mm.default,
            tp_ops._scaled_mm_v2.default,
        ) or is_view(node.target):
            return False
        if node.target == _CONVERT_ELEMENT_TYPE:
            return not (
                len(node.args) >= 2 and node.args[1] in _FLOAT8_E8M0FNU_TO_FLOAT_DTYPES
            )
        return True

    return False


def unsupported_output_tensor(t: tp.Tensor, node=None):
    "Do not support writing tensor but can read from it"
    supported_complex_views = (
        tp_ops.view.dtype,
        _CONVERT_ELEMENT_TYPE,
    )
    if node is not None and node.target in supported_complex_views and t.is_complex():
        return False
    if unsupported_input_tensor(t, node):
        return True
    if not is_triton_fp8_dtype_supported(t.dtype, t.device):
        return True
    return t.is_cpu and config.disable_cpp_codegen


def fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=True):
    # Custom fallback lowering
    if node.target is tp_ops.view_as_complex.default:
        return False

    if node.op == "placeholder":
        return False

    # We should be able to remove this special case once `disable_cpp_codegen` is killed.
    if node.target is tp_ops.lift_fresh_copy.default:
        return False

    def check_skip_condition(inp_out_node, is_output):
        if not isinstance(inp_out_node, Node):
            return False

        if "val" not in inp_out_node.meta:
            return False

        for meta in tree_leaves(inp_out_node.meta["val"]):
            # The value recorded on a node is what a lowering reads the type
            # from, and only a tensor carries a type; anything else in the
            # position says nothing about what the node holds.
            if not isinstance(meta, tp.Tensor):
                continue

            if is_output:
                if unsupported_output_tensor(meta, node):
                    return True
            else:
                if unsupported_input_tensor(meta, node):
                    return True

        return False

    # only skip codegen if there is a cpu output, not input
    for arg in arg_tree_leaves(*node.args, **node.kwargs):
        if check_skip_condition(arg, is_output=False):
            return True

    return check_skip_condition(node, is_output=True)

LOWERINGS: dict[str, Callable[..., Any]] = {}

#: Lowerings a program supplied for its own operations, by the operation rather
#: than by the name it is called under.  Kept apart from the built-in lowerings
#: because it is consulted first: an operation a program wrote a lowering for is
#: that program's own, and the built-in lowering of a name it happens to share
#: describes a different operation that goes by the same name.
user_lowerings: dict[Any, Callable[..., Any]] = {}


#: The promotion each operation's operands go through, for the operations whose
#: result is the promotion of what they were given.  An operation that is not
#: here either says what it produces or is one of the special functions, whose
#: promotion is recorded beside the spelling of each of them.
POINTWISE_TYPE_PROMOTION_KIND: dict = {}


#: The operations that have no lowering of their own and are computed by the
#: framework instead.  A call to one of these reaches the framework whole, which
#: is slower than a kernel and correct, and the point of recording it is that the
#: set is what a caller checks before deciding something can be done here.
FALLBACKS: set = set()


def select_decomp_table() -> dict:
    """The decompositions a captured graph may be written in terms of.

    Asked for rather than imported, because which decompositions are available
    depends on how the program was configured, and a graph captured against one
    set and compiled against another would have operations in it that no longer
    have a meaning.  So the set is chosen at the moment of capture and travels
    with the capture.
    """

    # The module is imported for what importing it does rather than for
    # anything in it: each decomposition registers itself as it is defined, so a
    # table read before the module has been read is a table of nothing.  The
    # import is here, at the one place the table is read, rather than at the top
    # of this file because a table of decompositions is only wanted by a capture
    # and a program that never captures should not pay for reading them.
    import tensorplay._decomp.decompositions  # noqa: F401
    import tensorplay._decomp.decompositions_for_rng  # noqa: F401
    from tensorplay._decomp import decomposition_table

    # Read at the moment the table is read rather than at import, so that a
    # program which never captures does not pay for reading them -- and asked
    # for rather than assumed, because whether randomness is a read at a
    # position or a call that consults a generator is a question about how the
    # program was configured.
    tensorplay._decomp.decompositions_for_rng.register_rng_decompositions()

    # The random table is merged in rather than kept apart, because a graph
    # captured is meant to be written in terms of reads at a position.  It was
    # kept apart while being written so that a read could ask the framework
    # for values without the framework expanding the ask into another read.
    table = dict(decomposition_table)
    table.update(tensorplay._decomp.decompositions_for_rng.rng_decompositions)
    return table


#: Operations a region's trace keeps whole although the table can decompose
#: them: a random draw is the framework's own generator's, so a compiled region
#: draws the values the program would have drawn without it.
_TRACE_KEEPS_RANDOM_DRAWS = frozenset({
    "rand", "rand_like", "randn", "randn_like", "randint", "randint_like",
    "randperm", "bernoulli", "bernoulli_", "dropout", "native_dropout",
    "normal", "normal_", "uniform", "uniform_", "exponential", "exponential_",
    "cauchy", "cauchy_", "geometric", "geometric_", "log_normal", "log_normal_",
})


def trace_decompositions() -> dict:
    """The decompositions a region's operator-level trace is written in.

    An operation with a lowering, a template or a library call of its own
    reaches the region whole, where it is written better than its parts would
    be.  Any other operation the table can decompose is traced as its parts,
    which are lowered, instead of being handed to the framework as a call: the
    region keeps its loops around it and fuses through it.
    """

    from tensorplay._decomp import decompositions_for_rng

    from .graph_lowering import _TEMPLATE_OPERATORS
    from .operator_coverage import LIBRARY_CALLS

    load_lowering_modules()
    table = select_decomp_table()
    random_draws = set(decompositions_for_rng.rng_decompositions)
    traced = {}
    for op, fn in table.items():
        name = target_name(op)
        if (
            op in random_draws
            or name.split(".", 1)[0] in _TRACE_KEEPS_RANDOM_DRAWS
            or op in user_lowerings
            or find_lowering(name) is not None
            or name in _TEMPLATE_OPERATORS
            or name in LIBRARY_CALLS
        ):
            continue
        traced[op] = fn
    return traced


def in_namespace(op: Any, namespace: str) -> bool:
    """Whether an operation belongs to a namespace.

    Asked by name rather than by where it was found, because the same operation
    is reached from several places and only its own name says where it lives.
    """

    qualified = getattr(op, "_qualified_op_name", None)
    if qualified is not None:
        return namespace in qualified
    name = getattr(op, "name", None)
    if callable(name):
        return namespace in name()
    return False


def _record_symbolic_input_source(tensor, dim, expr, kind) -> None:
    """Which input a shape expression was read from, and at which element.

    Only an expression that is a plain symbol read out of a region input is
    worth recording: anything else is either not a symbol or not something a
    wrapper could bind to a single element of an input.
    """

    if not isinstance(expr, sympy.Symbol) or not isinstance(tensor, TensorBox):
        return

    if not isinstance(tensor.data, StorageBox) or not isinstance(
        tensor.data.data, InputBuffer
    ):
        return

    name = tensor.get_name()
    if name not in V.graph.graph_inputs:
        return

    V.graph.symbolic_input_sources.setdefault(expr, (name, kind, int(dim)))
    from .ir import InputBuffer


def unsupported_input_tensor(t, node=None) -> bool:
    """Whether a value cannot be read or written by a generated kernel.

    A complex value has two parts per element that no kernel here addresses,
    a meta value has no memory, and a sparse one has its elements somewhere
    other than its positions.
    """

    return bool(t.is_complex() or t.is_meta or t.is_sparse)


def unsupported_output_tensor(t, node=None) -> bool:
    """Whether a value cannot be written by a generated kernel.

    Reinterpreting the bits of a complex value as another type writes nothing
    of the complex value itself, so that one is allowed.
    """

    if (
        node is not None
        and t.is_complex()
        and target_name(node.target) == "view.dtype"
    ):
        return False
    if unsupported_input_tensor(t, node):
        return True
    return t.device.type == "cpu" and config.disable_cpp_codegen


def fallback_node_due_to_unsupported_type(node) -> bool:
    """Whether a call reads or writes a value no generated kernel can hold.

    Such a call is handed to the framework on its own; the calls around it are
    still generated, reading and writing ordinary values in memory.
    """

    if node.op != "call_function" or node.target is operator.getitem:
        return False

    def check(item, is_output: bool) -> bool:
        if not hasattr(item, "meta") or "val" not in item.meta:
            return False
        for value in tree_leaves(item.meta["val"]):
            if not isinstance(value, tp.Tensor):
                continue
            if is_output:
                if unsupported_output_tensor(value, node):
                    return True
            elif unsupported_input_tensor(value, node):
                return True
        return False

    for arg in arg_tree_leaves(*node.args, **node.kwargs):
        if check(arg, is_output=False):
            return True
    return check(node, is_output=True)


def fallback_handler(kernel, add_to_fallback_set: bool = True):
    """A way to compute an operation by handing the whole call to the framework.

    Returned rather than registered, because whether an operation should be
    computed this way is a decision made where the operation is lowered, not one
    made for the operation: the same operation is a fallback in one place and a
    kernel in another, and only the place that knows which can say.

    The arguments are wrapped first, because a lowered value is not the value
    itself and the framework wants the value -- so a call that is only being
    handed over still has to give the framework something it can read.
    """

    if add_to_fallback_set:
        FALLBACKS.add(kernel)

    def handler(*args, **kwargs):
        def wrap_tensors(x):
            return x.wrap_for_lowering() if isinstance(x, IRNode) else x

        return tree_map(
            wrap_tensors, FallbackKernel.create(kernel, *args, **kwargs)
        )

    return handler


def broadcast_symbolic_shapes(a, b):
    """The shape two shapes broadcast to, said as far as it can be said exactly.

    A shape of zero or one takes the other side's extent, whichever it is, because
    a value repeated once is that value and a value repeated never is nothing.
    Beyond that the two extents have to agree, and of the two ways of saying they
    agree the shorter formula is kept -- which is not a preference but a fact
    about what a kernel can be checked against.
    """

    b = tuple(b)
    if not a or a == b:
        return b

    output = []
    for x, y in itertools.zip_longest(
        reversed(a), reversed(b), fillvalue=sympy.S.One
    ):
        if V.graph.sizevars.is_size_one_or_false(y):
            output.append(x)
        elif V.graph.sizevars.is_size_one_or_false(x):
            output.append(y)
        else:
            V.graph.sizevars.check_equals(x, y)
            if len(sympy.expand(y).free_symbols) < len(sympy.expand(x).free_symbols):
                output.append(y)  # prefer shorter formula
            else:
                output.append(x)
    return tuple(reversed(output))


def broadcast_tensors(*inputs):
    """The values, each seen in the space all of them share.

    A value whose shape is already the shared one is left alone; one that is not
    is viewed into the shared shape, which is free when the value has one element
    to repeat and a copy when it does not.  Deciding which happens per value is
    why this returns the values and not just the shape.
    """

    if len(inputs) == 1:
        if isinstance(inputs[0], (list, tuple)):
            return broadcast_tensors(*inputs[0])
        return inputs
    target: list[sympy.Expr] = functools.reduce(
        broadcast_symbolic_shapes, (x.get_size() for x in inputs), ()
    )
    outputs = []
    for x in inputs:
        if (sizes := tuple(x.get_size())) == target:
            pass
        elif len(sizes) != len(target) or any(
            V.graph.sizevars.is_size_one_or_false(a)
            != V.graph.sizevars.is_size_one_or_false(b)
            for a, b in zip(sizes, target)
        ):
            x = ExpandView.create(x, target)
        outputs.append(x)
    return outputs


def maybe_copy_cpu_scalar(x: TensorBox, device: Any) -> TensorBox:
    """A value of no elements, or of one, moved onto the device it is used on.

    Only a value with nothing to iterate is worth moving: anything larger is read
    where it is anyway, and a copy of it would cost more than the read.  A view
    rather than a value is left alone, because what a view holds is not what was
    read and copying it would copy something else.
    """

    if not isinstance(x.data, ReinterpretView) or has_free_unbacked_symbols(
        x.get_size()
    ):
        return x
    size = V.graph.sizevars.guarding_hints_or_throw(x.get_size())
    cur_device = x.get_device()
    if (
        cur_device is not None
        and getattr(cur_device, "type", None) == "cpu"
        and cur_device != device
        and (len(size) == 0 or (len(size) == 1 and size[0] == 1))
    ):
        return TensorBox(StorageBox(DeviceCopy.create(x, cur_device, False)))
    return x


def transform_args(
    args: list[Any],
    kwargs: dict[str, Any],
    broadcast: bool,
    type_promotion_kind: Any = None,
    convert_input_to_bool: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """The arguments of a call, made to be computed together.

    Three things have to be settled before a call can be computed, and all three
    are settled here rather than in each lowering: the values have to be of one
    type, or the result is of a type nobody asked for; they have to be seen in
    one shape, or the result is of a shape nobody asked for; and a value of no
    elements has to be on the device the others are on, or the call is a copy per
    use.
    """

    args_indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    kwargs_indices = [k for k, v in kwargs.items() if isinstance(v, TensorBox)]
    if not args_indices and (not kwargs_indices):
        return (args, kwargs)
    if type_promotion_kind or convert_input_to_bool:
        if convert_input_to_bool:
            dtype = tp.bool
        else:
            promoting_args = [
                a
                for a in args
                if isinstance(a, (Number, sympy.Basic)) or hasattr(a, "dtype")
            ]
            promoting_args.extend(
                (a for a in kwargs.values() if hasattr(a, "dtype"))
            )
            dtype = promoted_dtype_of_values(
                *promoting_args, type_promotion_kind=type_promotion_kind
            )
        device = (
            args[args_indices[0]] if args_indices else kwargs[kwargs_indices[0]]
        ).get_device()
        for i in args_indices:
            args[i] = maybe_copy_cpu_scalar(args[i], device)
        for k in kwargs_indices:
            kwargs[k] = maybe_copy_cpu_scalar(kwargs[k], device)

        def promote(arg: Any) -> Any:
            if isinstance(arg, TensorBox):
                return cast_to(arg, dtype)
            elif isinstance(arg, ir.Constant):
                return ir.Constant(value=arg.value, dtype=dtype, device=device)
            else:
                return arg

        args = [promote(a) for a in args]
        kwargs = {k: promote(v) for k, v in kwargs.items()}
    if broadcast:
        broadcasted = broadcast_tensors(
            *list(
                itertools.chain(
                    (args[i] for i in args_indices), (kwargs[k] for k in kwargs_indices)
                )
            )
        )
        size = list(broadcasted[0].get_size())
        for i, x in zip(args_indices, broadcasted[: len(args_indices)]):
            args[i] = x
        for k, x in zip(kwargs_indices, broadcasted[len(args_indices) :]):
            kwargs[k] = x
        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], size)
        for k in kwargs:
            if isinstance(kwargs[k], ir.Constant):
                kwargs[k] = ExpandView.create(kwargs[k], size)
    return (args, kwargs)


def _register_lowering(
    op: Any,
    decomp_fn: Callable[..., Any],
    broadcast: bool,
    type_promotion_kind: Any,
    convert_input_to_bool: bool,
    lowering_dict: dict,
):
    """Put one operation's lowering into the table, wrapped so it is called right.

    The wrapping is what makes every lowering callable the same way: a caller
    hands over arguments as they were written, and this settles their types and
    shapes before the lowering sees them.  A lowering that had to do that itself
    would do it differently from every other one.
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args: list[Any] = list(args)
        kwargs: dict[str, Any] = dict(kwargs)
        unpacked = False
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(args[0])

        if not all(
            (fn in FALLBACKS or in_namespace(fn, "_collective_functional"))
            for fn in (op if isinstance(op, (tuple, list)) else (op,))
        ):
            # an out= call has no lowering here, and saying so plainly beats
            # letting it fail later as something unrelated
            if any(x == "out" for x in kwargs):
                raise AssertionError("out= ops aren't yet supported")

        args, kwargs = transform_args(
            args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
        )

        if unpacked:
            args = [args]

        out = decomp_fn(*args, **kwargs)
        # What came back is about to be held as a graph value, so it is checked
        # while the lowering that produced it is still the thing being read.
        validate_ir(out)
        return out

    lowering_dict.update(dict.fromkeys(get_overloads(op), wrapped))
    return wrapped


def realize_inputs(*args: Any) -> Any:
    """The values as they must be handed to something outside this compiler.

    Realized and made to run along their last axis, because a caller that is not
    this compiler reads the values it is given and cannot be told what a stride
    means.
    """

    if len(args) == 1:
        realized = args[0]
        if not realized.has_tensor_output():
            return realized
        return ir.ExternKernel.require_stride1(realized)
    return [realize_inputs(x) for x in args]


def decode_device(device: Any) -> Any:
    """The device a value is on, as a device rather than as a description of one.

    A device named without saying which one means whichever is current, and a
    device that is not a processor is always a particular one even when the
    description left the number out.  So this is where a description becomes
    something a value can be on.
    """

    if device is None:
        # Unnamed is the device a new tensor would be made on, which is the
        # processor unless the program said otherwise.
        return tp.get_default_device()
    if isinstance(device, str):
        device = tp.device(device)
    if device.type not in ("cpu", "meta") and device.index is None:
        from .runtime.benchmarking import get_interface_for_device

        device_interface = get_interface_for_device(device.type)
        return tp.device(device.type, index=device_interface.current_device())
    return device


def to_device(x: TensorBox, device: Any, *, copy: bool = False, non_blocking: bool = False):
    """The same value, on another device.

    A value that is already there is left alone, or copied if the caller asked
    for a copy rather than a move -- which is a different request, because a move
    that turned out to be a no-op would not have produced a second value.
    """

    device = decode_device(device)
    if x.get_device() == device:
        return clone(x) if copy else x
    return TensorBox.create(ir.DeviceCopy.create(x, device, non_blocking))


def clone(x: TensorBox, *, memory_format: Any = None):
    """A second value with the same contents.

    The layout is deliberately not settled here: what shape of memory the copy
    has is the scheduler's to decide, and deciding it now would take that
    choice away.  The loader carries whatever strides the value had, and what
    comes after sorts it out.
    """

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=list(x.get_size()),
    )


def mutate_to(changed: Any, val: Any, unsafe_alias: bool = False):
    """Make one value's contents become another's, in the memory the other already has.

    Writing into a buffer that already exists is not a new value: it is the old
    one, changed.  Which is why this hands back the buffer it was given rather
    than what was written into it -- a caller that got a new value back would
    have to be told to write it, and the whole point is that it does not.

    Where the value to write is a view of something, it is first copied into
    memory of its own, because what is being promised is that the destination
    holds these contents afterwards, and a view would stop holding them the
    moment its source changed.
    """

    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data

    if not isinstance(val, ir.StorageBox):
        # A view cannot be written through, so give the value memory of its own
        # to be written into.
        node = Pointwise.create(
            device=changed.get_device(),
            dtype=changed.get_dtype(),
            inner_fn=val.make_loader(),
            ranges=changed.get_size(),
        )
        if not (isinstance(node, (BaseView, MutableBox))):
            raise AssertionError("expected: isinstance(node, (BaseView, MutableBox))")
        val = node.data
        if not (isinstance(val, ir.StorageBox)):
            raise AssertionError("expected: isinstance(val, ir.StorageBox)")

    if isinstance(changed_data, ir.StorageBox) and not (
        changed_data.is_input_buffer()
        # A parameter or a buffer of a module is not an input to the graph, and
        # swapping what a node points at is how a module's own value would be
        # replaced rather than written to.
        or changed_data.is_module_buffer()
        or isinstance(changed_data.data, ir.NopKernel)
    ):
        # Nothing else holds this memory, so the data pointer can simply be
        # moved across.
        val.realize()
        changed_data.data = val.data
        return changed

    ir.MutationLayoutSHOULDREMOVE.realize_into(
        val, changed_data, unsafe_alias=unsafe_alias
    )
    return changed


#: The operations that take a list of tensors and do the same thing to each.
#: Read to decide whether a value produced by one of them is still only wanted
#: by other ones -- if something else wants it, fusing them together would
#: change what that something else sees.
foreach_ops: set = set()

#: The ones that write into their input, and so cannot be reordered against
#: anything else that reads it.
inplace_foreach_ops: set = set()

#: Which of them has a non-writing form to lower to, keyed by the writing one.
inplaceable_foreach_ops: dict = {}


def cur_node_has_non_foreach_users() -> bool:
    """Whether anything other than a sibling list operation wants this result.

    Fusing a list of same operations into one program is only sound while
    nothing else reads the individual results, because a fused program produces
    them together or not at all.  So this is asked before fusing, and a yes
    means the results have to exist on their own.
    """

    for node in V.graph.current_node.users:
        for user in node.users:
            if not (user.op == "call_function" and (user.target in foreach_ops)):
                return True

    return False


def group_foreach_args(arg_pairs: Iterable[Any]) -> dict:
    """Sort the list's entries by where they run and whether they can be fused.

    Two things decide it: the device, because one program covers one device,
    and whether the shapes are known, because a program written for a shape can
    only cover that shape.
    """

    out = defaultdict(list)
    unpack_args = False
    for i, args in enumerate(arg_pairs):
        if not isinstance(args, Iterable):
            unpack_args = True
            args = (args,)
        use_foreach = (
            not is_dynamic(*args) or config.combo_kernel_foreach_dynamic_shapes
        )
        device = None
        for t in args:
            if isinstance(t, TensorBox):
                device = t.data.get_device()
                break
        if device is None:
            raise AssertionError("foreach op should have at least one tensor arg")
        if unpack_args:
            (args,) = args
        out[(device, use_foreach)].append((i, args))
    return out


def _register_foreach_lowering(decomp_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a lowering for one of the list-of-tensors operations.

    The result is checked the way any lowering's is, because a list operation
    produces a list and a list that is not a value is as wrong as a value that
    is not.
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out

    return wrapped


def make_foreach_pointwise(
    pw_fn: Callable[..., Any],
    allow_alpha: bool = False,
    scalar_kwarg: str = "alpha",
) -> Callable[..., list]:
    """Turn a one-tensor-at-a-time lowering into one over a whole list.

    The scalar an operation like add-and-multiply carries arrives in a different
    place depending on the operation's shape, so where to look for it is said
    here rather than guessed at each use.  A scalar that is not a list is
    repeated to the length of the list that is, so that what the per-tensor
    lowering receives is always one entry per tensor.
    """

    def inner(*inputs: list, alpha=1, value=1) -> list:
        # For ops like addcmul/addcdiv, the scalar `value` arrives as a
        # positional arg (not keyword) due to the schema. Extract it
        # from the end of inputs if present.
        inputs = list(inputs)
        if (
            scalar_kwarg == "value"
            and inputs
            and not isinstance(inputs[-1], (list, tuple))
        ):
            scalar_val = inputs.pop()
        elif scalar_kwarg == "value":
            scalar_val = value
        else:
            scalar_val = alpha

        realize_outputs = (
            len(V.graph.current_node.users) == 0
            or V.graph.current_node.target in inplace_foreach_ops
            or cur_node_has_non_foreach_users()
        )

        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        if a_list_input is None:
            raise AssertionError("at least one input must be a list to a foreach op")

        # broadcast scalar inputs to match length of list inputs
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                broadcast_inputs.append(input)

        groups = group_foreach_args(zip(*broadcast_inputs))

        def apply_fn(args) -> Any:
            if allow_alpha:
                return pw_fn(*args, **{scalar_kwarg: scalar_val})
            else:
                return pw_fn(*args)

        return foreach_group_loop(groups, len(a_list_input), apply_fn, realize_outputs)

    return inner


def foreach_group_loop(
    groups: dict,
    num_outputs: int,
    apply_fn: Callable[..., Any],
    realize_outputs: bool,
) -> list:
    """Apply one operation across each group, and say which results may be fused.

    A result is only offered for fusing once it has been written down: a fused
    program produces its results at the point it runs, and a result that is
    still only a description of a computation has nothing to fuse with yet.
    """

    outputs: list = [None] * num_outputs
    for (device, use_foreach), group in groups.items():
        operation_list: list[str] = []
        for output_ind, args in group:
            output = apply_fn(args)
            outputs[output_ind] = output

            if (
                V.graph.has_feature(device, BackendFeature.FOREACH)
                and use_foreach
                and realize_outputs
            ):
                output.realize()
                operation_list.append(output.get_operation_name())

        if operation_list:
            V.graph.register_operation_list(operation_list)

    if not all(x is not None for x in outputs):
        raise AssertionError("expected: all(x is not None for x in outputs)")

    return outputs


def register_foreach_pointwise(
    pointwise_lowering_fn: Callable[..., Any],
    allow_alpha: bool = False,
    scalar_kwarg: str = "alpha",
    *,
    names: tuple[str, ...],
):
    """Register the list form of an operation, from the one-tensor form of it.

    The names are the operation's own, without a namespace: the table is keyed
    that way, and saying so here is what keeps a list operation from being
    registered under a name nothing will look it up by.
    """

    fn = make_foreach_pointwise(
        pointwise_lowering_fn, allow_alpha=allow_alpha, scalar_kwarg=scalar_kwarg
    )
    wrapped = _register_foreach_lowering(fn)
    for name in names:
        foreach_ops.add(name)
        LOWERINGS[name] = wrapped
    return wrapped


def register_foreach_inplace(
    names: tuple[str, ...],
    outplace_names: tuple[str, ...],
    outplace_op: Callable[..., Any],
):
    """Register the writing form of a list operation, from its non-writing one.

    Writing into each input is the non-writing operation's result made to be
    that input, and the aliasing is declared unsafe because a later write
    through one of them would otherwise be seen by whoever still reads the
    result.
    """

    for name in outplace_names:
        inplaceable_foreach_ops[name] = names
    for name in names:
        inplace_foreach_ops.add(name)

    def fn(*args: Any, **kwargs: Any) -> Any:
        results = outplace_op(*args, **kwargs)
        mut_results = []
        for arg, result in zip(args[0], results):
            mut_results.append(mutate_to(arg, result, unsafe_alias=True))

        return mut_results

    wrapped = _register_foreach_lowering(fn)
    for name in names:
        foreach_ops.add(name)
        LOWERINGS[name] = wrapped


def register_inplace(*names: str, outplace_op: Callable[..., Any]):
    """Register the writing form of an operation, from its non-writing one.

    What is written keeps the type of what was there, not the type the
    non-writing form produced: the buffer is the caller's and its type was
    chosen when it was allocated.
    """

    def fn(*args: Any, **kwargs: Any) -> Any:
        result = outplace_op(*args, **kwargs)
        result = ops.to_dtype(result, args[0].get_dtype())
        return mutate_to(args[0], result)

    return register(*names)(fn)
def get_overloads(op: Any) -> list[Any]:
    """The table's keys for an operation, one for each of its forms.

    A whole operation names its forms itself, and registering the operation means
    registering every form of it -- a caller who asks for the operation has not
    said which form it meant, and the forms differ enough that answering for the
    wrong one would compute a different thing.  A form already in the table is
    left out, so a form registered on its own keeps the lowering it was given.
    """

    if not isinstance(op, (list, tuple)):
        op = [op]
    else:
        op = list(op)

    for fn in list(op):
        overloads = getattr(fn, "overloads", None)
        if not callable(overloads):
            continue
        for other_name in overloads():
            other_fn = getattr(fn, other_name, None)
            if other_fn is not None and _lowering_key(other_fn) not in LOWERINGS:
                op.append(other_fn)

    return [_lowering_key(fn) for fn in op]


def _lowering_key(op: Any) -> str:
    """The table's key for one form of an operation.

    The operation and which of its forms, because two forms of one operation are
    two lowerings -- a call that writes its result somewhere and a call that
    returns it are not the same call, and a table that could not tell them apart
    could not answer either.
    """

    name = getattr(op, "__name__", None)
    if name:
        return name
    overload = getattr(op, "default", None)
    if overload is not None and getattr(overload, "__name__", None):
        return overload.__name__
    return str(op)


def register_lowering(
    op: Any,
    broadcast: bool = False,
    type_promotion_kind: Any = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool: bool = False,
    lowering_dict: dict = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a lowering of one operation, to be used as a decorator.

    ``broadcast`` says the operation's values are made to share a shape first,
    and ``type_promotion_kind`` says what the result's type is -- ``None`` for an
    operation whose result is not the promotion of what it was given, which is
    most of the products.
    """

    return functools.partial(
        _register_lowering,
        op,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
        lowering_dict=LOWERINGS if lowering_dict is None else lowering_dict,
    )


def register(*names: str):
    """Register the lowering of one or more operations by name."""

    def wrap(fn):
        for name in names:
            LOWERINGS[name] = fn
        return fn

    return wrap


def register_pointwise(
    *names: str,
    type_promotion_kind=None,
    override_return_dtype=None,
):
    """Register a lowering that operates on one element at a time.

    Registering it also states what the operation's result type is, because a
    lowering and the type of the value it produces are two facts about the same
    operation and are declared together: an operation registered without a
    promotion would leave the type of its result unstated, which is caught where
    the types of all operations are collected rather than at the point of use.

    An operation that reads a number and produces a number is declared with the
    promotion that turns a number into a number, so an integer argument does not
    silently produce an integer where the operation means a real one.
    """

    kind = (
        type_promotion_kind
        if type_promotion_kind is not None
        else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    for name in names:
        POINTWISE_TYPE_PROMOTION_KIND[name] = kind
        register_op_dtype_propagation_rules(name, kind, override_return_dtype)
    return names


def register_pointwise_numeric(*names: str):
    """Register an operation that takes a number and produces a real number.

    An integer argument does not make the result an integer: a reciprocal of an
    integer is not an integer, and an operation that took one as such would
    answer with a value nobody asked for.
    """

    return register_pointwise(
        *names, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


def target_name(target) -> str:
    if target is operator.getitem:
        return "getitem"
    return str(getattr(target, "__name__", target))


#: The overloads a call named without one is read as, in the order they are
#: tried.  A call through the operator's packet, the function, or a tensor
#: method carries the bare name, and the table is written per overload.
_OVERLOAD_SUFFIXES = (
    "", ".Tensor", ".Scalar", ".default", ".int", ".dim", ".dims", ".dtype",
    ".device", ".dtype_layout",
)


def find_lowering(name: str):
    """The lowering a call by this name is written by, or None.

    Every walk that lowers calls asks this, so a call reads the same in a region
    and in a region nested inside one.
    """

    for suffix in _OVERLOAD_SUFFIXES:
        lowering = LOWERINGS.get(f"{name}{suffix}")
        if lowering is not None:
            return lowering
    return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def node_val(node=None, index: int | None = None):
    """The value the node being lowered stands for.

    Asked for without saying which node by the parts of a lowering that are
    about the result rather than the operation -- how big it is, what it is
    made of, where it lives -- which is the same node for the whole of the
    call and is published for its duration rather than passed down through
    every helper that might want to ask.
    """

    if node is None:
        node = V.current_node
    val = node.meta.get("val")
    if index is not None and isinstance(val, (tuple, list)):
        val = val[index]
    return val


def val_info(val):
    return (
        tuple(int(s) for s in val.shape),
        val.dtype,
        val.device,
    )


def _extent(s):
    """One extent of a loop nest: a number when it is one, its symbol otherwise.

    Most extents are known when the region is lowered and are kept as plain
    numbers.  Some are only known while the kernel runs -- how many positions
    one group of a split takes, say, which the kernel decides from what it
    reads -- and those stay the symbols they were made as, because there is no
    number to give them yet.
    """
    if isinstance(s, int):
        return s
    if isinstance(s, sympy.Expr) and s.is_number:
        return int(s)
    if isinstance(s, sympy.Expr):
        return s
    return int(s)


def is_tensor_box(x) -> bool:
    return isinstance(x, TensorBox)


def as_value_node(x, dtype, device):
    """``x`` as a node a loader can be asked of.

    A value that is already a node is one.  One that is not is a number, and a
    number is read the same way whatever the output's extents are: as itself.
    A number that came from a shape is an expression over the loop variables
    rather than a number, so reading it produces an index instead.
    """

    if isinstance(x, (TensorBox, IRNode)):
        return x
    if isinstance(x, sympy.Expr):
        return IndexingConstant(index=x, dtype=dtype, device=device)
    return Constant(value=x, dtype=dtype, device=device)


def _broadcast_loader(x, size):
    """A loader of ``x`` asked at positions of the broadcast extents ``size``.

    Axes line up from the last; an axis ``x`` lacks, or holds once where the
    result holds many, is read at its only position.
    """

    load = x.make_loader()
    own = list(x.get_size())
    lead = len(size) - len(own)
    if lead == 0 and all(
        V.graph.sizevars.statically_known_equals(a, b) for a, b in zip(own, size)
    ):
        return load
    single = [
        V.graph.sizevars.statically_known_equals(a, 1)
        and not V.graph.sizevars.statically_known_equals(b, 1)
        for a, b in zip(own, size[lead:])
    ]

    def reindexed(index):
        index = list(index)[lead:]
        return load([sympy.S.Zero if one else i for i, one in zip(index, single)])

    return reindexed


def pointwise(fn, *inputs, val=None, out_dtype=None):
    # What the result is like is read from the inputs first: the first input's
    # extents and the device they live on say what the operation runs over,
    # and this is the one reading that holds whatever the call was handed --
    # a region inside a region records its own calls without the node that
    # holds the region being the node being lowered, so asking that one about
    # this operation's result would answer with someone else's answer.  The
    # recorded value is a fallback for a call whose inputs say nothing.
    if val is None:
        val = next((x for x in inputs if hasattr(x, "get_size")), None)
    if val is None:
        val = next((x for x in inputs if hasattr(x, "shape")), None)
    if val is None:
        val = node_val()
    if hasattr(val, "get_size"):
        size = tuple(_extent(s) for s in val.get_size())
        dtype = val.get_dtype()
        device = val.get_device()
    else:
        size, dtype, device = val_info(val)
    tensor_inputs = [x for x in inputs if hasattr(x, "get_size")]
    if len(tensor_inputs) > 1:
        target = functools.reduce(
            broadcast_symbolic_shapes, (x.get_size() for x in tensor_inputs), ()
        )
        size = tuple(_extent(s) for s in target)
    loaders = [
        _broadcast_loader(x, size) if hasattr(x, "get_size")
        else as_value_node(x, dtype, device).make_loader()
        for x in inputs
    ]

    def inner(index):
        return fn(*[load(index) for load in loaders])

    # A number among the inputs is read in the inputs' type; the result may be
    # of another type altogether, as a comparison's is.
    return Pointwise.create(
        device=device,
        dtype=dtype if out_dtype is None else out_dtype,
        inner_fn=inner,
        ranges=size,
    )


def cast_to(value, dtype):
    """The value read as this type, unchanged when it already is.

    A value already of the type is returned as it is rather than converted to
    itself: a conversion to the type a value already has is a no-op that still
    costs a kernel, and for a value that is not a loop nest there is no loop to
    put it in.  A value with a loop of its own is cast by a kernel of its own;
    only a scalar read inside a loop is cast by the loop's own arithmetic.
    """
    if is_tensor_box(value) and value.get_dtype() == dtype:
        return value
    if is_tensor_box(value):
        return to_dtype(value, dtype)
    return ops.to_dtype(value, dtype)


def normalize_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


# ---------------------------------------------------------------------------
# pointwise
# ---------------------------------------------------------------------------


def _alpha(args, kwargs, position):
    if len(args) > position:
        return args[position]
    return kwargs.get("alpha", 1)


@register_lowering(
    ["add.Tensor", "add.Scalar"],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lower_add(a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(ops.add, a, b)
    return pointwise(lambda x, y: ops.add(x, ops.mul(y, ops.constant(float(alpha), tp.float32))), a, b)


@register_lowering(
    ["sub.Tensor", "sub.Scalar"],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lower_sub(a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(ops.sub, a, b)
    return pointwise(lambda x, y: ops.sub(x, ops.mul(y, ops.constant(float(alpha), tp.float32))), a, b)


@register_lowering(
    ["rsub.Scalar", "rsub.Tensor"],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lower_rsub(a, b, *rest, **kwargs):
    return pointwise(lambda x, y: ops.sub(y, x), a, b)


# Two operands of different element types are computed in the type both fit
# in, as the sum above is: read in the first operand's type, a half value times
# a float one would be laid down as a half and lose what the float carried.
@register_lowering(
    ["mul.Tensor", "mul.Scalar"],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lower_mul(a, b):
    return pointwise(ops.mul, a, b)


# A quotient is a real number whatever it was a quotient of, so whole-number
# operands are read as real ones before they are divided.
@register_lowering(
    ["div.Tensor", "div.Scalar", "truediv", "truediv.Tensor", "truediv.Scalar",
     "true_divide.Tensor", "true_divide.Scalar"],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def lower_div(a, b):
    return pointwise(ops.truediv, a, b)


def _unary(op_name):
    # Named rather than reached for now: which handler is active is not known
    # until a region is being lowered, and a unary op is written down when the
    # module is read rather than when a region is walked.  Resolving the name
    # here would capture whatever was active at import, which is nothing.
    fn = ops_wrapper(op_name)

    def lower(x):
        return pointwise(fn, x)

    return lower


for _name, _op in {
    "neg.default": "neg", "exp.default": "exp", "log.default": "log",
    "sigmoid.default": "sigmoid", "rsqrt.default": "rsqrt", "sqrt.default": "sqrt",
    "reciprocal.default": "reciprocal", "abs.default": "abs", "sin.default": "sin",
    "cos.default": "cos", "tanh.default": "tanh", "relu.default": "relu",
    "sign.default": "sign",
}.items():
    LOWERINGS[_name] = _unary(_op)
    # The same walk under the bare method/functional spelling, which is how
    # a captured tensor method (``x.sin()``) or functional wrapper
    # (``tp.sin(x)``) reaches the graph.
    LOWERINGS[_op] = _unary(_op)

#: Two raised to a power, and the base-two logarithm.  Not the same operations
#: as the ones next to them: a kernel that has a device unit for squaring does
#: not have one for doubling, and a logarithm is a division by a constant
#: somewhere.  So they are named separately rather than written as the
#: operations they resemble.
register_pointwise_numeric("exp2.default", "log2.default")
LOWERINGS["exp2.default"] = lower_exp2 = _unary("exp2")
LOWERINGS["log2.default"] = lower_log2 = _unary("log2")

#: Whether two values are the same.  The answer is not a number: comparing two
#: values is how one of them is chosen, and a value that could be chosen as one
#: of two numbers is not a number.  So the type is stated here rather than
#: worked out from the arguments, which is what a comparison of numbers would
#: give.
register_pointwise("eq.default", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)


def _binary(_op: str):
    """A lowering of a two-argument operation, one element at a time.

    What the operation is called is decided by the table above rather than
    written here, so that one name means one operation everywhere it is asked
    for -- including from a whole list of tensors at once, which is the same
    operation applied to each of them.
    """

    fn = ops_wrapper(_op)

    def lower(x, y):
        return pointwise(fn, x, y)

    return lower


for _name, _op in {
    "maximum.default": "maximum", "minimum.default": "minimum",
}.items():
    LOWERINGS[_name] = _binary(_op)

#: The ways two values are ordered against each other.  The answer is not a
#: number: comparing two values is how one of them is chosen, and a value that
#: could be chosen as one of two numbers is not a number.  So the type is
#: stated rather than worked out from the arguments.
register_pointwise("le.default", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
register_pointwise("lt.default", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
register_pointwise("ge.default", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
register_pointwise("gt.default", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
register_pointwise("ne.default", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)


def _comparison(_op: str):
    """A lowering of a comparison: its answer is a truth value whatever it compared."""

    fn = ops_wrapper(_op)

    def lower(x, y):
        return pointwise(fn, x, y, out_dtype=tp.bool)

    return lower


LOWERINGS["le.default"] = lower_le = _comparison("le")
LOWERINGS["lt.default"] = lower_lt = _comparison("lt")
LOWERINGS["ge.default"] = lower_ge = _comparison("ge")
LOWERINGS["gt.default"] = lower_gt = _comparison("gt")
LOWERINGS["ne.default"] = lower_ne = _comparison("ne")
LOWERINGS["eq.default"] = lower_eq = _comparison("eq")
#: The forms where one side is a plain number: a value asked whether it is
#: below a bound is answered by the same comparison, with the number on the
#: other side of it.
LOWERINGS["le.Scalar"] = _comparison("le")
LOWERINGS["lt.Scalar"] = _comparison("lt")
LOWERINGS["ge.Scalar"] = _comparison("ge")
LOWERINGS["gt.Scalar"] = _comparison("gt")
LOWERINGS["ne.Scalar"] = _comparison("ne")
LOWERINGS["eq.Scalar"] = _comparison("eq")
#: The form between two values under the name the operation's own declaration
#: gives it.  A region recorded operation by operation holds the comparison
#: under that name, and without it the comparison is handed to the framework
#: whole instead of being written into the loop that reads it.
for _comparison in ("le", "lt", "ge", "gt", "ne", "eq"):
    LOWERINGS[f"{_comparison}.Tensor"] = LOWERINGS[f"{_comparison}.default"]


# ---------------------------------------------------------------------------
# The rest of the elementwise vocabulary.  Every operation below is one value
# per element computed from the same element of its operands, so each is a
# loop body the kernels around it can absorb; handed to the framework instead
# it would be a launch of its own and a round trip of its operands through
# memory.  The ones the device kernels have a unit for are named by that unit;
# the rest are written in terms of those.
# ---------------------------------------------------------------------------


def _is_real(dtype) -> bool:
    return dtype in (tp.float16, tp.bfloat16, tp.float32, tp.float64)


def _real_unary(op_name):
    """An operation that reads a number and produces a real number: an
    integer operand is read in the default real type first."""

    fn = ops_wrapper(op_name)

    def lower(x):
        if _is_real(x.get_dtype()):
            return pointwise(fn, x)
        dtype = tp.get_default_dtype()
        return pointwise(lambda v: fn(ops.to_dtype(v, dtype)), x, out_dtype=dtype)

    return lower


def _rounding_unary(op_name):
    """A rounding: an integer is already whole and is its own answer."""

    fn = ops_wrapper(op_name)

    def lower(x):
        if _is_real(x.get_dtype()):
            return pointwise(fn, x)
        return pointwise(lambda v: v, x)

    return lower


for _op in (
    "erf", "erfc", "erfinv", "expm1", "log1p", "log10", "sinh", "cosh", "tan",
    "asin", "acos", "atan", "asinh", "acosh", "atanh", "lgamma",
):
    LOWERINGS[f"{_op}.default"] = _real_unary(_op)
for _op in ("floor", "ceil", "trunc"):
    LOWERINGS[f"{_op}.default"] = _rounding_unary(_op)


def _real_predicate(op_name, integral_answer: bool):
    """A question about a real value; every integer gets the same answer."""

    fn = ops_wrapper(op_name)

    def lower(x):
        if _is_real(x.get_dtype()):
            return pointwise(fn, x, out_dtype=tp.bool)
        return pointwise(
            lambda v: ops.constant(integral_answer, tp.bool), x, out_dtype=tp.bool
        )

    return lower


LOWERINGS["isnan.default"] = _real_predicate("isnan", False)
LOWERINGS["isinf.default"] = _real_predicate("isinf", False)


def _overridden_pointwise(funcname, data, op_overload):
    """An element-wise operation each target spells for itself.

    The operands are promoted the way the operation says and read in that
    type.  A target with no spelling of the operation hands the call to the
    framework, which has one.
    """

    fn = ops_wrapper(funcname)

    def lower(*args):
        tensors = [a for a in args if isinstance(a, TensorBox)]
        if not tensors:
            raise NotImplementedError(f"{funcname} of numbers only")
        device = tensors[0].get_device()
        if (device.type == "cpu" and data.cpp is None) or (
            device.type != "cpu" and data.triton is None
        ):
            return fallback_handler(op_overload, add_to_fallback_set=False)(*args)
        from .dtype_propagation import get_promoted_dtype

        dtype = get_promoted_dtype(
            *[_promotion_input(a) for a in args], type_promotion_kind=data.type_promotion_kind
        )
        args = [
            to_dtype(a, dtype) if isinstance(a, TensorBox) and a.get_dtype() != dtype else a
            for a in args
        ]
        if len(tensors) > 1:
            shaped = iter(broadcast_tensors(*[a for a in args if isinstance(a, TensorBox)]))
            args = [next(shaped) if isinstance(a, TensorBox) else a for a in args]
        return pointwise(
            lambda *values: fn(*[ops.to_dtype(v, dtype) for v in values]),
            *args,
            val=next(a for a in args if isinstance(a, TensorBox)),
            out_dtype=dtype,
        )

    return lower


def _register_overridden_pointwise() -> None:
    """The special functions, each under the operation names it is reached by:
    its own and, where the framework names it differently, that name too."""

    from .codegen.common import pointwise_overrides_data

    for funcname, data in pointwise_overrides_data.items():
        for op_name in dict.fromkeys((data.name, funcname)):
            packet = getattr(tp_ops, op_name, None)
            if packet is None or not callable(getattr(packet, "overloads", None)):
                continue
            for overload in packet.overloads():
                key = f"{op_name}.{overload}"
                op_overload = getattr(packet, overload)
                schema = getattr(op_overload, "_schema", None)
                if schema is not None and schema.is_mutable:
                    continue
                if key in LOWERINGS:
                    continue
                LOWERINGS[key] = _overridden_pointwise(funcname, data, op_overload)


_register_overridden_pointwise()


@register("isfinite.default")
def lower_isfinite(x):
    if not _is_real(x.get_dtype()):
        return pointwise(lambda v: ops.constant(True, tp.bool), x, out_dtype=tp.bool)
    return pointwise(
        lambda v: ops.logical_not(ops.logical_or(ops.isnan(v), ops.isinf(v))),
        x,
        out_dtype=tp.bool,
    )


@register("signbit.default")
def lower_signbit(x):
    if _is_real(x.get_dtype()):
        return pointwise(ops.signbit, x, out_dtype=tp.bool)
    return pointwise(
        lambda v: ops.lt(v, ops.constant(0, x.get_dtype())), x, out_dtype=tp.bool
    )


@register("logical_not.default")
def lower_logical_not(x):
    return pointwise(
        lambda v: ops.logical_not(ops.to_dtype(v, tp.bool)), x, out_dtype=tp.bool
    )


@register("bitwise_not.default")
def lower_bitwise_not(x):
    if x.get_dtype() == tp.bool:
        return pointwise(ops.logical_not, x)
    return pointwise(ops.bitwise_not, x)


@register("square.default")
def lower_square(x):
    return pointwise(lambda v: ops.mul(v, v), x)


@register("frac.default")
def lower_frac(x):
    return pointwise(lambda v: ops.sub(v, ops.trunc(v)), x)


@register("sgn.default", "positive.default")
def lower_sgn_or_positive(x):
    node = V.current_node
    name = target_name(node.target) if node is not None else ""
    if name.startswith("positive"):
        return pointwise(lambda v: v, x)
    return pointwise(ops.sign, x)


def _promoted_pair(a, b):
    """The type two operands are computed in, the way an arithmetic
    operation promotes them: a tensor's type wins over a number's."""

    from .dtype_propagation import get_promoted_dtype

    return get_promoted_dtype(
        _promotion_input(a),
        _promotion_input(b),
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


def _binary_on(
    op_name,
    *,
    real: bool,
    out_bool: bool = False,
    to_bool: bool = False,
    whole_op: str | None = None,
):
    """A two-operand operation, broadcast and computed in the operands'
    promoted type (a real one when the operation produces a real number).

    ``whole_op`` names the operation computed instead when that type is a
    whole number: the floating-point form of some operations has no meaning
    for whole numbers, and another one computes the same result for them.
    """

    fn = ops_wrapper(op_name)
    whole_fn = ops_wrapper(whole_op) if whole_op is not None else fn

    def lower(a, b):
        # An operand may arrive as a bare view rather than a box.
        a, b = _box_view(a), _box_view(b)
        dtype = _promoted_pair(a, b)
        if real and not _is_real(dtype):
            dtype = tp.get_default_dtype()
        compute = tp.bool if to_bool else dtype
        # The operands are brought to the type they are computed in before a
        # number among them is made into a value, since a number is made in
        # the type of the operand beside it: 2.5 beside a whole-number tensor
        # would otherwise be read as 2.
        a, b = (
            to_dtype(x, compute) if isinstance(x, TensorBox) and x.get_dtype() != compute else x
            for x in (a, b)
        )
        tensors = [x for x in (a, b) if isinstance(x, TensorBox)]
        if len(tensors) == 2:
            a, b = broadcast_tensors(a, b)
        anchor = a if isinstance(a, TensorBox) else b
        op = whole_fn if is_integer_dtype(compute) or is_boolean_dtype(compute) else fn

        def inner(x, y):
            return op(ops.to_dtype(x, compute), ops.to_dtype(y, compute))

        return pointwise(
            inner, a, b, val=anchor, out_dtype=tp.bool if out_bool else dtype
        )

    return lower


for _op in ("atan2", "hypot", "nextafter"):
    LOWERINGS[f"{_op}.default"] = _binary_on(_op, real=True)
for _overload in ("Tensor", "Scalar"):
    LOWERINGS[f"copysign.{_overload}"] = _binary_on("copysign", real=True)
    # The remainder of whole numbers that truncates toward zero is ``mod``.
    LOWERINGS[f"fmod.{_overload}"] = _binary_on("fmod", real=False, whole_op="mod")
    for _op in ("bitwise_and", "bitwise_or", "bitwise_xor"):
        LOWERINGS[f"{_op}.{_overload}"] = _binary_on(_op, real=False)
for _op in ("bitwise_left_shift", "bitwise_right_shift"):
    for _overload in ("Tensor", "Tensor_Scalar"):
        LOWERINGS[f"{_op}.{_overload}"] = _binary_on(_op, real=False)
for _op in ("logical_and", "logical_or", "logical_xor"):
    LOWERINGS[f"{_op}.default"] = _binary_on(_op, real=False, out_bool=True, to_bool=True)
#: The symbol spellings of the bitwise operations: the language reads these as
#: operators rather than names, and behind each is the bitwise operation it is
#: spelled with.
for _dunder, _bitwise in (
    ("__and__", "bitwise_and"),
    ("__or__", "bitwise_or"),
    ("__xor__", "bitwise_xor"),
    ("__lshift__", "bitwise_left_shift"),
    ("__rshift__", "bitwise_right_shift"),
):
    for _overload in ("Tensor", "Scalar"):
        LOWERINGS[f"{_dunder}.{_overload}"] = _binary_on(_bitwise, real=False)


@register_lowering(tp_ops.ldexp, broadcast=True, type_promotion_kind=None)
def lower_ldexp(x, n):
    """The value scaled by a power of two, with the exponent read from the other
    input one place at a time.

    An exponent that is a whole number is what the device's own scaling answers,
    so it is asked for directly.  Any other pair falls back to a product with
    two raised to the exponent, which is the same number written the long way.
    """

    x_dtype = x.get_dtype()
    n_dtype = n.get_dtype() if hasattr(n, "get_dtype") else None
    n_is_int = n_dtype is None or (not _is_real(n_dtype) and n_dtype != tp.bool)
    if _is_real(x_dtype) and n_is_int:

        def inner(value, exponent):
            return ops.ldexp(value, exponent)

        return pointwise(inner, x, n, out_dtype=x_dtype)

    out_dtype = tp.float32 if is_integer_dtype(x_dtype) else x_dtype

    def inner(value, exponent):
        two = ops.constant(2.0, out_dtype)
        return ops.mul(
            ops.to_dtype(value, out_dtype),
            ops.pow(two, ops.to_dtype(exponent, out_dtype)),
        )

    return pointwise(inner, x, n, out_dtype=out_dtype)


def _scalar_const(value, like):
    return ops.constant(value, like.get_dtype() if _is_real(like.get_dtype()) else tp.float32)


def _lower_remainder(a, b):
    """The remainder whose sign follows the divisor (floor division's)."""

    # An operand may arrive as a bare view rather than a box.
    a, b = _box_view(a), _box_view(b)
    dtype = _promoted_pair(a, b)
    tensors = [x for x in (a, b) if isinstance(x, TensorBox)]
    if len(tensors) == 2:
        a, b = broadcast_tensors(a, b)
    anchor = a if isinstance(a, TensorBox) else b

    def fn(x, y):
        x = ops.to_dtype(x, dtype)
        y = ops.to_dtype(y, dtype)
        if _is_real(dtype):
            # x - floor(x / y) * y
            return ops.sub(x, ops.mul(ops.floor(ops.truediv(x, y)), y))
        return ops.remainder(x, y)

    return pointwise(fn, a, b, val=anchor, out_dtype=dtype)


for _overload in ("Tensor", "Scalar", "Scalar_Tensor"):
    LOWERINGS[f"remainder.{_overload}"] = _lower_remainder
LOWERINGS["mod"] = _lower_remainder


def _clamp_one_side(x, bound, op):
    """``x`` held on one side of ``bound``, in the type the two promote to: a
    fractional bound beside whole numbers is not read as a whole number."""

    dtype = _promoted_pair(x, bound)
    if x.get_dtype() != dtype:
        x = to_dtype(x, dtype)
    if isinstance(bound, TensorBox):
        if bound.get_dtype() != dtype:
            bound = to_dtype(bound, dtype)
        x, bound = broadcast_tensors(x, bound)
        return pointwise(op, x, bound)
    return pointwise(lambda v: op(v, ops.constant(bound, dtype)), x)


@register("clamp_min.default", "clamp_min.Tensor")
def lower_clamp_min(x, min):
    return _clamp_one_side(x, min, ops.maximum)


@register("clamp_max.default", "clamp_max.Tensor")
def lower_clamp_max(x, max):
    return _clamp_one_side(x, max, ops.minimum)


@register("lerp.Scalar", "lerp.Tensor", "lerp.default")
def lower_lerp(start, end, weight):
    """start + weight * (end - start), evaluated from whichever end the
    weight is nearer so the endpoints are reproduced exactly."""

    def fn(s, e, w):
        half = ops.constant(0.5, tp.float32)
        one = ops.constant(1.0, tp.float32)
        diff = ops.sub(e, s)
        near_start = ops.add(s, ops.mul(w, diff))
        near_end = ops.sub(e, ops.mul(diff, ops.sub(one, w)))
        return ops.where(ops.lt(ops.abs(w), half), near_start, near_end)

    tensors = [t for t in (start, end, weight) if isinstance(t, TensorBox)]
    if len(tensors) > 1:
        shaped = broadcast_tensors(*tensors)
        it = iter(shaped)
        start, end, weight = (
            next(it) if isinstance(t, TensorBox) else t for t in (start, end, weight)
        )
    return pointwise(fn, start, end, weight, val=start)


@register("nan_to_num.default")
def lower_nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    dtype = x.get_dtype()
    if not _is_real(dtype):
        return pointwise(lambda v: v, x)
    info = tp.finfo(dtype)
    nan = 0.0 if nan is None else nan
    posinf = info.max if posinf is None else posinf
    neginf = info.min if neginf is None else neginf

    def fn(v):
        zero = ops.constant(0.0, dtype)
        inf = ops.isinf(v)
        out = ops.where(ops.isnan(v), ops.constant(nan, dtype), v)
        out = ops.where(
            ops.logical_and(inf, ops.gt(v, zero)), ops.constant(posinf, dtype), out
        )
        return ops.where(
            ops.logical_and(inf, ops.lt(v, zero)), ops.constant(neginf, dtype), out
        )

    return pointwise(fn, x)


# Activations and the gradients of activations.  A gradient formula is as
# elementwise as the activation it differentiates, and in a training region it
# sits between the gradient of the layer after it and the reduction of the
# layer before it -- exactly where a separate launch costs the most.


def _f32(value):
    return ops.constant(float(value), tp.float32)


@register("tanh_backward.default")
def lower_tanh_backward(grad, output):
    # grad * (1 - y^2)
    return pointwise(
        lambda g, y: ops.mul(g, ops.sub(_f32(1.0), ops.mul(y, y))), grad, output
    )


@register("sigmoid_backward.default")
def lower_sigmoid_backward(grad, output):
    # grad * y * (1 - y)
    return pointwise(
        lambda g, y: ops.mul(ops.mul(g, y), ops.sub(_f32(1.0), y)), grad, output
    )


@register("threshold.default")
def lower_threshold(x, threshold, value, inplace=False):
    return pointwise(
        lambda v: ops.where(ops.le(v, _f32(threshold)), _f32(value), v), x
    )


@register("threshold_backward.default")
def lower_threshold_backward(grad, self, threshold):
    return pointwise(
        lambda g, v: ops.where(ops.le(v, _f32(threshold)), _f32(0.0), g), grad, self
    )


@register("leaky_relu.default")
def lower_leaky_relu(x, negative_slope=0.01, inplace=False):
    return pointwise(
        lambda v: ops.where(
            ops.gt(v, _f32(0.0)), v, ops.mul(v, _f32(negative_slope))
        ),
        x,
    )


@register("leaky_relu_backward.default")
def lower_leaky_relu_backward(grad, self, negative_slope, self_is_result=False):
    return pointwise(
        lambda g, v: ops.where(
            ops.gt(v, _f32(0.0)), g, ops.mul(g, _f32(negative_slope))
        ),
        grad,
        self,
    )


@register("hardtanh.default")
def lower_hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False):
    return pointwise(
        lambda v: ops.minimum(ops.maximum(v, _f32(min_val)), _f32(max_val)), x
    )


@register("hardtanh_backward.default")
def lower_hardtanh_backward(grad, self, min_val, max_val):
    return pointwise(
        lambda g, v: ops.where(
            ops.logical_or(ops.le(v, _f32(min_val)), ops.ge(v, _f32(max_val))),
            _f32(0.0),
            g,
        ),
        grad,
        self,
    )


@register("softplus.default")
def lower_softplus(x, beta=1.0, threshold=20.0):
    # log(1 + exp(beta x)) / beta, and x itself where beta x is past threshold
    def fn(v):
        scaled = ops.mul(v, _f32(beta))
        soft = ops.truediv(ops.log1p(ops.exp(scaled)), _f32(beta))
        return ops.where(ops.gt(scaled, _f32(threshold)), v, soft)

    return pointwise(fn, x)


@register("softplus_backward.default")
def lower_softplus_backward(grad, self, beta, threshold):
    def fn(g, v):
        scaled = ops.mul(v, _f32(beta))
        z = ops.exp(scaled)
        return ops.where(
            ops.gt(scaled, _f32(threshold)),
            g,
            ops.truediv(ops.mul(g, z), ops.add(z, _f32(1.0))),
        )

    return pointwise(fn, grad, self)


@register("elu.default")
def lower_elu(x, alpha=1.0, scale=1.0, input_scale=1.0, inplace=False):
    def fn(v):
        negative = ops.mul(
            _f32(alpha * scale), ops.expm1(ops.mul(v, _f32(input_scale)))
        )
        return ops.where(ops.gt(v, _f32(0.0)), ops.mul(v, _f32(scale)), negative)

    return pointwise(fn, x)


@register("elu_backward.default")
def lower_elu_backward(grad, alpha, scale, input_scale, is_result, self_or_result):
    def fn(g, v):
        if is_result:
            negative = ops.mul(
                ops.mul(g, _f32(input_scale)), ops.add(v, _f32(alpha * scale))
            )
        else:
            negative = ops.mul(
                ops.mul(g, _f32(input_scale * alpha * scale)),
                ops.exp(ops.mul(v, _f32(input_scale))),
            )
        return ops.where(ops.le(v, _f32(0.0)), negative, ops.mul(g, _f32(scale)))

    return pointwise(fn, grad, self_or_result)


@register("hardsigmoid.default")
def lower_hardsigmoid(x, inplace=False):
    return pointwise(
        lambda v: ops.minimum(
            ops.maximum(ops.add(ops.truediv(v, _f32(6.0)), _f32(0.5)), _f32(0.0)),
            _f32(1.0),
        ),
        x,
    )


@register("hardsigmoid_backward.default")
def lower_hardsigmoid_backward(grad, self):
    return pointwise(
        lambda g, v: ops.where(
            ops.logical_and(ops.gt(v, _f32(-3.0)), ops.lt(v, _f32(3.0))),
            ops.truediv(g, _f32(6.0)),
            _f32(0.0),
        ),
        grad,
        self,
    )


@register("hardswish.default")
def lower_hardswish(x, inplace=False):
    return pointwise(
        lambda v: ops.truediv(
            ops.mul(
                v,
                ops.minimum(
                    ops.maximum(ops.add(v, _f32(3.0)), _f32(0.0)), _f32(6.0)
                ),
            ),
            _f32(6.0),
        ),
        x,
    )


@register("hardswish_backward.default")
def lower_hardswish_backward(grad, self):
    def fn(g, v):
        middle = ops.mul(g, ops.add(ops.truediv(v, _f32(3.0)), _f32(0.5)))
        return ops.where(
            ops.lt(v, _f32(-3.0)),
            _f32(0.0),
            ops.where(ops.le(v, _f32(3.0)), middle, g),
        )

    return pointwise(fn, grad, self)


_SQRT1_2 = 0.7071067811865476
_INV_SQRT_2PI = 0.3989422804014327
_SQRT_2_OVER_PI = 0.7978845608028654
_GELU_KAPPA = 0.044715


@register("gelu.default")
def lower_gelu(x, approximate="none"):
    def exact(v):
        cdf = ops.mul(
            _f32(0.5), ops.add(_f32(1.0), ops.erf(ops.mul(v, _f32(_SQRT1_2))))
        )
        return ops.mul(v, cdf)

    def tanh(v):
        inner = ops.mul(
            _f32(_SQRT_2_OVER_PI),
            ops.add(v, ops.mul(_f32(_GELU_KAPPA), ops.mul(v, ops.mul(v, v)))),
        )
        return ops.mul(ops.mul(_f32(0.5), v), ops.add(_f32(1.0), ops.tanh(inner)))

    return pointwise(tanh if approximate == "tanh" else exact, x)


@register("gelu_backward.default")
def lower_gelu_backward(grad, self, approximate="none"):
    def exact(g, v):
        cdf = ops.mul(
            _f32(0.5), ops.add(_f32(1.0), ops.erf(ops.mul(v, _f32(_SQRT1_2))))
        )
        pdf = ops.mul(
            _f32(_INV_SQRT_2PI), ops.exp(ops.mul(_f32(-0.5), ops.mul(v, v)))
        )
        return ops.mul(g, ops.add(cdf, ops.mul(v, pdf)))

    def tanh(g, v):
        v2 = ops.mul(v, v)
        inner = ops.mul(
            _f32(_SQRT_2_OVER_PI),
            ops.add(v, ops.mul(_f32(_GELU_KAPPA), ops.mul(v2, v))),
        )
        t = ops.tanh(inner)
        left = ops.mul(_f32(0.5), v)
        right = ops.add(_f32(1.0), t)
        left_derivative = ops.mul(_f32(0.5), right)
        right_derivative = ops.mul(
            ops.mul(left, ops.sub(_f32(1.0), ops.mul(t, t))),
            ops.mul(
                _f32(_SQRT_2_OVER_PI),
                ops.add(_f32(1.0), ops.mul(_f32(3.0 * _GELU_KAPPA), v2)),
            ),
        )
        return ops.mul(g, ops.add(left_derivative, right_derivative))

    return pointwise(tanh if approximate == "tanh" else exact, grad, self)


@register("clamp.default")
def lower_clamp(x, min=None, max=None):
    """A value held between two others.

    Written as the two operations it is rather than as one, because there is no
    operation for holding a value between two others: it is a lower one and an
    upper one, and saying which is which is the whole of what clamping is.
    """

    if min is not None:
        x = lower_clamp_min(x, min)
    if max is not None:
        x = lower_clamp_max(x, max)
    return x


@register("silu.default", "silu", "swish.default", "swish")
def lower_silu(x):
    # silu(x) = x * sigmoid(x)
    return pointwise(lambda v: ops.mul(v, ops.sigmoid(v)), x)


@register("silu_backward.default")
def lower_silu_backward(grad, x):
    # grad * s * (1 + x * (1 - s)),  s = sigmoid(x)
    def fn(g, v):
        s = ops.sigmoid(v)
        one = ops.constant(1.0, tp.float32)
        return ops.mul(ops.mul(g, s), ops.add(one, ops.mul(v, ops.sub(one, s))))

    return pointwise(fn, grad, x)


def _copy_meets_a_write() -> bool:
    """Whether the region writes into what the copy being lowered reads or makes.

    An input updated in place, or a copy written into afterwards, changes the
    memory after the copy was asked for; the copy then has to be memory of
    its own, holding the value as it stood.
    """

    written = V.graph.written_storages()
    if not written:
        return False
    from .fx_utils import get_node_storage

    node = V.graph.current_node
    if node is None:
        return True
    source = node.args[0] if node.args else None
    for value in (source, node):
        if isinstance(value, Node) and get_node_storage(value) in written:
            return True
    return False


def _identity_copy_source(x):
    """The value itself, when a copy of it would arrive at the same bytes.

    Inside a region whose values are written once and only read, a copy that
    changes neither elements nor arrangement is the value.  That holds when
    its storage has not been laid down yet -- a fresh realization settles
    row-major, which is the arrangement a copy makes -- when the storage is
    already row-major, and when the value is a plain window onto row-major
    storage.  Anything else (a reshaped view whose reading is not a plain
    window, a frozen non-row-major buffer) keeps its copy, and so does a copy
    of memory the region writes into, or that is itself written into.
    """

    if _copy_meets_a_write():
        return None
    if isinstance(x, TensorBox) and isinstance(x.data, StorageBox):
        inner = x.data.data
        if isinstance(inner, (Pointwise, Reduction)):
            return x
        if isinstance(inner, Buffer) and inner.layout.is_contiguous():
            return x
        return None
    if isinstance(x, TensorBox) and isinstance(x.data, ReinterpretView):
        if x.data.get_layout().is_contiguous():
            return x
    return None


def _flat_window_name(x):
    """The storage's name, when the value reads that storage in order.

    A window with contiguous strides and no offset over row-major storage
    holds exactly the storage's bytes under another shape, so a copy of the
    window is a copy of the storage and both are one buffer.  Anything else --
    a shifted or strided window, a reshaped reading -- names only itself.
    """

    node = _underlying(x)
    if not isinstance(node, ReinterpretView):
        return None
    cursor = node
    while isinstance(cursor, ReinterpretView):
        layout = cursor.get_layout()
        size = [int(s) for s in cursor.get_size()]
        if int(layout.offset) != 0 or list(layout.stride) != contiguous_strides(size):
            return None
        cursor = cursor.data
    if isinstance(cursor, Buffer) and cursor.layout.is_contiguous():
        return cursor.get_name()
    return None


def _shared_copy(x, dtype):
    """One materialized copy per (value, element type), however many ask.

    Several consumers of the same value each ask for the same copy -- a
    gradient handed to several backward calls, each wanting it in the value's
    element type.  Answering every ask from the same result means one buffer
    where there would have been one per ask, and the reads still see the same
    elements.  The cache lives on the region, so it never outlives the values
    it names.  The value is named by its storage when the ask reads the
    storage in order, and by the node itself otherwise.
    """

    node = _underlying(x)
    # Only a window that reads row-major storage in order may name the copy
    # by the storage; anything else shares by its own node, since a view is
    # named by its base and two views of one base need not read it alike.
    name = _flat_window_name(x)
    # A value named by its own node is named by that node's identity.  The
    # entry keeps the node, both so the identity cannot be handed to a later
    # node once this one is collected and so a hit can be checked against the
    # node that is asking.
    anchor = None
    if name is None:
        name = id(node)
        anchor = node
    try:
        shape = tuple(int(s) for s in x.get_size())
    except (TypeError, ValueError):
        shape = tuple(str(s) for s in x.get_size())
    key = (name, shape, dtype_name(dtype))
    cache = getattr(V.graph, "_shared_copy_cache", None)
    if cache is None:
        cache = {}
        V.graph._shared_copy_cache = cache
    hit = cache.get(key)
    if hit is not None and hit[0] is anchor:
        return hit[1]
    # The copy is described in the element type that was asked for: described
    # in the source's, the converted values would be laid down in a buffer of
    # the old type and every reader would see the type the copy was meant to
    # leave behind.
    out = to_dtype(x, dtype, copy=True)
    cache[key] = (anchor, out)
    return out


# The element-type methods are conversions by another name; a region that
# records the method rather than the conversion it stands for is lowered the
# same way, with the type read off the result.
@register(
    "to.dtype", "to.device", "to.dtype_layout", "_to_copy.default",
    "half", "float", "double", "bfloat16", "long", "int", "short", "bool",
    "char", "byte",
)
def lower_to(x, *args, **kwargs):
    size, dtype, _ = val_info(node_val())
    if not kwargs.get("copy", False) and hasattr(x, "get_dtype") and dtype_name(x.get_dtype()) == dtype_name(dtype):
        # A request that changes no element type moves no bytes, so the value
        # itself answers it.  A spelling that also carries an arrangement is
        # only free for a boxed value, whose arrangement is still to be
        # decided; for a bare node it keeps the old path.
        spelling = "to.dtype"
        try:
            spelling = target_name(V.current_node.target)
        except Exception:
            pass
        if spelling in ("to.dtype", "to.device") or is_tensor_box(x):
            return x
        source = _identity_copy_source(x)
        if source is not None:
            return source
    return _shared_copy(x, dtype)


@register("clone.default", "contiguous.default")
def lower_clone(x, *args, **kwargs):
    source = _identity_copy_source(x)
    if source is not None:
        return source
    return _shared_copy(x, x.get_dtype())


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def make_view(x: TensorBox, size, reindex) -> TensorBox:
    # A view reads the value it looks at, not the box that value happens to be
    # held in, so the box is unwrapped here.
    if reindex is None:
        return View.create(x, size)
    return TensorBox(View(data=_underlying(x), size=size, reindex=reindex))


def _underlying(box):
    """The node a box holds, with the boxes themselves peeled off.

    A box says what may still be written into a value; what is inside it is
    the value.  Only boxes are peeled: a view is a value in its own right, and
    reading past one would answer a question about a different shape than the
    one asked about.
    """

    from .ir import MutableBox

    node = box
    while isinstance(node, MutableBox):
        node = node.data
    return node


def _flat_index(index, size):
    expr = sympy.Integer(0)
    for i, extent in zip(index, size):
        expr = expr * int(extent) + as_index(i)
    return expr


def _unflatten_index(flat, size):
    out = []
    stride = prod(size)
    for extent in size:
        stride //= max(int(extent), 1)
        if int(extent) == 1:
            out.append(sympy.Integer(0))
        elif stride == 1:
            out.append(modular_indexing(flat, 1, int(extent)) if out else flat)
        else:
            out.append(
                modular_indexing(flat, stride, int(extent))
                if out
                else FloorDiv(flat, sympy.Integer(stride))
            )
    return out


#: What a body may do besides reading and converting while still counting as
#: a lone element-type conversion.
_CAST_ONLY_OPS = frozenset({"load", "to_dtype", "constant"})


def _is_lone_cast(node) -> bool:
    """Whether a loop's body is its reads converted to another element type.

    Laying such a body down would put a whole copy of the input in memory,
    holding what every reader re-derives from the input itself: nothing a
    conversion of the reads cannot give them.  Anything the body computes
    beyond the conversion means real work whose result is worth keeping.
    """

    try:
        opcount = node.inner_fn_opcount()
    except Exception:
        return False
    return opcount.used_ops <= _CAST_ONLY_OPS


def reshape(x: TensorBox, new_size) -> TensorBox:
    """This value under a different shape, without moving anything if it can be.

    Memory that already lies in consecutive elements is described by the new
    shape rather than copied: the elements are where they were, and a shape, a
    stride and an offset say how to read them as the new shape.  The view is
    recorded before any storage exists, so a producer that is still only a
    loop stays unwritten and each consumer reads straight through the view;
    where the new shape cannot be reached by arithmetic on the position, the
    view says so and the elements are moved when the value is laid down.
    """

    old_size = tuple(_extent(s) for s in x.get_size())
    new_size = tuple(_extent(s) for s in new_size)
    if old_size == new_size:
        return x
    node = _underlying(x)
    viewed = _reinterpret_consecutive(node, new_size)
    if viewed is not None:
        return viewed
    if isinstance(node, (Pointwise, Reduction)) and isinstance(x.data, StorageBox):
        # A loop that has not been materialized has no storage to view, so it
        # is realized first; the resulting buffer is contiguous and the new
        # shape can be described as a plain reinterpret view of it.  A bare
        # element-type conversion is excluded: writing it down would put a
        # whole copy of the value in memory that every reader re-derives
        # anyway, so it stays unwritten and each reader converts on load.
        if not _is_lone_cast(node):
            x.data.realize()
            node = _underlying(x)
            viewed = _reinterpret_consecutive(node, new_size)
            if viewed is not None:
                return viewed
    return TensorBox(View.create(_underlying(x), new_size))


def _reinterpret_consecutive(node, new_size):
    """A buffer laid out as consecutive elements, read under a new shape.

    The view reads the buffer as consecutive elements, so the buffer is held to
    that arrangement first: a layout left open could be settled differently
    later, and the view would then read the wrong elements.  Settling it is
    also where rows may be padded apart, so whether the elements are still
    consecutive is asked of the settled layout, not of the open one; a padded
    buffer is read through the view's own index arithmetic instead.
    """

    if not isinstance(node, Buffer) or not node.layout.is_contiguous():
        return None
    node.freeze_layout()
    settled = node.layout
    if not settled.is_contiguous():
        return None
    return TensorBox(
        ReinterpretView(
            data=node,
            layout=FixedLayout(
                settled.device,
                settled.dtype,
                new_size,
                contiguous_strides(new_size),
                settled.offset,
                settled.is_pinned,
            ),
        )
    )


def _resolve_size(size, numel):
    size = [_extent(s) for s in size]
    if size.count(-1) > 1:
        raise ValueError("only one dimension can be inferred")
    if -1 in size:
        known = sympy.prod(s for s in size if s != -1)
        V.graph.sizevars.check(sympy.Ne(known, 0))
        V.graph.sizevars.check(sympy.Eq(sympy.Mod(numel, known), 0))
        size[size.index(-1)] = sympy.simplify(FloorDiv(numel, known))
    V.graph.sizevars.check_equals(sympy.prod(size), numel)
    return size


@register("view.default", "reshape.default", "_unsafe_view.default", "view.dtype_unused")
def lower_view(x, size):
    return reshape(x, _resolve_size(size, x.get_numel()))


@register("flatten.default", "flatten.int", "flatten", "flatten.using_ints")
def lower_flatten(x, start_dim=0, end_dim=-1):
    # Flatten merges the axes in ``[start_dim, end_dim]`` into one: the
    # leading and trailing axes stay as they are, and the merged axis holds
    # the product of the extents it spans.
    size = [_extent(s) for s in x.get_size()]
    rank = len(size)
    start = normalize_dim(start_dim, rank)
    end = normalize_dim(end_dim, rank)
    merged = sympy.prod(size[start : end + 1])
    return reshape(x, size[:start] + [merged] + size[end + 1 :])


@register("permute.default")
def lower_permute(x, dims):
    # A permutation is which axis each position is read along, so the view that
    # says so is told the order and works out the addressing from it.
    rank = len(x.get_size())
    dims = [normalize_dim(d, rank) for d in dims]
    return PermuteView.create(_underlying(x), tuple(dims))


@register("permute_backward.default")
def lower_permute_backward(grad, _input, dims):
    rank = len(grad.get_size())
    dims = [normalize_dim(d, rank) for d in dims]
    inverse = [0] * rank
    for position, d in enumerate(dims):
        inverse[d] = position
    return lower_permute(grad, inverse)


@register("transpose.default", "transpose.int")
def lower_transpose(x, d0, d1):
    rank = len(x.get_size())
    dims = list(range(rank))
    a, b = normalize_dim(d0, rank), normalize_dim(d1, rank)
    dims[a], dims[b] = dims[b], dims[a]
    return lower_permute(x, dims)


@register("t.default")
def lower_t(x):
    return lower_permute(x, list(reversed(range(len(x.get_size())))))


@register("unsqueeze.default")
def lower_unsqueeze(x, dim):
    # A dimension of extent one holds one element, so adding one names no
    # memory that was not already there.
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size) + 1)
    new_size = size[:dim] + [1] + size[dim:]
    return View.create(_underlying(x), new_size)


@register("squeeze.dim", "squeeze.dims", "squeeze.default")
def lower_squeeze(x, dim=None):
    size = list(x.get_size())
    rank = len(size)

    def extent_is_one(extent):
        # An extent carried in the index language reads as one only through
        # the equality it is asked, not through comparison against the number.
        return V.graph.sizevars.guard_or_false(sympy.Eq(extent, 1))

    if dim is None:
        dims = [d for d in range(rank) if extent_is_one(size[d])]
    elif isinstance(dim, (list, tuple)):
        dims = [
            normalize_dim(d, rank)
            for d in dim
            if extent_is_one(size[normalize_dim(d, rank)])
        ]
    else:
        d = normalize_dim(dim, rank)
        dims = [d] if extent_is_one(size[d]) else []
    if not dims:
        return x
    if dim is None:
        return SqueezeView.create(_underlying(x))
    new_size = [s for d, s in enumerate(size) if d not in dims]
    return View.create(_underlying(x), new_size)


@register("expand.default")
def lower_expand(x, size, *args, **kwargs):
    # A dimension of extent one reads the same element everywhere, so growing
    # one needs no memory and no copy.  The view that says so has to be the one
    # that knows how a shorter shape lines up with a longer one, since a value
    # of fewer dimensions is lined up by its innermost axes.
    return ExpandView.create(x, [_extent(s) for s in size])


@register("broadcast_tensors.default")
def lower_broadcast_tensors(tensors):
    # Each value seen in the shape they all share: a view, like expanding, so
    # nothing is copied for a value that only has to be repeated.
    return list(broadcast_tensors(*tensors))


def _slice(x, dim, start, end, step):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    extent = size[dim]
    start = 0 if start is None else int(start)
    end = extent if end is None else int(end)
    if start < 0:
        start += extent
    if end < 0:
        end += extent
    start = max(0, min(start, extent))
    end = max(start, min(end, extent))
    step = int(step or 1)
    new_size = list(size)
    new_size[dim] = (end - start + step - 1) // step
    if start == 0 and step == 1 and end == extent:
        return x
    return SliceView.create(
        _underlying(x), dim, start, end, step, clamp=True
    )


@register("slice.Tensor")
def lower_slice(x, dim=0, start=None, end=None, step=1):
    return _slice(x, dim, start, end, step)


def _as_box(node):
    """A lowering's result held the way a graph value is: a view is boxed as
    the window it is, anything else in storage of its own."""

    if is_tensor_box(node):
        return node
    if isinstance(node, BaseView):
        return TensorBox(node)
    return TensorBox.create(node)


def lower_basic_getitem(x, index):
    """``x[index]`` as views, for an index made of integers, slices, ``None``
    and at most one ``...``; None for any other index (tensors, booleans,
    lists), which the caller computes as a call instead."""

    if not isinstance(index, tuple):
        index = (index,)
    basic = (int, slice, type(None), type(Ellipsis))
    if any(isinstance(i, bool) or not isinstance(i, basic) for i in index):
        return None
    for part in index:
        if isinstance(part, slice):
            for bound in (part.start, part.stop, part.step):
                if bound is not None and not isinstance(bound, int):
                    return None
            if part.step is not None and part.step <= 0:
                return None
    size = list(x.get_size())
    if _static_ints(size) is None:
        return None
    consumed = sum(1 for i in index if isinstance(i, (int, slice)))
    if sum(1 for i in index if i is Ellipsis) > 1 or consumed > len(size):
        return None
    expanded = []
    for part in index:
        if part is Ellipsis:
            expanded.extend([slice(None)] * (len(size) - consumed))
        else:
            expanded.append(part)
    # Axes the index does not reach are kept whole.
    result = x
    dim = 0
    for part in expanded:
        if part is None:
            result = _as_box(LOWERINGS["unsqueeze.default"](result, dim))
            dim += 1
        elif isinstance(part, int):
            result = _as_box(lower_select(result, dim, part))
        else:
            if part != slice(None):
                result = _as_box(
                    _slice(result, dim, part.start, part.stop, part.step or 1)
                )
            dim += 1
    return result


@register("select.int")
def lower_select(x, dim, index):
    """One position along an axis, the axis dropped: a window, no copy."""

    size = list(x.get_size())
    dim = normalize_dim(int(dim), len(size))
    index = int(index)
    if index < 0:
        index += int(size[dim])
    window = _slice(x, dim, index, index + 1, 1)
    new_size = size[:dim] + size[dim + 1 :]
    return View.create(_underlying(window), new_size)


_COMPLEX_DTYPES = tuple(
    d for d in (getattr(tp, "complex32", None), getattr(tp, "complex64", None),
                getattr(tp, "complex128", None)) if d is not None
)


def _real_identity(op):
    """An operation that is the value itself for a real value: a conjugate,
    a detach (history is not part of a value), an alias."""

    fallback = fallback_handler(op, add_to_fallback_set=False)

    def lower(x, *args, **kwargs):
        if x.get_dtype() in _COMPLEX_DTYPES:
            return fallback(x, *args, **kwargs)
        return x

    return lower


for _op_name in ("conj", "_conj", "resolve_conj", "resolve_neg", "detach", "alias"):
    _packet = getattr(tp.ops.tp, _op_name, None)
    if _packet is not None:
        LOWERINGS[f"{_op_name}.default"] = _real_identity(_packet.default)

#: A value handed to a region is retagged as one the region owns: the values
#: and the sharing underneath are the ones it already has, so what is asked
#: for is the value itself.
@register("lift.default", "lift_fresh.default")
def lift(x, *args, **kwargs):
    return x


@register("slice_backward.default")
def lower_slice_backward(grad, self, dim=0, start=None, end=None, step=1):
    """The gradient of a slice: zeros the input's shape, the gradient laid
    into the sliced run."""

    size = list(self.get_size()) if hasattr(self, "get_size") else list(self)
    zeros = Pointwise.create(
        device=grad.get_device(),
        dtype=grad.get_dtype(),
        inner_fn=lambda index: ops.constant(0, grad.get_dtype()),
        ranges=size,
    )
    return slice_scatter(zeros, grad, dim, start, end, step)


@register("select_backward.default")
def lower_select_backward(grad, self, dim, index):
    """The gradient of a selection: zeros the input's shape, the gradient at
    the one position it was read from."""

    size = list(self.get_size()) if hasattr(self, "get_size") else list(self)
    zeros = Pointwise.create(
        device=grad.get_device(),
        dtype=grad.get_dtype(),
        inner_fn=lambda index: ops.constant(0, grad.get_dtype()),
        ranges=size,
    )
    return select_scatter(zeros, grad, dim, index)


# The forms a kernel template addresses its operands through.  A template
# names parts of a buffer as it writes the kernel -- a block of rows, the heads
# in another order -- and gets back a box whose contents are a view of the
# same storage, so the part it names is the memory it will read.  The extents
# it names them by may be symbols the kernel gives numbers to while it runs,
# which is why these keep every extent as the expression it was written in.


def permute(x: TensorBox, dims: Any) -> TensorBox:
    """The same buffer with its axes read in another order."""

    if not isinstance(x, TensorBox):
        raise AssertionError(f"expected a box, got {type(x)}")
    return TensorBox(PermuteView.create(x.data, tuple(dims)))


def slice_(x: TensorBox, dim: int = 0, start: Any = 0, end: Any = 2**63 - 1,
           step: Any = 1, clamp: bool = True) -> TensorBox:
    """Part of one axis of a buffer, as a view of it.

    ``clamp`` brings the two ends inside the axis, which is what a slice in a
    program means.  A template that has already worked its ends out says so
    with ``clamp=False``, since an end it computed is the end it wants even
    when it is a symbol the axis cannot be compared with.
    """

    if not isinstance(x, TensorBox):
        raise AssertionError(f"expected a box, got {type(x)}")
    dim = normalize_dim(dim, len(x.get_size()))
    return TensorBox(SliceView.create(x.data, dim, start, end, step, clamp=clamp))


def squeeze(x: TensorBox, dim: Any = None) -> TensorBox:
    """A buffer without the axes of extent one it was asked to drop."""

    if not isinstance(x, TensorBox):
        raise AssertionError(f"expected a box, got {type(x)}")
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))
    size = list(x.get_size())
    dims = {normalize_dim(d, len(size)) for d in (dim if isinstance(dim, (list, tuple)) else (dim,))}
    new_size = [
        s for d, s in enumerate(size)
        if not (d in dims and V.graph.sizevars.guard_or_false(sympy.Eq(s, 1)))
    ]
    return view(x, new_size) if new_size != size else x


def copy(dst: Any, src: Any, non_blocking: bool = False) -> TensorBox:
    """A fresh buffer holding ``src`` as ``dst`` is: its device, type and extents."""

    x = src
    if x.get_device() != dst.get_device():
        x = to_device(x, dst.get_device())
    if x.get_dtype() != dst.get_dtype():
        x = to_dtype(x, dst.get_dtype())
    if list(x.get_size()) != list(dst.get_size()):
        x = TensorBox(ExpandView.create(x.data if isinstance(x, TensorBox) else x, list(dst.get_size())))
    return clone(x)


@register("chunk.default")
def lower_chunk(x, chunks, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    piece = (size[dim] + chunks - 1) // chunks
    out = []
    start = 0
    while start < size[dim]:
        out.append(_slice(x, dim, start, min(start + piece, size[dim]), 1))
        start += piece
    return tuple(out)


@register("split.Tensor", "split.default", "unsafe_split.Tensor")
def lower_split(x, split_size, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    out = []
    start = 0
    while start < size[dim]:
        out.append(_slice(x, dim, start, min(start + int(split_size), size[dim]), 1))
        start += int(split_size)
    return tuple(out)


@register("split_with_sizes.default", "split.sizes")
def lower_split_with_sizes(x, sizes, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    out = []
    start = 0
    for s in sizes:
        out.append(_slice(x, dim, start, start + int(s), 1))
        start += int(s)
    return tuple(out)


@register("stack.default")
def lower_stack(tensors, dim=0):
    """A stack is a concatenation of the inputs, each given a new axis."""

    rank = len(tensors[0].get_size()) + 1
    dim = normalize_dim(int(dim), rank)
    unsqueeze = LOWERINGS["unsqueeze.default"]
    return lower_cat([unsqueeze(t, dim) for t in tensors], dim)


@register("cat.default")
def lower_cat(tensors, dim=0):
    """Concatenation as one pointwise loop selecting its source per index."""

    size, dtype, device = val_info(node_val())
    dim = normalize_dim(dim, len(size))
    # A view handed over by another lowering reads as the value it views.
    tensors = [_as_box(t) if isinstance(t, IRNode) else t for t in tensors]
    inputs = [t for t in tensors if is_tensor_box(t) and t.get_size()[dim] > 0]
    if not inputs:
        # Every operand was a constant or empty, so there is no source to read
        # per index.  Say so instead of emitting a body that yields nothing.
        raise NotImplementedError(
            "cat with no tensor operand to read per index"
            f" (operands={len(tensors)}, dim={dim})"
        )
    starts = []
    start = 0
    for t in inputs:
        starts.append(start)
        start += int(t.get_size()[dim])
    loaders = [t.make_loader() for t in inputs]

    def inner(index):
        position = ops.index_expr(index[dim], tp.int64)
        value = None
        for k in range(len(inputs)):
            lo = starts[k]
            hi = lo + int(inputs[k].get_size()[dim])
            shifted = list(index)
            shifted[dim] = index[dim] - lo
            if k == 0:
                cond = ops.lt(position, ops.constant(hi, tp.int64))
            elif k == len(inputs) - 1:
                cond = ops.ge(position, ops.constant(lo, tp.int64))
            else:
                cond = ops.and_(
                    ops.ge(position, ops.constant(lo, tp.int64)),
                    ops.lt(position, ops.constant(hi, tp.int64)),
                )
            loaded = ops.masked(cond, lambda k=k, shifted=shifted: loaders[k](shifted), 0.0)
            value = loaded if value is None else ops.where(cond, loaded, value)
        return value

    result = Pointwise.create(device=device, dtype=dtype, inner_fn=inner, ranges=size)
    stores_channels_last = getattr(V.graph, "stores_channels_last", None)
    if (
        getattr(V.graph, "layout_opt", False)
        and stores_channels_last is not None
        and stores_channels_last(V.graph.current_node)
    ):
        # A join the region stores channels-last is written in that order
        # here, once, rather than being stored row-major by whichever reader
        # asks first and repacked for the convolutions that read it.
        result = V.graph.in_channels_last_order(result)
    return result


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype, default):
    """An optional dtype, with the graph's "unspecified" sentinel read as none.

    The dispatch graph spells an omitted optional dtype as ``tensorplay.undefined``
    rather than as ``None``, so both have to be treated as "not given" before the
    value the caller actually asked for (or the input's own type) is chosen.
    """

    if dtype is None or dtype == tp.undefined:
        return default
    return dtype


def make_reduction(x: TensorBox, dims, keepdim, dtype, device, rtype="sum", prologue=None) -> TensorBox:
    # A lowering is handed a value, not a node: what may still be written into
    # it is part of what it is, and realizing below is a question about the
    # value rather than about whatever happens to be holding it.
    if not isinstance(x, TensorBox):
        x = TensorBox.create(x)
    src_dtype = x.get_dtype()
    size = list(x.get_size())
    rank = len(size)
    dims = sorted({normalize_dim(d, rank) for d in dims})
    out_ranges = [size[d] for d in range(rank) if d not in dims]
    red_ranges = [size[d] for d in dims]
    loader = x.make_loader()
    # A total or a product asked for in another type is accumulated in that
    # type: counting truths is a sum of whole numbers, not of truths, and a
    # sum of truths stays a truth however many there are.
    convert = (
        rtype in ("sum", "prod")
        and dtype is not None
        and dtype != src_dtype
        and prologue is None
    )
    if convert:
        src_dtype = dtype

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dims else next(it) for d in range(rank)]
        value = loader(full)
        if prologue is not None:
            value = prologue(value, full)
        if convert:
            value = ops.to_dtype(value, dtype)
        return value

    box = Reduction.create(
        device=device,
        dst_dtype=dtype,
        src_dtype=src_dtype,
        inner_fn=inner,
        ranges=out_ranges,
        reduction_ranges=red_ranges,
        reduction_type=rtype,
    )
    if isinstance(box.data.data, Reduction):
        # A body that was unrolled into a body of its own is already settled;
        # one that stayed a reduction can still take more work, and realizing
        # it is what ends that.
        box.realize()
    if keepdim:
        kept = [1 if d in dims else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dims]

        return make_view(box, kept, reindex)
    return box


@register("sum.dim_IntList", "sum.default")
def lower_sum(x, dim=None, keepdim=False, dtype=None, **kwargs):
    # An empty list of dimensions reduces over all of them, as no list does.
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    if dtype is None or dtype == tp.undefined:
        # A sum of whole numbers is not a whole number unless it is asked to be.
        if is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype()):
            dtype = tp.int64
        else:
            dtype = x.get_dtype()
    return make_reduction(
        x, dim, keepdim, dtype, x.get_device(), "sum"
    )


@register("sum_to_size.default")
def lower_sum_to_size(x, size):
    """The sum over the axes a broadcast to ``x``'s shape added or widened:
    the leading ones it added go away, the ones it widened from one stay."""

    own = list(x.get_size())
    size = list(size)
    lead = len(own) - len(size)
    widened = [
        lead + i for i, (want, have) in enumerate(zip(size, own[lead:]))
        if V.graph.sizevars.statically_known_equals(want, 1)
        and not V.graph.sizevars.statically_known_equals(have, 1)
    ]
    reduced = list(range(lead)) + widened
    if not reduced:
        return x
    total = lower_sum(x, reduced, keepdim=True)
    return view(total, size)


@register("mean", "mean.dim")
def lower_mean(x, dim=None, keepdim=False, dtype=None, **kwargs):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    dtype = _resolve_dtype(dtype, x.get_dtype())
    count = sympy.prod(x.get_size()[normalize_dim(d, len(x.get_size()))] for d in dim)
    total = make_reduction(
        x, dim, keepdim, dtype, x.get_device(), "sum"
    )
    return pointwise(lambda v: ops.truediv(v, ops.index_expr(count, dtype)), total)


@register("var.dim", "var.correction")
def lower_var(x, dim=None, correction=1, keepdim=False, **kwargs):
    if correction is None:
        correction = 1
    """Variance as the mean of squared deviations from the mean.

    The mean and the total of the squared differences from it are carried in a
    single walk over the reduced axes, so the variance is that total divided by
    the count, or by the count less the requested correction.  Keeping the two
    running values instead of a sum of squares keeps the accumulation in range
    for a long group.
    """

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    dtype = x.get_dtype()
    device = x.get_device()
    size = list(x.get_size())
    rank = len(size)
    dim = sorted({normalize_dim(d, rank) for d in dim})
    out_ranges = [size[d] for d in range(rank) if d not in dim]
    red_ranges = [size[d] for d in dim]
    loader = x.make_loader()

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dim else next(it) for d in range(rank)]
        return ops.to_dtype(loader(full), dtype)

    _mean, m2, _weight = ir.WelfordReduction.create(
        device=device,
        dtype=dtype,
        inner_fns=(inner,),
        ranges=out_ranges,
        reduction_ranges=red_ranges,
        reduction_type="welford_reduce",
    )
    m2.realize()
    n_elems = prod(red_ranges)
    denom = Max(sympy.Integer(n_elems) - sympy.sympify(correction), 0)
    m2_loader = m2.make_loader()

    var = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda index: ops.truediv(
            m2_loader(index), ops.constant(float(denom), tp.float32)
        ),
        ranges=out_ranges,
    )
    if keepdim:
        kept = [1 if d in dim else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dim]

        return make_view(var, kept, reindex)
    return var


@register("var_mean.default", "var_mean.dim", "var_mean.correction", "var_mean")
def lower_var_mean(x, dim=None, unbiased=True, keepdim=False, *, correction=None, **kwargs):
    """The mean and variance of the reduced axes in one walk.

    Both statistics come out of the same running mean and total of squared
    differences, so the reduced data is read once rather than once for the
    mean and once for the variance.  The variance is biased when no correction
    is asked for and divided by the count less one otherwise.
    """

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    # The result element type is what the graph declares for this node, which
    # can be wider than the input's: a normalization computes its moments in
    # float32 even when the input is float16.  The input's own type is only a
    # fallback for when the graph has not declared one.
    dtype = x.get_dtype()
    val = node_val(index=0)
    if val is not None and getattr(val, "dtype", None) is not None:
        dtype = val.dtype
    device = x.get_device()
    size = list(x.get_size())
    rank = len(size)
    dim = sorted({normalize_dim(d, rank) for d in dim})
    out_ranges = [size[d] for d in range(rank) if d not in dim]
    red_ranges = [size[d] for d in dim]
    loader = x.make_loader()

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dim else next(it) for d in range(rank)]
        return ops.to_dtype(loader(full), dtype)

    mean, m2, _weight = ir.WelfordReduction.create(
        device=device,
        dtype=dtype,
        inner_fns=(inner,),
        ranges=out_ranges,
        reduction_ranges=red_ranges,
        reduction_type="welford_reduce",
    )
    mean.realize()
    m2.realize()
    n_elems = prod(red_ranges)
    if correction is None:
        # Unstated, the variance is the unbiased one.
        correction = 1 if (unbiased is None or unbiased) else 0
    denom = Max(sympy.Integer(n_elems) - sympy.sympify(correction), 0)
    m2_loader = m2.make_loader()

    var = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda index: ops.truediv(
            m2_loader(index), ops.constant(float(denom), tp.float32)
        ),
        ranges=out_ranges,
    )
    if keepdim:
        kept = [1 if d in dim else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dim]

        return make_view(var, kept, reindex), make_view(mean, kept, reindex)
    return var, mean


@register("amax.default")
def lower_amax(x, dim=None, keepdim=False, dtype=None, **kwargs):
    # What the reduction produces is read from the value it reduces: reducing
    # over axes changes how many there are and leaves a value with the same
    # element type.  A dtype the caller asked for is a different thing, and
    # overrides what that would say.
    # An empty list of dimensions reduces over all of them, as no list does.
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    return make_reduction(x, dim, keepdim, _resolve_dtype(dtype, x.get_dtype()), x.get_device(), "max")


@register("amin.default")
def lower_amin(x, dim=None, keepdim=False, dtype=None, **kwargs):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    return make_reduction(x, dim, keepdim, _resolve_dtype(dtype, x.get_dtype()), x.get_device(), "min")


@register("prod.default", "prod.dim_int", "prod.dim_IntList")
def lower_prod(x, dim=None, keepdim=False, dtype=None, **kwargs):
    """The product along the axes; a product of whole numbers widens unless
    another type is asked for, as a total does."""

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    dtype = _resolve_dtype(dtype, None)
    if dtype is None:
        whole = is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        dtype = tp.int64 if whole else x.get_dtype()
    return make_reduction(x, dim, keepdim, dtype, x.get_device(), "prod")


@register("mean.default")
def lower_mean_all(x, dtype=None, **kwargs):
    """The mean of every element: the mean over all the axes."""

    return lower_mean(x, None, False, dtype)


@register("max.dim")
def lower_max_dim(x, dim, keepdim=False):
    """The largest value along an axis, and where it is."""

    return lower_amax(x, [dim], keepdim), reduce_argmax(x, dim, keepdim)


@register("min.dim")
def lower_min_dim(x, dim, keepdim=False):
    """The smallest value along an axis, and where it is."""

    return lower_amin(x, [dim], keepdim), reduce_argmin(x, dim, keepdim)


@register("min.default")
def lower_min_all(x):
    """The smallest of every element."""

    return lower_amin(x, None, False)


@register("any.default", "any.dim", "any.dims")
def lower_any(x, dim=None, keepdim=False):
    """Whether any element along the axes is true: a reduction by ``or`` of
    the elements read as truth values."""

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(len(x.get_size())))
    elif isinstance(dim, (int, sympy.Integer)):
        dim = [dim]
    if x.get_dtype() != tp.bool:
        x = to_dtype(x, tp.bool)
    return make_reduction(x, dim, keepdim, tp.bool, x.get_device(), "any")


# ---------------------------------------------------------------------------
# Normalizations and the softmax family, written as the reductions and
# elementwise passes they are.  Each is a statistic over some axes followed by
# an elementwise pass that reads it; written this way the scheduler puts the
# statistic, the pass and whatever surrounds them (the residual add before a
# layer norm, the activation after it, the scaling before a softmax) into one
# kernel, where a library call would be a wall none of them can cross.
# ---------------------------------------------------------------------------


def _given(value) -> bool:
    """Whether an optional operand was given: a value reaches a lowering
    boxed, or bare when it is a view of one (an expanded weight)."""

    return isinstance(value, ir.IRNode)


def _accumulation_dtype(dtype):
    """The type a statistic of values of this type is accumulated in."""

    return tp.float32 if dtype in (tp.float16, tp.bfloat16) else dtype


def _static_ints(values) -> list[int] | None:
    out = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return None
    return out


def _reduce_rows(device, dtype, rows, cols, value, reduction_type):
    """One statistic per row of a rows x cols reading: value(row, col)."""

    box = Reduction.create(
        device=device,
        dst_dtype=dtype,
        src_dtype=dtype,
        inner_fn=lambda index, rindex: value(index[0], rindex[0]),
        ranges=[rows],
        reduction_ranges=[cols],
        reduction_type=reduction_type,
    )
    box.realize()
    return box


def _reduce_cols(device, dtype, rows, cols, value, reduction_type):
    """One statistic per column of a rows x cols reading: value(row, col)."""

    box = Reduction.create(
        device=device,
        dst_dtype=dtype,
        src_dtype=dtype,
        inner_fn=lambda index, rindex: value(rindex[0], index[0]),
        ranges=[cols],
        reduction_ranges=[rows],
        reduction_type=reduction_type,
    )
    box.realize()
    return box


class _RowView:
    """A tensor read as rows x cols: the leading axes flattened into the row
    and the trailing ``inner`` axes into the column, in row-major order."""

    def __init__(self, size, inner_rank):
        self.size = list(size)
        self.axis = len(self.size) - inner_rank
        self.outer = self.size[: self.axis]
        self.inner = self.size[self.axis :]
        self.rows = prod(self.outer) if self.outer else 1
        self.cols = prod(self.inner) if self.inner else 1

    def full(self, row, col):
        return _unflatten_index(row, self.outer) + _unflatten_index(col, self.inner)

    def row_of(self, index):
        flat = sympy.Integer(0)
        for i, extent in zip(index[: self.axis], self.outer):
            flat = flat * int(extent) + i
        return flat

    def col_of(self, index):
        flat = sympy.Integer(0)
        for i, extent in zip(index[self.axis :], self.inner):
            flat = flat * int(extent) + i
        return flat


_fallback_layer_norm = fallback_handler(tp.ops.tp.native_layer_norm.default)
_fallback_layer_norm_backward = fallback_handler(
    tp.ops.tp.native_layer_norm_backward.default
)


@register("native_layer_norm.default")
def lower_native_layer_norm(x, normalized_shape, weight, bias, eps):
    """(x - mean) * rstd * weight + bias over the trailing axes, with the
    per-row mean and reciprocal standard deviation as further results."""

    size = _static_ints(x.get_size())
    if size is None or len(normalized_shape) > len(size) or 0 in size:
        return _fallback_layer_norm(x, normalized_shape, weight, bias, eps)
    view = _RowView(size, len(normalized_shape))
    device = x.get_device()
    in_dtype = x.get_dtype()
    acc = _accumulation_dtype(in_dtype)
    x_l = x.make_loader()

    def element(row, col):
        return ops.to_dtype(x_l(view.full(row, col)), acc)

    mean, m2, _weight = ir.WelfordReduction.create(
        device=device,
        dtype=acc,
        inner_fns=(lambda index, rindex: element(index[0], rindex[0]),),
        ranges=[view.rows],
        reduction_ranges=[view.cols],
        reduction_type="welford_reduce",
    )
    mean.realize()
    m2.realize()
    m2_l = m2.make_loader()
    mean_l = mean.make_loader()
    rstd = Pointwise.create(
        device=device,
        dtype=acc,
        inner_fn=lambda index: ops.rsqrt(
            ops.add(
                ops.truediv(m2_l(index), ops.constant(float(view.cols), acc)),
                ops.constant(float(eps), acc),
            )
        ),
        ranges=[view.rows],
    )
    rstd.realize()
    rstd_l = rstd.make_loader()
    w_l = weight.make_loader() if _given(weight) else None
    b_l = bias.make_loader() if _given(bias) else None

    def out_fn(index):
        row = [view.row_of(index)]
        inner = list(index[view.axis :])
        v = ops.mul(
            ops.sub(ops.to_dtype(x_l(index), acc), mean_l(row)), rstd_l(row)
        )
        if w_l is not None:
            v = ops.mul(v, ops.to_dtype(w_l(inner), acc))
        if b_l is not None:
            v = ops.add(v, ops.to_dtype(b_l(inner), acc))
        return ops.to_dtype(v, in_dtype)

    out = Pointwise.create(device=device, dtype=in_dtype, inner_fn=out_fn, ranges=size)
    return out, mean, rstd


@register("layer_norm.default", "layer_norm")
def lower_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, *rest):
    result = lower_native_layer_norm(x, normalized_shape, weight, bias, eps)
    return result[0] if isinstance(result, (tuple, list)) else result


@register("native_layer_norm_backward.default")
def lower_native_layer_norm_backward(
    grad_out, x, normalized_shape, mean, rstd, weight, bias, output_mask
):
    """The three gradients of a layer norm from the saved row statistics.

    With x̂ = (x - mean) * rstd and ĝ = grad * weight, the input gradient is
    rstd / N * (N ĝ - Σ ĝ - x̂ Σ x̂ ĝ) with the sums over the row; the weight
    and bias gradients are Σ grad x̂ and Σ grad down the columns.
    """

    size = _static_ints(x.get_size())
    if size is None or 0 in size:
        return _fallback_layer_norm_backward(
            grad_out, x, normalized_shape, mean, rstd, weight, bias, output_mask
        )
    view = _RowView(size, len(normalized_shape))
    device = x.get_device()
    in_dtype = x.get_dtype()
    acc = _accumulation_dtype(in_dtype)
    g_l = grad_out.make_loader()
    x_l = x.make_loader()
    mean_l = mean.make_loader()
    rstd_l = rstd.make_loader()
    w_l = weight.make_loader() if _given(weight) else None

    def x_hat(full, row):
        return ops.mul(
            ops.sub(ops.to_dtype(x_l(full), acc), ops.to_dtype(mean_l([row]), acc)),
            ops.to_dtype(rstd_l([row]), acc),
        )

    def g_hat(full):
        g = ops.to_dtype(g_l(full), acc)
        if w_l is not None:
            g = ops.mul(g, ops.to_dtype(w_l(full[view.axis :]), acc))
        return g

    results = [None, None, None]
    if output_mask[0]:
        sum_g = _reduce_rows(
            device, acc, view.rows, view.cols,
            lambda r, c: g_hat(view.full(r, c)), "sum",
        )
        sum_gx = _reduce_rows(
            device, acc, view.rows, view.cols,
            lambda r, c: ops.mul(g_hat(view.full(r, c)), x_hat(view.full(r, c), r)),
            "sum",
        )
        sum_g_l = sum_g.make_loader()
        sum_gx_l = sum_gx.make_loader()

        def grad_input(index):
            row = view.row_of(index)
            count = ops.constant(float(view.cols), acc)
            inner = ops.sub(
                ops.sub(ops.mul(g_hat(list(index)), count), sum_g_l([row])),
                ops.mul(x_hat(list(index), row), sum_gx_l([row])),
            )
            scale = ops.truediv(ops.to_dtype(rstd_l([row]), acc), count)
            return ops.to_dtype(ops.mul(scale, inner), in_dtype)

        results[0] = Pointwise.create(
            device=device, dtype=in_dtype, inner_fn=grad_input, ranges=size
        )
    inner_shape = view.inner

    def by_column(value):
        flat = _reduce_cols(device, acc, view.rows, view.cols, value, "sum")
        flat_l = flat.make_loader()
        return Pointwise.create(
            device=device,
            dtype=in_dtype,
            inner_fn=lambda index: ops.to_dtype(
                flat_l([view.col_of([sympy.Integer(0)] * view.axis + list(index))]),
                in_dtype,
            ),
            ranges=inner_shape,
        )

    if output_mask[1] and _given(weight):
        results[1] = by_column(
            lambda r, c: ops.mul(
                ops.to_dtype(g_l(view.full(r, c)), acc), x_hat(view.full(r, c), r)
            )
        )
    if output_mask[2] and _given(bias):
        results[2] = by_column(lambda r, c: ops.to_dtype(g_l(view.full(r, c)), acc))
    return tuple(results)


def _softmax_like(x, dim, out_dtype, log):
    size = _static_ints(x.get_size())
    rank = len(x.get_size())
    if rank == 0:
        return None
    d = normalize_dim(int(dim), rank)
    if size is None or size[d] == 0:
        return None
    device = x.get_device()
    in_dtype = x.get_dtype()
    acc = _accumulation_dtype(in_dtype)
    others = [s for i, s in enumerate(size) if i != d]
    view_rows = prod(others) if others else 1
    x_l = x.make_loader()

    def full(row, col):
        outer = _unflatten_index(row, others)
        return outer[:d] + [col] + outer[d:]

    def row_of(index):
        flat = sympy.Integer(0)
        for i, (idx, extent) in enumerate(zip(index, size)):
            if i != d:
                flat = flat * int(extent) + idx
        return flat

    peak = _reduce_rows(
        device, acc, view_rows, size[d],
        lambda r, c: ops.to_dtype(x_l(full(r, c)), acc), "max",
    )
    peak_l = peak.make_loader()
    total = _reduce_rows(
        device, acc, view_rows, size[d],
        lambda r, c: ops.exp(ops.sub(ops.to_dtype(x_l(full(r, c)), acc), peak_l([r]))),
        "sum",
    )
    total_l = total.make_loader()

    def fn(index):
        row = [row_of(index)]
        shifted = ops.sub(ops.to_dtype(x_l(index), acc), peak_l(row))
        if log:
            value = ops.sub(shifted, ops.log(total_l(row)))
        else:
            value = ops.truediv(ops.exp(shifted), total_l(row))
        return ops.to_dtype(value, out_dtype)

    return Pointwise.create(device=device, dtype=out_dtype, inner_fn=fn, ranges=size)


def _softmax_out_dtype(x, dtype=None, half_to_float=False):
    if dtype is not None and dtype != tp.undefined:
        return dtype
    if half_to_float:
        return tp.float32
    return x.get_dtype()


def _softmax_lowering(op, log):
    fallback = fallback_handler(op, add_to_fallback_set=False)

    def lower(x, dim, third=None, **kwargs):
        # ``_softmax(x, dim, half_to_float)`` and ``softmax(x, dim, dtype)``
        # differ only in how the result type is named.
        if isinstance(third, bool):
            out_dtype = _softmax_out_dtype(x, half_to_float=third)
        else:
            out_dtype = _softmax_out_dtype(x, dtype=kwargs.get("dtype", third))
        if out_dtype != x.get_dtype() and not log:
            x = to_dtype(x, out_dtype)
        result = _softmax_like(x, dim, out_dtype, log)
        if result is None:
            return fallback(x, dim, third) if third is not None else fallback(x, dim)
        return result

    return lower


for _name, _log in (
    ("_softmax.default", False), ("softmax.int", False), ("softmax.default", False),
    ("_log_softmax.default", True), ("log_softmax.int", True), ("log_softmax.default", True),
):
    _op_name, _overload = _name.split(".")
    LOWERINGS[_name] = _softmax_lowering(
        getattr(getattr(tp.ops.tp, _op_name), _overload), _log
    )


def _softmax_backward_lowering(op, log):
    fallback = fallback_handler(op, add_to_fallback_set=False)

    def lower(grad, output, dim, input_dtype):
        size = _static_ints(grad.get_size())
        rank = len(grad.get_size())
        if size is None or rank == 0:
            return fallback(grad, output, dim, input_dtype)
        d = normalize_dim(int(dim), rank)
        device = grad.get_device()
        acc = _accumulation_dtype(grad.get_dtype())
        others = [s for i, s in enumerate(size) if i != d]
        rows = prod(others) if others else 1
        g_l = grad.make_loader()
        y_l = output.make_loader()

        def full(row, col):
            outer = _unflatten_index(row, others)
            return outer[:d] + [col] + outer[d:]

        def row_of(index):
            flat = sympy.Integer(0)
            for i, (idx, extent) in enumerate(zip(index, size)):
                if i != d:
                    flat = flat * int(extent) + idx
            return flat

        def summand(r, c):
            g = ops.to_dtype(g_l(full(r, c)), acc)
            if log:
                return g
            return ops.mul(g, ops.to_dtype(y_l(full(r, c)), acc))

        total = _reduce_rows(device, acc, rows, size[d], summand, "sum")
        total_l = total.make_loader()
        out_dtype = input_dtype if input_dtype is not None else grad.get_dtype()

        def fn(index):
            row = [row_of(index)]
            g = ops.to_dtype(g_l(index), acc)
            y = ops.to_dtype(y_l(index), acc)
            if log:
                value = ops.sub(g, ops.mul(ops.exp(y), total_l(row)))
            else:
                value = ops.mul(y, ops.sub(g, total_l(row)))
            return ops.to_dtype(value, out_dtype)

        return Pointwise.create(device=device, dtype=out_dtype, inner_fn=fn, ranges=size)

    return lower


LOWERINGS["_softmax_backward_data.default"] = _softmax_backward_lowering(
    tp.ops.tp._softmax_backward_data.default, False
)
LOWERINGS["_log_softmax_backward_data.default"] = _softmax_backward_lowering(
    tp.ops.tp._log_softmax_backward_data.default, True
)


_fallback_batch_norm = fallback_handler(tp.ops.tp.batch_norm.default, add_to_fallback_set=False)


def _lower_batch_norm_training(x, weight, bias, running_mean, running_var, momentum, eps):
    """A training batch norm: the batch's statistics, the normalized value,
    and the running statistics moved toward the batch's.

    The mean and the biased variance per channel come from one Welford pass
    over every other axis; the running variance takes the unbiased one.  The
    running statistics are written in place, as the call does.  None when the
    shape is not known or the call needs what this does not spell (a
    cumulative average, a batch of one element per channel).
    """

    size = _static_ints(x.get_size())
    has_running = _given(running_mean) and _given(running_var)
    if size is None or len(size) < 2 or 0 in size:
        return None
    if has_running and momentum is None:
        return None
    device = x.get_device()
    in_dtype = x.get_dtype()
    acc = _accumulation_dtype(in_dtype)
    channels = size[1]
    rest = [size[0], *size[2:]]
    count = prod(rest)
    if has_running and count <= 1:
        return None
    x_l = x.make_loader()
    w_l = weight.make_loader() if _given(weight) else None
    b_l = bias.make_loader() if _given(bias) else None

    def full(c, r):
        parts = _unflatten_index(r, rest)
        return [parts[0], c, *parts[1:]]

    mean, m2, _count = ir.WelfordReduction.create(
        device=device,
        dtype=acc,
        inner_fns=(
            lambda index, rindex: ops.to_dtype(x_l(full(index[0], rindex[0])), acc),
        ),
        ranges=[channels],
        reduction_ranges=[count],
        reduction_type="welford_reduce",
    )
    mean.realize()
    m2.realize()
    mean_l = mean.make_loader()
    m2_l = m2.make_loader()

    def fn(index):
        c = [index[1]]
        rstd = ops.rsqrt(
            ops.add(
                ops.truediv(m2_l(c), ops.constant(float(count), acc)),
                ops.constant(float(eps), acc),
            )
        )
        v = ops.mul(ops.sub(ops.to_dtype(x_l(index), acc), mean_l(c)), rstd)
        if w_l is not None:
            v = ops.mul(v, ops.to_dtype(w_l(c), acc))
        if b_l is not None:
            v = ops.add(v, ops.to_dtype(b_l(c), acc))
        return ops.to_dtype(v, in_dtype)

    out = Pointwise.create(device=device, dtype=in_dtype, inner_fn=fn, ranges=list(size))
    if has_running:
        keep = 1.0 - float(momentum)
        rm_l = running_mean.make_loader()
        rv_l = running_var.make_loader()
        stat_dtype = running_mean.get_dtype()

        def moved(old_l, batch):
            def body(index):
                old = ops.to_dtype(old_l(index), acc)
                value = ops.add(
                    ops.mul(old, ops.constant(keep, acc)),
                    ops.mul(batch(index), ops.constant(float(momentum), acc)),
                )
                return ops.to_dtype(value, stat_dtype)

            return Pointwise.create(
                device=device, dtype=stat_dtype, inner_fn=body, ranges=[channels]
            )

        new_mean = moved(rm_l, mean_l)
        new_var = moved(
            rv_l,
            lambda index: ops.truediv(m2_l(index), ops.constant(float(count - 1), acc)),
        )
        LOWERINGS["copy_.default"](running_mean, new_mean)
        LOWERINGS["copy_.default"](running_var, new_var)
    return out


@register("batch_norm.default")
def lower_batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps, *rest):
    """A batch norm that reads its running statistics is an affine map per
    channel: (x - mean) * rsqrt(var + eps) * weight + bias.  One that uses the
    batch's statistics computes them in one pass and moves the running ones
    toward them."""

    if training:
        lowered = _lower_batch_norm_training(
            x, weight, bias, running_mean, running_var, momentum, eps
        )
        if lowered is not None:
            return lowered
        return _fallback_batch_norm(
            x, weight, bias, running_mean, running_var, training, momentum, eps, *rest
        )
    if not (_given(running_mean) and _given(running_var)) or len(x.get_size()) < 2:
        return _fallback_batch_norm(
            x, weight, bias, running_mean, running_var, training, momentum, eps, *rest
        )
    in_dtype = x.get_dtype()
    acc = _accumulation_dtype(in_dtype)
    x_l = x.make_loader()
    m_l = running_mean.make_loader()
    v_l = running_var.make_loader()
    w_l = weight.make_loader() if _given(weight) else None
    b_l = bias.make_loader() if _given(bias) else None

    def fn(index):
        channel = [index[1]]
        v = ops.mul(
            ops.sub(ops.to_dtype(x_l(index), acc), ops.to_dtype(m_l(channel), acc)),
            ops.rsqrt(
                ops.add(ops.to_dtype(v_l(channel), acc), ops.constant(float(eps), acc))
            ),
        )
        if w_l is not None:
            v = ops.mul(v, ops.to_dtype(w_l(channel), acc))
        if b_l is not None:
            v = ops.add(v, ops.to_dtype(b_l(channel), acc))
        return ops.to_dtype(v, in_dtype)

    return Pointwise.create(
        device=x.get_device(), dtype=in_dtype, inner_fn=fn, ranges=list(x.get_size())
    )


_fallback_batch_norm_backward = fallback_handler(
    tp.ops.tp.batch_norm_backward.default, add_to_fallback_set=False
)


@register("batch_norm_backward.default")
def lower_batch_norm_backward(
    grad_out, x, weight=None, running_mean=None, running_var=None, training=True, eps=1e-5
):
    """The three gradients of a batch norm over every axis but the channel.

    In training the statistics are those of the batch: with x̂ the normalized
    input and M the count per channel, the input gradient is
    w · rstd / M · (M g - Σ g - x̂ Σ g x̂); the weight and bias gradients are
    Σ g x̂ and Σ g.  With running statistics the normalization is an affine
    map and the input gradient is g · w · rstd.
    """

    size = _static_ints(x.get_size())
    if size is None or len(size) < 2 or 0 in size or (
        not training and not (_given(running_mean) and _given(running_var))
    ):
        return _fallback_batch_norm_backward(
            grad_out, x, weight, running_mean, running_var, training, eps
        )
    device = x.get_device()
    in_dtype = x.get_dtype()
    acc = _accumulation_dtype(in_dtype)
    channels = size[1]
    rest = [size[0], *size[2:]]
    count = prod(rest)
    g_l = grad_out.make_loader()
    x_l = x.make_loader()
    w_l = weight.make_loader() if _given(weight) else None

    def full(c, r):
        parts = _unflatten_index(r, rest)
        return [parts[0], c, *parts[1:]]

    def per_channel(value, reduction_type="sum"):
        box = Reduction.create(
            device=device,
            dst_dtype=acc,
            src_dtype=acc,
            inner_fn=lambda index, rindex: value(index[0], rindex[0]),
            ranges=[channels],
            reduction_ranges=[count],
            reduction_type=reduction_type,
        )
        box.realize()
        return box.make_loader()

    if training:
        mean, m2, _weight = ir.WelfordReduction.create(
            device=device,
            dtype=acc,
            inner_fns=(
                lambda index, rindex: ops.to_dtype(x_l(full(index[0], rindex[0])), acc),
            ),
            ranges=[channels],
            reduction_ranges=[count],
            reduction_type="welford_reduce",
        )
        mean.realize()
        m2.realize()
        mean_l = mean.make_loader()
        m2_l = m2.make_loader()

        def centre(c):
            return mean_l([c])

        def rstd(c):
            return ops.rsqrt(
                ops.add(
                    ops.truediv(m2_l([c]), ops.constant(float(count), acc)),
                    ops.constant(float(eps), acc),
                )
            )
    else:
        rm_l = running_mean.make_loader()
        rv_l = running_var.make_loader()

        def centre(c):
            return ops.to_dtype(rm_l([c]), acc)

        def rstd(c):
            return ops.rsqrt(
                ops.add(ops.to_dtype(rv_l([c]), acc), ops.constant(float(eps), acc))
            )

    def x_hat(index, c):
        return ops.mul(ops.sub(ops.to_dtype(x_l(index), acc), centre(c)), rstd(c))

    sum_g = per_channel(lambda c, r: ops.to_dtype(g_l(full(c, r)), acc))
    sum_gx = per_channel(
        lambda c, r: ops.mul(ops.to_dtype(g_l(full(c, r)), acc), x_hat(full(c, r), c))
    )

    def scale(c):
        if w_l is None:
            return rstd(c)
        return ops.mul(ops.to_dtype(w_l([c]), acc), rstd(c))

    def grad_input(index):
        c = index[1]
        g = ops.to_dtype(g_l(index), acc)
        if not training:
            return ops.to_dtype(ops.mul(g, scale(c)), in_dtype)
        m = ops.constant(float(count), acc)
        inner = ops.sub(
            ops.sub(ops.mul(g, m), sum_g([c])), ops.mul(x_hat(index, c), sum_gx([c]))
        )
        return ops.to_dtype(ops.mul(ops.truediv(scale(c), m), inner), in_dtype)

    gi = Pointwise.create(device=device, dtype=in_dtype, inner_fn=grad_input, ranges=size)
    param_dtype = weight.get_dtype() if _given(weight) else acc
    gw = Pointwise.create(
        device=device, dtype=param_dtype,
        inner_fn=lambda index: ops.to_dtype(sum_gx(index), param_dtype),
        ranges=[channels],
    )
    gb = Pointwise.create(
        device=device, dtype=param_dtype,
        inner_fn=lambda index: ops.to_dtype(sum_g(index), param_dtype),
        ranges=[channels],
    )
    return gi, gw, gb


def _adaptive_avg_pool_windows(x, h, w, oh, ow, lead):
    """The average of each window when the windows do not tile the input.

    Where each window starts and how far it runs follows from the two extents
    alone, so the positions are written as a small table of whole numbers and
    the answer reads the input once per window position.  A place a window
    would reach past its end reads zero, and the value each window is divided
    by is the window's own length, so a short window averages over what it
    actually holds rather than over the room it leaves empty.
    """
    dev = x.get_device()
    acc = _accumulation_dtype(x.get_dtype())

    def window_plan(in_len, out_len):
        # How far the longest window runs, and whether the windows differ in
        # length at all: they all run the same unless one extent misses the
        # other's measure and misses it unevenly.
        maxlength = in_len // out_len + 1
        mod = in_len % out_len
        adaptive = not (mod == 0 or out_len % mod == 0)
        if adaptive:
            maxlength += 1
        elif mod == 0:
            maxlength -= 1
        return maxlength, adaptive

    def window_starts(in_len, out_len):
        iota = arange_start_step(0, out_len, 1, dtype=tp.int64, device=dev)
        return pointwise(
            lambda o: ops.floordiv(
                ops.mul(o, ops.constant(in_len, tp.int64)),
                ops.constant(out_len, tp.int64),
            ),
            iota,
            out_dtype=tp.int64,
        )

    def window_ends(in_len, out_len):
        iota = arange_start_step(0, out_len, 1, dtype=tp.int64, device=dev)
        return pointwise(
            lambda o: ops.floordiv(
                ops.add(
                    ops.mul(o, ops.constant(in_len, tp.int64)),
                    ops.constant(in_len + out_len - 1, tp.int64),
                ),
                ops.constant(out_len, tp.int64),
            ),
            iota,
            out_dtype=tp.int64,
        )

    max_h, adaptive_h = window_plan(h, oh)
    max_w, adaptive_w = window_plan(w, ow)
    if max_h * max_w > 256:
        # A window too long to unroll is read by the framework kernel, whose
        # loop does not grow with the window.
        return _fallback_adaptive_avg_pool2d(x, [oh, ow])
    starts_h = window_starts(h, oh)
    ends_h = window_ends(h, oh)
    starts_w = window_starts(w, ow)
    ends_w = window_ends(w, ow)
    iota_h = arange_start_step(0, max_h, 1, dtype=tp.int64, device=dev)
    iota_w = arange_start_step(0, max_w, 1, dtype=tp.int64, device=dev)
    idx_h = pointwise(
        ops.add, unsqueeze(starts_h, -1), iota_h, out_dtype=tp.int64
    )
    idx_w = pointwise(
        ops.add, unsqueeze(starts_w, -1), iota_w, out_dtype=tp.int64
    )
    if adaptive_h:
        idx_h = pointwise(
            lambda v: ops.minimum(v, ops.constant(h - 1, tp.int64)),
            idx_h,
            out_dtype=tp.int64,
        )
        # A column, so that it lines up with the row axis of a window grid
        # rather than with the trailing column axis.
        length_h = unsqueeze(lower_sub(ends_h, starts_h), -1)
    else:
        length_h = max_h
    if adaptive_w:
        idx_w = pointwise(
            lambda v: ops.minimum(v, ops.constant(w - 1, tp.int64)),
            idx_w,
            out_dtype=tp.int64,
        )
        length_w = lower_sub(ends_w, starts_w)
    else:
        length_w = max_w

    vals = index_tensor(
        x,
        [
            None,
            None,
            view(idx_h, [oh, max_h, 1, 1]),
            view(idx_w, [1, 1, ow, max_w]),
        ],
    )
    total = None
    for i in range(max_h):
        for j in range(max_w):
            cell = _cast_to(
                lower_basic_getitem(
                    _as_box(vals),
                    (
                        slice(None),
                        slice(None),
                        slice(None),
                        i,
                        slice(None),
                        j,
                    ),
                ),
                acc,
            )
            if adaptive_h:
                cell = lower_where(lower_ge(i, length_h), 0.0, cell)
            if adaptive_w:
                cell = lower_where(lower_ge(j, length_w), 0.0, cell)
            total = cell if total is None else lower_add(total, cell)

    if adaptive_h and adaptive_w:
        count = lower_mul(
            _cast_to(length_h, acc),
            unsqueeze(_cast_to(length_w, acc), 0),
        )
    elif adaptive_h:
        count = lower_mul(_cast_to(length_h, acc), float(max_w))
    elif adaptive_w:
        count = lower_mul(_cast_to(length_w, acc), float(max_h))
    else:
        count = float(max_h * max_w)

    averaged = pointwise(
        lambda v, k: ops.truediv(v, k), total, count, out_dtype=acc
    )
    return _cast_to(averaged, x.get_dtype())


_fallback_adaptive_avg_pool2d = fallback_handler(
    tp.ops.tp.adaptive_avg_pool2d.default, add_to_fallback_set=False
)


@register("adaptive_avg_pool2d.default", "_adaptive_avg_pool2d.default")
def lower_adaptive_avg_pool2d(x, output_size):
    """An adaptive average pool whose windows tile the input exactly is a
    mean over each window: one reduction, fusable with what reads it.  Windows
    that stop mid-stride are read position by position instead, each averaged
    over what it actually holds."""

    size = _static_ints(x.get_size())
    out = _static_ints(output_size) if output_size is not None else None
    if size is None or out is None or len(size) not in (3, 4) or len(out) != 2:
        return _fallback_adaptive_avg_pool2d(x, output_size)
    h, w = size[-2:]
    oh, ow = out
    if oh == 0 or ow == 0:
        return _fallback_adaptive_avg_pool2d(x, output_size)
    if h % oh or w % ow:
        if is_integer_dtype(x.get_dtype()) or x.get_dtype() == tp.bool:
            return _fallback_adaptive_avg_pool2d(x, output_size)
        return _adaptive_avg_pool_windows(x, h, w, oh, ow, size[:-2])
    kh, kw = h // oh, w // ow
    lead = size[:-2]
    acc = _accumulation_dtype(x.get_dtype())
    x_l = x.make_loader()

    def inner(index, rindex):
        *prefix, i, j = index
        r = rindex[0]
        return ops.to_dtype(
            x_l([*prefix, i * kh + FloorDiv(r, sympy.Integer(kw)), j * kw + modular_indexing(r, 1, kw)]),
            acc,
        )

    total = Reduction.create(
        device=x.get_device(),
        dst_dtype=acc,
        src_dtype=acc,
        inner_fn=inner,
        ranges=[*lead, oh, ow],
        reduction_ranges=[kh * kw],
        reduction_type="sum",
    )
    total.realize()
    t_l = total.make_loader()
    count = float(kh * kw)
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda index: ops.to_dtype(
            ops.truediv(t_l(index), ops.constant(count, acc)), x.get_dtype()
        ),
        ranges=[*lead, oh, ow],
    )


@register("conv2d_grad_bias.default", "conv_grad_bias.default")
def lower_conv_grad_bias(grad_out, *args):
    """The bias gradient is the output gradient summed over all but channels."""

    size, dtype, device = val_info(node_val())
    rank = len(grad_out.get_size())
    return make_reduction(grad_out, [0] + list(range(2, rank)), False, dtype, device, "sum")


_fallback_conv2d_grad_input = fallback_handler(
    tp.ops.tp.conv2d_grad_input.default, add_to_fallback_set=False
)
_fallback_conv2d_grad_weight = fallback_handler(
    tp.ops.tp.conv2d_grad_weight.default, add_to_fallback_set=False
)


def _conv_grad_channels_last(input, groups) -> bool:
    """Whether a gradient call's operands are handed over channels last.

    The same rule as the forward call's: the library computes the gradients of
    a reduced-precision ungrouped call channels last, and an operand in the
    other order is repacked inside the call on every step.  The input gradient
    reads the output gradient and the weight, the weight gradient reads the
    output gradient and the input; the operand only read for its shape keeps
    its order.
    """

    from .templates.conv import channels_last_call

    return len(input.get_size()) == 4 and channels_last_call(input, 2, groups, False)


@register("conv2d_grad_input.default")
def lower_conv2d_grad_input(grad_output, input, weight, stride, padding, dilation, groups):
    """The input gradient, asked of the framework kernel.

    The graph declares the result element type (a half-precision forward
    produces a half-precision input gradient), but the activation and weight
    saved for the backward can still be the wider parameters they came from,
    and the rest of the backward expects the result in that wider type.  The
    kernel runs in the result's type, so the operands are brought to it; the
    result is then returned in the type the caller supplied.
    """

    result_dtype = grad_output.get_dtype()
    val = node_val(index=0)
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        if grad_output.get_dtype() != dtype:
            grad_output = to_dtype(grad_output, dtype)
        if input.get_dtype() != dtype:
            input = to_dtype(input, dtype)
        if weight.get_dtype() != dtype:
            weight = to_dtype(weight, dtype)
        if _conv_grad_channels_last(input, groups):
            grad_output = ir.ExternKernel.require_channels_last(grad_output)
            weight = ir.ExternKernel.require_channels_last(weight)
        result = _fallback_conv2d_grad_input(
            grad_output, input, weight, stride, padding, dilation, groups
        )
        if result.get_dtype() != result_dtype:
            result = to_dtype(result, result_dtype)
        return result
    return _fallback_conv2d_grad_input(
        grad_output, input, weight, stride, padding, dilation, groups
    )


@register("conv2d_grad_weight.default")
def lower_conv2d_grad_weight(grad_output, input, weight, stride, padding, dilation, groups):
    """The weight gradient, asked of the framework kernel.

    The graph declares the result element type (a half-precision forward
    produces a half-precision weight gradient), but the activation and weight
    saved for the backward can still be the wider parameters they came from,
    and the rest of the backward expects the result in that wider type.  The
    kernel runs in the result's type, so the operands are brought to it; the
    result is then returned in the type the caller supplied.
    """

    result_dtype = weight.get_dtype()
    val = node_val(index=0)
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        if grad_output.get_dtype() != dtype:
            grad_output = to_dtype(grad_output, dtype)
        if input.get_dtype() != dtype:
            input = to_dtype(input, dtype)
        if weight.get_dtype() != dtype:
            weight = to_dtype(weight, dtype)
        if _conv_grad_channels_last(input, groups):
            grad_output = ir.ExternKernel.require_channels_last(grad_output)
            input = ir.ExternKernel.require_channels_last(input)
        result = _fallback_conv2d_grad_weight(
            grad_output, input, weight, stride, padding, dilation, groups
        )
        if result.get_dtype() != result_dtype:
            result = to_dtype(result, result_dtype)
        return result
    return _fallback_conv2d_grad_weight(
        grad_output, input, weight, stride, padding, dilation, groups
    )


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


_fallback_group_norm = fallback_handler(
    tp.ops.tp.native_group_norm.default, add_to_fallback_set=False
)
_fallback_group_norm_backward = fallback_handler(
    tp.ops.tp.native_group_norm_backward.default, add_to_fallback_set=False
)


@register("native_group_norm.default")
def lower_native_group_norm(x, weight, bias, n, c, hxw, groups, eps):
    return _fallback_group_norm(x, weight, bias, n, c, hxw, groups, eps)


def _lower_gn_bwd_template(grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask):
    """Group-norm backward through a fused template kernel, if the call fits.

    The template kernel reads the whole backward pass in one launch: it reduces
    the two group sums per group, writes the input gradient, and accumulates the
    per-channel weight and bias gradients atomically.  Calls that the template
    does not cover -- a partial output mask, a missing weight, a non-four-
    dimensional input, or dynamic shapes -- return None so the caller falls
    back to the framework call.
    """

    if list(output_mask) != [True, True, True]:
        return None
    if gamma is None:
        return None
    x_size = x.get_size()
    if len(x_size) != 4:
        return None
    if is_dynamic(*x_size, *grad_out.get_size(), *mean.get_size(), *rstd.get_size()):
        return None
    if not V.graph.sizevars.statically_known_equals(
        sympy.sympify(c) % sympy.sympify(groups), 0
    ):
        return None

    from .templates.group_norm import group_norm_backward_template
    from .templates.select_algorithm import autotune_select_algorithm

    device = grad_out.get_device()
    dtype = grad_out.get_dtype()
    c_int = int(c)
    dgamma0 = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda i: ops.constant(0, dtype),
        ranges=(c_int,),
    )
    dbeta0 = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda i: ops.constant(0, dtype),
        ranges=(c_int,),
    )
    dgamma0.data.realize()
    dbeta0.data.realize()
    if isinstance(grad_out, TensorBox):
        grad_out.realize()
    layout = grad_out.get_layout()
    if isinstance(layout, ir.FlexibleLayout):
        layout = layout.get_fixed_layout_without_freezing()
    choices = []
    err = group_norm_backward_template.maybe_append_choice(
        choices,
        input_nodes=(grad_out, x, mean, rstd, gamma, dgamma0, dbeta0),
        mutated_inputs=[dgamma0, dbeta0],
        layout=layout,
        GROUPS=int(groups),
        BLOCK=1024,
        num_stages=1,
        num_warps=4,
    )
    if err is not None or not choices:
        import sys as _sys
        print("GN_TEMPLATE_ERR:", repr(err), "groups:", groups, "x_size:", [str(s) for s in x_size], "layout_stride:", getattr(x.get_layout(), "stride", None), file=_sys.stderr, flush=True)
        return None
    node, _ = autotune_select_algorithm(
        "group_norm_backward",
        choices,
        [grad_out, x, mean, rstd, gamma, dgamma0, dbeta0],
        layout,
    )
    return (node, dgamma0, dbeta0)


def _lower_gn_bwd_decomp(grad_out, x, mean, rstd, gamma, n, c, hxw, groups,
                         output_mask):
    """The group-norm backward written as reduction and pointwise passes.

    The native kernel is a single launch, but it is also a wall the scheduler
    cannot fuse through: the gradient of the surrounding silu, the half to
    float widening, and the weight gradients all stay separate calls.  This
    path re-expresses the backward as the same reductions and pointwise passes
    the scheduler already knows, so the input gradient can be produced in whatever
    dtype the graph asks for and fused with its neighbours.
    """

    if len(x.get_size()) not in (3, 4):
        return None
    vals = node_val()
    cpg = c // groups
    device = grad_out.get_device()
    rank = len(x.get_size())
    spatial = list(x.get_size()[2:])
    dy_loader = grad_out.make_loader()
    x_loader = x.make_loader()
    mean_loader = mean.make_loader()
    rstd_loader = rstd.make_loader()
    gamma_loader = gamma.make_loader() if is_tensor_box(gamma) else None
    f32 = tp.float32

    def full_index(index, rindex):
        s = rindex[0]
        return [index[0], index[1]] + _unflatten_index(s, spatial)

    def ds_inner(index, rindex):
        full = full_index(index, rindex)
        return ops.mul(
            ops.to_dtype(dy_loader(full), f32),
            ops.to_dtype(x_loader(full), f32),
        )

    def db_inner(index, rindex):
        return ops.to_dtype(dy_loader(full_index(index, rindex)), f32)

    ds = Reduction.create(
        device=device,
        dst_dtype=f32,
        src_dtype=f32,
        inner_fn=ds_inner,
        ranges=(n, c),
        reduction_ranges=(hxw,),
        reduction_type="sum",
    )
    db = Reduction.create(
        device=device,
        dst_dtype=f32,
        src_dtype=f32,
        inner_fn=db_inner,
        ranges=(n, c),
        reduction_ranges=(hxw,),
        reduction_type="sum",
    )
    ds.realize()
    db.realize()
    ds_loader = ds.make_loader()
    db_loader = db.make_loader()

    def gamma_at(ch):
        if gamma_loader is not None:
            return ops.to_dtype(gamma_loader([ch]), f32)
        return ops.constant(1.0, f32)

    results = [None, None, None]
    s = 1.0 / (hxw * cpg)
    if output_mask[0]:
        def dsv_inner(index, rindex):
            ch = index[1] * cpg + rindex[0]
            return ops.mul(ds_loader([index[0], ch]), gamma_at(ch))

        def dbv_inner(index, rindex):
            ch = index[1] * cpg + rindex[0]
            return ops.mul(db_loader([index[0], ch]), gamma_at(ch))

        ds_val = Reduction.create(
            device=device,
            dst_dtype=f32,
            src_dtype=f32,
            inner_fn=dsv_inner,
            ranges=(n, groups),
            reduction_ranges=(cpg,),
            reduction_type="sum",
        )
        db_val = Reduction.create(
            device=device,
            dst_dtype=f32,
            src_dtype=f32,
            inner_fn=dbv_inner,
            ranges=(n, groups),
            reduction_ranges=(cpg,),
            reduction_type="sum",
        )
        ds_val.realize()
        db_val.realize()
        dsv = ds_val.make_loader()
        dbv = db_val.make_loader()

        def c2_at(ng):
            r = ops.to_dtype(rstd_loader(ng), f32)
            m = ops.to_dtype(mean_loader(ng), f32)
            num = ops.sub(ops.mul(dbv(ng), m), dsv(ng))
            return ops.mul(
                ops.mul(ops.mul(ops.mul(num, r), r), r), ops.constant(s, f32)
            )

        def c3_at(ng):
            r = ops.to_dtype(rstd_loader(ng), f32)
            m = ops.to_dtype(mean_loader(ng), f32)
            left = ops.mul(ops.neg(c2_at(ng)), m)
            right = ops.mul(ops.mul(dbv(ng), r), ops.constant(s, f32))
            return ops.sub(left, right)

        c2 = Pointwise.create(
            device=device, dtype=f32, inner_fn=c2_at, ranges=(n, groups)
        )
        c3 = Pointwise.create(
            device=device, dtype=f32, inner_fn=c3_at, ranges=(n, groups)
        )
        c2.realize()
        c3.realize()
        c2_loader = c2.make_loader()
        c3_loader = c3.make_loader()
        dx_size, _, _ = val_info(vals[0])
        # The input gradient is wanted in the input's stored precision: the
        # framework's backward returns it there, and the downstream gradient
        # calls expect it in that precision too.  The declared output type of
        # the native kernel (float) would force a widening cast here that the
        # framework's own decomposed path does not make.
        dx_dtype = input.get_dtype()

        dx_rank = len(dx_size)

        def dx_inner(index):
            if dx_rank == 4:
                n_idx, ch = index[0], index[1]
                full = list(index)
            elif dx_rank == 3:
                n_idx, ch = index[0], index[1]
                full = [index[0], index[1]] + _unflatten_index(
                    index[2], spatial
                )
            else:
                flat = as_index(index[0])
                total = sympy.Integer(c * hxw)
                n_idx = FloorDiv(flat, total)
                rem = modular_indexing(flat, 1, total)
                ch = FloorDiv(rem, sympy.Integer(hxw))
                s = modular_indexing(rem, 1, sympy.Integer(hxw))
                full = [n_idx, ch] + _unflatten_index(s, spatial)
            ng = [n_idx, FloorDiv(as_index(ch), sympy.Integer(cpg))]
            c1 = ops.mul(
                ops.to_dtype(rstd_loader(ng), f32), gamma_at(ch)
            )
            dy = ops.to_dtype(dy_loader(full), f32)
            xv = ops.to_dtype(x_loader(full), f32)
            return ops.add(
                ops.add(ops.mul(dy, c1), ops.mul(xv, c2_loader(ng))),
                c3_loader(ng),
            )

        results[0] = Pointwise.create(
            device=device, dtype=dx_dtype, inner_fn=dx_inner, ranges=dx_size
        )
    if output_mask[1]:
        dg_size, dg_dtype, _ = val_info(vals[1])

        def dgamma_inner(index, rindex):
            ch = index[0]
            ng = [rindex[0], FloorDiv(as_index(ch), sympy.Integer(cpg))]
            m = ops.to_dtype(mean_loader(ng), f32)
            r = ops.to_dtype(rstd_loader(ng), f32)
            nc = [rindex[0], ch]
            return ops.mul(
                ops.sub(ds_loader(nc), ops.mul(db_loader(nc), m)), r
            )

        results[1] = Reduction.create(
            device=device,
            dst_dtype=dg_dtype,
            src_dtype=f32,
            inner_fn=dgamma_inner,
            ranges=(c,),
            reduction_ranges=(n,),
            reduction_type="sum",
        )
        results[1].realize()
    if output_mask[2]:
        db_size, db_dtype, _ = val_info(vals[2])

        def dbeta_inner(index, rindex):
            return db_loader([rindex[0], index[0]])

        results[2] = Reduction.create(
            device=device,
            dst_dtype=db_dtype,
            src_dtype=f32,
            inner_fn=dbeta_inner,
            ranges=(c,),
            reduction_ranges=(n,),
            reduction_type="sum",
        )
        results[2].realize()
    return tuple(results)


@register("native_group_norm_backward.default")
def lower_native_group_norm_backward(grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask):
    if config.use_gn_bwd_template:
        lowered = _lower_gn_bwd_template(
            grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask
        )
        if lowered is not None:
            return lowered
    if config.use_gn_bwd_decomp:
        lowered = _lower_gn_bwd_decomp(
            grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask
        )
        if lowered is not None:
            return lowered
    # The framework's mixed-precision group-norm backward keeps the
    # activation-side inputs in their stored half precision and takes the
    # gradient in float, returning the parameter gradients in float.  A
    # half gradient next to a half activation would be re-cast by the
    # framework kernel into half parameter gradients, which this graph does
    # not expect, so the gradient is brought to float first.
    if grad_out.get_dtype() != tp.float32:
        grad_out = to_dtype(grad_out, tp.float32)
    return _fallback_group_norm_backward(
        grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask
    )


__all__ = [
    "FALLBACKS",
    "user_lowerings",
    "select_decomp_table",
    "LOWERINGS",
    "fallback_handler",
    "make_reduction",
    "pointwise",
    "register",
    "reshape",
    "target_name",
]


# ---------------------------------------------------------------------------
# sampling operators
# ---------------------------------------------------------------------------


def _pair(value, count: int) -> list:
    """A kernel/stride/padding argument as one entry per spatial axis."""

    if isinstance(value, int):
        return [int(value)] * count
    items = [int(v) for v in value]
    if len(items) == 1:
        return items * count
    return items


def _spatial_ndim() -> int:
    """How many trailing axes of the operation being lowered are spatial.

    An operation comes in a two-dimensional and a three-dimensional form which
    differ only in that, and which of the two is being lowered is a property of
    the node rather than of the arguments -- both forms take the same arguments
    and would be told apart by looking at nothing else.  The node is read the
    same way the rest of a lowering reads it, as the one currently published.
    """

    return 3 if "3d" in target_name(V.current_node.target) else 2


def _upsample_nearestnd(x, output_size, ndim, exact=False, **kwargs):
    """Nearest upsampling as an index remap of the source.

    Each output element reads the input element the scale maps it to, so the
    operator is a view with a remapped address rather than a call: it fuses
    with whatever consumes it instead of standing on its own.

    ``ndim`` is how many trailing axes are spatial, which is what says how much
    of the output size is a size rather than a leading extent.  ``exact``
    picks the flavor of the mapping: the plain one anchors the sample at the
    output position itself, the exact one at the midpoint between positions,
    so the same extent can read a different source element under the two.

    The exact mapping is carried out in integer arithmetic — the output
    position is doubled on both sides so the half stays a whole number —
    where a float scale could round a boundary position the wrong way.
    """

    size, dtype, device = val_info(node_val())
    in_size = list(x.get_size())
    out_spatial = [int(s) for s in output_size][-ndim:]
    in_spatial = in_size[-ndim:]
    prefix = in_size[:-ndim]

    def reindex(index):
        # Plain: floor(i * in / out); exact: floor((i + 1/2) * in / out).
        out = []
        for axis, (i, o) in enumerate(zip(in_spatial, out_spatial)):
            at = as_index(index[len(prefix) + axis])
            if exact:
                out.append(FloorDiv((2 * at + 1) * i, sympy.Integer(2 * o)))
            else:
                out.append(FloorDiv(at * i, sympy.Integer(o)))
        return [*index[: len(prefix)], *out]

    return make_view(x, size, reindex)


def _no_scale_hint(scales):
    # A scale handed alongside an extent is a hint the lowering cannot carry
    # through the integer mapping; that call stays a boundary.
    for scale in scales:
        if scale is not None:
            raise NotImplementedError("a scale handed alongside an extent")


@register("upsample_nearest2d.default")
def lower_upsample_nearest2d(x, output_size, scales_h=None, scales_w=None,
                            **kwargs):
    _no_scale_hint((scales_h, scales_w))
    return _upsample_nearestnd(x, output_size, 2, **kwargs)


@register("_upsample_nearest_exact2d.default")
def lower_upsample_nearest_exact2d(x, output_size, scales_h=None,
                                   scales_w=None, **kwargs):
    _no_scale_hint((scales_h, scales_w))
    return _upsample_nearestnd(x, output_size, 2, exact=True, **kwargs)


@register("upsample_nearest3d.default")
def lower_upsample_nearest3d(x, output_size, scales_d=None, scales_h=None,
                             scales_w=None, **kwargs):
    _no_scale_hint((scales_d, scales_h, scales_w))
    return _upsample_nearestnd(x, output_size, 3, **kwargs)


@register("_upsample_nearest_exact3d.default")
def lower_upsample_nearest_exact3d(x, output_size, scales_d=None,
                                   scales_h=None, scales_w=None, **kwargs):
    _no_scale_hint((scales_d, scales_h, scales_w))
    return _upsample_nearestnd(x, output_size, 3, exact=True, **kwargs)


@register("upsample_nearest1d.default")
def lower_upsample_nearest1d(x, output_size, scales_d=None, **kwargs):
    _no_scale_hint((scales_d,))
    return _upsample_nearestnd(x, output_size, 1, **kwargs)


# ---------------------------------------------------------------------------
# Grid sampling and bicubic resampling: every output element reads the source
# at a position of its own, computed from the coordinates the call carries, so
# each is a gather the kernels around it absorb.
# ---------------------------------------------------------------------------


def _cast_to(value, dtype):
    return pointwise(lambda v: ops.to_dtype(v, dtype), value, out_dtype=dtype)


def _cubic_convolution1(t, dtype):
    """The weight of a source at a distance of at most one from the sample:
    ``((A + 2) t - (A + 3)) t^2 + 1`` with A = -0.75."""
    A = -0.75
    return pointwise(
        lambda x: ops.add(
            ops.mul(
                ops.sub(
                    ops.mul(ops.constant(A + 2.0, dtype), x),
                    ops.constant(A + 3.0, dtype),
                ),
                ops.mul(x, x),
            ),
            ops.constant(1.0, dtype),
        ),
        t,
        out_dtype=dtype,
    )


def _cubic_convolution2(t, dtype):
    """The weight of a source one or two steps from the sample:
    ``((A t - 5A) t + 8A) t - 4A`` with A = -0.75."""
    A = -0.75
    return pointwise(
        lambda x: ops.sub(
            ops.mul(
                ops.add(
                    ops.mul(
                        ops.sub(
                            ops.mul(ops.constant(A, dtype), x),
                            ops.constant(5.0 * A, dtype),
                        ),
                        x,
                    ),
                    ops.constant(8.0 * A, dtype),
                ),
                x,
            ),
            ops.constant(4.0 * A, dtype),
        ),
        t,
        out_dtype=dtype,
    )


def _cubic_weights(t, dtype):
    """The four weights a cubic interpolation at distance t asks for, one per
    source element counting from one step below the sample."""
    return (
        _cubic_convolution2(lower_add(t, 1), dtype),
        _cubic_convolution1(t, dtype),
        _cubic_convolution1(lower_sub(1, t), dtype),
        _cubic_convolution2(lower_sub(2, t), dtype),
    )


_fallback_grid_sampler_2d = fallback_handler(
    tp_ops.grid_sampler_2d.default, add_to_fallback_set=False
)


@register_lowering(tp_ops.grid_sampler_2d, type_promotion_kind=None)
def lower_grid_sampler_2d(a, grid, interpolation_mode=0, padding_mode=0,
                          align_corners=False):
    """Read the input at the positions the grid names.

    The grid names a position in the unit square for every output element, and
    the answer gathers the input there, blending the neighbours the requested
    smoothness asks for: the two-by-two neighbourhood a linear blend reads, the
    one nearest element, or the sixteen a cubic blend reads.  A position named
    outside the input is answered by the padding rule: zeros, the nearest
    border, or the input mirrored across its edge.  A named position with no
    input there contributes nothing, so the positions outside cost a zero
    weight rather than a read past the end.
    """
    if interpolation_mode not in (0, 1, 2):
        raise RuntimeError(f"Invalid interpolation mode {interpolation_mode}")
    if padding_mode not in (0, 1, 2):
        raise RuntimeError(f"Invalid padding mode {padding_mode}")
    a_size = list(a.get_size())
    grid_size = list(grid.get_size())
    in_dtype = a.get_dtype()
    static = _static_ints([*a_size, *grid_size])
    if (
        static is None
        or len(a_size) != 4
        or len(grid_size) != 4
        or static[0] != static[4]
        or static[7] != 2
        or is_integer_dtype(in_dtype)
        or in_dtype == tp.bool
        or in_dtype in (tp.complex64, tp.complex128)
    ):
        return _fallback_grid_sampler_2d(
            a, grid, interpolation_mode, padding_mode, align_corners
        )
    n, c, ih, iw, _gn, oh, ow, _two = static
    dev = a.get_device()
    acc = tp.float32 if in_dtype in (tp.float16, tp.bfloat16) else in_dtype

    grid_box = _as_box(grid)
    xg = _cast_to(
        lower_basic_getitem(
            grid_box, (slice(None), slice(None), slice(None), 0)
        ),
        acc,
    )
    yg = _cast_to(
        lower_basic_getitem(
            grid_box, (slice(None), slice(None), slice(None), 1)
        ),
        acc,
    )

    def unnormalize(coords, size):
        mul = size * 0.5 - 0.5 if align_corners else size * 0.5
        ofs = size * 0.5 - 0.5
        return pointwise(
            lambda v: ops.add(
                ops.mul(v, ops.constant(mul, acc)), ops.constant(ofs, acc)
            ),
            coords,
            out_dtype=acc,
        )

    def reflect(coords, twice_low, twice_high):
        # A position outside the pair of bounds is folded back in: the distance
        # past an edge is read again inward, and every other fold reads the
        # remaining span rather than the distance, which is what the odd or
        # even fold count tells apart.
        if twice_low == twice_high:
            return _full(0, dev, acc, coords.get_size())
        coords_min = twice_low / 2
        span = (twice_high - twice_low) / 2
        distance = pointwise(
            lambda v: ops.abs(ops.sub(v, ops.constant(coords_min, acc))),
            coords,
            out_dtype=acc,
        )
        extra = pointwise(
            lambda v: ops.fmod(v, ops.constant(span, acc)),
            distance,
            out_dtype=acc,
        )
        flips = pointwise(
            lambda v: ops.floor(ops.truediv(v, ops.constant(span, acc))),
            distance,
            out_dtype=acc,
        )
        even = pointwise(
            lambda v: ops.eq(
                ops.fmod(v, ops.constant(2.0, acc)), ops.constant(0.0, acc)
            ),
            flips,
            out_dtype=tp.bool,
        )
        return lower_where(
            even,
            lower_add(extra, coords_min),
            lower_sub(lower_add(span, coords_min), extra),
        )

    def coordinates(coords, size):
        if padding_mode == 0:
            return coords
        if padding_mode == 1:
            return lower_clamp(coords, 0, size - 1)
        if align_corners:
            reflected = reflect(coords, 0, 2 * (size - 1))
        else:
            reflected = reflect(coords, -1, 2 * size - 1)
        return lower_clamp(reflected, 0, size - 1)

    def source_index(coords, size):
        return coordinates(unnormalize(coords, size), size)

    n_idx = view(
        arange_start_step(0, n, 1, dtype=tp.int64, device=dev), [n, 1, 1, 1]
    )
    c_idx = view(
        arange_start_step(0, c, 1, dtype=tp.int64, device=dev), [1, c, 1, 1]
    )

    def within(ix, iy):
        return pointwise(
            lambda x, y: ops.logical_and(
                ops.logical_and(
                    ops.ge(x, ops.constant(0.0, acc)),
                    ops.lt(x, ops.constant(iw, acc)),
                ),
                ops.logical_and(
                    ops.ge(y, ops.constant(0.0, acc)),
                    ops.lt(y, ops.constant(ih, acc)),
                ),
            ),
            ix,
            iy,
            out_dtype=tp.bool,
        )

    def summand(ix, iy, weight):
        # One corner of the blend: read where the position lands inside, and
        # nowhere (with no weight) where it does not, so a position outside
        # costs the input's border element rather than a read past the end.
        cond = within(ix, iy)

        def index_of(value):
            return pointwise(
                lambda f, v: ops.where(
                    f, ops.to_dtype(v, tp.int64), ops.constant(0, tp.int64)
                ),
                cond,
                value,
                out_dtype=tp.int64,
            )

        if isinstance(weight, (int, float)):
            weight = lower_where(cond, float(weight), 0.0)
        else:
            weight = lower_where(cond, weight, 0.0)
        gathered = index_tensor(
            a,
            [n_idx, c_idx, unsqueeze(index_of(iy), 1), unsqueeze(index_of(ix), 1)],
        )
        return lower_mul(gathered, unsqueeze(weight, 1))

    if interpolation_mode == 0:
        ix = source_index(xg, iw)
        iy = source_index(yg, ih)
        ix_nw = pointwise(ops.floor, ix, out_dtype=acc)
        iy_nw = pointwise(ops.floor, iy, out_dtype=acc)
        ix_nw_p1 = lower_add(ix_nw, 1)
        iy_nw_p1 = lower_add(iy_nw, 1)
        total = lower_add(
            lower_add(
                summand(
                    ix_nw,
                    iy_nw,
                    lower_mul(
                        lower_sub(ix_nw_p1, ix), lower_sub(iy_nw_p1, iy)
                    ),
                ),
                summand(
                    ix_nw_p1,
                    iy_nw,
                    lower_mul(lower_sub(ix, ix_nw), lower_sub(iy_nw_p1, iy)),
                ),
            ),
            lower_add(
                summand(
                    ix_nw,
                    iy_nw_p1,
                    lower_mul(lower_sub(ix_nw_p1, ix), lower_sub(iy, iy_nw)),
                ),
                summand(
                    ix_nw_p1,
                    iy_nw_p1,
                    lower_mul(lower_sub(ix, ix_nw), lower_sub(iy, iy_nw)),
                ),
            ),
        )

    elif interpolation_mode == 1:
        nearest_x = pointwise(
            ops.round, source_index(xg, iw), out_dtype=acc
        )
        nearest_y = pointwise(
            ops.round, source_index(yg, ih), out_dtype=acc
        )
        total = summand(nearest_x, nearest_y, 1)

    else:
        ix = unnormalize(xg, iw)
        iy = unnormalize(yg, ih)
        ix_nw = pointwise(ops.floor, ix, out_dtype=acc)
        iy_nw = pointwise(ops.floor, iy, out_dtype=acc)
        wx = _cubic_weights(lower_sub(ix, ix_nw), acc)
        wy = _cubic_weights(lower_sub(iy, iy_nw), acc)
        total = None
        for row in range(4):
            y_ofs = lower_add(iy_nw, row - 1)
            line = None
            for col in range(4):
                piece = lower_mul(
                    summand(lower_add(ix_nw, col - 1), y_ofs, 1.0), wx[col]
                )
                line = piece if line is None else lower_add(line, piece)
            piece = lower_mul(line, wy[row])
            total = piece if total is None else lower_add(total, piece)

    return _cast_to(total, in_dtype)


_fallback_upsample_bicubic2d = fallback_handler(
    tp_ops.upsample_bicubic2d.default, add_to_fallback_set=False
)


@register_lowering(tp_ops.upsample_bicubic2d, type_promotion_kind=None)
def lower_upsample_bicubic2d(input, output_size, align_corners,
                             scales_h=None, scales_w=None):
    """Resample the two trailing axes with a cubic blend.

    Every output element is a weighted read of the four-by-four source
    neighbourhood its position maps to, the weights set by how far the mapped
    position sits inside it.  A mapped position outside the source reads the
    border the clamped index names, and the blend weights make the edge hold
    still: a weight of one on the border element and none on the rest.
    """
    _no_scale_hint((scales_h, scales_w))
    in_dtype = input.get_dtype()
    out = _static_ints(output_size) if output_size is not None else None
    in_size = _static_ints(input.get_size())
    if (
        out is None
        or len(out) != 2
        or in_size is None
        or len(in_size) != 4
        or is_integer_dtype(in_dtype)
        or in_dtype == tp.bool
        or in_dtype in (tp.complex64, tp.complex128)
    ):
        return _fallback_upsample_bicubic2d(
            input, output_size, align_corners, scales_h, scales_w
        )
    _n, _c, ih, iw = in_size
    oh, ow = out
    dev = input.get_device()
    acc = tp.float32 if in_dtype in (tp.float16, tp.bfloat16) else in_dtype

    def scale(in_len, out_len):
        if align_corners:
            return (in_len - 1.0) / (out_len - 1.0) if out_len > 1 else 0.0
        return in_len / out_len

    def source_positions(scale_value, out_len):
        # Where each output position sits in the source: at the position
        # itself when the corners are pinned, half a step in from it otherwise.
        iota = arange_start_step(0, out_len, 1, dtype=tp.int64, device=dev)
        if align_corners:
            return pointwise(
                lambda i: ops.mul(
                    ops.constant(scale_value, acc), ops.to_dtype(i, acc)
                ),
                iota,
                out_dtype=acc,
            )
        return pointwise(
            lambda i: ops.sub(
                ops.mul(
                    ops.constant(scale_value, acc),
                    ops.add(ops.to_dtype(i, acc), ops.constant(0.5, acc)),
                ),
                ops.constant(0.5, acc),
            ),
            iota,
            out_dtype=acc,
        )

    x_float = source_positions(scale(iw, ow), ow)
    y_float = source_positions(scale(ih, oh), oh)
    x_nw = pointwise(ops.floor, x_float, out_dtype=acc)
    y_nw = pointwise(ops.floor, y_float, out_dtype=acc)
    tx = lower_clamp(lower_sub(x_float, x_nw), 0.0, 1.0)
    ty = lower_clamp(lower_sub(y_float, y_nw), 0.0, 1.0)
    x_i = _cast_to(x_nw, tp.int64)
    y_i = unsqueeze(_cast_to(y_nw, tp.int64), -1)
    wx = _cubic_weights(tx, acc)
    wy = tuple(unsqueeze(w, -1) for w in _cubic_weights(ty, acc))
    y_ofs = tuple(lower_add(y_i, ofs) for ofs in (-1, 0, 1, 2))
    x_ofs = tuple(lower_add(x_i, ofs) for ofs in (-1, 0, 1, 2))

    def read(yv, xv):
        y_idx = lower_clamp(yv, 0, ih - 1)
        x_idx = lower_clamp(xv, 0, iw - 1)
        return index_tensor(input, [None, None, y_idx, x_idx])

    total = None
    for row in range(4):
        line = None
        for col in range(4):
            piece = lower_mul(read(y_ofs[row], x_ofs[col]), wx[col])
            line = piece if line is None else lower_add(line, piece)
        piece = lower_mul(line, wy[row])
        total = piece if total is None else lower_add(total, piece)
    return _cast_to(total, in_dtype)


@register("_upsample_nearest_exact1d.default")
def lower_upsample_nearest_exact1d(x, output_size, scales_d=None, **kwargs):
    _no_scale_hint((scales_d,))
    return _upsample_nearestnd(x, output_size, 1, exact=True, **kwargs)


@register("_upsample_nearest_exact1d.vec", "_upsample_nearest_exact2d.vec",
          "_upsample_nearest_exact3d.vec")
def lower_upsample_nearest_exact_vec(x, output_size, scale_factors=None,
                                     **kwargs):
    """The bundled spelling of a nearest-exact upsampling, every spatial axis.

    The bundled call carries one scale where the axis-by-axis call carries
    one per axis; the extent it works on is what the operation's own name
    says, and a scale without an extent is not something an address can be
    read from, so that call stays a boundary.
    """

    if not output_size:
        raise NotImplementedError("a scale given without an extent")
    ndim = int(target_name(V.current_node.target).split(".")[0][-2])
    return _upsample_nearestnd(x, output_size, ndim, exact=True, **kwargs)


def _prod_ints(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _window_overhangs(extent: int, kernel: int, stride: int, padding: int,
                      pooled: int) -> bool:
    """Whether the last window of an axis reaches past the padded extent.

    Rounding the window count up is what lets a window start inside the padded
    extent and end outside it; that window is divided by what it covers rather
    than by the kernel, and reads nothing where it hangs over.
    """

    return (pooled - 1) * stride + kernel > extent + 2 * padding


def _constant_outside(x, ndim: int, fill: float):
    """A loader of ``x`` that answers ``fill`` for an address outside it."""

    extents = list(x.get_size())[-ndim:]
    loader = x.make_loader()

    def load(index):
        index = list(index)
        inside = None
        for axis in range(ndim):
            at = ops.index_expr(index[len(index) - ndim + axis], tp.int64)
            term = ops.and_(
                ops.ge(at, ops.index_expr(sympy.Integer(0), tp.int64)),
                ops.lt(at, ops.index_expr(as_index(extents[axis]), tp.int64)),
            )
            inside = term if inside is None else ops.and_(inside, term)
        return ops.masked(inside, lambda: loader(index), fill)

    return load


_fallback_avg_pool = {
    2: fallback_handler(tp.ops.tp.avg_pool2d.default, add_to_fallback_set=False),
    3: fallback_handler(tp.ops.tp.avg_pool3d.default, add_to_fallback_set=False),
}


@register("avg_pool2d.default", "avg_pool3d.default")
def lower_avg_poolnd(x, kernel_size, stride=(), padding=0, ceil_mode=False,
                     count_include_pad=True, divisor_override=None, **kwargs):
    """Average pooling as the window's values added up and divided, per element.

    Each output element reads its window and adds it up in the loop of
    whatever consumes it, so a pooling of a few positions costs no kernel of
    its own.  A window of many positions is asked of the framework kernel:
    written out, it is a body too long to be worth having.
    """

    size, _, device = val_info(node_val())
    ndim = _spatial_ndim()
    kernel = _pair(kernel_size, ndim)
    stride = _pair(stride, ndim) if stride else list(kernel)
    padding = _pair(padding, ndim) if padding else [0] * ndim
    in_size = list(x.get_size())
    window = _prod_ints(kernel)
    if (
        window > 25
        or len(in_size) not in (ndim + 1, ndim + 2)
        or is_dynamic(*in_size, *size)
    ):
        return _fallback_avg_pool[ndim](
            x, kernel, stride, padding, ceil_mode, count_include_pad,
            divisor_override,
        )
    lead = len(in_size) - ndim
    spatial_in = [int(s) for s in in_size[lead:]]
    spatial_out = [int(s) for s in list(size)[-ndim:]]
    overhang = [
        _window_overhangs(extent, k, s, p, pooled)
        for extent, k, s, p, pooled in zip(
            spatial_in, kernel, stride, padding, spatial_out
        )
    ]
    # A window that leaves the source reads nothing there, so the read has to
    # be one that knows where the source ends.
    had_padding = any(padding) or any(overhang)

    # Read once per position of the window, so it is put in memory when it is
    # more than a plain read.
    x.realize_hint()
    loader = _constant_outside(x, ndim, 0.0) if had_padding else x.make_loader()
    dtype = x.get_dtype()
    floating = dtype.is_floating_point

    def window_sum(index):
        base = index[lead:]
        total = None
        for offset in itertools.product(*[range(k) for k in kernel]):
            at = [
                as_index(base[axis]) * stride[axis] + offset[axis] - padding[axis]
                for axis in range(ndim)
            ]
            value = loader([*index[:lead], *at])
            total = value if total is None else ops.add(value, total)
        return total

    if divisor_override:
        fixed = int(divisor_override)
    elif not had_padding or (count_include_pad and not any(overhang)):
        fixed = window
    else:
        fixed = None

    def covered(index):
        """How many positions the window of this element is divided by."""

        base = index[lead:]
        factor = None
        for axis in range(ndim):
            start = as_index(base[axis]) * stride[axis] - padding[axis]
            end = Min(start + kernel[axis], spatial_in[axis] + padding[axis])
            if not count_include_pad:
                start = Max(start, 0)
                end = Min(end, spatial_in[axis])
            term = ops.index_expr(end - start, tp.int32)
            factor = term if factor is None else ops.mul(factor, term)
        return factor

    def fn(index):
        total = window_sum(index)
        if fixed is None:
            divisor = covered(index)
            if floating:
                return ops.truediv(total, divisor)
            return ops.truncdiv(total, divisor)
        if not floating:
            return ops.truncdiv(total, ops.constant(fixed, dtype))
        if fixed > 0 and fixed & (fixed - 1) == 0:
            # A power of two divides exactly as a product, which is the cheaper
            # of the two ways to write the same answer.
            return ops.mul(total, ops.constant(1.0 / fixed, dtype))
        return ops.truediv(total, ops.constant(fixed, dtype))

    return Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=[*in_size[:lead], *spatial_out],
    )


_fallback_max_pool = {
    2: fallback_handler(tp.ops.tp.max_pool2d.default, add_to_fallback_set=False),
    3: fallback_handler(tp.ops.tp.max_pool3d.default, add_to_fallback_set=False),
}

_fallback_max_pool_with_indices = {
    2: fallback_handler(tp.ops.tp.max_pool2d_with_indices.default, add_to_fallback_set=False),
    3: fallback_handler(tp.ops.tp.max_pool3d_with_indices.default, add_to_fallback_set=False),
}


def _max_pool_common(x, kernel_size, stride, padding, dilation, ndim):
    """The window of a pooling call, spread over its spatial axes.

    Every argument may arrive as a single number meant for each axis, and an
    empty stride means the windows do not overlap: the stride is the window.
    """

    kernel = _pair(kernel_size, ndim)
    stride = _pair(stride, ndim) if stride else list(kernel)
    padding = _pair(padding, ndim) if padding else [0] * ndim
    dilation = _pair(dilation, ndim)
    return kernel, stride, padding, dilation


def _max_pool_checks(x, size, kernel, ndim, window):
    """Whether this call is one whose window can be written out.

    A window of many positions is a body too long to be worth having, and a
    call whose extents are only numbers the kernel gives later has no window
    to unroll -- those are asked of the framework kernel.
    """

    in_size = list(x.get_size())
    return (
        window <= 25
        and len(in_size) in (ndim + 1, ndim + 2)
        and not is_dynamic(*in_size, *size)
    )


_fallback_adaptive_max_pool2d = fallback_handler(tp_ops.adaptive_max_pool2d.default)


@register("adaptive_max_pool2d.default")
def adaptive_max_pool2d(x, output_size):
    """The largest value of each window, the window grown to fit a shape.

    The window's shape is not given: it is the one that makes the asked-for
    output shape out of the input's, and saying which positions it covers is
    the walk this lowering does not write -- so the call is handed over to the
    kernel that owns that walk.
    """

    return _fallback_adaptive_max_pool2d(x, output_size)


@register("max_pool2d.default", "max_pool3d.default")
def lower_max_poolnd(x, kernel_size, stride=(), padding=0, dilation=1,
                     ceil_mode=False, **kwargs):
    """Max pooling as the largest value of each window.

    Like the average, each output element reads its window in the loop of
    whatever consumes it, so no kernel of its own is launched; a window that
    reaches past the source reads the smallest value its type can hold there,
    which is the one value that can never win.
    """

    size, _, _ = val_info(node_val())
    ndim = _spatial_ndim()
    kernel, stride, padding, dilation = _max_pool_common(
        x, kernel_size, stride, padding, dilation, ndim
    )
    in_size = list(x.get_size())
    window = _prod_ints(kernel)
    if not _max_pool_checks(x, size, kernel, ndim, window):
        return _fallback_max_pool[ndim](
            x, kernel, stride, padding, dilation, ceil_mode
        )
    lead = len(in_size) - ndim
    spatial_out = [int(s) for s in list(size)[-ndim:]]

    dtype = x.get_dtype()
    loader = _max_pool_window_loader(x, ndim, dtype)

    def fn(index):
        base = index[lead:]
        best = None
        for offset in itertools.product(*[range(k) for k in kernel]):
            value = loader([*index[:lead], *_max_pool_at(base, offset, stride,
                                                         padding, dilation)])
            best = value if best is None else ops.maximum(best, value)
        return best

    return Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=[*in_size[:lead], *spatial_out],
    )


def _max_pool_at(base, offset, stride, padding, dilation):
    """Where one position of a window sits in the source."""

    return [
        as_index(base[axis]) * stride[axis] + offset[axis] * dilation[axis]
        - padding[axis]
        for axis in range(len(base))
    ]


def _max_pool_window_loader(x, ndim, dtype):
    """A reader of the window's source that reads the losing value outside.

    The value a window reads where the source has nothing is the smallest one
    its type can hold, so that no position of the source is beaten by a
    position that is not there.  A truth is its own scale: nothing wins over
    something, which is what the false reads as.
    """

    if dtype == tp.bool:
        fill = False
    elif dtype.is_floating_point:
        fill = float("-inf")
    else:
        fill = tp.iinfo(dtype).min
    return _constant_outside(x, ndim, fill)


@register("max_pool2d_with_indices.default", "max_pool3d_with_indices.default")
def lower_max_pool_with_indices(x, kernel_size, stride=(), padding=0, dilation=1,
                                ceil_mode=False, **kwargs):
    """Max pooling together with where each largest value was read from.

    The largest value and its place are asked for together, but the place is
    not worth a second pass over memory: the winner is chosen again in the
    loop that writes the places, by the same rule -- the earliest position of
    the window holds when two read equal, which is what a scan in order gives.
    """

    size, _, _ = val_info(node_val(index=0))
    ndim = _spatial_ndim()
    kernel, stride, padding, dilation = _max_pool_common(
        x, kernel_size, stride, padding, dilation, ndim
    )
    in_size = list(x.get_size())
    window = _prod_ints(kernel)
    if not _max_pool_checks(x, size, kernel, ndim, window):
        return _fallback_max_pool_with_indices[ndim](
            x, kernel, stride, padding, dilation, ceil_mode
        )
    lead = len(in_size) - ndim
    spatial_in = [int(s) for s in in_size[lead:]]
    spatial_out = [int(s) for s in list(size)[-ndim:]]

    dtype = x.get_dtype()
    loader = _max_pool_window_loader(x, ndim, dtype)
    positions = list(itertools.product(*[range(k) for k in kernel]))

    def largest(index):
        base = index[lead:]
        best = None
        for offset in positions:
            value = loader([*index[:lead], *_max_pool_at(base, offset, stride,
                                                         padding, dilation)])
            best = value if best is None else ops.maximum(best, value)
        return best

    def fn(index):
        base = index[lead:]
        best = largest(index)
        # The places are flat positions in the source's spatial volume, one
        # number however many spatial axes there are.
        found = ops.constant(False, tp.bool)
        where = ops.constant(0, tp.int64)
        for offset in positions:
            at = _max_pool_at(base, offset, stride, padding, dilation)
            value = loader([*index[:lead], *at])
            take = ops.logical_and(ops.logical_not(found), ops.eq(value, best))
            flat = as_index(at[0])
            for axis in range(1, ndim):
                flat = flat * spatial_in[axis] + as_index(at[axis])
            where = ops.where(take, ops.index_expr(flat, tp.int64), where)
            found = ops.logical_or(found, take)
        return where

    values = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=largest,
        ranges=[*in_size[:lead], *spatial_out],
    )
    indices = Pointwise.create(
        device=x.get_device(),
        dtype=tp.int64,
        inner_fn=fn,
        ranges=[*in_size[:lead], *spatial_out],
    )
    return values, indices


# ---------------------------------------------------------------------------
# Scattering into a tensor: writing the elements a value chooses
# ---------------------------------------------------------------------------


def needs_fallback_due_to_atomic_add_limitations(dtype: Any) -> bool:
    """Whether adding into memory atomically is ruled out for a type.

    Adding in place from many threads asks the memory system for a read that
    also writes, and not every type can: a 64-bit whole number or a truth
    value has no atomic add to begin with, and a 16-bit brain float only once
    the hardware grew one.
    """

    if dtype == tp.bfloat16 and tp.cuda.is_available():
        return tp.cuda.get_device_capability() < (9, 0)
    return dtype in (tp.int64, tp.bool)


def use_scatter_fallback(
    op_overload: Any,
    reduction_type: Any,
    self_dtype: Any,
    src_dtype: Any,
    src_device_type: Any,
    src_is_tensor: bool,
) -> bool:
    """Whether a scatter is handed to the framework rather than written out.

    A written-out scatter is a store, or an atomic add when the contributions
    accumulate.  Every other way of combining -- taking the larger, the
    smaller, the product, the mean -- has no atomic form here, so it is run
    through; and even an add is run through for a type that cannot be added
    atomically, for a whole-number or truth-valued destination, for a source
    of such a type on a device where threads run apart, or when the whole
    program has been asked to settle every tie the same way.
    """

    packet_name = getattr(op_overload, "__name__", str(op_overload))
    if packet_name.startswith("scatter_reduce") and reduction_type is None:
        return False

    accumulate_name = "add" if packet_name.startswith("scatter_") else "sum"

    return (
        reduction_type not in (None, accumulate_name)
        or (
            src_is_tensor
            and is_gpu(src_device_type)
            and needs_fallback_due_to_atomic_add_limitations(src_dtype)
        )
        or (
            packet_name.startswith("scatter_reduce_")
            and reduction_type == "sum"
            and src_is_tensor
            and src_device_type == "cpu"
            and getattr(config.cpp, "fallback_scatter_reduce_sum", False)
            and (
                getattr(config.cpp, "dynamic_threads", False)
                or parallel_num_threads() != 1
            )
        )
        or (reduction_type == accumulate_name and self_dtype in (tp.bool, tp.int64))
        or tp.are_deterministic_algorithms_enabled()
    )


def _box_view(value: Any) -> Any:
    """A tensor argument held in a box, whether it arrived boxed or as a bare
    view; anything that is not a tensor -- a scalar source -- is left as it is."""

    return _as_box(value) if isinstance(value, ir.IRNode) else value


def scatter_fallback(
    op_overload: Any,
    self: Any,
    dim: Any,
    index: Any,
    src: Any,
    *,
    reduce: Any = None,
    include_self: bool = True,
) -> Any:
    """Hand one scatter to the framework, or answer with nothing.

    Nothing is the answer that says the caller may write the scatter out
    itself; the framework call, once made, writes into the value it was given
    and the value is the answer.
    """

    self, index, src = (_box_view(v) for v in (self, index, src))
    src_is_tensor = isinstance(src, TensorBox)
    if use_scatter_fallback(
        op_overload,
        reduce,
        self.get_dtype(),
        src.get_dtype() if src_is_tensor else type(src),
        src.get_device().type if src_is_tensor else "not impl",
        src_is_tensor,
    ):
        ir.ScatterFallback(
            op_overload,
            self,
            dim,
            index,
            src,
            reduce=reduce,
            include_self=include_self,
        )
        return self

    return None


def index_put_as_masked_fill(self: Any, indices: Any, value: Any, accumulate: bool):
    """One truth-valued position list, one value: a masked fill.

    Where the mask is set the value is written, and where it is not what was
    there stays -- which is the whole of what writing one value at the
    positions a mask names does.  Accumulating adds the value everywhere
    first, so the positions the mask names end up holding what was there plus
    the value.
    """

    if value.get_device() != self.get_device():
        value = to_device(value, self.get_device())
    if accumulate:
        value = lower_add(self, value)
    return mutate_to(self, lower_where(indices[0], value, self))


def index_put_fallback(self: Any, indices: Any, values: Any, accumulate: bool):
    """Hand a write-by-position-list to the framework.

    A list of positions can name one place twice, and which of the values
    named for one place wins is not settled here -- the framework settles it,
    so the whole call is run through.
    """

    op_overload = getattr(
        tp_ops.index_put_, V.graph.current_node.target._overloadname
    )
    ir.IndexPutFallback(op_overload, self, indices, values, accumulate)
    return self


def index_put_impl_(
    self: Any,
    indices: Any,
    values: Any,
    accumulate: bool,
    check: bool,
    may_realize: bool = False,
):
    """Write a value into a tensor at a list of positions.

    Each position names one place for every element of the value -- the value's
    leading axes are given by the position's shape -- so the write is a loop
    over the value's positions, each asking the indices where its element
    belongs.  A position naming a place outside the tensor is a mistake and is
    refused when asked to be checked; accumulating adds instead of replaces,
    which several threads do at once through atomic adds.

    The value is read at every position it covers even where the indices
    collapse many positions onto one place, because which of those wins is
    what a scatter settles -- so the value is lined up against the positions'
    shape first, and what it would repeat is repeated.
    """

    if may_realize:
        name = ir.try_get_name(self)
        if name is not None and name in values.get_read_names() and not all(
            _index_is_a_permutation_slice(i) for i in indices
        ):
            # The value being written may be read out of the memory being
            # written to.  The write then races the read, since a position
            # written early changes what a later element reads; give the
            # value its own memory first.
            values.realize()

    # Dispatch to masked fill for single boolean index with single value
    if V.graph.sizevars.statically_known_true(
        sympy.Eq(values.get_numel(), 1)
    ) and len(indices) == 1 and indices[0].get_dtype() in (tp.bool, tp.uint8):
        mask = indices[0]
        for _ in range(len(mask.get_size()), len(self.get_size())):
            mask = unsqueeze(mask, -1)
        return index_put_as_masked_fill(self, [mask], values, accumulate)

    # Fallback in deterministic mode
    if tp.are_deterministic_algorithms_enabled():
        return index_put_fallback(self, indices, values, accumulate)

    # Fallback if there is a boolean index
    for index in indices:
        if index is not None and index.get_dtype() in (tp.bool, tp.uint8):
            return index_put_fallback(self, indices, values, accumulate)

    x_size = self.get_size()
    x_ndim = len(x_size)

    device = self.get_device()
    if (
        accumulate
        and device is not None
        and is_gpu(device.type)
        and needs_fallback_due_to_atomic_add_limitations(self.get_dtype())
    ):
        # self is a scalar tensor
        if x_ndim == 0:
            self = view(self, [1])
        self = index_put_fallback(self, indices, values, accumulate)
        if x_ndim == 0:
            self = view(self, [])
        return self

    values = to_dtype(values, self.get_dtype())

    try:
        indices, tensor_indices = check_and_broadcast_indices(
            indices, self.get_device()
        )
    except NotImplementedError:
        return index_put_fallback(self, indices, values, accumulate)

    indices_loaders = [i.make_loader() if i is not None else None for i in indices]

    if not (isinstance(self, TensorBox)):
        raise AssertionError("expected: isinstance(self, TensorBox)")
    self.realize()

    # self is a scalar tensor
    if x_ndim == 0:
        self = view(self, [1])

    # They are all required to be the same size, so the first names it
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
        check=check,
    )
    # The scatter reads the value at every position of the positions' shape;
    # see expand.
    values = lower_expand(values, expected_vals_size)
    # all guards are set above during broadcast_tensors and expand

    device = self.get_device()
    if device is None:
        raise AssertionError("expected: device is not None")
    scatter = ir.Scatter(
        device=device,
        dtype=self.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,
        output_indexer=inner_fn,
        scatter_mode="atomic_add" if accumulate else None,
    )
    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.MutationLayoutSHOULDREMOVE(self),
        data=scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    if x_ndim == 0:
        self = view(self, [])
    return self


def _index_is_a_permutation_slice(indice: Any) -> bool:
    """Whether an index is a fresh list of every position in some order.

    A list of positions that names each place exactly once, made on its own,
    cannot make the value written race its own reads: no place is written
    twice, so what an element reads is never changed by the write beside it.
    Such a list is recognized by where it came from -- a draw of every
    position once -- and anything else is refused.
    """

    if isinstance(indice, TensorBox) and isinstance(indice.data, ir.BaseView):
        indice = indice.data.unwrap_view()
        if not (isinstance(indice, ir.StorageBox) and isinstance(indice.data, ir.ExternKernel)):
            return False
        node = getattr(indice.data, "fx_node", None)
        if node is None:
            return False
        return node.target is tp_ops.randperm.default
    return False


@register_lowering(tp_ops.index_put, type_promotion_kind=None)
def index_put(x: Any, indices: Any, values: Any, accumulate: bool = False):
    return index_put_impl_(
        clone(x), indices, values, accumulate, check=True, may_realize=False
    )


@register_lowering(tp_ops._unsafe_index_put, type_promotion_kind=None)
def _unsafe_index_put(x: Any, indices: Any, values: Any, accumulate: bool = False):
    return index_put_impl_(
        clone(x), indices, values, accumulate, check=False, may_realize=False
    )


@register_lowering(tp_ops.index_put_, type_promotion_kind=None)
def index_put_(self: Any, indices: Any, values: Any, accumulate: bool = False):
    return index_put_impl_(
        self, indices, values, accumulate, check=True, may_realize=True
    )


def _positions_along_one_axis(x: Any, index: Any, dim: int) -> list:
    """A single axis's index as a whole position list.

    An index that names positions along one axis, with every other axis taken
    whole, is a position list with one entry for that axis and nothing for the
    rest.  The axis is counted from the front of ``x``, so one counted from the
    back is first turned around.
    """

    ndim = len(x.get_size())
    if ndim:
        dim = dim % ndim
    return [None] * dim + [index]


@register_lowering(tp_ops.index_add, type_promotion_kind=None)
def index_add(x: Any, dim: Any, index: Any, tensor: Any, *, alpha: Any = 1):
    if alpha != 1:
        tensor = lower_mul(tensor, alpha)
    return index_put_impl_(
        clone(x),
        _positions_along_one_axis(x, index, dim),
        tensor,
        True,
        check=True,
        may_realize=False,
    )


@register_lowering(tp_ops.index_add_, type_promotion_kind=None)
def index_add_(x: Any, dim: Any, index: Any, tensor: Any, *, alpha: Any = 1):
    if alpha != 1:
        tensor = lower_mul(tensor, alpha)
    return index_put_impl_(
        x, _positions_along_one_axis(x, index, dim), tensor, True, check=True, may_realize=True
    )


@register_lowering(tp_ops.index_copy, type_promotion_kind=None)
def index_copy(x: Any, dim: Any, index: Any, tensor: Any):
    return index_put_impl_(
        clone(x),
        _positions_along_one_axis(x, index, dim),
        tensor,
        False,
        check=True,
        may_realize=False,
    )


@register_lowering(tp_ops.index_copy_, type_promotion_kind=None)
def index_copy_(x: Any, dim: Any, index: Any, tensor: Any):
    return index_put_impl_(
        x, _positions_along_one_axis(x, index, dim), tensor, False, check=True, may_realize=True
    )


@register_lowering(tp_ops.diagonal_scatter, type_promotion_kind=None)
def diagonal_scatter(input: Any, src: Any, offset: Any = 0, dim1: Any = 0, dim2: Any = 1):
    output = clone(input)
    target = lower_diagonal(output, offset, dim1, dim2)
    mutate_to(target, src)
    return output


@register_lowering(tp_ops.scatter, type_promotion_kind=None)
def scatter(x: Any, dim: Any, index: Any, src: Any, **kwargs: Any):
    return scatter_(clone(x), dim, index, src, **kwargs)


@register_lowering(tp_ops.scatter_, type_promotion_kind=None)
def scatter_(self: Any, dim: Any, index: Any, src: Any, *, reduce: Any = None):
    if reduce not in (None, "add", "multiply"):
        raise AssertionError('expected: reduce in (None, "add", "multiply")')
    if reduce is None:
        op_overload = getattr(
            tp_ops.scatter_, V.graph.current_node.target._overloadname
        )
        fallback_result = scatter_fallback(
            op_overload, self, dim, index, src, reduce=reduce
        )
        if fallback_result is not None:
            return fallback_result

    if reduce == "add":
        reduce = "sum"
    elif reduce == "multiply":
        reduce = "prod"
    return scatter_reduce_(self, dim, index, src, reduce)


@register_lowering(tp_ops.scatter_add, type_promotion_kind=None)
def scatter_add(x: Any, dim: Any, index: Any, src: Any):
    return scatter_add_(clone(x), dim, index, src)


@register_lowering(tp_ops.scatter_add_, type_promotion_kind=None)
def scatter_add_(x: Any, dim: Any, index: Any, src: Any):
    return scatter_reduce_(x, dim, index, src, "sum")


@register_lowering(tp_ops.scatter_reduce, type_promotion_kind=None)
def scatter_reduce(x: Any, dim: Any, index: Any, src: Any, reduction_type: Any, **kwargs: Any):
    return scatter_reduce_(clone(x), dim, index, src, reduction_type, **kwargs)


@register_lowering(tp_ops.scatter_reduce_, type_promotion_kind=None)
def scatter_reduce_(self: Any, dim: Any, index: Any, src: Any, reduce: Any, *, include_self: bool = True):
    """Write a value into a tensor along one axis, combining what lands together.

    Each element of the value is written at the place its index names along
    that axis, its other axes taken as they are.  Several elements can name
    the same place; how they combine is the reduction: added, or replaced --
    replaced being an add of nothing, since the place already holds what the
    destination started with.  What was there can also be left out, which
    zeroes each destination first so only what arrives is kept.

    Adding is written as an atomic add, so many threads can land on one place
    at once; every other combining is handed to the framework, which settles
    the order among the contributions itself.
    """

    if reduce not in (None, "sum", "prod", "mean", "amax", "amin"):
        raise AssertionError(
            'expected: reduce in (None, "sum", "prod", "mean", "amax", "amin")'
        )
    self, index, src = (_box_view(v) for v in (self, index, src))
    if "two" not in getattr(tp_ops.scatter_reduce_, "overloads", list)():
        raise AssertionError(
            "tp.scatter_reduce_.two is not the unique overload of tp.scatter_reduce_"
        )

    fallback_result = scatter_fallback(
        tp_ops.scatter_reduce_.two,
        self,
        dim,
        index,
        src,
        reduce=reduce,
        include_self=include_self,
    )
    if fallback_result:
        return fallback_result

    if not (isinstance(self, TensorBox)):
        raise AssertionError("expected: isinstance(self, TensorBox)")

    ndim = len(self.get_size())
    if ndim == 0:
        self = view(self, [1])

    if isinstance(src, TensorBox) and len(src.get_size()) == 0:
        src = view(src, [1])

    if isinstance(index, TensorBox) and len(index.get_size()) == 0:
        index = view(index, [1])

    if V.graph.sizevars.statically_known_true(sympy.Eq(index.get_numel(), 0)):
        return self

    if "int" not in str(index.get_dtype()):
        raise AssertionError('expected: "int" in str(index.get_dtype())')

    dim = _validate_dim(self, dim)

    self.realize()
    index_loader = index.make_loader()
    src_loader = src.make_loader() if isinstance(src, TensorBox) else None

    def output_indexer(idx: Any):
        shape = self.get_size()
        ndim = len(shape)
        indirect_idx = list(idx)
        indirect_idx[dim] = ops.indirect_indexing(
            index_loader(idx), 1 if ndim == 0 else shape[dim], wrap_neg=False
        )
        return indirect_idx

    def fn(idx: Any):
        if src_loader:
            return src_loader(idx)
        # src is a scalar
        return ops.constant(src, self.get_dtype())

    def backend_reduce_str(reduce: Any):
        if reduce == "sum":
            return "atomic_add"
        if reduce is not None:
            raise AssertionError("expected: reduce is None")
        return None

    device = self.get_device()
    if device is None:
        raise AssertionError("expected: device is not None")

    if not include_self:
        # zero out the corresponding elements first
        zero_out = ir.Scatter(
            device=device,
            dtype=self.get_dtype(),
            inner_fn=lambda index: ops.constant(0, self.get_dtype()),
            ranges=index.get_size(),
            output_indexer=output_indexer,
            scatter_mode=None,
        )
        buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.MutationLayoutSHOULDREMOVE(self),
            data=zero_out,
        )
        buffer.name = V.graph.register_buffer(buffer)
        V.graph.register_operation(buffer)

    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    # self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
    scatter = ir.Scatter(
        device=device,
        dtype=self.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
        output_indexer=output_indexer,
        scatter_mode=backend_reduce_str(reduce),
    )
    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.MutationLayoutSHOULDREMOVE(self),
        data=scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    if ndim == 0:
        self = view(self, [])
    return self


# ---------------------------------------------------------------------------
# Pooling backwards
# ---------------------------------------------------------------------------


_fallback_avg_pool2d_backward = fallback_handler(
    tp.ops.tp.avg_pool2d_backward.default, add_to_fallback_set=False
)


@register("avg_pool2d_backward.default")
def lower_avg_pool2d_backward(grad, _input, kernel_size, stride=(), padding=0,
                              ceil_mode=False, count_include_pad=True,
                              divisor_override=None, **kwargs):
    """2D average pooling gradient, read from the few windows that cover a position.

    A position of the input was read by the windows whose start lies within a
    kernel's width before it, which for the usual strides is one window or a
    handful.  Each position therefore reads those few gradient values and adds
    them, in the loop of whatever consumes the result, instead of a kernel of
    its own walking every window.  A kernel wide enough that a position sits
    under many windows is asked of the framework kernel instead.
    """

    def fallback():
        return _fallback_avg_pool2d_backward(
            grad, _input, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override,
        )

    kernel = _pair(kernel_size, 2)
    stride = _pair(stride, 2) if stride else list(kernel)
    padding = _pair(padding, 2) if padding else [0, 0]
    if divisor_override is not None and divisor_override == 0:
        return fallback()
    in_size = list(_input.get_size())
    grad_size = list(grad.get_size())
    if len(in_size) not in (3, 4) or is_dynamic(*in_size, *grad_size):
        return fallback()
    height, width = int(in_size[-2]), int(in_size[-1])
    pooled_height, pooled_width = int(grad_size[-2]), int(grad_size[-1])
    # A window that hangs over the padded extent is divided by what it covers
    # rather than by the kernel, whichever way the padding is counted.
    overhang = _window_overhangs(
        height, kernel[0], stride[0], padding[0], pooled_height
    ) or _window_overhangs(width, kernel[1], stride[1], padding[1], pooled_width)
    if divisor_override is not None:
        fixed = divisor_override
    elif not overhang and (count_include_pad or not (padding[0] or padding[1])):
        fixed = kernel[0] * kernel[1]
    else:
        fixed = None

    h_window = max(
        max(h // stride[0] - max(0, (h - kernel[0]) // stride[0]), 1)
        for h in range(kernel[0] * 2)
    )
    w_window = max(
        max(w // stride[1] - max(0, (w - kernel[1]) // stride[1]), 1)
        for w in range(kernel[1] * 2)
    )
    if h_window * w_window > 25:
        return fallback()

    # Read once per covering window, so it is put in memory when it is more
    # than a plain read.
    grad.realize_hint()
    loader = grad.make_loader()
    i32 = tp.int32

    def covered(ph, pw):
        """How many positions the window at (ph, pw) is divided by."""

        hstart = ops.sub(ops.mul(ph, ops.constant(stride[0], i32)), ops.constant(padding[0], i32))
        wstart = ops.sub(ops.mul(pw, ops.constant(stride[1], i32)), ops.constant(padding[1], i32))
        hend = ops.minimum(
            ops.add(hstart, ops.constant(kernel[0], i32)),
            ops.constant(height + padding[0], i32),
        )
        wend = ops.minimum(
            ops.add(wstart, ops.constant(kernel[1], i32)),
            ops.constant(width + padding[1], i32),
        )
        if not count_include_pad:
            hstart = ops.maximum(hstart, ops.constant(0, i32))
            wstart = ops.maximum(wstart, ops.constant(0, i32))
            hend = ops.minimum(hend, ops.constant(height, i32))
            wend = ops.minimum(wend, ops.constant(width, i32))
        return ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))

    def fn(idx):
        *prefix, h, w = idx
        h = as_index(h) + padding[0]
        w = as_index(w) + padding[1]
        phstart = ops.index_expr(FloorDiv(h - kernel[0] + stride[0], stride[0]), i32)
        pwstart = ops.index_expr(FloorDiv(w - kernel[1] + stride[1], stride[1]), i32)
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, i32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, i32)
        phstart = ops.maximum(phstart, ops.constant(0, i32))
        pwstart = ops.maximum(pwstart, ops.constant(0, i32))
        phend = ops.minimum(phend, ops.constant(pooled_height, i32))
        pwend = ops.minimum(pwend, ops.constant(pooled_width, i32))

        gradient = None
        for ph_ in range(h_window):
            for pw_ in range(w_window):
                ph = ops.add(phstart, ops.constant(ph_, i32))
                pw = ops.add(pwstart, ops.constant(pw_, i32))
                scale = fixed if fixed is not None else covered(ph, pw)
                part = ops.truediv(
                    loader([
                        *prefix,
                        ops.indirect_indexing(
                            ops.minimum(ph, ops.sub(phend, ops.constant(1, i32))),
                            pooled_height,
                            check=False,
                        ),
                        ops.indirect_indexing(
                            ops.minimum(pw, ops.sub(pwend, ops.constant(1, i32))),
                            pooled_width,
                            check=False,
                        ),
                    ]),
                    scale,
                )
                mask = ops.and_(ops.lt(ph, phend), ops.lt(pw, pwend))
                if gradient is None:
                    gradient = ops.where(mask, part, ops.constant(0.0, tp.float32))
                else:
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)
        return gradient

    return Pointwise.create(
        device=grad.get_device(),
        dtype=_input.get_dtype(),
        inner_fn=fn,
        ranges=in_size,
    )


_fallback_upsample_nearest2d_backward = fallback_handler(
    tp.ops.tp.upsample_nearest2d_backward.default, add_to_fallback_set=False
)


@register("upsample_nearest2d_backward.default")
def lower_upsample_nearest2d_backward(grad, output_size, input_size,
                                      scales_h=None, scales_w=None, **kwargs):
    """Nearest upsampling's gradient, as the sum of the positions that read one.

    Upsampling by a whole factor reads each input position from a block of
    output positions, so the input's gradient at a position is the sum of the
    output's gradient over that block: a few reads and additions per element,
    done in the loop of whatever consumes the result.  A factor that is not a
    whole number, or one spelled differently from the sizes, is asked of the
    framework kernel.
    """

    def fallback():
        return _fallback_upsample_nearest2d_backward(
            grad, output_size, input_size, scales_h, scales_w
        )

    grad_size = list(grad.get_size())
    if len(grad_size) != 4 or is_dynamic(*grad_size):
        return fallback()
    try:
        out_h, out_w = (int(s) for s in list(output_size)[-2:])
        in_h, in_w = (int(s) for s in list(input_size)[-2:])
    except (TypeError, ValueError):
        return fallback()
    if (int(grad_size[-2]), int(grad_size[-1])) != (out_h, out_w):
        return fallback()
    if in_h <= 0 or in_w <= 0 or out_h % in_h or out_w % in_w:
        return fallback()
    rh, rw = out_h // in_h, out_w // in_w
    for given, ratio in ((scales_h, rh), (scales_w, rw)):
        if given is not None and float(given) != float(ratio):
            return fallback()
    if rh * rw > 25:
        return fallback()

    # Read once per position of the block, so it is put in memory when it is
    # more than a plain read.
    grad.realize_hint()
    loader = grad.make_loader()

    def fn(idx):
        *prefix, i, j = idx
        total = None
        for a in range(rh):
            for b in range(rw):
                value = loader([*prefix, as_index(i) * rh + a, as_index(j) * rw + b])
                total = value if total is None else ops.add(total, value)
        return total

    return Pointwise.create(
        device=grad.get_device(),
        dtype=grad.get_dtype(),
        inner_fn=fn,
        ranges=[*grad_size[:-2], in_h, in_w],
    )


@register("shallow_copy_data_.default")
def lower_shallow_copy_data_(self_tensor, storage_tensor):
    """Pointing one value at the memory of another, contents and all.

    This is told apart from making one value refer to another because what it
    does to the memory's shape is not the same: here the first value keeps
    describing the region it was describing, and only the memory underneath it
    is taken over.  A value that was a transposed or narrowed view of another's
    memory therefore stays that view, and a result written through it lands in
    the memory both names.
    """
    self_tensor.realize()
    storage_tensor.realize()
    return TensorBox.create(ir.ShallowCopyDataKernel(self_tensor, storage_tensor))


# The pointwise operations, declared with the promotion each one applies.
#
# Registering a lowering and stating the promotion together is what keeps the
# two from drifting apart: an operation whose result is the promotion of its
# operands says so here, so the type of its result is known without asking the
# operation, and an operation that is not declared here either says what it
# produces or is caught where the types are collected.
#
# The ones declared as numeric read a number and produce a real number, so an
# integer argument does not make the result an integer.
register_pointwise(
    "add",
    "rsub",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "floordiv_tensor",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "and_",
    "or_",
    "xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "maximum",
    "minimum",
    "fmaximum",
    "clamp_min",
    "clamp_max",
    "where",
    "remainder",
    "fmod",
    "aten_add",
    "sum",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "rsqrt",
    "erf",
    "erfc",
    "erfinv",
    "lgamma",
    "sigmoid",
    "tanh",
    "cosh",
    "sinh",
    "acos",
    "asin",
    "atan",
    "atan2",
    "atanh",
    "asinh",
    "acosh",
    "cos",
    "sin",
    "tan",
    "sign",
    "abs",
    "neg",
    "square",
    "ceil",
    "floor",
    "round",
    "trunc",
    "nan_to_num",
    "nextafter",
    "hypot",
    "copysign",
    "ldexp",
    "sigmoid_backward",
    "relu",
    "lu",
    "index",
)
register_pointwise_numeric(
    "reciprocal",
    "sigmoid",
    "softplus",
    "elu",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "silu",
    "mish",
    "tanh",
    "softsign",
    "selu",
)
register_op_dtype_propagation_rules(
    "ldexp",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    override_return_dtype=None,
)
register_op_dtype_propagation_rules(
    "fmaximum",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    override_return_dtype=None,
)
register_op_dtype_propagation_rules(
    "remainder",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    override_return_dtype=None,
)
# The predicates and the logical operations answer with a truth value
# whatever they were asked about.
for _name in ("isnan", "isinf", "signbit", "logical_not", "logical_and", "logical_or", "logical_xor"):
    register_op_dtype_propagation_rules(
        _name,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
        override_return_dtype=tp.bool,
    )
# The operations that produce no value, and so have no result type at all.
for _name in ("output", "placeholder", "device_assert_async", "check_bounds"):
    register_op_dtype_propagation_rules(
        _name, type_promotion_kind=None, override_return_dtype=None
    )


def _per_tensor(key: str) -> Callable[..., Any]:
    """The one-tensor-at-a-time lowering behind an operation, without its wrapper.

    What a registered lowering carries around it is the promotion and
    broadcasting its arguments need when one caller passes a number and another
    passes a tensor.  A list operation hands each of its entries values that are
    already alike -- that is what a list of alike operations means -- so what is
    wanted here is the arithmetic itself, and asking for the wrapped form would
    do that work once per entry to arrive at the same answer.
    """

    fn = LOWERINGS[key]
    return getattr(fn, "__wrapped__", fn)


def _register_foreach_all() -> None:
    """The list form of every operation whose one-tensor form already exists.

    Each entry says which list operations are this operation applied to a list,
    and which is the one-tensor lowering they all share.  The scalar an
    operation carries is named where it arrives, because the schema decides that
    per operation and a guess would be a wrong guess.
    """

    table: list[tuple] = [
        # (per-tensor key, allow_alpha, scalar_kwarg, [(foreach names)])
        ("add.Tensor", True, "alpha", [
            ("_foreach_add.List",), ("_foreach_add.Scalar",), ("_foreach_add.Tensor",),
        ]),
        ("mul.Tensor", False, "alpha", [
            ("_foreach_mul.List",), ("_foreach_mul.Tensor",), ("_foreach_mul.Scalar",),
        ]),
        ("sub.Tensor", True, "alpha", [
            ("_foreach_sub.List",), ("_foreach_sub.Scalar",),
        ]),
        ("div.Tensor", False, "alpha", [
            ("_foreach_div.List",), ("_foreach_div.Tensor",), ("_foreach_div.Scalar",),
        ]),
        ("neg.default", False, "alpha", [("_foreach_neg.default",)]),
        ("abs.default", False, "alpha", [("_foreach_abs.default",)]),
        ("sqrt.default", False, "alpha", [("_foreach_sqrt.default",)]),
        ("rsqrt.default", False, "alpha", [("_foreach_rsqrt.default",)]),
        ("reciprocal.default", False, "alpha", [("_foreach_reciprocal.default",)]),
        ("clone.default", False, "alpha", [("_foreach_clone.default",)]),
        ("sign.default", False, "alpha", [("_foreach_sign.default",)]),
        ("maximum.default", False, "alpha", [
            ("_foreach_maximum.List",), ("_foreach_maximum.Scalar",),
        ]),
        ("minimum.default", False, "alpha", [
            ("_foreach_minimum.List",), ("_foreach_minimum.Scalar",),
        ]),
    ]
    for key, allow_alpha, scalar_kwarg, groups in table:
        pw = _per_tensor(key)
        for names in groups:
            register_foreach_pointwise(
                pw, allow_alpha=allow_alpha, scalar_kwarg=scalar_kwarg, names=names
            )

    # A clamp is a lower bound and an upper bound, so its list form is whichever
    # of the two it is named for; the arithmetic is the same either way.
    for lo, hi, suffix in (("minimum.default", "maximum.default", "clamp_min"),
                           ("maximum.default", "minimum.default", "clamp_max")):
        pw = _per_tensor(lo if suffix == "clamp_min" else hi)
        for names in ((f"_foreach_{suffix}.List",), (f"_foreach_{suffix}.Scalar",)):
            register_foreach_pointwise(pw, allow_alpha=False, scalar_kwarg="alpha", names=names)


def _registered_foreach(name: str) -> Callable[..., Any]:
    """The list form already registered under a name.

    The writing form of a list operation is the non-writing one's result made
    to be its input, so it is registered from that rather than from the
    one-tensor form again -- which is what keeps the two answering alike.
    """

    return LOWERINGS[name]


def _register_foreach_inplace_all() -> None:
    """The writing form of every list operation that has one.

    Each entry pairs a writing form with the non-writing form it is made from,
    because the difference between them is where the result goes and nothing
    else: the arithmetic is the non-writing operation's.
    """

    for inplace_name, outplace_name in (
        ("_foreach_add_.List", "_foreach_add.List"),
        ("_foreach_add_.Scalar", "_foreach_add.Scalar"),
        ("_foreach_mul_.List", "_foreach_mul.List"),
        ("_foreach_mul_.Scalar", "_foreach_mul.Scalar"),
        ("_foreach_div_.List", "_foreach_div.List"),
        ("_foreach_div_.Scalar", "_foreach_div.Scalar"),
    ):
        register_foreach_inplace(
            names=(inplace_name,),
            outplace_names=(outplace_name,),
            outplace_op=_registered_foreach(outplace_name),
        )


_register_foreach_all()


def pow_recursive(x: Any, y: int, dtype: Any) -> Any:
    """A whole-number power by squaring, and the value is written out.

    A power is not one operation but a number of them, and which number depends
    on the power: squaring halves how many multiplications there are, and a
    power of one is the value itself rather than a multiplication of it.  The
    result is assembled from operations the loop body already has, so a power
    costs no operation this compiler did not already have.
    """

    if y < 0:
        return pow_recursive(ops.reciprocal(x), -y, dtype)
    if y == 0:
        return ops.constant(1, dtype)
    if y == 1:
        return x

    result = pow_recursive(x, y // 2, dtype)
    result = ops.mul(result, result)
    if (y % 2) == 1:
        result = ops.mul(result, x)
    return result


#: The whole-number power, handed to the framework.  The device's own power is
#: for real numbers, so this is not a shortcut but the only answer for a
#: whole-number power of whole numbers; and an exponent large enough to need it
#: is one where writing out multiplications would cost more than the call.
_fallback_pow = fallback_handler(tp.ops.tp.pow)


@register("pow.Tensor_Scalar")
@register("pow.Scalar")
def lower_pow(a: Any, b: Any) -> Any:
    """One value raised to a power.

    The powers worth writing out are the ones a program actually asks for: a
    square root is a square root, a power of one is the value, and a small whole
    number is a number of multiplications rather than a call.  A power too large
    to write out, and a whole-number power of whole numbers where the
    arithmetic is not the one the device does, are handed to the framework --
    which is a slower answer rather than a wrong one.
    """

    if isinstance(b, float) and b.is_integer():
        return lower_pow(a, int(b))
    elif isinstance(b, float) and b == 0.5:
        return LOWERINGS["sqrt.default"](a)
    elif isinstance(b, int) and b == 1:
        return clone(a)

    # The arguments have been made to agree on a type by now, so either will do.
    dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.IRNode))
    is_integer_pow = is_integer_dtype(dtype)

    embed_exponent = isinstance(b, int) and (
        -32 < b < 32 or (is_integer_pow and b >= 0)
    )
    if embed_exponent:
        loader = a.make_loader()

        def fn(idx):
            return pow_recursive(loader(idx), b, a.get_dtype())

        return Pointwise.create(
            device=a.get_device(),
            dtype=a.get_dtype(),
            inner_fn=fn,
            ranges=a.get_size(),
        )

    if is_integer_pow:
        # The device's own power is for real numbers; a whole-number power of
        # whole numbers is not what it computes.
        return _fallback_pow(a, b)

    return pointwise(ops.pow, a, b)


def _use_fma(dtype: Any, device: Any) -> bool:
    """Whether a product and a sum of it should be done in one rounding.

    A device that can multiply and add without rounding between the two does it
    in one step, which is not the same answer as rounding the product first:
    the product keeps digits the addition would have dropped.  Which is more
    accurate is not the question -- the question is which one the program
    expects, and whole numbers have no such step to take.
    """

    return (
        dtype.is_floating_point
        and device is not None
        and device.type in ["cuda", "xpu"]
    )


@register("addcmul.default")
def lower_addcmul(self: Any, tensor1: Any, tensor2: Any, *, value: Any = 1) -> Any:
    """This, plus a scaled product of two others.

    The order is the whole of it: the product is rounded before it is scaled and
    added, because that is the order the answer is defined in, and a device that
    would otherwise do the multiply and the add in one step is asked to round in
    between so that it arrives at the same number.  Whole numbers have no such
    step, so they are added the long way round.
    """

    dtype = promoted_dtype_of_values(
        self, tensor1, tensor2, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    self_loader = self.make_loader()
    t1_loader = tensor1.make_loader()
    t2_loader = tensor2.make_loader()

    device = self.get_device()
    use_fma = _use_fma(dtype, device)

    def inner_fn(idx):
        self_val = self_loader(idx)
        t1_val = t1_loader(idx)
        t2_val = t2_loader(idx)

        if value == 1 and use_fma:
            return ops.fma(t1_val, t2_val, self_val)

        # Round the product before it is scaled and added.
        if use_fma:
            t1_times_t2 = ops.mul_rn(t1_val, t2_val)
        else:
            t1_times_t2 = ops.mul(t1_val, t2_val)

        # A power that came from a value only known while the program runs is an
        # expression rather than a number, and is written as one.
        if isinstance(value, sympy.Basic):
            value_expr = ops.index_expr(value, dtype)
        else:
            value_expr = ops.constant(value, dtype)

        if use_fma:
            return ops.fma(value_expr, t1_times_t2, self_val)
        else:
            return ops.add(self_val, ops.mul(value_expr, t1_times_t2))

    return Pointwise.create(
        device=self.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )


@register("addcdiv.default")
def lower_addcdiv(self: Any, tensor1: Any, tensor2: Any, *, value: Any = 1) -> Any:
    """This, plus a scaled quotient of two others.

    The quotient is rounded before it is scaled and added, for the same reason
    and in the same order as the product above: the answer is defined as this
    plus the scaled quotient, not as this plus a quotient of a scaled pair.
    """

    dtype = promoted_dtype_of_values(
        self, tensor1, tensor2, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )

    self_loader = self.make_loader()
    t1_loader = tensor1.make_loader()
    t2_loader = tensor2.make_loader()

    device = self.get_device()
    use_fma = _use_fma(dtype, device)

    def inner_fn(idx):
        self_val = self_loader(idx)
        t1_val = t1_loader(idx)
        t2_val = t2_loader(idx)

        # Round the quotient before it is scaled and added.
        if use_fma:
            quot = ops.div_rn(t1_val, t2_val)
        else:
            quot = ops.div(t1_val, t2_val)

        if isinstance(value, sympy.Basic):
            value_expr = ops.index_expr(value, dtype)
        else:
            value_expr = ops.constant(value, dtype)

        if value == 1 and use_fma:
            return ops.fma(quot, value_expr, self_val)
        if use_fma:
            return ops.fma(value_expr, quot, self_val)
        return ops.add(self_val, ops.mul(value_expr, quot))

    return Pointwise.create(
        device=self.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )


@register("copy_.default")
def lower_copy_(dst: Any, src: Any, non_blocking: Any = False) -> Any:
    """One value's contents become another's, where the other already exists.

    Everything about the source is made to match the destination first --
    where it is, what type it is, how big it is -- because what is being
    promised is that the destination holds these contents, and a copy that
    changed any of those would be copying something else.  A copy onto itself is
    not a copy, and is left alone rather than done.
    """

    if dst is src:
        return dst
    if not isinstance(src, ir.IRNode):
        # A source that is a number rather than a value is a value of that
        # number, shaped like the destination and held in it.
        src = Pointwise.create(
            device=dst.get_device(),
            dtype=dst.get_dtype(),
            inner_fn=lambda idx: ops.constant(src, dst.get_dtype()),
            ranges=dst.get_size(),
        )
    x = src
    if dst.get_device() != src.get_device():
        x = to_device(x, dst.get_device())
    if dst.get_dtype() != src.get_dtype():
        x = ops.to_dtype(x, dst.get_dtype())

    if list(dst.get_size()) != list(src.get_size()):
        out = LOWERINGS["expand.default"](x, dst.get_size())
        result = clone(out)
    else:
        result = clone(x)

    return mutate_to(dst, result)


def _register_foreach_remaining() -> None:
    """The list forms of the operations whose single form has an argument of
    its own.

    A power, a scaled product and a scaled quotient each carry a number beside
    their tensors, and where that number arrives is part of the operation's
    shape rather than something to be guessed at here.  A copy is a list
    operation too, and writing into the destination is a copy whose result is
    made to be its source.
    """

    for allow_alpha, scalar_kwarg, key, names in (
        (False, "alpha", "pow.Tensor_Scalar", (
            "_foreach_pow.Scalar", "_foreach_pow.List", "_foreach_pow.ScalarAndTensor",
        )),
        (True, "value", "addcmul.default", ("_foreach_addcmul.Scalar",)),
        (True, "value", "addcdiv.default", ("_foreach_addcdiv.Scalar",)),
    ):
        register_foreach_pointwise(
            _per_tensor(key),
            allow_alpha=allow_alpha,
            scalar_kwarg=scalar_kwarg,
            names=names,
        )

    register_foreach_pointwise(
        _per_tensor("copy_.default"), names=("_foreach_copy.default",)
    )

    register_foreach_inplace(
        names=("_foreach_copy_.default",),
        outplace_names=("_foreach_copy.default",),
        outplace_op=_registered_foreach("_foreach_copy.default"),
    )


_register_foreach_all()
_register_foreach_inplace_all()
_register_foreach_remaining()


#: The operations that are computed by handing the whole call to the framework
#: rather than by compiling them.  Kept apart from the lowering table, because
#: "how this is computed" and "what this is" are two different facts about the
#: same operation, and code that asks one of them is not asking the other.
fallbacks: set = set()

#: The operations whose arguments must be written down before they are handed
#: over, because the framework is handed the values rather than a description
#: of how to compute them.
needs_realized_inputs: set = set()

#: What each operation's arguments must look like for the operation to be
#: computed at all -- asked before the operation is chosen, not after.
_maybe_layout_constraints: dict = {}


def add_needs_realized_inputs(fn: Any) -> None:
    """Say that an operation's arguments have to exist as values already.

    An operation computed by handing the call over is given what the operation
    was called with, not a description of how to produce it, so an argument that
    is only ever described has to be written down first.
    """

    if isinstance(fn, (list, set, tuple)):
        return [add_needs_realized_inputs(x) for x in fn]
    needs_realized_inputs.add(fn)


def add_layout_constraint(fn: Any, constraint: Callable[..., Any]) -> None:
    """Say what an operation's arguments must look like for it to be computable.

    Some operations are only defined for arguments of a particular shape, and
    finding that out by trying is how a program ends up failing rather than
    taking a different path.  The constraint is asked instead, before the
    operation is committed to.
    """

    _maybe_layout_constraints[fn] = constraint


def require_dense(_, *args, **kwargs):
    """Ask that every tensor argument already be laid out as it will be read.

    An operation that reads its arguments through a kernel written for one
    layout cannot be handed a value whose layout is merely describable, so the
    arguments that are tensors are asked to be written down with the layout
    they are to be read in.
    """

    args, kwargs = tree_map(_densify_argument, (args, kwargs))
    return args, kwargs


def _densify_argument(value: Any) -> Any:
    if isinstance(value, TensorBox):
        return ir.ExternKernel.require_stride1(value)
    return value


def get_constraint_for_op(fn: Any) -> Callable[..., Any] | None:
    """What this operation's arguments must look like, if anything was said."""

    return _maybe_layout_constraints.get(fn)


def make_fallback(
    op: Any,
    layout_constraint: Callable[..., Any] | None = None,
    warn: bool = True,
) -> Callable[..., Any]:
    """Register an operation as computed by handing the call to the framework.

    Whether an operation should be computed this way is a decision made where
    the operation is lowered rather than made for the operation, which is why
    this returns what to call instead of being the call.  What it registers is
    the operation in the table, in the set of operations handled this way, and
    -- where one was given -- the constraint that says whether it can be.
    """

    def register_fallback(op_overload: Any) -> None:
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        handler = fallback_handler(op_overload, add_to_fallback_set=False)
        fallbacks.add(op_overload)
        # Through the table's own registration, which is what turns an
        # operation into the name the table is keyed by; going in directly
        # would file it under the operation rather than under its name, and the
        # two would then be two entries for one operation.
        register_lowering(op_overload, type_promotion_kind=None)(handler)

    if callable(getattr(op, "overloads", None)):
        # What an operation names are its forms; each is reached by name on the
        # operation itself, and a form is what a graph node can be.
        for op_overload in op.overloads():
            register_fallback(getattr(op, op_overload))
    else:
        register_fallback(op)
    return fallback_handler


def _register_attention_fallbacks() -> None:
    """Every way of asking for attention that is computed by handing it over.

    Each of these is one operation the framework already has a fused kernel
    for, in a form the compiler does not decompose and would not improve on by
    trying.  What registering them says is that they are computed that way, so
    that a graph containing one has somewhere to go for it -- and so that
    whatever walks the graph afterwards can tell a value that exists from one
    that is only a description of a call.

    They are grouped by which of the framework's own attention operations they
    are, because that is what decides which one is used: they are the same
    attention under different names, and a program picks one.
    """

    for name in (
        # The fused attention the framework provides, and its gradients.
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_efficient_attention_backward",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_flash_attention_backward",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_cudnn_attention_backward",
        "_scaled_dot_product_flash_attention_for_cpu",
        "_scaled_dot_product_flash_attention_for_cpu_backward",
        "_scaled_dot_product_fused_attention_overrideable",
        "_scaled_dot_product_fused_attention_overrideable_backward",
        # The same attention named directly rather than through the scaled form.
        "_flash_attention_forward",
        "_flash_attention_backward",
        "_efficient_attention_forward",
        "_efficient_attention_backward",
    ):
        packet = getattr(tp_ops, name, None)
        if packet is None:
            continue
        # A name bound to a plain function is an adapter rather than a set of
        # overloads, and an adapter is not something a graph node can be.
        if not callable(getattr(packet, "overloads", None)):
            continue
        make_fallback(packet, warn=False)


_register_attention_fallbacks()


@register("set_.source_Tensor")
def lower_set_source_tensor(self: Any, source_tensor: Any) -> Any:
    """One value becomes another's, by taking over what the other reads.

    Not a copy: the destination stops being what it was and starts being the
    source, which is why both have to be written down first -- what the
    destination was is still being read by whatever read it, and what the
    source is has to exist before it can be what they read.
    """

    self.realize()
    source_tensor.realize()
    return TensorBox.create(ir.SetSourceTensorKernel(self, source_tensor))


def _register_writing_forms() -> None:
    """The operations that write into their first argument.

    Each is the non-writing operation's result made to be that argument, so
    registering it from the non-writing form is what keeps the two computing the
    same thing -- the only difference between them is where the result goes.
    """

    for names, key in (
        (("div_.Tensor", "div_.Scalar", "div_.Tensor_mode", "div_.Scalar_mode"), "div.Tensor"),
        (("sub_.Tensor", "sub_.Scalar"), "sub.Tensor"),
        (("mul_.Tensor", "mul_.Scalar"), "mul.Tensor"),
        (("add_.Tensor", "add_.Scalar"), "add.Tensor"),
        (("bitwise_and_.Tensor", "bitwise_and_.Scalar"), "bitwise_and.Tensor"),
        (("bitwise_or_.Tensor", "bitwise_or_.Scalar"), "bitwise_or.Tensor"),
        (("bitwise_xor_.Tensor", "bitwise_xor_.Scalar"), "bitwise_xor.Tensor"),
        (("bitwise_not_.default",), "bitwise_not.default"),
        (
            ("bitwise_left_shift_.Tensor", "bitwise_left_shift_.Tensor_Scalar"),
            "bitwise_left_shift.Tensor",
        ),
        (
            ("bitwise_right_shift_.Tensor", "bitwise_right_shift_.Tensor_Scalar"),
            "bitwise_right_shift.Tensor",
        ),
        (("logical_and_.default",), "logical_and.default"),
        (("logical_not_.default",), "logical_not.default"),
        (("logical_or_.default",), "logical_or.default"),
        (("logical_xor_.default",), "logical_xor.default"),
        (("relu_.default",), "relu.default"),
        (("sigmoid_.default",), "sigmoid.default"),
        (("__iand__.Tensor", "__iand__.Scalar"), "__and__.Tensor"),
        (("__ior__.Tensor", "__ior__.Scalar"), "__or__.Tensor"),
        (("__ixor__.Tensor", "__ixor__.Scalar"), "__xor__.Tensor"),
        (("__ilshift__.Tensor", "__ilshift__.Scalar"), "__lshift__.Tensor"),
        (("__irshift__.Tensor", "__irshift__.Scalar"), "__rshift__.Tensor"),
    ):
        if key not in LOWERINGS:
            continue
        fn = LOWERINGS[key]
        register_inplace(*names, outplace_op=getattr(fn, "__wrapped__", fn))


_register_writing_forms()


def is_integer_type(x: Any) -> bool:
    """Whether a value is a whole number, whatever kind of value it is.

    A lowered value, a number that is only known while the program runs, and a
    number that is already a number are three different things that can all be
    whole numbers, and a question about whole numbers has to be able to ask it
    of any of them.
    """

    if isinstance(x, (TensorBox, IRNode)):
        return is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    elif isinstance(x, sympy.Expr):
        return x.is_integer is True
    else:
        return isinstance(x, int)


def is_boolean_type(x: Any) -> bool:
    """Whether a value is a truth value, whatever kind of value it is."""

    if isinstance(x, (TensorBox, IRNode)):
        return is_boolean_dtype(x.get_dtype())
    else:
        return isinstance(x, bool)


def floordiv(a: Any, b: Any) -> Any:
    """The quotient rounded towards minus infinity."""

    return ops.floordiv(a, b)


def truncdiv(a: Any, b: Any) -> Any:
    """The quotient with its fractional part dropped."""

    return ops.truncdiv(a, b)


def _div_rn(a: Any, b: Any) -> Any:
    """The quotient rounded to nearest, which is not what dividing usually does.

    A device usually divides by multiplying by a reciprocal it worked out in
    advance, which is one multiply away from the quotient.  Where the quotient
    is about to be rounded again -- a floor of it, say -- that one bit is the
    difference between the right answer and the one below it, so this asks for
    the quotient itself.
    """

    return ops.div_rn(a, b)


def _div_mode_body(a: Any, b: Any, rounding_mode: Any, both_integer: bool) -> Any:
    """The quotient, rounded as asked, for one pair of values.

    This is what a device computes rather than what the graph is built from:
    whether rounding down drops the fractional part or is a floor is a question
    about the two values, and it cannot be answered until the graph has been
    turned into a loop that produces them.
    """

    if rounding_mode == "floor":
        if both_integer:
            return ops.floordiv(a, b)
        # A floor of the reciprocal-rounded quotient can come out one too
        # small, so the quotient is rounded to nearest first -- which is then
        # the floor the exact quotient would have given, wherever the two round
        # the same way.
        return ops.floor(ops.div_rn(a, b))
    if rounding_mode == "trunc":
        if both_integer:
            return ops.truncdiv(a, b)
        return ops.trunc(ops.truediv(a, b))
    return ops.truediv(a, b)


def div_mode(a: Any, b: Any, rounding_mode: Any = None) -> Any:
    """A quotient, rounded the way the caller said to round it.

    Whole numbers and values that are not whole numbers are rounded differently
    and for different reasons, and neither can be done by rounding a quotient
    the usual way: a floor of an approximate quotient can come out one too small,
    and a device's own division of whole numbers is not the floor of anything.

    So which of the two comes out is decided here, from the dtypes, and the
    arithmetic itself is left to the loop that will produce the values.
    """

    both_integer = is_integer_type(a) and is_integer_type(b)
    both_boolean = is_boolean_type(a) and is_boolean_type(b)

    if rounding_mode in ("floor", "trunc") and both_boolean:
        raise AssertionError(
            f"{rounding_mode}div operands can not be boolean at the same time"
        )
    # ``pointwise`` loads every argument after the first as though it were a
    # value to read at each index, so the two answers settled above are closed
    # over rather than passed: they are the same for every element, and passing
    # them would ask for a value at each index instead.
    def body(x: Any, y: Any) -> Any:
        return _div_mode_body(x, y, rounding_mode, both_integer)

    return pointwise(body, a, b)


def _register_div_writing_forms() -> None:
    """The dividing operations that write into their first argument.

    Rounding is not an operation of its own here: it is a mode of dividing, and
    whether it comes out as a floor or a truncation depends on whether the
    operands are whole numbers.  That question cannot be asked of an operation
    before the operation has been chosen, which is why it is settled by a
    lowering of its own rather than while registering one.
    """

    fn = LOWERINGS["div.Tensor"]
    bare = getattr(fn, "__wrapped__", fn)
    register_lowering("div.Tensor_mode", type_promotion_kind=None)(div_mode)

    register_inplace("div_.Tensor", "div_.Scalar", outplace_op=bare)
    for name in ("div_.Tensor_mode", "div_.Scalar_mode"):
        register(name)(
            lambda a, b, rounding_mode=None: mutate_to(a, div_mode(a, b, rounding_mode))
        )


_register_div_writing_forms()


def to_dtype(
    x: Any,
    dtype: Any,
    copy: bool = False,
    use_compute_types: bool = True,
) -> Any:
    """The same values, read as another element type.

    Whether the values are moved is a separate question from how they are read,
    and asking for the same type again is the one case where they are the same
    value rather than two: a conversion that produces no change is only a copy
    if a copy was asked for.  Low-precision types are computed in a wider one
    and converted back, so that what a consumer reads is a value it can do
    arithmetic on without narrowing first.
    """

    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return clone(x) if copy else x

    loader = x.make_loader()
    size, device = x.get_size(), x.get_device()

    def _to_dtype(index: Any) -> Any:
        result = ops.to_dtype(
            loader(index),
            dtype,
            src_dtype=src_dtype,
            use_compute_types=use_compute_types,
        )
        if not use_compute_types and dtype in (tp.bfloat16, tp.float16):
            result = ops.to_dtype(result, dtype)
        return result

    # Described from the value being converted rather than from whatever node
    # is current: this runs while some other node is being lowered, and that
    # node's shape and type are not this value's.
    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=_to_dtype,
        ranges=size,
    )


#: An axis that cannot be walked by the loop this compiler generates is walked
#: by the framework instead.  Both are the same computation; which one runs is
#: decided per call, because whether an axis can be walked is a fact about its
#: size that is not known until the program runs.
fallback_cumsum = fallback_handler(tp_ops.cumsum.default)
fallback_cumprod = fallback_handler(tp_ops.cumprod.default)
fallback_logcumsumexp = fallback_handler(tp_ops.logcumsumexp.default)
fallback_cummax = fallback_handler(tp_ops.cummax.default)
fallback_cummin = fallback_handler(tp_ops.cummin.default)


def _validate_dim(x: Any, dim: Any, offset: int = 0) -> Any:
    """A dimension named the way it will be counted from the front.

    A dimension can be named from the back -- the last one is -1 -- and which
    one that is cannot be answered until the shape is known, so a negative name
    is turned into a positive one here rather than being carried as a negative
    number into the code that walks the shape.
    """

    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    if not (0 <= dim < ndim + offset):
        raise AssertionError(f"expected: 0 <= dim < ndim + offset, got {dim}")
    return dim


def _make_scan_inner(x: Any, *, axis: Any, dtype: Any) -> dict:
    """The description every cumulative operation shares.

    Walking an axis and carrying a value along it is the same walk whichever
    value is carried, so what describes the walk is written once here: which
    device, what type the carried value has, what to read at each step, how big
    the whole thing is, and which axis to walk.  Only the way two running
    values become one differs between them, and that is left out because it is
    the part that differs.
    """

    if dtype is not None:
        x = to_dtype(x, dtype)
    axis = _validate_dim(x, axis)

    return dict(
        device=x.get_device(),
        dtypes=(x.get_dtype(),),
        inner_fns=(x.make_loader(),),
        size=x.get_size(),
        axis=axis,
    )


@register_lowering(tp_ops.cumsum)
def cumsum(x: Any, dim: Any = 0, dtype: Any = None) -> Any:
    """Running totals along one axis.

    A total of whole numbers is not a whole number until it is asked to be, and
    counting through a list of them is not a sum of them either, so a running
    total of whole numbers widens unless the caller said otherwise.
    """

    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and (dtype is None or dtype == tp.undefined):
        dtype = tp.int64

    if len(x.get_size()) == 0:
        if dim not in [0, -1]:
            raise AssertionError("expected: axis in [0, -1]")
        dtype = _resolve_dtype(dtype, x.get_dtype())
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple: Any, b_tuple: Any) -> Any:
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.add(a, b),)

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumsum(x, dim=dim, dtype=dtype)
    return result


@register_lowering(tp_ops.cumprod)
def cumprod(x: Any, dim: Any = 0, dtype: Any = None) -> Any:
    """Running products along one axis.

    Widens for whole numbers for the same reason a running total does: a product
    of whole numbers is not a whole number until it is asked to be.
    """

    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and (dtype is None or dtype == tp.undefined):
        dtype = tp.int64

    if len(x.get_size()) == 0:
        if dim not in [0, -1]:
            raise AssertionError("expected: axis in [0, -1]")
        dtype = _resolve_dtype(dtype, x.get_dtype())
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple: Any, b_tuple: Any) -> Any:
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.mul(a, b),)

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumprod(x, dim=dim, dtype=dtype)
    return result


@register_lowering(tp_ops.logcumsumexp)
def logcumsumexp(x: Any, dim: Any, dtype: Any = None) -> Any:
    """Running totals of exponents, kept in their logarithm.

    Adding two exponents overflows once either is large, so what is added is
    their logarithms: the larger value, and the smaller one folded into it
    after being scaled by the difference between them.  Where the two are equal
    the smaller is the larger, and adding a number to itself is what the caller
    asked for -- so the folding is skipped there, which is also what keeps an
    infinite value from being turned into a number.
    """

    def log_add_exp_helper(a_tuple: Any, b_tuple: Any) -> Any:
        (a,) = a_tuple
        (b,) = b_tuple
        min_v = ops.minimum(a, b)
        max_v = ops.maximum(a, b)
        mask = ops.logical_or(ops.ne(min_v, max_v), ops.logical_not(ops.isinf(min_v)))
        return (ops.where(mask, ops.add(ops.log1p(ops.exp(ops.sub(min_v, max_v))), max_v), a),)

    dtype = x.get_dtype()
    if len(x.get_size()) == 0:
        if dim not in [0, -1]:
            raise AssertionError("expected: dim in [0, -1]")
        return clone(x)

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=log_add_exp_helper)
    if result is None:
        return fallback_logcumsumexp(x, dim=dim)
    return result


def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    """A dimension named relative to the back, as one counted from the front.

    The last dimension is -1 and the one before it -2, which is a naming that
    needs the rank to resolve -- and a rank of zero has no last dimension, so a
    single value is given one dimension to be the last of.  Every way of asking
    for a dimension resolves to the same positive number here, so what walks a
    shape never has to know which convention it was called with.
    """

    if rank < 0:
        raise IndexError(f"Rank cannot be negative but got {rank}")

    if rank == 0:
        if not wrap_scalar:
            raise IndexError(
                f"Dimension specified as {idx} but tensor has no dimensions"
            )
        rank = 1

    if idx >= 0 and idx < rank:
        return idx

    _idx = idx + rank if idx < 0 else idx

    if _idx < 0 or _idx >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-rank}, {rank - 1}], but got {idx})"
        )

    return _idx


def iota(
    length: Any,
    *,
    start: Any,
    step: Any,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> Any:
    """The numbers from ``start`` by ``step``, as many as there are positions.

    A walk that needs to know which position it is at asks for this rather than
    being handed the numbers, because the numbers are a function of the
    position: the same position is the same number in every walk, so producing
    them is the same operation however many walks ask for it.
    """

    def fn(index: Any) -> Any:
        return ops.index_expr(step * index[0] + start, dtype=dtype)

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=fn,
        ranges=[length],
    )


def _full(fill_value: Any, device: Any, dtype: Any, size: Any) -> Any:
    """A tensor of one value, from a number or from a value that is not known yet.

    What to fill with is one of three things -- a number, an expression of the
    program, or a value of no dimensions -- and which one it is decides how the
    fill is written, so it is asked here rather than assumed by the caller.
    """

    value = fill_value
    if not isinstance(fill_value, (int, float)) and hasattr(value, "value"):
        value = value.value

    if isinstance(value, (int, float)):

        def inner_fn(index: Any) -> Any:
            return ops.constant(value, dtype)

    elif isinstance(value, sympy.Basic):

        def inner_fn(index: Any) -> Any:
            return ops.index_expr(value, dtype)

    else:
        if len(value.get_size()) != 0:
            raise AssertionError("expected: len(value.get_size()) == 0")
        value_loader = value.make_loader()

        def inner_fn(index: Any) -> Any:
            return value_loader([])

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=size,
    )


def view(x: Any, sizes: Any) -> Any:
    """The same values read as a different shape.

    A shape with the same elements in it is not a copy: what changes is how a
    position is turned into an offset, and the values do not move.  A value that
    is already a view of something keeps that view -- taking the underlying
    buffer of a view of a view would throw away the view in between, and the
    shape asked for here would then describe the wrong bytes.
    """

    data = x.data if isinstance(x, TensorBox) else x
    return TensorBox(View.create(data, sizes))


#: An axis too long to be sorted by the network this compiler generates is
#: sorted by the framework instead.  Both are the same computation; which one
#: runs is decided per call, because whether an axis can be sorted is a fact
#: about its length that is not known until the program runs.
sort_fallback = fallback_handler(tp_ops.sort.stable, add_to_fallback_set=False)


@register_lowering(tp_ops.sort.stable, type_promotion_kind=None)
def sort_stable(x: Any, *, stable: Any = None, dim: Any = -1, descending: Any = False) -> Any:
    """The values of an axis in order, and which of them each one was.

    The values and the positions are two halves of one answer and are produced
    by one walk, so they are built together: the positions start as the numbers
    of the axis and are carried alongside the values, which is what makes the
    second half free.

    Where the positions are kept decides how long an axis can be.  A network
    that holds every position at once spends a register on each, so the narrower
    type is worth its smaller range: an axis that fits in the narrow type is
    sorted here, and one that does not is left to the framework rather than
    risking a number that no longer fits.
    """

    if stable is None:
        stable = False

    shape = x.get_size()
    device = x.get_device()
    dim = canonicalize_dim(len(shape), dim)
    if len(shape) == 0:
        return clone(x), _full(0, device, tp.int64, shape)

    dim_size = shape[dim] if len(shape) else 1
    if config.triton.decompose_sort_ops:
        idx_dtype = tp.int32
    else:
        idx_dtype = tp.int16
    if not V.graph.sizevars.guard_or_false(
        sympy.Lt(dim_size, tp.iinfo(idx_dtype).max)
    ):
        return sort_fallback(x, stable=stable, dim=dim, descending=descending)

    indices = iota(
        dim_size, start=0, step=1, dtype=idx_dtype, device=device, requires_grad=False
    )
    view_shape = [1] * len(shape)
    if len(shape):
        view_shape[dim] = dim_size
    indices = view(indices, view_shape)
    indices = lower_expand(indices, shape)

    values, indices = ir.Sort.create(
        device=device,
        dtypes=(x.dtype, indices.dtype),
        inner_fns=(x.make_loader(), indices.make_loader()),
        size=shape,
        axis=dim,
        stable=stable,
        descending=descending,
    )
    if values is None:
        return sort_fallback(x, stable=stable, dim=dim, descending=descending)

    if indices is None:
        raise AssertionError("expected: indices is not None")
    return values, to_dtype(indices, tp.int64)


@register_lowering(tp_ops.sort.default, type_promotion_kind=None)
def sort(x: Any, dim: Any = -1, descending: Any = False) -> Any:
    return sort_stable(x, stable=False, dim=dim, descending=descending)


select_fallback = fallback_handler(tp_ops.select.int, add_to_fallback_set=False)


def unsqueeze(x: Any, dim: Any) -> Any:
    """The same values with a dimension of one added.

    A dimension of one holds the single value that was there before, so adding
    it moves nothing; what changes is which position that value is read at,
    because a value of more dimensions is lined up by its innermost axes.
    """

    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.S.One)
    return view(x, new_shape)


def select(x: Any, dim: Any, idx: Any) -> Any:
    """One position along one axis, with that axis gone.

    Taking a position out is a slice of one followed by a shape that no longer
    has room for it, and both halves matter: the slice says which value, the
    shape says that the axis is not there any more.  An index named from the
    back is the same position, and which one it is cannot be answered until the
    axis length is known, so a negative index is resolved here rather than being
    carried into the slice.

    A position whose value is not known until the program runs cannot be turned
    into either half here -- the shape would be a guess -- so it is left to the
    framework rather than being written down as though it were known.
    """

    idx = sympy.expand(idx)
    size = sympy.expand(x.get_size()[dim])
    actual_index = None

    if V.graph.sizevars.guard_or_false(sympy.Lt(idx, 0)):
        actual_index = idx + size
    elif V.graph.sizevars.guard_or_false(sympy.Ge(idx, 0)):
        actual_index = idx

    if actual_index is not None:
        if has_free_unbacked_symbols(idx):
            # A shape written down before the program runs would be a guess,
            # and a guess about which position is read is a guess about the
            # values themselves.  So the position is resolved where it is read.
            return fallback_select(x, dim, idx)

        slice_result = _slice(x, dim, actual_index, actual_index + 1, 1)
        return lower_squeeze(slice_result, dim)

    return fallback_select(x, dim, idx)


#: These are one sort and a read from it, and are written as such when the sort
#: is written.  When it is not, they are the same computation done by the
#: framework -- which is not a second way of doing it but the first way, reached
#: differently, and the choice is made per call because whether the sort will be
#: written is not known until the shapes are.
topk_fallback = fallback_handler(tp_ops.topk.default, add_to_fallback_set=False)
kthvalue_fallback = fallback_handler(tp_ops.kthvalue.default, add_to_fallback_set=False)
median_fallback = fallback_handler(tp_ops.median.default, add_to_fallback_set=False)
median_dim_fallback = fallback_handler(tp_ops.median.dim, add_to_fallback_set=False)
mode_fallback = fallback_handler(tp_ops.mode.default, add_to_fallback_set=False)


@register_lowering(tp_ops.median.default, type_promotion_kind=None)
def median_default(self: Any) -> Any:
    """The middle value of all of them, with no axis to speak of.

    A value with no axis is flattened first, because "the middle" is only a
    question once everything is in one line, and sorting a flat list is the same
    walk as sorting any other.
    """

    if not config.triton.decompose_sort_ops:
        return median_fallback(self)
    size = self.get_size()
    numel = functools.reduce(operator.mul, size, sympy.Integer(1))
    flat = view(self, [numel])
    sorted_vals, _ = sort_stable(flat, dim=0)
    k = (numel - 1) // 2
    return select(sorted_vals, 0, k)


@register_lowering(tp_ops.median.dim, type_promotion_kind=None)
def median_dim(self: Any, dim: Any, keepdim: Any = False) -> Any:
    """The middle value along one axis, and which position it was at.

    Two answers rather than one because the position is not recoverable from
    the value: two equal values are the same value and are not the same
    position, and which one a caller means is the question being asked.

    Even length leaves no single middle, so the smaller of the two is taken --
    the one at ``(n-1)//2`` -- which is the same one a sort-based decomposition
    gives however the ties fell.
    """

    if not config.triton.decompose_sort_ops:
        return median_dim_fallback(self, dim, keepdim)
    shape = self.get_size()
    ndim = len(shape)
    if ndim == 0:
        return clone(self), _full(0, self.get_device(), tp.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(self, stable=True, dim=dim)
    n = shape[dim]
    k = (n - 1) // 2
    values = select(sorted_vals, dim, k)
    indices = select(sorted_idxs, dim, k)
    if keepdim:
        values = unsqueeze(values, dim)
        indices = unsqueeze(indices, dim)
    return values, indices


@register_lowering(tp_ops.topk.default, type_promotion_kind=None)
def topk(
    self: Any,
    k: Any,
    dim: Any = -1,
    largest: Any = True,
    sorted: Any = True,
    impl: Any = 0,
) -> Any:
    """The k largest or smallest values along one axis, and where they were.

    Which end counts as large is part of what was asked rather than something
    applied to the result, so it goes to the sort: a descending sort and a
    prefix of it is the whole of this, and the positions come along because the
    sort produces them anyway.

    The order the caller asked to be sorted is the sort's own stability: values
    that compare equal keep the order they were in, which is what makes the
    answer the same however the ties fell.
    """

    # Which implementation to use is a choice between ways of doing this, and
    # this is the one way it is done here -- so the choice is not a parameter.
    del impl

    if not config.triton.decompose_sort_ops:
        return topk_fallback(self, k, dim, largest, sorted)
    shape = self.get_size()
    ndim = len(shape)
    if ndim == 0:
        return clone(self), _full(0, self.get_device(), tp.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(
        self, stable=True, dim=dim, descending=largest
    )
    values = _slice(sorted_vals, dim, 0, k, 1)
    indices = _slice(sorted_idxs, dim, 0, k, 1)
    return values, indices


@register_lowering(tp_ops.kthvalue.default, type_promotion_kind=None)
def kthvalue(self: Any, k: Any, dim: Any = -1, keepdim: Any = False) -> Any:
    """The kth smallest value along one axis, and where it was.

    Only the values and their positions are wanted, not the order they came out
    in, so this reads one position out of a sort rather than keeping a prefix
    of it.  Which position is the kth is counted from one, as it is named here,
    and one is subtracted because a position in a sorted list is not.
    """

    if not config.triton.decompose_sort_ops:
        return kthvalue_fallback(self, k, dim, keepdim)
    shape = self.get_size()
    ndim = len(shape)
    if ndim == 0:
        return clone(self), _full(0, self.get_device(), tp.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(self, stable=True, dim=dim)
    values = select(sorted_vals, dim, k - 1)
    indices = select(sorted_idxs, dim, k - 1)
    if keepdim:
        values = unsqueeze(values, dim)
        indices = unsqueeze(indices, dim)
    return values, indices


def new_empty(x: Any, size: Any, *, dtype: Any = None, device: Any = None) -> Any:
    """A value of a shape and type, whose contents are not yet anything.

    What comes back is a place to write rather than a value: nothing has been
    put in it, so asking what is in it has no answer.  Its type and where it
    lives come from what it will hold, and its type is the type asked for if one
    was -- otherwise the type of the value it stands in for, since that is what
    it is standing in for.
    """

    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    # Written as a walk of zeros rather than as storage nothing has been put
    # in: a value whose contents are unset has no defined contents to read, and
    # a walk of zeros does -- at the cost of writing them, which for a shape
    # with no positions costs nothing at all.
    return _full(0, device, dtype, list(size))


def empty_strided(
    size: Any,
    stride: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
) -> Any:
    """A place to write, at a shape and at distances between positions.

    A shape on its own does not say where an element sits: that is the strides'
    job, and for most values the compiler is free to pick them, which is what
    ``new_empty`` lets it do by leaving them out.  Some values are not free to
    pick them.  A kernel handed two buffers has to agree with the caller about
    how the second one is laid out before either of them reads it, and a value
    that is also written by a kernel that is not the one computing it cannot be
    re-laid out afterwards without the two disagreeing.  So the distances are
    part of what is asked for here, and given, not chosen.

    What the shape is for is the caller to say; what is in the buffer is not the
    caller's business, which is why this writes a walk of zeros like
    ``new_empty`` rather than leaving it unset.  That walk is also what gives
    the buffer a name and a place in the graph, which is what a buffer a kernel
    fills needs to have before the kernel is written.

    Asked for with no type and no device, the two are the ones a program gets by
    not saying: its own default type, and the device this runs on.  A layout
    other than the ordinary one is not something this can honour, and is said
    out loud rather than quietly treated as the ordinary one.
    """

    if not isinstance(size, (list, tuple)):
        raise AssertionError("expected: isinstance(size, (list, tuple))")
    if not isinstance(stride, (list, tuple, type(None))):
        raise AssertionError("expected: isinstance(stride, (list, tuple, None))")
    if layout is not None:
        raise NotImplementedError(f"layout={layout}")
    if dtype is None:
        dtype = tp.get_default_dtype()
    if device is None:
        device = tp.device("cuda" if tp.cuda.is_available() else "cpu")

    pointwise = _full(0, device, dtype, list(size))
    pointwise.realize()
    buffer = pointwise.data.data
    if not isinstance(buffer, ir.ComputedBuffer):
        raise AssertionError("expected: isinstance(buffer, ir.ComputedBuffer)")
    # Every position is a zero-length range, so the walk that wrote the zeros
    # has nothing left to do by the time a kernel writes over it.  Saying so
    # here is what keeps a buffer nobody reads from being written twice.
    buffer.data = dataclasses.replace(buffer.data, ranges=[0] * len(size))
    size = [sympy.expand(s) for s in size]
    stride = (
        [sympy.expand(s) for s in stride]
        if stride
        else ir.FlexibleLayout.contiguous_strides(size)
    )
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=size,
        stride=stride,
        is_pinned=pin_memory or False,
    )
    return pointwise


register("empty_strided.default")(empty_strided)


@register("new_empty_strided.default")
def new_empty_strided(
    x: Any, size: Any, stride: Any, *, dtype: Any = None, layout: Any = None,
    device: Any = None, pin_memory: Any = None,
) -> Any:
    """A place to write, asked of another value.

    What is not said about the type and where the value lives is taken from
    the value asked of, since that is the one thing the ask names; the rest is
    the same place ``empty_strided`` makes.
    """

    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    return empty_strided(
        size, stride, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def _new_like(
    x: Any,
    size: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
) -> Any:
    """A place to write, at a shape the program writes down.

    The value this was called with is asked for its type and its device rather
    than for its shape: what to write into is at a shape the program states,
    and what the call gives is where and of what that shape is.  The distances
    are worked out from the extents, because nothing here says the result is
    laid out like anything.
    """

    device = device or x.get_device()
    dtype = dtype or x.get_dtype()
    return empty_strided(
        size,
        None,
        dtype=dtype,
        device=device,
    )


def _new_filled(
    x: Any,
    size: Any,
    fill: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
) -> Any:
    """A tensor of one value, shaped by another value.

    The shape is written down in the program rather than read off the value the
    call was given, and the value fills it: the shape the program states and the
    value the call names are two separate things, and either of them may be the
    one that differs from what the value would have had.
    """

    device = device or x.get_device()
    dtype = dtype or x.get_dtype()
    return _full(fill, device, dtype, size)


def _new_zeros(x: Any, size: Any, **kwargs: Any) -> Any:
    """A tensor of zeros, shaped by another value."""

    return _new_filled(x, size, 0, **kwargs)


def _new_ones(x: Any, size: Any, **kwargs: Any) -> Any:
    """A tensor of ones, shaped by another value."""

    return _new_filled(x, size, 1, **kwargs)


register("new_empty.default")(_new_like)
register("new_zeros.default")(_new_zeros)
register("new_ones.default")(_new_ones)


def _filled_like(fill: Any):
    """A tensor of one value at another value's shape: a constant every
    reader computes in its own loop, never a buffer of its own."""

    def lower(x: Any, *args: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Any:
        value = args[0] if fill is None else fill
        return _full(
            value,
            decode_device(device) if device is not None else x.get_device(),
            dtype if isinstance(dtype, tp.dtype) and dtype != tp.undefined else x.get_dtype(),
            list(x.get_size()),
        )

    return lower


@register("eye", "eye.default", "eye.m")
def lower_eye(n, m=None, *, dtype=None, layout=None, device=None, pin_memory=None, requires_grad=False):
    """Ones where the row is the column, zeros elsewhere: computed by every
    reader from its position, never written down."""

    # The contract spells "as many columns as rows" as a negative count.
    m = n if m is None or (isinstance(m, int) and m < 0) else m
    if not isinstance(dtype, tp.dtype) or dtype == tp.undefined:
        dtype = tp.get_default_dtype()

    def fn(index):
        on_diagonal = ops.eq(ops.index_expr(index[0], tp.int64), ops.index_expr(index[1], tp.int64))
        return ops.to_dtype(on_diagonal, dtype)

    return Pointwise.create(
        device=decode_device(device if device is not None else "cpu"),
        dtype=dtype,
        inner_fn=fn,
        ranges=[n, m],
    )


register("zeros_like.default", "zeros_like")(_filled_like(0))
register("ones_like.default", "ones_like")(_filled_like(1))
register("full_like.default", "full_like")(_filled_like(None))


def lower_full(size: Any, fill_value: Any, **kwargs: Any) -> Any:
    """A tensor of one value, at a shape the program writes down.

    An extent may be a symbol the kernel only gives a number to while it runs,
    so the shape is kept as the expressions it was written in.
    """

    dtype = kwargs.get("dtype")
    # The dispatch graph spells an unsaid type as the undefined one, which is
    # a value in its own right and not the absence of one.
    if not isinstance(dtype, tp.dtype) or dtype == tp.undefined:
        dtype = tp.get_default_dtype()
    device = decode_device(kwargs.get("device"))
    size = [sympy.expand(s) if isinstance(s, sympy.Expr) else s for s in size]
    return _full(fill_value, device, dtype, size)


def lower_zeros(size: Any, **kwargs: Any) -> Any:
    """A tensor of zeros, at a shape the program writes down."""

    return lower_full(size, 0, **kwargs)


def lower_ones(size: Any, **kwargs: Any) -> Any:
    """A tensor of ones, at a shape the program writes down."""

    return lower_full(size, 1, **kwargs)


# Also under the bare name, which is how a program that calls the function
# itself rather than the operator reaches a region -- the mask a processor
# kernel applies is one.
register("full.default", "full")(lower_full)
register("zeros.default")(lower_zeros)
register("ones.default")(lower_ones)



def gather(x: Any, dim: Any, index: Any, sparse_grad: Any = False) -> Any:
    """One value per position asked for, each read at an index of its own.

    The result is shaped like what was asked for rather than like what was read:
    the two are the same here, and saying so is what makes each output position
    know which input position it corresponds to.  An index is clamped to the
    axis rather than trusted, because an index past the end has no value to
    read and returning the last one is a defined answer where reading nothing
    is not.

    Whether a gradient for this is sparse is not decided here: it is a fact
    about the backward pass, and the forward pass is the same computation either
    way.
    """

    # Whether a gradient for this is sparse changes only the backward pass, and
    # the backward pass is not this.
    del sparse_grad

    # A view reaches here bare as often as boxed; it is read the same way.
    x = _as_box(x)
    if index.get_numel() == 0:
        return new_empty(x, index.get_size())

    size = x.get_size()
    offset = len(size) == 0
    dim = _validate_dim(x, dim, offset)

    if offset:
        x = lower_expand(x, [1])
        size = [1]

    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx: Any) -> Any:
        idx = list(idx)
        gather_idx = ops.indirect_indexing(
            index_loader(idx), size[dim], wrap_neg=False
        )
        if len(idx) == 0:
            idx = [gather_idx]
        else:
            idx[dim] = gather_idx
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
    )


@register_lowering(tp_ops.cummax, type_promotion_kind=None)
def cummax(x: Any, dim: Any = 0) -> Any:
    """The largest value so far along one axis, and where it was.

    Carrying the position along with the value is what makes the second answer
    possible: the value alone cannot say where it was, because two equal values
    are the same value and are not the same position.

    Which of two equal values is kept is not decided here but by how two running
    values are compared -- and it is not the same choice as the one a reduction
    over the whole axis makes, so it is asked of that comparison rather than
    assumed.  Taking the later of two equals is what makes this the running
    maximum rather than the first one reached.
    """

    if len(x.get_size()) == 0:
        if dim not in [0, -1]:
            raise AssertionError("expected: dim in [0, -1]")
        return clone(x), _full(0, x.get_device(), tp.int64, x.get_size())

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmax", dtype=dtype, arg_break_ties_left=False
    )

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    kwargs["dtypes"] = (dtype, tp.int64)
    kwargs["inner_fns"] = (
        x.make_loader(),
        lambda idx: ops.index_expr(idx[dim], tp.int64),
    )
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if values is None:
        return fallback_cummax(x, dim=dim)
    return values, indices


@register_lowering(tp_ops.cummin, type_promotion_kind=None)
def cummin(x: Any, dim: Any = 0) -> Any:
    """The smallest value so far along one axis, and where it was.

    The mirror of the running maximum, and the same in every respect but which
    end of the comparison is taken -- which is the whole difference, and is
    asked of the comparison rather than written out again here.
    """

    if len(x.get_size()) == 0:
        if dim not in [0, -1]:
            raise AssertionError("expected: dim in [0, -1]")
        return clone(x), _full(0, x.get_device(), tp.int64, x.get_size())

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmin", dtype=dtype, arg_break_ties_left=False
    )

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    kwargs["dtypes"] = (dtype, tp.int64)
    kwargs["inner_fns"] = (
        x.make_loader(),
        lambda idx: ops.index_expr(idx[dim], tp.int64),
    )
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if values is None:
        return fallback_cummin(x, dim=dim)
    return values, indices


def _compute_slice_index(index: Any, size: Any, default: Any = None) -> Any:
    """Where a slice boundary lands, counted the way a slice counts it.

    A boundary can be named from the back, clamped to the axis, or left out
    entirely, and which of those it is may not be known until the program runs.
    So the cases are asked in the order they can be settled, and each one that
    can be settled says what the boundary is: inside the axis as written, wrapped
    around from the back, clamped to either end, or -- when only the sign is
    known -- the boundary clamped and wrapped.  The two clamping cases are
    separated because clamping a name from the back and clamping a name from the
    front are different operations on the same number.

    Returns nothing when the boundary is a number whose value is not known
    until the program runs: the index it stands for cannot be written down
    before the program does, and a guess would be a guess about which elements
    are read.
    """

    if index is None:
        return default

    guard = V.graph.sizevars.guard_or_false
    index = sympy.expand(index)
    size = sympy.expand(size)
    if guard(sympy.And(sympy.Ge(index, 0), sympy.Le(index, size))):
        return index
    elif guard(sympy.And(sympy.Lt(index, 0), sympy.Ge(index, -size))):
        return index + size
    elif guard(sympy.Gt(index, size)):
        return size
    elif guard(sympy.Lt(index, -size)):
        return 0
    elif guard(sympy.Ge(index, 0)):
        return Min(index, size)
    elif guard(sympy.Lt(index, 0)):
        return Max(index + size, 0)
    return None


def _clamp_slice_end_to_start(end: Any, start: Any) -> Any:
    """A slice's end, never before its start.

    An end before the start describes no positions at all, which is a slice of
    nothing rather than an error -- so the end is raised to the start, and where
    the two are not known to be in either order the smaller of them is taken,
    which is the only answer that holds either way.
    """

    if V.graph.sizevars.statically_known_geq(end, start):
        return end
    if V.graph.sizevars.statically_known_leq(end, start):
        return start
    return Max(end, start)


@register_lowering(tp_ops.select_scatter, type_promotion_kind=None)
def select_scatter(x: Any, src: Any, dim: Any, index: Any) -> Any:
    """One position of an axis replaced, the rest kept.

    Every position of the axis asks whether it is the one being replaced, and
    answers with the replacement or with what was there -- so the whole value is
    written, and what makes it a scatter is the one position that reads
    elsewhere.  The replacement is lined up against the whole shape first, since
    it is asked at every position and a value of fewer dimensions would be lined
    up by its innermost axes.

    A position whose value is not known until the program runs is left to the
    framework: which position is replaced would be a guess, and a guess here is
    a guess about which value is written.
    """

    src = to_dtype(src, x.get_dtype())
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    if V.graph.sizevars.guard_or_false(sympy.Lt(index, 0)):
        index = index + x.get_size()[dim]
    elif V.graph.sizevars.guard_or_false(sympy.Ge(index, 0)):
        pass
    else:
        return fallback_handler(tp_ops.select_scatter.default)(x, src, dim, index)

    V.graph.sizevars.check_leq(0, index)
    V.graph.sizevars.check_lt(index, x.get_size()[dim])
    src = lower_expand(unsqueeze(src, dim), x.get_size())
    src_loader = src.make_loader()

    def inner_fn(idx: Any) -> Any:
        return ops.where(
            ops.eq(
                ops.index_expr(idx[dim], tp.int32),
                ops.index_expr(index, tp.int32),
            ),
            src_loader(idx),
            x_loader(idx),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


@register_lowering(tp_ops.slice_scatter, type_promotion_kind=None)
def slice_scatter(
    x: Any, src: Any, dim: Any = 0, start: Any = None, end: Any = None, step: Any = 1
) -> Any:
    """A run of positions along an axis replaced, the rest kept.

    Which positions are replaced is a run described by its two boundaries, and
    the boundaries can be named from the back, clamped, or left out -- so they
    are resolved to positions first, and how many positions that is decides how
    far along the axis the replacement is read.  Every position then asks whether
    it falls in the run, and answers with the replacement or with what was there.

    The two boundaries are checked against each other rather than trusted: a run
    whose end is before its start is a run of no positions, not a run running
    backwards.
    """

    src = to_dtype(src, x.get_dtype())
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]

    if any(has_free_unbacked_symbols(v) for v in (start, end, dim_size)):
        start_index = _compute_slice_index(start, dim_size, 0)
        if end is not None and V.graph.sizevars.statically_known_equals(
            end, sys.maxsize
        ):
            end_index = dim_size
        else:
            end_index = _compute_slice_index(end, dim_size, dim_size)

        if start_index is None or end_index is None:
            return fallback_handler(tp_ops.slice_scatter.default)(
                x, src, dim, start, end, step
            )

        start = start_index
        end = _clamp_slice_end_to_start(end_index, start)
    else:
        start, end = ir.SliceView.normalize_start_end(x, dim, start, end)

    src_size = list(x.get_size())
    src_size[dim] = FloorDiv(end - start + (step - 1), step)
    if len(src.get_size()) != len(src_size):
        raise AssertionError("expected src and slice to have the same rank")
    for actual, expected in zip(src.get_size(), src_size):
        V.graph.sizevars.check_equals(actual, expected)
    src = lower_expand(src, src_size)
    src_loader = src.make_loader()

    def inner_fn(idx: Any) -> Any:
        if start == 0 and end == dim_size and step == 1:
            # A run covering the whole axis at every position is the
            # replacement and nothing else, so there is nothing to choose.
            return src_loader(idx)

        idx_dim = ops.index_expr(idx[dim], tp.int64)
        src_idx = list(idx)
        src_idx[dim] = FloorDiv(idx[dim] - start, step)

        mask = []
        if start != 0:
            mask.append(
                ops.ge(
                    idx_dim,
                    ops.index_expr(sympy.expand(start), tp.int64),
                )
            )
        if end != dim_size:
            mask.append(
                ops.lt(
                    idx_dim,
                    ops.index_expr(sympy.expand(end), tp.int64),
                )
            )
        if step != 1:
            mask.append(
                ops.eq(
                    ops.index_expr(
                        ModularIndexing(idx[dim] - start, 1, step), tp.int64
                    ),
                    ops.constant(0, tp.int64),
                )
            )
        if not (mask):
            raise AssertionError("expected: mask")
        mask = functools.reduce(ops.and_, mask)
        src_val = ops.masked(
            mask,
            lambda: src_loader(src_idx),
            0 if is_integer_type(x) else 0.0,
        )
        return ops.where(
            mask,
            src_val,
            x_loader(idx),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


@register("argmax.default")
def reduce_argmax(x: Any, dim: Any = None, keepdim: Any = False) -> Any:
    """Where the largest value along an axis is, counted from the front.

    A position rather than a value, and counted from the front of the axis
    rather than from wherever the value happened to be found -- so two equal
    values give the same answer however the walk that found them was ordered.
    """

    # This is reached while some other node is being lowered, and that
    # node's result is named by which of its results is wanted rather
    # than being the whole of them.
    device = x.get_device()
    if dim is None:
        dims = list(range(len(x.get_size())))
    elif isinstance(dim, (list, tuple)):
        dims = list(dim)
    else:
        dims = [dim]
    return make_reduction(x, dims, keepdim, tp.int64, device, "argmax")


@register("argmin.default")
def reduce_argmin(x: Any, dim: Any = None, keepdim: Any = False) -> Any:
    """Where the smallest value along an axis is, counted from the front.

    The mirror of the largest, and the same in every respect but which end of
    the comparison is taken.
    """

    # This is reached while some other node is being lowered, and that
    # node's result is named by which of its results is wanted rather
    # than being the whole of them.
    device = x.get_device()
    if dim is None:
        dims = list(range(len(x.get_size())))
    elif isinstance(dim, (list, tuple)):
        dims = list(dim)
    else:
        dims = [dim]
    return make_reduction(x, dims, keepdim, tp.int64, device, "argmin")


@register_lowering(tp_ops.mode.default, type_promotion_kind=None)
def mode_default(self: Any, dim: Any = -1, keepdim: Any = False) -> Any:
    """The value that occurs most often along an axis, and where it was.

    A sorted axis groups equal values together, so the most frequent value is
    the longest run of equal values in it -- and the run is found by asking each
    position whether it starts one and then carrying the last such position
    forward, which is what turns "this position starts a run" into "this position
    is in a run" without counting anything.

    Where two runs are the same length, the one that starts first is taken: the
    last starting position carried forward is the earliest of the longest runs,
    and the position with the longest run behind it is that run's end.  Ties
    among equal values are broken by position rather than by which came first in
    memory, so the answer does not depend on the order the values happened to be
    read in.
    """

    if not config.triton.decompose_sort_ops:
        return mode_fallback(self, dim, keepdim)
    shape = self.get_size()
    ndim = len(shape)
    device = self.get_device()
    if ndim == 0:
        return clone(self), _full(0, device, tp.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(self, stable=True, dim=dim)
    n = shape[dim]

    positions = iota(
        n, start=0, step=1, dtype=tp.int64, device=device, requires_grad=False
    )
    pos_view_shape = [sympy.Integer(1)] * ndim
    pos_view_shape[dim] = n
    positions = view(positions, pos_view_shape)
    positions = lower_expand(positions, shape)

    positions_loader0 = positions.make_loader()

    def prev_pos_fn(idx: Any) -> Any:
        return ops.maximum(
            ops.sub(positions_loader0(idx), ops.constant(1, tp.int64)),
            ops.constant(0, tp.int64),
        )

    prev_positions = Pointwise.create(
        device=decode_device(device),
        dtype=tp.int64,
        inner_fn=prev_pos_fn,
        ranges=shape,
    )

    shifted_vals = gather(sorted_vals, dim, prev_positions)

    sorted_loader = sorted_vals.make_loader()
    shifted_loader = shifted_vals.make_loader()
    positions_loader = positions.make_loader()

    def is_boundary_fn(idx: Any) -> Any:
        return ops.or_(
            ops.ne(sorted_loader(idx), shifted_loader(idx)),
            ops.eq(positions_loader(idx), ops.constant(0, tp.int64)),
        )

    is_boundary = Pointwise.create(
        device=decode_device(device),
        dtype=tp.bool,
        inner_fn=is_boundary_fn,
        ranges=shape,
    )

    is_boundary_loader = is_boundary.make_loader()
    positions_loader2 = positions.make_loader()

    def boundary_pos_fn(idx: Any) -> Any:
        return ops.where(
            is_boundary_loader(idx),
            positions_loader2(idx),
            ops.constant(-1, tp.int64),
        )

    boundary_pos = Pointwise.create(
        device=decode_device(device),
        dtype=tp.int64,
        inner_fn=boundary_pos_fn,
        ranges=shape,
    )

    last_boundary, _ = cummax(boundary_pos, dim)

    positions_loader3 = positions.make_loader()
    last_boundary_loader = last_boundary.make_loader()

    def run_len_fn(idx: Any) -> Any:
        return ops.add(
            ops.sub(positions_loader3(idx), last_boundary_loader(idx)),
            ops.constant(1, tp.int64),
        )

    run_len = Pointwise.create(
        device=decode_device(device),
        dtype=tp.int64,
        inner_fn=run_len_fn,
        ranges=shape,
    )

    max_pos = reduce_argmax(run_len, dim, True)
    mode_vals = gather(sorted_vals, dim, max_pos)
    mode_idxs = gather(sorted_idxs, dim, max_pos)

    if not keepdim:
        # Reshaped rather than squeezed: what came back is a walk rather than a
        # view of something, and a view says which bytes it reads -- which is a
        # question about where the values came from rather than about the shape
        # the answer has.
        def _drop_axis(value: Any, axis: Any) -> Any:
            new_shape = [s for d, s in enumerate(value.get_size()) if d != axis]
            return view(value, new_shape)

        mode_vals = _drop_axis(mode_vals, dim)
        mode_idxs = _drop_axis(mode_idxs, dim)

    return mode_vals, mode_idxs


def tensor(
    data: Any,
    *,
    dtype: Any = None,
    device: Any = None,
    layout: Any = None,
    pin_memory: Any = False,
) -> Any:
    """One number, or a short run of them, written into the code itself.

    A value known before the program runs is a constant rather than a
    computation, and the type of a number is int where it came from an integer
    and the program's own default where it came as a real number -- asking
    which of a handful of representations was meant is not the same as asking
    what the program defaults to.

    A short run of numbers is searched rather than indexed, because a search
    over a handful of positions needs no storage and no load: the position asks
    which half it is in, and each half asks the same of the one below it.  That
    is why the length is bounded here -- the walk is as long as the run, and a
    long run is better off being stored.
    """

    if isinstance(data, int) and not isinstance(data, bool):
        dtype = dtype or "int64"
    else:
        dtype = dtype or "float32"

    ranges: list = []

    # An index is the type the kernel indexes by, and the index machinery asks
    # the type of an index as a type -- to know what it can hold and what it can
    # be arithmetic on -- rather than as a name to look up later. So the name a
    # constant is written under is turned into the type here, once, where the
    # name is still in hand, and every index written below is handed the type.
    _index_dtype = getattr(tp, "int64", "int64")

    # Narrower than a float is a type a real number is rounded to on the way in,
    # so the rounding is done here -- where it is a fact about the constant
    # rather than about the code generated from it.
    _truncate_fp = dtype in ("bfloat16", "float16")

    if isinstance(data, sympy.Basic):

        def inner_fn(index: Any) -> Any:
            result = ops.index_expr(data, _index_dtype)
            if _truncate_fp:
                result = ops.to_dtype(result, tp.float32)
                result = ops.to_dtype(result, dtype)
            return result

    elif isinstance(data, (float, int, bool)):
        # A real number that will not fit the type it is written as would be
        # rounded on the way in, so it is rounded here instead -- where the
        # rounding is a fact about the constant rather than about the code
        # generated from it.
        if _truncate_fp and isinstance(data, float):
            data = tp.tensor(data, dtype=dtype).item()

        def inner_fn(index: Any) -> Any:
            return ops.constant(data, dtype)

    elif not len(data) or (
        isinstance(data[0], (float, int)) and len(data) <= 8
    ):
        # A short run of numbers is written into the body by asking which
        # number the position being written is, and halving the run to find
        # out -- so there is no buffer to allocate and no name to keep.  A
        # longer run is a value worth storing, because then it is searched for
        # once rather than laid out again at every use.
        ranges.append(sympy.Integer(len(data)))

        def inner_fn(index: Any) -> Any:
            def binary_search(start: int, end: int) -> Any:
                if start >= end:
                    raise AssertionError("expected: start < end")
                if end - start == 1:
                    return ops.constant(data[start], dtype)
                mid = (end - start) // 2 + start
                return ops.where(
                    ops.lt(
                        ops.index_expr(index[0], _index_dtype),
                        ops.constant(mid, _index_dtype),
                    ),
                    binary_search(start, mid),
                    binary_search(mid, end),
                )

            if len(data) == 0:
                return ops.constant(0, dtype)
            return binary_search(0, len(data))

    else:
        # A run of numbers too long to search is a value that has to be stored,
        # and this has no way to put one in the graph -- so it is refused here
        # rather than silently becoming a walk of the wrong length.
        raise AssertionError(
            f"expected a number or an expression, got {type(data).__name__}"
        )

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=ranges,
    )


def philox_rand_offset(shape: Any) -> Any:
    """How many values reading a shape of this size reads.

    Deliberately not the number the eager generator would have counted: the two
    read the stream differently -- one asks a device for values, the other asks
    a device for values at a position -- so counting them the same way would
    make two graphs that read the same amount land on the same position where
    they do not.
    """

    numel = 1
    for s in shape:
        numel = numel * s
    return tensor(numel, dtype=tp.int64)


@register_lowering(prims.philox_rand, type_promotion_kind=None)
def philox_rand(
    size: Any, seed: Any, offset: Any, stride: Any, device: Any, dtype: Any
) -> Any:
    """Values read at a position, and how many of them there were.

    The position asked for is where the stream was left off plus where in the
    shape this element is, so two elements are two positions rather than the
    same one twice -- which is what makes the values a function of the element
    rather than of how many threads happened to read before it.

    A stride would say the stream is shared across devices, which is a
    distributed concern and not one this walks: a stream with a stride has
    positions on it that are not this position.
    """

    # Which stream is a fact about the caller rather than about how one is read.
    del stride

    random_pos = ir.FixedLayout(
        device,
        dtype,
        size,
        ir.FlexibleLayout.contiguous_strides(size),
    ).make_indexer()
    seed_loader = seed.make_loader()
    offset_loader = offset.make_loader()

    def inner_fn(index: Any) -> Any:
        # The seed and the offset are values rather than numbers, and a device
        # reads a position as a number: so both are read and then read as
        # numbers, which is the same conversion the values come back through.
        seed_index_expr = ops.to_dtype(seed_loader([]), tp.int32)
        offset_index_expr = ops.to_dtype(offset_loader([]), tp.int32)
        rand_index_expr = ops.add(
            ops.index_expr(random_pos(index), tp.int32), offset_index_expr
        )
        result = ops.rand(
            seed_index_expr,
            rand_index_expr,
        )
        return ops.to_dtype(result, dtype)

    random_values_node = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )

    offset_node = philox_rand_offset(size)
    return random_values_node, offset_node


_fallback_randn_default = fallback_handler(tp_ops.randn.default)
_fallback_randn_generator = fallback_handler(tp_ops.randn.generator)


@register("randn.default", "randn.generator")
def randn(*args: Any, **kwargs: Any) -> Any:
    """Normal values handed to the framework rather than drawn here.

    A draw reads the generator the program runs with, so the values a
    compiled call makes are the ones an eager call would: the call goes to
    the framework whole, and the generator's state moves with it.  A draw
    naming a generator of its own reaches that generator's overload the same
    way.
    """

    if kwargs.get("generator") is not None:
        return _fallback_randn_generator(*args, **kwargs)
    return _fallback_randn_default(*args, **kwargs)


def to_dtype_bitcast(x: Any, dtype: Any, *, copy: bool = False) -> Any:
    """The same bits read as another element type, without moving them.

    A different element type over the same number of bits is a different way of
    reading the same bytes -- which is what makes this different from a
    conversion, where a value is turned into a value of another type.  Two types
    of different widths are not that: the number of elements would change, so
    the values themselves have to be produced, and that is the framework's job
    rather than this one's.
    """

    x_dtype = x.get_dtype()
    if x_dtype == dtype:
        return clone(x) if copy else x

    def _get_primitive_bitwidth(dt: Any) -> int:
        if dt in (tp.float16, tp.bfloat16, tp.float32, tp.float64):
            return tp.finfo(dt).bits
        elif dt == tp.bool:
            # A truth value is stored as the smallest whole number, so its
            # width is that one's rather than one bit.
            return 8
        else:
            return tp.iinfo(dt).bits

    src_bits = _get_primitive_bitwidth(x_dtype)
    dst_bits = _get_primitive_bitwidth(dtype)
    if src_bits != dst_bits:
        x_cont = ir.ExternKernel.require_contiguous_strides(x)
        return fallback_handler(tp_ops.view.dtype)(x_cont, dtype)
    return TensorBox.create(ir.DtypeView.create(x, dtype))


@register_lowering(tp_ops.view.dtype, type_promotion_kind=None)
def _view_dtype(x: Any, dtype: Any) -> Any:
    """The same values read as another element type.

    A pair of real numbers read as one complex number is the same bits read a
    different way rather than a different value, and which of the two it is
    depends on whether either side is complex -- a real value has no imaginary
    half to pair with, so reading it as complex is a widening of the values
    rather than a re-reading of them.
    """

    if _is_complex(dtype) or _is_complex(x.get_dtype()):
        return TensorBox.create(
            ir.ComplexView.create(tp_ops.view.dtype, x, dtype)
        )
    return to_dtype_bitcast(x, dtype)


@register("prims::convert_element_type")
def _convert_element_type(x: Any, dtype: Any) -> Any:
    """The same values turned into another element type.

    Different from reading the same bits another way: here the values are
    produced, and a value that does not fit the type it is written as is
    rounded on the way rather than being reinterpreted.
    """

    if _is_complex(dtype) or _is_complex(x.get_dtype()):
        if x.get_size():
            # Decomposed rather than handed over: a fallback here is friendlier
            # to the other side than one that has to widen in place.
            dst = new_empty(x, x.get_size(), dtype=dtype)
            ir.InplaceCopyFallback.create(dst, x)
            return dst
        return new_empty(x, x.get_size(), dtype=dtype)
    return to_dtype(x, dtype)


@register_lowering(tp_ops.round.default)
def round(x: Any) -> Any:
    """The nearest whole number, halves away from zero.

    A whole number is already as near a whole number as it can be, so rounding
    one produces it rather than a new one -- which is why the two are not the
    same operation and rounding one is not the general case.
    """

    if is_integer_type(x):
        return clone(x)
    else:
        fn = ops_wrapper("round")
        return pointwise(fn, x)


#: A greatest common divisor is the framework's rather than this one's: which
#: of two answers is meant differs by sign convention, and a whole loop over
#: the Euclidean steps is a lot of work for a question asked rarely.
make_fallback(tp_ops.gcd.default, warn=False)


@register_lowering(tp_ops.pow.Tensor_Tensor)
def pow_tensor_tensor(a: Any, b: Any) -> Any:
    """One whole number raised to another, exponent and all.

    Exists as a form of its own because a graph records which form was called:
    a power with an exponent that is a value and a power with an exponent that
    is a number are the same computation and two calls, and a graph that used
    one is not a graph that used the other.
    """

    return pointwise(ops.pow, a, b)


def _is_complex(dtype: Any) -> bool:
    """Whether a type is a pair of real numbers read as one value."""

    return dtype in (tp.complex64, tp.complex128, tp.complex32)


@register_lowering(tp_ops.sym_size.int)
def sym_size(a: Any, dim: Any) -> Any:
    """How long an axis is, as a number rather than as a value.

    Asked of a value that is not yet one -- the shape is a number before the
    value that has that shape is -- so the answer is a function of shapes and
    not a read of anything.
    """

    return a.get_size()[dim]


@register_lowering(tp_ops.sym_stride.int)
def sym_stride(a: Any, dim: Any) -> Any:
    """How far apart an axis's elements are, as a number rather than as a value.

    The companion of the size above: a position is turned into an offset by
    both, and neither is a read of anything while the value is still to be made.
    """

    return a.get_stride()[dim]


@register_lowering(tp_ops.sym_numel.default)
def sym_numel(a: Any) -> Any:
    """How many positions a value has, as a number rather than as a read.

    The count of a shape is a product of what the shape already says, so the
    answer is there before the value is: no part of it is read from memory.
    """

    return a.get_numel()


@register_lowering(tp_ops.lift_fresh_copy.default)
def lift_fresh_copy(x: Any) -> Any:
    """The same values, as something nothing else shares.

    What is shared is the storage underneath, and a value is asked to stop
    sharing it when something might write to it and something else might still
    read the old contents.  Whether it does is a question about the value's
    own definition rather than about where it is used, so the copy is made when
    the value is lifted and not at the point of use.
    """

    return clone(x)


@register_lowering(tp_ops._to_dense.default)
def _to_dense(x: Any) -> Any:
    """A sparse value as a dense one, with room for every position.

    The positions the value does not have read as nothing, which is what makes
    the answer a value of the dense shape rather than a smaller one.
    """

    return clone(x)


@register_lowering(tp_ops.view_as_complex.default)
def view_as_complex(x: Any) -> Any:
    """Two real numbers read as one complex one.

    The same bits read a different way rather than a different value: the pairs
    were already side by side, and what changes is how many numbers each one
    value is.
    """

    return TensorBox.create(ir.ComplexView.create(tp_ops.view_as_complex.default, x, None))


@register_lowering(tp_ops._assert_async.msg)
def _assert_async(msg: Any) -> None:
    """A check that does not stop the program, deferred to the device.

    A check whose answer is not known while the graph is being built is asked
    where it can be answered, and a failure there is a failure of the running
    program rather than of the one being built.
    """

    return None


@register_lowering(tp_ops._functional_assert_async.msg)
def _functional_assert_async(t: Any, msg: Any) -> None:
    """A check on a value, deferred to the device.

    The value is not read while the graph is built -- the check is about what
    the program will do -- so nothing is produced and nothing is read.
    """

    return None


#: A backward pass written as a walk over the forward pass's own choices is the
#: framework's rather than this one's: which of several equally valid ways to
#: spread a gradient over the positions that produced one is a question about
#: the operation's meaning rather than about how to walk it, and the answers
#: differ between the two sides in ways a single walk would have to know about.
#: These are therefore computed by handing the call over, and the ones whose
#: result has to be dense say so, because a sparse result is a different shape
#: of answer rather than a sparser one.
for _name, _warn in (
    ("_adaptive_avg_pool2d_backward", False),
    ("_adaptive_avg_pool3d_backward", False),
    ("adaptive_max_pool2d_backward", False),
    ("adaptive_max_pool3d_backward", False),
    ("fractional_max_pool2d_backward", False),
    ("fractional_max_pool3d_backward", False),
    ("replication_pad1d_backward", False),
    ("replication_pad2d_backward", False),
    ("upsample_linear1d_backward", False),
    ("upsample_bicubic2d_backward", False),
    ("upsample_trilinear3d_backward", False),
    ("grid_sampler_2d_backward", False),
    ("_pdist_backward", False),
    ("max_pool2d_with_indices_backward", False),
    ("max_pool3d_with_indices_backward", False),
    ("avg_pool3d_backward", False),
    ("adaptive_avg_pool2d_backward", False),
    ("adaptive_avg_pool3d_backward", False),
):
    _op = getattr(tp_ops, _name, None)
    if _op is not None:
        make_fallback(_op, warn=_warn)


#: Operations whose own walk would have to know something the framework already
#: knows: which distribution a set of values came from, how a recurrent layer
#: carries its state from one step to the next, how a group of matrices is laid
#: out, and how a distance is counted.  Each of those is a meaning rather than
#: a computation, and a walk written here would be a second answer to the same
#: question rather than the one.  Handing the call over is what keeps the two
#: from existing at once.
#
#: The random ones are here rather than written as a read at a position because
#: they are the eager forms: they consult a generator rather than a position,
#: and which stream a generator is on is a fact about the call rather than
#: about the graph.  The read at a position is what the decompositions use, and
#: these are what a call that has not been decomposed reaches.
for _name, _warn in (
    ("randint", False),
    ("rand_like", False),
    ("randn_like", False),
    ("randint_like", False),
    ("normal", False),
    ("_pdist_forward", False),
    ("soft_margin_loss_backward", False),
    ("_fused_rms_norm", False),
    ("_cdist_forward", False),
    ("_cdist_backward", False),
    ("_trilinear", False),
    ("segment_reduce", False),
    ("_segment_reduce_backward", False),
    ("histc", False),
    ("_histogramdd_bin_edges", False),
    ("_histogramdd_from_bin_cts", False),
    ("addbmm", False),
    ("_addmm_activation", False),
    ("_grouped_mm", False),
    ("_cudnn_rnn", False),
    ("_cudnn_rnn_backward", False),
    ("_embedding_bag", False),
    ("_embedding_bag_forward_only", False),
    ("_adaptive_avg_pool3d", False),
    ("adaptive_max_pool3d", False),
):
    _op = getattr(tp_ops, _name, None)
    if _op is not None:
        make_fallback(_op, warn=_warn)

#: A histogram whose edges are given rather than counted is a different
#: operation from one that counts them: where the edges are is a fact about the
#: call, and which bin a value falls in follows from them.
_histogram = getattr(tp_ops, "histogram", None)
if _histogram is not None:
    for _ov in _histogram.overloads():
        make_fallback(getattr(_histogram, _ov), warn=False)


#: Operations whose answer has a size nothing here knows in advance: how many
#: distinct values there are, which positions are not zero, how many fall in
#: each bin.  The shape of the answer is the answer, so it cannot be written as
#: a walk over a range of known length -- and a walk that could be would have
#: to guess the length first, which is the one thing that cannot be guessed.
for _name, _warn in (
    ("bincount", False),
    ("_unique2", False),
    ("unique_dim", False),
    ("unique_dim_consecutive", False),
    ("unique_consecutive", False),
    ("nonzero", True),
    ("nonzero_static", True),
):
    _op = getattr(tp_ops, _name, None)
    if _op is None:
        continue
    _overloads = (
        [_op] if not hasattr(_op, "overloads") else
        [getattr(_op, _ov) for _ov in _op.overloads()]
    )
    for _ov in _overloads:
        make_fallback(_ov, warn=_warn)

#: A gather that does not check its indices is a gather whose answer is only
#: defined when every index is in range, and a scatter that accumulates into
#: positions several indices name is one where the order of the additions is
#: part of the answer.  Both are asked for where the check or the order is
#: known rather than assumed.
for _name in (
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
):
    _op = getattr(tp_ops, _name, None)
    if _op is not None:
        make_fallback(_op, warn=False)

#: Reducing a list of values into a smaller one, and the pass that puts the
#: gradient back, are a question about which of several ways of splitting the
#: work is meant -- and the two halves have to agree on which, which is why they
#: are asked of the framework together rather than written here separately.
for _name in ("segment_reduce", "_segment_reduce_backward"):
    _op = getattr(tp_ops, _name, None)
    if _op is not None:
        make_fallback(_op, warn=False)

#: A scatter that reduces says which of several values landing on the same
#: position wins -- a meaning rather than a walk, which is why it is asked of
#: the framework together rather than written here.  The indexed form of the
#: same question is asked of a list of positions at once, and is handed over
#: for the same reason.
for _name in ("scatter_reduce_", "index_reduce"):
    _op = getattr(tp_ops, _name, None)
    if _op is None:
        continue
    _overloads = (
        [_op] if not hasattr(_op, "overloads") else
        [getattr(_op, _ov) for _ov in _op.overloads()]
    )
    for _ov in _overloads:
        make_fallback(_ov, warn=False)


# ---------------------------------------------------------------------------
# searches among ordered boundaries
# ---------------------------------------------------------------------------


def _bucketize_lookup(tb: Any) -> Any:
    """The ordered list as the search's walk reads it: one named buffer.

    The walk addresses the list directly rather than loading it element by
    element, so it has to stand in memory with a name and strides before the
    search is written.
    """

    return TensorBox(ir.ExternKernel.realize_input(tb))


def _bucketize_boundaries(tb: Any) -> Any:
    """The ordered list's buffer, length, storage and stride, in one tuple."""

    size = tb.get_size()
    return (
        tb.get_name(),
        size[-1],
        tb.get_layout().storage_size(),
        tb.get_stride()[-1],
    )


def _bucketize_sorter(tb: Any) -> Any:
    """The list's companion order, as a buffer and a stride along its last axis."""

    return tb.get_name(), tb.get_stride()[-1]


def _bucketize_indices(tb: Any, index: Any, index_dtype: Any) -> Any:
    """Where one search starts reading inside the flattened list.

    The leading positions of the output name which row of the list is being
    read, so they fold into an offset along the list's own strides; the last
    position of the output has no part in it, because the walk covers the whole
    row however long it is.
    """

    strides = tb.get_stride()
    flattened_index = tb.get_layout().offset + sum(
        (s * i for s, i in zip(strides[:-1], index[:-1])), sympy.S.Zero
    )
    if not index and isinstance(flattened_index, sympy.Integer):
        return int(flattened_index)
    return ops.index_expr(flattened_index, index_dtype)


searchsorted_fallback = fallback_handler(
    tp_ops.searchsorted.Tensor, add_to_fallback_set=False
)


@register_lowering(tp_ops.searchsorted.Tensor, type_promotion_kind=None)
def lower_searchsorted(
    sorted_sequence: Any,
    self: Any,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Any = None,
    sorter: Any = None,
) -> Any:
    """For each value, the position it would keep in an ordered list.

    Every answer is found by one walk of the same list, and a walk answers one
    element at a time, so each element of the result is written as a search
    rather than the whole answer being handed to the framework.  Whether a
    value equal to a boundary belongs on its left or its right is part of what
    was asked, and the side the caller named is the walk's own choice of edge.
    """

    if not (
        V.graph.has_feature(sorted_sequence, BackendFeature.BUCKETIZE)
        and V.graph.has_feature(self, BackendFeature.BUCKETIZE)
        and (sorter is None or V.graph.has_feature(sorter, BackendFeature.BUCKETIZE))
    ):
        return searchsorted_fallback(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
        )

    if side is not None and side == "right":
        right = True

    index_dtype = tp.int32 if out_int32 else tp.int64
    values_loader = self.make_loader()

    sorted_sequence = _bucketize_lookup(sorted_sequence)
    if sorter is not None:
        sorter = _bucketize_lookup(sorter)

    boundaries = _bucketize_boundaries(sorted_sequence)
    sorter_arg = None if sorter is None else _bucketize_sorter(sorter)

    def walk(value: Any, index: Any) -> Any:
        return ops.bucketize(
            value,
            boundaries,
            _bucketize_indices(sorted_sequence, index, index_dtype),
            index_dtype,
            right,
            sorter=sorter_arg,
            sorter_indices=(
                None
                if sorter is None
                else _bucketize_indices(sorter, index, index_dtype)
            ),
        )

    if len(sorted_sequence.get_size()) == 1:
        # One long row: every read starts at its head, whatever the output
        # position is.
        def inner_fn(index: Any) -> Any:
            return walk(values_loader(index), ())

    else:

        def inner_fn(index: Any) -> Any:
            return walk(values_loader(index), index)

    result = ir.Pointwise.create(
        device=self.get_device(),
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )
    # A walk of an ordered list is not cheap, and a result that is read many
    # times -- a broadcast over it, say -- would walk once per read if it were
    # left inside whatever reads it.  It is written to memory once instead.
    result.realize()
    return result


bucketize_fallback = fallback_handler(tp_ops.bucketize.Tensor, add_to_fallback_set=False)


@register_lowering(
    tp_ops.bucketize.Tensor,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def lower_bucketize(
    input: Any,
    boundaries: Any,
    *,
    out_int32: bool = False,
    right: bool = False,
) -> Any:
    """Which bin each value falls into, told by one ordered list of edges.

    A bin is the stretch between two neighbours of the list, so the answer for
    one value is the same walk the position search makes, and it is written the
    same way: one search per element of the result.
    """

    if len(boundaries.get_size()) != 1:
        raise AssertionError("expected: len(boundaries.get_size()) == 1")

    if not (
        V.graph.has_feature(input, BackendFeature.BUCKETIZE)
        and V.graph.has_feature(boundaries, BackendFeature.BUCKETIZE)
    ):
        return bucketize_fallback(input, boundaries, out_int32=out_int32, right=right)

    boundaries = _bucketize_lookup(boundaries)
    input_loader = input.make_loader()
    index_dtype = tp.int32 if out_int32 else tp.int64
    edges = _bucketize_boundaries(boundaries)

    def inner_fn(index: Any) -> Any:
        return ops.bucketize(
            input_loader(index),
            edges,
            _bucketize_indices(boundaries, (), index_dtype),
            index_dtype,
            right,
        )

    result = ir.Pointwise.create(
        device=input.get_device(),
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )
    result.realize()
    return result


def ceildiv(number: Any, denom: Any) -> Any:
    """The quotient rounded up.

    Rounded up rather than to nearest because a range of whole positions from
    here to there has to contain the end: a range that stopped one short would
    be a range of the wrong length, and every position after it would be read
    from the wrong place.
    """

    if isinstance(number, sympy.Expr) or isinstance(denom, sympy.Expr):
        return CeilDiv(sympy.sympify(number), sympy.sympify(denom))
    return -(-number // denom)


@register_lowering(tp_ops.arange.start_step, type_promotion_kind=None)
def arange_start_step(
    start: Any,
    end: Any,
    step: Any = 1,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    requires_grad: Any = False,
) -> Any:
    """The numbers from here to there, by so much.

    How many there are is the distance divided by the step and rounded up,
    which is the smallest count that reaches the end: a count that stopped one
    short would be a range of the wrong length.  A type is required because
    which type these numbers are is not a question the numbers answer -- two
    whole numbers and two real ones are the same sequence of values read
    differently, and which one was asked for has to have been said.
    """

    if dtype is None or dtype == tp.undefined:
        # Unsaid, the numbers say it: whole numbers count in the wide whole
        # type, and any real one among them makes the range real.
        dtype = (
            tp.get_default_dtype()
            if any(isinstance(v, float) for v in (start, end, step))
            else tp.int64
        )
    device = device if device is not None else "cpu"
    numbers = all(isinstance(v, (int, float)) for v in (start, end, step))
    if numbers:
        # Counted as the numbers are: a real step is not something an index
        # expression can divide by.
        length = max(0, math.ceil((end - start) / step))
    else:
        length = ceildiv(sympy.sympify(end) - sympy.sympify(start), sympy.sympify(step))
    if not any(isinstance(v, float) for v in (start, end, step)):
        return iota(
            length, start=start, step=step, dtype=dtype, device=device,
            requires_grad=requires_grad,
        )
    # A real range is the position counted in whole numbers and then scaled:
    # ``start + step * i`` in double precision and rounded once to the range's
    # type, which is how each value is computed when the range is made eagerly.
    real_dtype = tp.float64

    def fn(index: Any) -> Any:
        position = ops.to_dtype(ops.index_expr(index[0], tp.int64), real_dtype)
        value = ops.add(
            ops.constant(start, real_dtype),
            ops.mul(ops.constant(step, real_dtype), position),
        )
        return ops.to_dtype(value, dtype)

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=fn,
        ranges=[length],
    )


@register_lowering(tp_ops.arange.end, type_promotion_kind=None)
def arange_end(
    end: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    requires_grad: Any = False,
) -> Any:
    """The numbers from zero to there, by one.

    A form of its own rather than the general one with the start left out,
    because a graph records which form was called: the same numbers written two
    ways are two calls, and a graph that used one is not a graph that used the
    other.
    """

    return arange_start_step(
        0,
        end,
        1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_lowering(tp_ops.arange.start, type_promotion_kind=None)
def arange_start(
    start: Any,
    end: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    requires_grad: Any = False,
) -> Any:
    """The numbers from here to there, by one."""

    return arange_start_step(
        start,
        end,
        1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_lowering(tp_ops.arange.default, type_promotion_kind=None)
def arange_default(
    start: Any,
    end: Any,
    step: Any = 1,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    requires_grad: Any = False,
) -> Any:
    """The numbers from here to there, by so much: the contract's general form."""

    return arange_start_step(
        start,
        end,
        step,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )




#: Operations whose answer is a property of the framework's own state or of a
#: library it holds: how far a scaling factor has grown, which positions a
#: normalization was built from, which order a set of values came out in, and
#: which values were drawn from which distribution.  Each of those is a fact
#: about something outside the graph, and a walk written here would have to
#: carry that something with it.
for _name in (
    "_amp_update_scale_",
    "linalg_pinv",
    "repeat_interleave",
    "randperm",
    "multinomial",
    "_weight_norm_interface_backward",
):
    _op = getattr(tp_ops, _name, None)
    if _op is None:
        continue
    _overloads = (
        [_op] if not hasattr(_op, "overloads") else
        [getattr(_op, _ov) for _ov in _op.overloads()]
    )
    for _ov in _overloads:
        make_fallback(_ov, warn=False)

#: The scaled forms of a matrix multiply take a scale that is a value rather
#: than a number, and which of several ways of applying it is meant is a
#: meaning rather than a walk.  The writing forms of a fused add-and-multiply
#: are here for the same reason as the non-writing ones: they differ only in
#: where the result goes, and having one without the other is how the two come
#: to compute different things.
for _name in (
    "_scaled_mm",
    "_scaled_mm_v2",
    "_scaled_dot_product_flash_attention",
    "mm",
    "addmm",
    "scatter_reduce_",
    "_foreach_addcdiv_",
    "_foreach_addcmul_",
    "avg_pool1d",
):
    _op = getattr(tp_ops, _name, None)
    if _op is None:
        continue
    _overloads = (
        [_op] if not hasattr(_op, "overloads") else
        [getattr(_op, _ov) for _ov in _op.overloads()]
    )
    for _ov in _overloads:
        make_fallback(_ov, warn=False)


def type_casts(
    f: Any,
    type_promotion: Any,
    compute_dtype_only: bool = False,
    include_non_tensor_args: bool = False,
) -> Any:
    """Run a decomposition in the type it should compute in, and answer in the type it should return.

    A value of a narrow type is computed on in a wider one and converted back
    afterwards.  Doing it any other way is how a sum of half-precision values
    ends up with the rounding of a half-precision accumulator: the wider type
    is where the arithmetic happens, and the narrow one is only how the answer
    is written down.  Which of the two a given argument is comes from the
    promotion, so the arguments are not all treated alike.

    Numbers are widened too where the operation takes them, because a narrow
    value added to a wide one is a wide one, and a narrow number standing for a
    wide value would be the one place the arithmetic stayed narrow.
    """

    @functools.wraps(f)
    def inner(*args: Any, **kwargs: Any) -> Any:
        allowed_types = (int, float, complex, bool) if include_non_tensor_args else ()
        flat_args = [
            x
            for x in arg_tree_leaves(*args, **kwargs)
            if isinstance(x, (TensorBox, IRNode, *allowed_types))
        ]
        compute_dtype = promoted_dtype_of_values(
            *flat_args,
            type_promotion_kind=type_promotion,
            return_compute_dtype=True,
        )
        result_dtype = promoted_dtype_of_values(
            *flat_args, type_promotion_kind=type_promotion
        )

        def increase_prec(x: Any) -> Any:
            if isinstance(x, (TensorBox, IRNode)):
                return to_dtype(x, compute_dtype)
            return x

        def decrease_prec(x: Any) -> Any:
            if isinstance(x, (TensorBox, IRNode)):
                return to_dtype(x, result_dtype)
            return x

        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        if compute_dtype_only:
            return r
        return tree_map(decrease_prec, r)

    return inner


pw_cast_for_opmath = functools.partial(
    type_casts, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
compute_only_pw_cast_for_opmath = functools.partial(
    type_casts,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    compute_dtype_only=True,
)
pw_cast_for_opmath_non_tensor_args = functools.partial(
    type_casts,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    include_non_tensor_args=True,
)


def check_and_broadcast_indices(indices: Any, device: Any) -> Any:
    """The indices of a gather, made to share a shape, and which axes they name.

    An index per axis, and an axis with no index keeps its whole length.  The
    indices that are values are made to share a shape with each other because
    they are read together -- one position of them names one element, and two
    indices read at two different positions would name two different elements
    from what was meant to be one.

    A truth value or a byte names positions by whether they are set rather than
    by what they are, which is a different question and is not answered here; a
    value on a different device than the value being read is likewise refused
    rather than moved, because moving it would be a copy this did not ask for.
    """

    if not (
        all(
            i.get_dtype() in (tp.int64, tp.int32, tp.bool, tp.uint8)
            for i in indices
            if i is not None
        )
    ):
        raise AssertionError(
            f"indices must be int64, byte or bool. Got "
            f"{[i.get_dtype() for i in indices if i is not None]}"
        )
    if any(
        i.get_dtype() in (tp.bool, tp.uint8) for i in indices if i is not None
    ):
        raise NotImplementedError("Fallback for bool indices")

    valid_idxs = [i for i, x in enumerate(indices) if isinstance(x, TensorBox)]
    if len(valid_idxs) <= 0:
        raise AssertionError("requires at least 1 non-None index")
    new_indices = [None] * len(indices)
    for i, x in zip(valid_idxs, broadcast_tensors(*[indices[i] for i in valid_idxs])):
        if x.get_device() != device:
            raise NotImplementedError("Fallback when indices is on a different device")
        new_indices[i] = x
    return new_indices, valid_idxs


def index_output_size_and_inner_fn(
    x_size: Any,
    indices: Any,
    tensor_indices: Any,
    tensor_size: Any,
    indices_loaders: Any,
    indexed_size: Any,
    x_loader: Any,
    check: Any,
    wrap_neg: Any = True,
) -> Any:
    """The shape of a gather's answer, and how to turn a position into an index.

    The answer's shape is not the indexed value's shape with some axes
    shortened: it is the indexed value's shape with some axes *replaced*, and
    the replacements have to go where the indices were rather than at the end.
    Which is why the two cases are told apart here -- indices that are next to
    each other take the positions they replaced, and indices that are not are
    pulled to the front, because a value read with two indices that are not
    next to each other has its answer's leading axes given by the indices rather
    than by the axes between them.
    """

    non_consecutive_tensors = False
    for previous, current in itertools.pairwise(tensor_indices):
        if current - previous != 1:
            non_consecutive_tensors = True

    output_size = [x_size[i] for i, val in enumerate(indices) if val is None]
    output_size = [*output_size, *x_size[len(output_size) + len(tensor_indices) :]]

    first_tensor_index = tensor_indices[0]
    if non_consecutive_tensors:
        output_size = tensor_size + output_size
    else:
        output_size = (
            output_size[:first_tensor_index]
            + tensor_size
            + output_size[first_tensor_index:]
        )

    def fn(idx: Any) -> Any:
        if len(idx) != len(output_size):
            raise AssertionError("expected: len(idx) == len(output_size)")
        if len(indices_loaders) != len(indexed_size):
            raise AssertionError("expected: len(indices_loaders) == len(indexed_size)")

        rank = len(tensor_size)
        new_index: list = []
        first_tensor_index = tensor_indices[0]
        start_offset = 0 if non_consecutive_tensors else first_tensor_index
        next_idx = 0
        for i in range(tensor_indices[-1] + 1):
            if i == start_offset:
                next_idx += rank
            if indices[i] is None:
                if next_idx >= len(idx):
                    raise AssertionError("expected: next_idx < len(idx)")
                new_index.append(idx[next_idx])
                next_idx += 1
            else:
                loader = indices_loaders[i]
                if loader is None:
                    raise AssertionError("expected: loader is not None")
                size = indexed_size[i]
                new_index.append(
                    ops.indirect_indexing(
                        loader(idx[start_offset : start_offset + rank]),
                        size,
                        check=check,
                        wrap_neg=wrap_neg,
                    )
                )
        new_index = [*new_index, *idx[next_idx:]]
        return new_index if x_loader is None else x_loader(new_index)

    return output_size, fn


def index_impl_helper(x: Any, indices: Any, check: Any, wrap_neg: Any = True) -> Any:
    """The shape of a gather's answer, and how to read one of its positions.

    Split from the walk that uses it because the shape and the read have to
    agree, and asking for them together is what makes them agree.
    """

    if not (isinstance(indices, (list, tuple))):
        raise AssertionError("expected: isinstance(indices, (list, tuple))")
    x_loader = x.make_loader()
    indices, tensor_indices = check_and_broadcast_indices(indices, x.get_device())
    if len(tensor_indices) <= 0:
        raise AssertionError("Must have at least one valid idx")

    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    tensor_size = list(indices[tensor_indices[0]].get_size())

    x_size = x.get_size()

    indexed_size = [x_size[i] for i in range(len(indices)) if indices[i] is not None]
    if check and 0 in indexed_size and 0 not in tensor_size:
        raise IndexError("index is out of bounds for dimension with size 0")

    indexed_size = [x_size[i] for i in range(len(indices))]
    output_size, index_inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=check,
        wrap_neg=wrap_neg,
    )

    def inner_fn(idx: Any) -> Any:
        return x_loader(index_inner_fn(idx))

    return output_size, inner_fn, index_inner_fn


def index_impl(x: Any, indices: Any, check: Any) -> Any:
    """One element per position asked for, each read at an index of its own.

    The answer is shaped like what was asked for rather than like what was read:
    the two are the same here, and saying so is what makes each output position
    know which input position it corresponds to.
    """

    output_size, inner_fn, _ = index_impl_helper(x, indices, check)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=output_size,
    )


@register_lowering(tp_ops.index.Tensor, type_promotion_kind=None)
def index_tensor(x: Any, indices: Any) -> Any:
    """One element per position asked for, each read at an index of its own.

    Whether the index is checked is asked for rather than settled here: a
    checked index refuses to read past the end of an axis, and an unchecked one
    does not, and which of the two is wanted is a property of where the index
    came from rather than of what an index is.

    A mask of truths selects as many elements as it holds truths, which is a
    size the data decides; that call is handed to the framework whole.
    """

    try:
        return index_impl(x, indices, check=True)
    except NotImplementedError:
        x.realize()
        return fallback_handler(tp_ops.index.Tensor, add_to_fallback_set=False)(
            x, indices
        )


def _promotion_input(value: Any) -> tuple:
    """What an operand contributes to a promotion: its type, and whether it is a number.

    A number promotes differently from a value of one dimension, so the two
    facts travel together rather than being read apart at each lattice step.
    A plain number contributes its kind -- a truth, a whole number, a real
    one -- and never a width: a whole number beside a 32-bit tensor stays a
    32-bit whole number, and only a real number turns a whole tensor real.
    """

    if isinstance(value, ir.Constant):
        return (value.get_dtype(), True)
    if isinstance(value, bool):
        return (tp.bool, True)
    if isinstance(value, int):
        return (tp.int64, True)
    if isinstance(value, float):
        return (tp.get_default_dtype(), True)
    if isinstance(value, complex):
        return (tp.complex64, True)
    return (value.get_dtype(), len(value.get_size()) == 0)


@register_lowering(
    ("where.default", "where.self", "where.ScalarSelf", "where.ScalarOther", "where.Scalar"),
    broadcast=False,
    type_promotion_kind=None,
)
def lower_where(cond, a, b):
    """One of two values, chosen by a third.

    The two are promoted against each other and not against the condition,
    because the condition does not become a value: it says which one to take,
    and a value that says which to take is not one of the two.  That is why the
    result is the type the two come to rather than anything the condition
    brings.

    A number given for either side is made into a value of the other's type
    first, so that a program which chose between a tensor and a number does not
    get a result whose type came from the number.
    """

    from .dtype_propagation import get_promoted_dtype

    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        # Two numbers: the type is the one the numbers come to by themselves.
        number_type = (
            tp.bool if isinstance(a, bool) and isinstance(b, bool)
            else tp.int64 if not isinstance(a, float) and not isinstance(b, float)
            else tp.get_default_dtype()
        )
        a = ir.Constant(value=a, dtype=number_type, device=cond.get_device())
    if isinstance(a, (float, int)):
        a = ir.Constant(value=a, dtype=b.get_dtype(), device=b.get_device())
    if isinstance(b, (float, int)):
        b = ir.Constant(value=b, dtype=a.get_dtype(), device=a.get_device())

    args = [cond, a, b]
    dtype = get_promoted_dtype(
        _promotion_input(a),
        _promotion_input(b),
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    # The condition is read as the truth value it is; only the two values are
    # brought to the type of the result.
    cond = args[0]
    if is_tensor_box(cond) and cond.get_dtype() != tp.bool:
        cond = to_dtype(cond, tp.bool)
    return pointwise(
        ops.where, cond, to_dtype(args[1], dtype), to_dtype(args[2], dtype),
        out_dtype=dtype,
    )


@register_lowering("max.default", type_promotion_kind=None)
def lower_max(x, dim=None, keepdim=False):
    """The largest of a set, and where it was.

    Two values rather than one when an axis was named, because a program that
    asked which is where is asking a different question from what the largest
    is -- and a caller that only wants the value should not pay for the position.
    Both are computed here rather than one derived from the other, because a
    value and the position of a value are found by different means and finding
    one does not find the other.

    With no axis named it is the largest of everything, which is one value and
    has no position to report.
    """

    if dim is not None:
        return (
            lower_amax(x, [dim], keepdim=keepdim),
            reduce_argmax(x, dim, keepdim),
        )
    return lower_amax(x, None, keepdim=keepdim)


# ---------------------------------------------------------------------------
# Copies, gathers, reversals, repetitions and padding: each element of the
# result is one element of the input, found by arithmetic on its position, so
# each is a loop that reads where the arithmetic says.
# ---------------------------------------------------------------------------


@register("copy.default")
def lower_copy(self, src, non_blocking=False):
    """``src`` written in ``self``'s place: on its device, in its type and
    broadcast to its shape."""

    x = src
    if not isinstance(x, ir.IRNode):
        x = _full(x, self.get_device(), self.get_dtype(), list(self.get_size()))
    if x.get_device() != self.get_device():
        x = to_device(x, self.get_device())
    if x.get_dtype() != self.get_dtype():
        x = to_dtype(x, self.get_dtype())
    if list(x.get_size()) != list(self.get_size()):
        x = lower_expand(x, list(self.get_size()))
    return clone(x)


@register("gather.default")
def lower_gather(x, dim, index, sparse_grad=False):
    """Each element read from ``x`` at the position ``index`` holds along
    ``dim``, the others in place."""

    if V.graph.sizevars.statically_known_equals(prod(index.get_size()), 0):
        return new_empty(x, index.get_size())
    size = list(x.get_size())
    scalar = len(size) == 0
    dim = _validate_dim(x, dim, 1 if scalar else 0)
    if scalar:
        x = lower_expand(x, [1])
        size = [1]
    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx):
        idx = list(idx)
        gathered = ops.indirect_indexing(index_loader(idx), size[dim])
        if len(idx) == 0:
            idx = [gathered]
        else:
            idx[dim] = gathered
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=list(index.get_size())
    )


@register("diagonal.default")
def lower_diagonal(x, offset=0, dim1=0, dim2=1):
    """The elements whose positions along two axes differ by ``offset``,
    along a new last axis: a view of ``x``."""

    shape = list(x.get_size())
    rank = len(shape)
    dim1, dim2 = normalize_dim(dim1, rank), normalize_dim(dim2, rank)
    if dim1 == dim2:
        raise RuntimeError(f"diagonal dimensions cannot be identical {dim1}, {dim2}")
    sizevars = V.graph.sizevars
    if offset < 0:
        diag = sizevars.evaluate_max(sizevars.evaluate_min(shape[dim1] + offset, shape[dim2]), 0)
        base = (-offset, 0)
    else:
        diag = sizevars.evaluate_max(sizevars.evaluate_min(shape[dim1], shape[dim2] - offset), 0)
        base = (0, offset)
    sizes = [s for d, s in enumerate(shape) if d not in (dim1, dim2)] + [diag]

    def reindex(idx):
        position = idx[-1]
        rest = iter(idx[:-1])
        out = []
        for d in range(rank):
            if d == dim1:
                out.append(position + base[0])
            elif d == dim2:
                out.append(position + base[1])
            else:
                out.append(next(rest))
        return out

    return TensorBox(ir.GenericView.create(x, sizes, reindex))


@register_lowering(tp_ops.unfold, type_promotion_kind=None)
def lower_unfold(x, dimension, size, step):
    """Sliding windows of one axis, laid out along a new trailing axis.

    Every window reads values the source already holds, so the result is an
    address of the source rather than a copy: a position in the output names a
    window and a place inside it, and the place it names in the source is the
    window's start plus how far into the window it stands.
    """

    sizes = x.get_size()
    ndim = len(sizes)
    dim = canonicalize_dim(ndim, dimension)

    if ndim == 0:
        return _slice(unsqueeze(x, 0), 0, 0, size, 1)

    dim_size = sizes[dim]
    sizevars = V.graph.sizevars
    if sizevars.statically_known_gt(sympy.sympify(size), dim_size):
        raise RuntimeError(
            f"maximum size for tensor at dimension {dimension} is {dim_size} "
            f"but size is {size}"
        )
    if sizevars.statically_known_leq(sympy.sympify(step), 0):
        raise RuntimeError(f"step must be greater than 0 but got step={step}")

    new_dim_size = FloorDiv(dim_size - size, step) + 1
    out_size = [*sizes[:dim], new_dim_size, *sizes[dim + 1 :], size]

    def reindexer(idx):
        window = idx[-1] + idx[dim] * step
        return (*idx[:dim], window, *idx[dim + 1 : -1])

    return TensorBox(ir.GenericView.create(x, out_size, reindexer))


@register_lowering(tp_ops.prelu, type_promotion_kind=None)
def lower_prelu(x, weight):
    """A pass with a learned slope: the values above zero go through, the rest
    are scaled -- by one slope shared by all of them, or by one slope per
    channel, the second axis when there is more than one.
    """

    count = functools.reduce(operator.mul, weight.get_size(), sympy.Integer(1))
    shape = list(x.get_size())
    if count == 1:
        slope = weight
    elif len(shape) >= 2:
        broadcast_shape = [sympy.S.One] * len(shape)
        broadcast_shape[1] = shape[1]
        slope = view(weight, broadcast_shape)
    else:
        slope = weight
    return lower_where(lower_gt(x, 0), x, lower_mul(slope, x))


@register("flip.default")
def lower_flip(x, dims=()):
    """The elements in reverse order along each of ``dims``."""

    size = list(x.get_size())
    rank = len(size)
    flipped = {normalize_dim(d, rank) for d in dims}
    loader = x.make_loader()

    def fn(idx):
        idx = list(idx)
        for d in flipped:
            idx[d] = size[d] - 1 - idx[d]
        return loader(idx)

    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=size)


@register("repeat.default")
def lower_repeat(x, repeats):
    """``x`` laid end to end ``repeats[d]`` times along each axis ``d``; extra
    leading repeats add leading axes."""

    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.S.One] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    new_size = [s * r for s, r in zip(old_size, repeats)]
    if any(r == 0 for r in repeats):
        return new_empty(x, new_size)
    if all(r == 1 or s == 1 for r, s in zip(repeats, old_size)):
        return clone(lower_expand(x, new_size))
    loader = x.make_loader()

    def fn(idx):
        idx = list(idx)
        for d, r in enumerate(repeats):
            if r != 1:
                idx[d] = sympy.S.Zero if old_size[d] == 1 else ModularIndexing(idx[d], 1, old_size[d])
        return loader(idx)

    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=new_size)


def _range_mask_low(i, low):
    return ops.ge(ops.index_expr(i, tp.int64), ops.index_expr(sympy.Integer(low), tp.int64))


def _range_mask_high(i, high):
    return ops.lt(ops.index_expr(i, tp.int64), ops.index_expr(high, tp.int64))


def _python_value_of(value, dtype):
    """A number as the kind of number ``dtype`` holds."""

    if dtype == tp.bool:
        return bool(value)
    if is_integer_dtype(dtype):
        return int(value)
    return float(value)


@register("constant_pad_nd.default")
def lower_constant_pad_nd(x, pad, value=0):
    """``x`` with ``value`` added around it; ``pad`` names the amount before
    and after each axis from the last backwards, and a negative amount cuts."""

    if len(pad) % 2:
        raise RuntimeError("Length of pad must be even")
    if all(p == 0 for p in pad):
        return clone(x)
    sizes = list(x.get_size())
    bounds = list(reversed(list(zip(pad[::2], pad[1::2]))))
    n = len(sizes) - len(bounds)
    output_size = list(sizes[:n])
    mask_sizes = []
    for (low, high), size in zip(bounds, sizes[n:]):
        mask_sizes.append(size)
        output_size.append(sympy.expand(size + low + high))
    fill = _python_value_of(value, x.get_dtype())
    loader = x.make_loader()

    def fn(index):
        shifted = list(index[:n])
        for idx, (low, _high) in zip(index[n:], bounds):
            shifted.append(idx - low)
        conds = []
        for idx, (low, high), length in zip(shifted[n:], bounds, mask_sizes):
            if low != 0:
                conds.append(_range_mask_low(idx, 0))
            if high != 0:
                conds.append(_range_mask_high(idx, length))
        cond = functools.reduce(ops.and_, conds)
        return ops.masked(cond, lambda: loader(shifted), fill)

    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=output_size)


_fallback_remap_pad = {
    name: fallback_handler(getattr(tp.ops.tp, name).default, add_to_fallback_set=False)
    for name in ("reflection_pad_nd", "replication_pad_nd", "circular_pad_nd")
}


@register("reflection_pad_nd.default", "replication_pad_nd.default", "circular_pad_nd.default")
def lower_remap_pad_nd(x, pad, **kwargs):
    """A pad that reads the source's own values again rather than filling.

    Each output element reads one source element, chosen by folding the
    position it stands at back into the source's extent: a reflection doubles
    back one short of each edge, a replication holds at the edge, a circular
    wrap continues from the other side.  The reading is an address of the
    source rather than a call, so it fuses with whatever consumes it.
    """

    op_name = target_name(V.current_node.target).split(".")[0]
    sizes = list(x.get_size())
    if len(pad) % 2:
        raise RuntimeError("Length of pad must be even")
    pad = [int(p) for p in pad]
    if all(p == 0 for p in pad):
        return clone(x)
    if any(p < 0 for p in pad) or is_dynamic(*sizes):
        return _fallback_remap_pad[op_name](x, pad)
    bounds = list(reversed(list(zip(pad[::2], pad[1::2]))))
    n = len(sizes) - len(bounds)
    extents = [int(s) for s in sizes[n:]]
    if op_name == "reflection_pad_nd" and any(
        m <= 1 or low >= m or high >= m for (low, high), m in zip(bounds, extents)
    ):
        return _fallback_remap_pad[op_name](x, pad)
    output_size = list(sizes[:n])
    for (low, high), size in zip(bounds, sizes[n:]):
        output_size.append(sympy.expand(size + low + high))
    loader = x.make_loader()

    def fold(at, extent):
        """One position past the edge, read as a position of the source."""

        if op_name == "reflection_pad_nd":
            # m - 1 - |m - 1 - |at||: the walk doubles back one short of each
            # edge.  Both folds read as a smallest of two, so the address
            # stays arithmetic the kernel computes.
            inner = Min(extent - 1 - at, extent - 1 + at)
            return Min(extent - 1 - inner, extent - 1 + inner)
        if op_name == "replication_pad_nd":
            return Max(Min(at, extent - 1), 0)
        return at % extent

    def fn(index):
        shifted = list(index[:n])
        for at, (low, _high), extent in zip(index[n:], bounds, extents):
            shifted.append(fold(at - low, extent))
        return loader(shifted)

    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=output_size)


@register("embedding.default")
def lower_embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """The rows of ``weight`` that ``indices`` name, one per index."""

    if sparse:
        return fallback_handler(tp_ops.embedding.default, add_to_fallback_set=False)(
            weight, indices, padding_idx, scale_grad_by_freq, sparse
        )
    weight_loader = weight.make_loader()
    indices_loader = indices.make_loader()
    indices_ndim = len(indices.get_size())
    weight_size = list(weight.get_size())
    new_size = [*indices.get_size(), *weight_size[1:]]

    def fn(idx):
        row = ops.indirect_indexing(indices_loader(idx[:indices_ndim]), weight_size[0])
        return weight_loader([row, *idx[indices_ndim:]])

    return Pointwise.create(
        device=weight.get_device(), dtype=weight.get_dtype(), inner_fn=fn, ranges=new_size
    )


def _nan_ignoring(op_name):
    """The larger or smaller of two values, a missing value (NaN) losing to
    any number."""

    op = ops_wrapper(op_name)

    def lower(a, b):
        # An operand may arrive as a bare view rather than a box.
        a, b = _box_view(a), _box_view(b)
        dtype = _promoted_pair(a, b)
        a, b = (to_dtype(t, dtype) if isinstance(t, TensorBox) and t.get_dtype() != dtype else t for t in (a, b))
        if isinstance(a, TensorBox) and isinstance(b, TensorBox):
            a, b = broadcast_tensors(a, b)
        anchor = a if isinstance(a, TensorBox) else b

        def fn(x, y):
            x, y = ops.to_dtype(x, dtype), ops.to_dtype(y, dtype)
            chosen = op(x, y)
            if not _is_real(dtype) or not dtype.is_floating_point:
                return chosen
            return ops.where(ops.isnan(x), y, ops.where(ops.isnan(y), x, chosen))

        return pointwise(fn, a, b, val=anchor, out_dtype=dtype)

    return lower


LOWERINGS["fmax.default"] = _nan_ignoring("maximum")
LOWERINGS["fmin.default"] = _nan_ignoring("minimum")


def convert_symint_to_expr(val: Any) -> Any:
    """A size as an expression, leaving a number as a number.

    Not the same as turning everything into an expression, which would make a
    number into an expression that merely happens to be constant: the two are
    used differently, and a layout that was given a number can be compared with a
    number.
    """

    return getattr(val, "node", None) and val.node.expr or val


@register_lowering("as_strided.default", type_promotion_kind=None)
def lower_as_strided(
    x: Any,
    size: Any,
    stride: Any,
    storage_offset: Any = None,
    *,
    storage_offset_relative_to_input_storage: bool = True,
) -> Any:
    """The same memory read as a different shape.

    A shape and a set of distances say which element is where without saying
    where the memory is, so this is a way of reading what is already there rather
    than a copy.  Which is why the value must already be in memory: there is
    nothing to read from until it is, and a shape over memory that does not
    exist is a shape over nothing.

    When what comes in is itself a view, its shape and distances are replaced
    rather than a second view being stacked on it -- two views of one buffer
    would have to agree about both, and the outer one is the one being asked
    for.  The type is carried across, because that is the one thing about the
    old view that still holds.
    """

    explicit_storage_offset = (
        storage_offset is not None and storage_offset_relative_to_input_storage
    )
    new_device = None
    new_dtype = None
    if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
        new_device = x.get_device()
        new_dtype = x.dtype
        if storage_offset is None and x.maybe_get_layout() is not None:
            storage_offset = x.get_layout().offset
        x = x.data.unwrap_view()
    x.realize()
    if not ir.is_storage_and_layout(x):
        raise NotImplementedError(f"unrealized as_strided({x}, ...)")
    storage, old_layout = ir.as_storage_and_layout(x)
    storage_offset = (
        convert_symint_to_expr(storage_offset) if storage_offset is not None else 0
    )
    storage_data = storage.data if isinstance(storage, ir.StorageBox) else storage
    if explicit_storage_offset and isinstance(storage_data, ir.InputBuffer):
        # A pointer the graph was handed already includes the offset the input
        # tensor was created with, while an offset named here is from the start
        # of the storage -- so the one already counted has to come off.
        storage_offset = sympy.expand(
            storage_offset
            - V.graph.graph_input_storage_offsets.get(storage_data.get_name(), 0)
        )
    new_layout = ir.FixedLayout(
        new_device if new_device else old_layout.device,
        new_dtype if new_dtype else old_layout.dtype,
        [sympy.expand(s) for s in size],
        [sympy.expand(s) for s in stride],
        sympy.expand(storage_offset),
    )
    return TensorBox(ir.ReinterpretView(data=storage, layout=new_layout))


@register_lowering("as_strided_scatter.default", type_promotion_kind=None)
def as_strided_scatter(self: Any, src: Any, size: Any, stride: Any, storage_offset: Any = None) -> Any:
    """The values of another written into a shape of its own, re-read.

    The value is not changed where it was asked for: what is written is a copy
    of it, laid out as it was and read at the shape and distances asked for,
    with the other value written into the positions that shape names.
    """

    output = clone(self)
    output_view = lower_as_strided(output, size, stride, storage_offset)
    lower_copy_(output_view, src)
    return output


# Small schema variants share the view and allocation paths above.  Keeping
# these registrations explicit lets captured graphs retain their native names.
@register("_assert_scalar.default", "_assert_tensor_metadata.default")
def _lower_metadata_assert(*args: Any, **kwargs: Any) -> None:
    return None


@register("_efficientzerotensor.default")
def _lower_efficientzerotensor(
    size: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = False,
) -> Any:
    del layout, pin_memory
    dtype = dtype or tp.get_default_dtype()
    return _full(0, decode_device(device), dtype, list(size))


@register("_local_scalar_dense.default")
def _lower_local_scalar_dense(data: Any) -> Any:
    bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env,
        V.graph.current_node.meta.get("unbacked_bindings"),
    )
    if bindings is None or len(bindings) != 1:
        raise AssertionError(bindings)
    symbol, keypath = next(iter(bindings.items()))
    dynamic = DynamicScalar(symbol, keypath, data)
    dynamic.name = V.graph.register_buffer(dynamic)
    V.graph.register_operation(dynamic)
    value = V.graph.current_node.meta.get("val")
    if isinstance(value, (tp.SymInt, tp.SymFloat, tp.SymBool)):
        return value.node.expr
    return sympy.sympify(value)


@register("_neg_view.default")
def _lower_neg_view(x: Any) -> Any:
    return LOWERINGS["neg.default"](x)


@register("_unsafe_index.Tensor")
def _lower_unsafe_index(x: Any, indices: Any) -> Any:
    return index_impl(x, indices, check=False)


@register("as_strided_.default")
def _lower_as_strided_(x: Any, size: Any, stride: Any, storage_offset: Any = None) -> Any:
    value = lower_as_strided(x, size, stride, storage_offset)
    x.data = value.data
    return x


@register("as_strided_copy.default")
def _lower_as_strided_copy(
    x: Any, size: Any, stride: Any, storage_offset: Any = None
) -> Any:
    return clone(lower_as_strided(x, size, stride, storage_offset))


_fallback_bernoulli = fallback_handler(tp_ops.bernoulli.default)
_fallback_bernoulli_p = fallback_handler(tp_ops.bernoulli.p)
LOWERINGS["bernoulli.default"] = _fallback_bernoulli
LOWERINGS["bernoulli.p"] = _fallback_bernoulli_p
LOWERINGS["bernoulli_.Tensor"] = fallback_handler(tp_ops.bernoulli_.Tensor)
LOWERINGS["bernoulli_.float"] = fallback_handler(tp_ops.bernoulli_.float)


@register("detach_.default")
def _lower_detach_(x: Any) -> Any:
    return x


@register("diagonal_copy.default")
def _lower_diagonal_copy(
    x: Any, offset: Any = 0, dim1: Any = 0, dim2: Any = 1
) -> Any:
    return clone(lower_diagonal(x, offset, dim1, dim2))


@register("empty.default", "empty.memory_format")
def _lower_empty(
    *size: Any,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    memory_format: Any = None,
) -> Any:
    del memory_format
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    return empty_strided(
        list(size), None, dtype=dtype, layout=layout,
        device=decode_device(device), pin_memory=pin_memory,
    )


@register("empty_like.default")
def _lower_empty_like(
    x: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = False,
    memory_format: Any = None,
) -> Any:
    del memory_format
    return empty_strided(
        list(x.get_size()), None, dtype=dtype or x.get_dtype(), layout=layout,
        device=decode_device(device) if device is not None else x.get_device(),
        pin_memory=pin_memory,
    )


@register("expand_as.default")
def _lower_expand_as(x: Any, y: Any) -> Any:
    return lower_expand(x, y.get_size())


@register("fill_.Scalar", "fill_.Tensor")
def _lower_fill_(x: Any, value: Any) -> Any:
    return mutate_to(x, _full(value, x.get_device(), x.get_dtype(), list(x.get_size())))


_fallback_fractional_max_pool2d = fallback_handler(tp_ops.fractional_max_pool2d.default)
_fallback_fractional_max_pool3d = fallback_handler(tp_ops.fractional_max_pool3d.default)
LOWERINGS["fractional_max_pool2d.default"] = _fallback_fractional_max_pool2d
LOWERINGS["fractional_max_pool3d.default"] = _fallback_fractional_max_pool3d


@register("glu.default")
def _lower_glu(x: Any, dim: Any = -1) -> Any:
    dim = normalize_dim(dim, len(x.get_size()))
    half = V.graph.sizevars.guard_int(x.get_size()[dim]) // 2
    left = _slice(x, dim, 0, half, 1)
    right = _slice(x, dim, half, 2 * half, 1)
    return lower_mul(left, LOWERINGS["sigmoid.default"](right))


_fallback_mkldnn_rnn_layer = fallback_handler(tp_ops.mkldnn_rnn_layer.default)
LOWERINGS["mkldnn_rnn_layer.default"] = _fallback_mkldnn_rnn_layer

_fallback_native_dropout = fallback_handler(tp_ops.native_dropout.default)
LOWERINGS["native_dropout.default"] = _fallback_native_dropout

_fallback_rand = fallback_handler(tp_ops.rand.default)
_fallback_rand_generator = fallback_handler(tp_ops.rand.generator)
LOWERINGS["rand.default"] = _fallback_rand
LOWERINGS["rand.generator"] = _fallback_rand_generator


@register("scalar_tensor.default")
def _lower_scalar_tensor(
    data: Any, *, dtype: Any = None, device: Any = None, pin_memory: Any = False
) -> Any:
    return tensor(data, dtype=dtype, device=device, pin_memory=pin_memory)


LOWERINGS["special_erf.default"] = LOWERINGS["erf.default"]


@register("squeeze_.default", "squeeze_.dim", "squeeze_.dims")
def _lower_squeeze_(x: Any, dim: Any = None) -> Any:
    value = lower_squeeze(x, dim)
    x.data = value.data
    return x


@register("squeeze_copy.default", "squeeze_copy.dim", "squeeze_copy.dims")
def _lower_squeeze_copy(x: Any, dim: Any = None) -> Any:
    return clone(lower_squeeze(x, dim))


@register("sym_constrain_range.default")
def _lower_sym_constrain_range(a: Any, min: Any = None, max: Any = None) -> None:
    del a, min, max
    return None


@register("unbind.default", "unbind.int")
def _lower_unbind(x: Any, dim: Any = 0) -> list[Any]:
    dim = normalize_dim(dim, len(x.get_size()))
    extent = V.graph.sizevars.guard_int(x.get_size()[dim])
    return [lower_select(x, dim, index) for index in range(extent)]


@register("unsqueeze_.default")
def _lower_unsqueeze_(x: Any, dim: Any) -> Any:
    value = lower_unsqueeze(x, dim)
    x.data = value.data
    return x


_fallback_jagged_to_padded_dense = fallback_handler(tp_ops._jagged_to_padded_dense_forward.default)
_fallback_padded_dense_to_jagged = fallback_handler(tp_ops._padded_dense_to_jagged_forward.default)
LOWERINGS["_jagged_to_padded_dense_forward.default"] = _fallback_jagged_to_padded_dense
LOWERINGS["_padded_dense_to_jagged_forward.default"] = _fallback_padded_dense_to_jagged

_fallback_weight_int4pack = fallback_handler(tp_ops._weight_int4pack_mm_for_cpu.default)
_fallback_weight_int8pack = fallback_handler(tp_ops._weight_int8pack_mm.default)
LOWERINGS["_weight_int4pack_mm_for_cpu.default"] = _fallback_weight_int4pack
LOWERINGS["_weight_int8pack_mm.default"] = _fallback_weight_int8pack


# ---------------------------------------------------------------------------
# Control flow: branches and loops run as pieces of their own
# ---------------------------------------------------------------------------


def _register_control_flow_lowerings() -> None:
    from tensorplay._higher_order_ops.cond import cond_op
    from tensorplay._higher_order_ops.while_loop import (
        while_loop_op,
        while_loop_stack_output_op,
    )

    @register_lowering(cond_op, type_promotion_kind=None)
    def cond(pred: Any, true_fn: Any, false_fn: Any, operands: Any) -> list:
        """One of two pieces, chosen on the host by a truth value.

        The pieces are listed false first: the choice is made by reading the
        truth value as a number, and false reads as the first of them.
        """

        result = ir.Switch.create(pred, [false_fn, true_fn], list(operands), is_cond=True)
        boxes = list(map(TensorBox.create, result))
        # Pieces that return one value make the operation return that value.
        if not isinstance(V.graph.current_node.meta.get("val"), (list, tuple)):
            return boxes[0]
        return boxes

    def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs, stack_output=False):
        """A piece run again until another piece says to stop, on the host."""

        result = ir.WhileLoop.create(
            cond_fn, body_fn, list(carried_inputs), list(additional_inputs), stack_output
        )
        if not isinstance(result, Sequence):
            raise AssertionError(f"expected a sequence of results, got {type(result)}")
        return list(map(ir.WhileLoop._maybe_wrap_as_tensor_box, result))

    register_lowering(while_loop_op, type_promotion_kind=None)(while_loop)
    register_lowering(while_loop_stack_output_op, type_promotion_kind=None)(
        functools.partial(while_loop, stack_output=True)
    )


_register_control_flow_lowerings()


def load_lowering_modules() -> None:
    """Pull in the lowerings that are kept in modules of their own.

    Such a lowering registers by being imported, and its module imports this
    one -- along with the template helpers, which import this one too -- so it
    cannot be loaded from here while this module is still being read.  It is
    asked for by whatever is about to lower a region instead; without that
    nothing would ever load it, and its operations would be handed to the
    framework whole.
    """

    from . import attention  # noqa: F401
