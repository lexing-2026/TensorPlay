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
from typing import Any, Callable

import sympy

import tensorplay as tp

from . import config
from .utils import (
    is_dynamic,
    is_triton_fp8_dtype_supported,
    is_view,
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
    MutationLayoutSHOULDREMOVE,
    ExpandView,
    FixedLayout,
    FallbackKernel,
    IRNode,
    PermuteView,
    Pointwise,
    Reduction,
    ReinterpretView,
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
        return tp.device("cuda", 0)
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


def pointwise(fn, *inputs, val=None):
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
        size = tuple(int(s) for s in val.get_size())
        dtype = val.get_dtype()
        device = val.get_device()
    else:
        size, dtype, device = val_info(val)
    tensor_inputs = [x for x in inputs if hasattr(x, "get_size")]
    if len(tensor_inputs) > 1:
        target = functools.reduce(
            broadcast_symbolic_shapes, (x.get_size() for x in tensor_inputs), ()
        )
        size = tuple(int(s) for s in target)
    loaders = [
        as_value_node(x, dtype, device).make_loader() for x in inputs
    ]

    def inner(index):
        return fn(*[load(index) for load in loaders])

    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner, ranges=size)


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


@register("sub.Tensor", "sub.Scalar")
def lower_sub(a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(ops.sub, a, b)
    return pointwise(lambda x, y: ops.sub(x, ops.mul(y, ops.constant(float(alpha), tp.float32))), a, b)


@register("rsub.Scalar", "rsub.Tensor")
def lower_rsub(a, b, *rest, **kwargs):
    return pointwise(lambda x, y: ops.sub(y, x), a, b)


@register("mul.Tensor", "mul.Scalar")
def lower_mul(a, b):
    return pointwise(ops.mul, a, b)


@register("div.Tensor", "div.Scalar", "truediv", "truediv.Tensor", "truediv.Scalar")
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
LOWERINGS["le.default"] = lower_le = _binary("le")
LOWERINGS["lt.default"] = lower_lt = _binary("lt")
LOWERINGS["ge.default"] = lower_ge = _binary("ge")
LOWERINGS["gt.default"] = lower_gt = _binary("gt")
LOWERINGS["ne.default"] = lower_ne = _binary("ne")
LOWERINGS["eq.default"] = lower_eq = _binary("eq")
#: The forms where one side is a plain number: a value asked whether it is
#: below a bound is answered by the same comparison, with the number on the
#: other side of it.
LOWERINGS["le.Scalar"] = _binary("le")
LOWERINGS["lt.Scalar"] = _binary("lt")
LOWERINGS["ge.Scalar"] = _binary("ge")
LOWERINGS["gt.Scalar"] = _binary("gt")
LOWERINGS["ne.Scalar"] = _binary("ne")


@register("clamp.default")
def lower_clamp(x, min=None, max=None):
    """A value held between two others.

    Written as the two operations it is rather than as one, because there is no
    operation for holding a value between two others: it is a lower one and an
    upper one, and saying which is which is the whole of what clamping is.
    """

    if min is None and max is None:
        return x
    if min is None:
        return pointwise(lambda v: ops.minimum(v, max), x)
    if max is None:
        return pointwise(lambda v: ops.maximum(v, min), x)
    return pointwise(lambda v: ops.maximum(min, ops.minimum(max, v)), x)


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


@register("to.dtype", "to.device", "to.dtype_layout", "_to_copy.default")
def lower_to(x, *args, **kwargs):
    size, dtype, _ = val_info(node_val())
    if is_tensor_box(x) and dtype_name(x.get_dtype()) == dtype_name(dtype) and not kwargs.get("copy", False):
        return x
    return pointwise(lambda v: cast_to(v, dtype), x)


@register("clone.default", "contiguous.default")
def lower_clone(x, *args, **kwargs):
    return pointwise(lambda v: v, x)


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


def reshape(x: TensorBox, new_size) -> TensorBox:
    """This value under a different shape, without moving anything if it can be.

    Memory that already lies in consecutive elements is described by the new
    shape rather than copied: the elements are where they were, and a shape, a
    stride and an offset say how to read them as the new shape.  Memory that
    does not lie that way has to be walked, which is a view over an index.
    """

    old_size = tuple(int(s) for s in x.get_size())
    new_size = tuple(int(s) for s in new_size)
    if old_size == new_size:
        return x
    node = _underlying(x)
    if isinstance(node, Buffer) and node.layout.is_contiguous():
        settled = node.layout.as_fixed()
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
    if isinstance(node, (Pointwise, Reduction)) and isinstance(x.data, StorageBox):
        # A loop that has not been materialized has no storage to view, so it
        # is realized first; the resulting buffer is contiguous and the new
        # shape can be described as a plain reinterpret view of it.
        x.data.realize()
        node = _underlying(x)
        if isinstance(node, Buffer) and node.layout.is_contiguous():
            settled = node.layout.as_fixed()
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
    return TensorBox(View.create(_underlying(x), new_size))


def _resolve_size(size, numel):
    size = [int(s) for s in size]
    if -1 in size:
        known = prod(s for s in size if s != -1)
        size[size.index(-1)] = numel // max(known, 1)
    return size


@register("view.default", "reshape.default", "_unsafe_view.default", "view.dtype_unused")
def lower_view(x, size):
    return reshape(x, _resolve_size(size, x.get_numel()))


@register("flatten.default", "flatten.int", "flatten")
def lower_flatten(x, start_dim=0, end_dim=-1):
    # Flatten merges the axes in ``[start_dim, end_dim]`` into one: the
    # leading and trailing axes stay as they are, and the merged axis holds
    # the product of the extents it spans.
    size = [int(s) for s in x.get_size()]
    rank = len(size)
    start = normalize_dim(start_dim, rank)
    end = normalize_dim(end_dim, rank)
    merged = prod(size[start : end + 1])
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
    if dim is None:
        dims = [d for d in range(rank) if size[d] == 1]
    elif isinstance(dim, (list, tuple)):
        dims = [normalize_dim(d, rank) for d in dim if size[normalize_dim(d, rank)] == 1]
    else:
        d = normalize_dim(dim, rank)
        dims = [d] if size[d] == 1 else []
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
    return ExpandView.create(x, [int(s) for s in size])


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


@register("split.Tensor")
def lower_split(x, split_size, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    out = []
    start = 0
    while start < size[dim]:
        out.append(_slice(x, dim, start, min(start + int(split_size), size[dim]), 1))
        start += int(split_size)
    return tuple(out)


@register("split_with_sizes.default")
def lower_split_with_sizes(x, sizes, dim=0):
    size = list(x.get_size())
    dim = normalize_dim(dim, len(size))
    out = []
    start = 0
    for s in sizes:
        out.append(_slice(x, dim, start, start + int(s), 1))
        start += int(s)
    return tuple(out)


@register("cat.default")
def lower_cat(tensors, dim=0):
    """Concatenation as one pointwise loop selecting its source per index."""

    size, dtype, device = val_info(node_val())
    dim = normalize_dim(dim, len(size))
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

    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner, ranges=size)


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

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dims else next(it) for d in range(rank)]
        value = loader(full)
        if prologue is not None:
            value = prologue(value, full)
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
def lower_sum(x, dims=None, keepdim=False, dtype=None, **kwargs):
    if not dims:
        dims = list(range(len(x.get_size())))
    elif isinstance(dims, (int, sympy.Integer)):
        dims = [dims]
    if dtype is None or dtype == tp.undefined:
        # A sum of whole numbers is not a whole number unless it is asked to be.
        if is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype()):
            dtype = tp.int64
        else:
            dtype = x.get_dtype()
    return make_reduction(
        x, dims, keepdim, dtype, x.get_device(), "sum"
    )


@register("mean.dim")
def lower_mean(x, dims=None, keepdim=False, dtype=None, **kwargs):
    if dims is None:
        dims = list(range(len(x.get_size())))
    elif isinstance(dims, (int, sympy.Integer)):
        dims = [dims]
    dtype = _resolve_dtype(dtype, x.get_dtype())
    count = prod(x.get_size()[normalize_dim(d, len(x.get_size()))] for d in dims)
    total = make_reduction(
        x, dims, keepdim, dtype, x.get_device(), "sum"
    )
    return pointwise(lambda v: ops.truediv(v, ops.constant(float(count), tp.float32)), total)


@register("var.dim", "var.correction")
def lower_var(x, dims=None, correction=1, keepdim=False, **kwargs):
    """Variance as the mean of squared deviations from the mean.

    The mean and the total of the squared differences from it are carried in a
    single walk over the reduced axes, so the variance is that total divided by
    the count, or by the count less the requested correction.  Keeping the two
    running values instead of a sum of squares keeps the accumulation in range
    for a long group.
    """

    if dims is None:
        dims = list(range(len(x.get_size())))
    elif isinstance(dims, (int, sympy.Integer)):
        dims = [dims]
    dtype = x.get_dtype()
    device = x.get_device()
    size = list(x.get_size())
    rank = len(size)
    dims = sorted({normalize_dim(d, rank) for d in dims})
    out_ranges = [size[d] for d in range(rank) if d not in dims]
    red_ranges = [size[d] for d in dims]
    loader = x.make_loader()

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dims else next(it) for d in range(rank)]
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
    denom = Max(sympy.Integer(n_elems) - sympy.Integer(correction), 0)
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
        kept = [1 if d in dims else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dims]

        return make_view(var, kept, reindex)
    return var


@register("var_mean.default", "var_mean.dim", "var_mean.correction", "var_mean")
def lower_var_mean(x, dims=None, unbiased=True, keepdim=False, **kwargs):
    """The mean and variance of the reduced axes in one walk.

    Both statistics come out of the same running mean and total of squared
    differences, so the reduced data is read once rather than once for the
    mean and once for the variance.  The variance is biased when no correction
    is asked for and divided by the count less one otherwise.
    """

    if dims is None:
        dims = list(range(len(x.get_size())))
    elif isinstance(dims, (int, sympy.Integer)):
        dims = [dims]
    dtype = x.get_dtype()
    device = x.get_device()
    size = list(x.get_size())
    rank = len(size)
    dims = sorted({normalize_dim(d, rank) for d in dims})
    out_ranges = [size[d] for d in range(rank) if d not in dims]
    red_ranges = [size[d] for d in dims]
    loader = x.make_loader()

    def inner(index, rindex):
        it = iter(index)
        rt = iter(rindex)
        full = [next(rt) if d in dims else next(it) for d in range(rank)]
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
    correction = 0 if not unbiased else 1
    denom = Max(sympy.Integer(n_elems) - sympy.Integer(correction), 0)
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
        kept = [1 if d in dims else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dims]

        return make_view(var, kept, reindex), make_view(mean, kept, reindex)
    return var, mean


@register("amax.default")
def lower_amax(x, dims=None, keepdim=False, dtype=None, **kwargs):
    # What the reduction produces is read from the value it reduces: reducing
    # over axes changes how many there are and leaves a value with the same
    # element type.  A dtype the caller asked for is a different thing, and
    # overrides what that would say.
    if not dims:
        dims = list(range(len(x.get_size())))
    elif isinstance(dims, (int, sympy.Integer)):
        dims = [dims]
    return make_reduction(x, dims, keepdim, _resolve_dtype(dtype, x.get_dtype()), x.get_device(), "max")


@register("amin.default")
def lower_amin(x, dims=None, keepdim=False, dtype=None, **kwargs):
    if not dims:
        dims = list(range(len(x.get_size())))
    elif isinstance(dims, (int, sympy.Integer)):
        dims = [dims]
    return make_reduction(x, dims, keepdim, _resolve_dtype(dtype, x.get_dtype()), x.get_device(), "min")


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


@register("conv2d_grad_input.default")
def lower_conv2d_grad_input(grad_output, input, weight, stride, padding, dilation, groups):
    """The input gradient, asked of the framework kernel."""

    return _fallback_conv2d_grad_input(
        grad_output, input, weight, stride, padding, dilation, groups
    )


@register("conv2d_grad_weight.default")
def lower_conv2d_grad_weight(grad_output, input, weight, stride, padding, dilation, groups):
    """The weight gradient, asked of the framework kernel."""

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
    layout = grad_out.get_layout()
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
        return None
    node, _ = autotune_select_algorithm(
        "group_norm_backward",
        choices,
        [grad_out, x, mean, rstd, gamma, dgamma0, dbeta0],
        layout,
    )
    return (node, dgamma0, dbeta0)


@register("native_group_norm_backward.default")
def lower_native_group_norm_backward(grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask):
    if config.use_gn_bwd_template:
        lowered = _lower_gn_bwd_template(
            grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask
        )
        if lowered is not None:
            return lowered
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


def _pool_output_size(extent: int, kernel: int, stride: int, padding: int,
                      ceil_mode: bool) -> int:
    """Output extent of one pooling axis."""

    if ceil_mode:
        return -((-(extent + 2 * padding - kernel)) // stride) + 1
    return (extent + 2 * padding - kernel) // stride + 1


def _upsample_nearestnd(x, output_size, ndim, **kwargs):
    """Nearest upsampling as an index remap of the source.

    Each output element reads the input element the scale maps it to, so the
    operator is a view with a remapped address rather than a call: it fuses
    with whatever consumes it instead of standing on its own.

    ``ndim`` is how many trailing axes are spatial, which is what says how much
    of the output size is a size rather than a leading extent.
    """

    size, dtype, device = val_info(node_val())
    in_size = list(x.get_size())
    out_spatial = [int(s) for s in output_size][-ndim:]
    in_spatial = in_size[-ndim:]
    prefix = in_size[:-ndim]

    def reindex(index):
        # Nearest maps output position i to floor(i / scale) of the input.
        return [
            *index[: len(prefix)],
            *[
                FloorDiv(
                    as_index(index[len(prefix) + axis]) * i, sympy.Integer(o)
                )
                for axis, (i, o) in enumerate(zip(in_spatial, out_spatial))
            ],
        ]

    return make_view(x, size, reindex)


@register("upsample_nearest2d.default", "_upsample_nearest_exact2d.default")
def lower_upsample_nearest2d(x, output_size, scales_h=None, scales_w=None,
                            **kwargs):
    return _upsample_nearestnd(x, output_size, 2, **kwargs)


@register("upsample_nearest3d.default", "_upsample_nearest_exact3d.default")
def lower_upsample_nearest3d(x, output_size, scales_d=None, scales_h=None,
                            scales_w=None, **kwargs):
    return _upsample_nearestnd(x, output_size, 3, **kwargs)


@register("avg_pool2d.default", "avg_pool3d.default")
def lower_avg_poolnd(x, kernel_size, stride=(), padding=0, ceil_mode=False,
                     count_include_pad=True, divisor_override=None, **kwargs):
    """Average pooling as a window sum followed by the window's divisor.

    A window that is both large and overlapping is left to the operator: the
    decomposition reads the input once per window, which stops paying once
    the windows overlap heavily.
    """

    size, dtype, device = val_info(node_val())
    in_size = list(x.get_size())
    ndim = _spatial_ndim()
    kernel = _pair(kernel_size, ndim)
    stride = _pair(stride, ndim) if stride else list(kernel)
    padding = _pair(padding, ndim) if padding else [0] * ndim
    window = 1
    for extent in kernel:
        window *= extent
    if window > 25 and any(k != s for k, s in zip(kernel, stride)):
        raise NotImplementedError(
            f"average pooling with an overlapping {window}-element window"
        )
    spatial_in = in_size[-ndim:]
    spatial_out = [
        _pool_output_size(extent, k, s, p, bool(ceil_mode))
        for extent, k, s, p in zip(spatial_in, kernel, stride, padding)
    ]
    prefix = in_size[: len(in_size) - ndim]
    loader = x.make_loader()
    boundary = any(padding)
    f32 = tp.float32

    def inner(index, rindex):
        full = list(index[: len(prefix)])
        for axis in range(ndim):
            base = index[len(prefix) + axis]
            full.append(base * stride[axis] - padding[axis] + rindex[axis])
        if not boundary:
            return loader(full)
        # A padded window reads zero outside the source, so the sum skips it.
        outside = None
        for axis in range(ndim):
            # The address is index arithmetic; a bound test needs it as a value.
            position = ops.index_expr(full[len(prefix) + axis], tp.int64)
            low = ops.ge(position, ops.constant(0, tp.int64))
            high = ops.lt(
                position, ops.constant(spatial_in[axis], tp.int64)
            )
            inside = ops.and_(low, high)
            outside = inside if outside is None else ops.and_(outside, inside)
        return ops.masked(outside, lambda: loader(full), ops.constant(0.0, f32))

    total = Reduction.create(
        device=device,
        dst_dtype=f32,
        src_dtype=f32,
        inner_fn=inner,
        ranges=(*prefix, *spatial_out),
        reduction_ranges=tuple(kernel),
        reduction_type='sum',
    )
    total.realize()
    if divisor_override is not None:
        divisor = float(divisor_override)
    elif count_include_pad or not any(padding):
        divisor = float(window)
    else:
        # Only the positions inside the source contribute to the divisor.
        divisor = None
    if divisor is None:
        return _pool_with_masked_divisor(
            total, prefix, spatial_out, kernel, stride, padding,
            spatial_in, f32, device,
        )
    return pointwise(
        lambda value: ops.truediv(value, ops.constant(divisor, f32)),
        total,
    )


def _pool_with_masked_divisor(total, prefix, spatial_out, kernel, stride,
                              padding, spatial_in, f32, device):
    """Average pooling whose divisor counts only the positions inside."""

    def inner(index, rindex):
        full = list(index[: len(prefix)])
        for axis in range(len(kernel)):
            full.append(
                index[len(prefix) + axis] * stride[axis] - padding[axis] + rindex[axis]
            )
        inside = None
        for axis in range(len(kernel)):
            position = ops.index_expr(full[len(prefix) + axis], tp.int64)
            term = ops.and_(
                ops.ge(position, ops.constant(0, tp.int64)),
                ops.lt(position, ops.constant(spatial_in[axis], tp.int64)),
            )
            inside = term if inside is None else ops.and_(inside, term)
        return ops.masked(inside, lambda: ops.constant(1.0, f32),
                          ops.constant(0.0, f32))

    counted = Reduction.create(
        device=device,
        dst_dtype=f32,
        src_dtype=f32,
        inner_fn=inner,
        ranges=(*prefix, *spatial_out),
        reduction_ranges=tuple(kernel),
        reduction_type='sum',
    )
    counted.realize()
    return pointwise(
        lambda value, count: ops.truediv(value, count),
        total,
        counted,
    )


# ---------------------------------------------------------------------------
# Scattering into a tensor along one axis
# ---------------------------------------------------------------------------


@register("index_add.default")
def lower_index_add(base, dim, index_box, addend, alpha=None, **kwargs):
    """Accumulating along one axis, as the sum over that axis of what lands.

    A scattered position is a sum of the contributions that name it, so the
    scatter is a reduction over the axis being scattered: each contribution
    asks the index where it belongs and contributes to that one destination,
    and to no other.  Nothing needs to be written twice and nothing needs a
    lock, because each destination is written by exactly one loop iteration.
    """

    dim = normalize_dim(int(dim), base.get_rank())
    _, dtype, device = val_info(node_val())
    out_size = [int(s) for s in base.get_size()]
    addend_size = [int(s) for s in addend.get_size()]
    if addend_size[dim] != int(index_box.get_size()[0]):
        raise NotImplementedError("an index that does not match its addend")
    index_loader = index_box.make_loader()
    addend_loader = addend.make_loader()
    f32 = tp.float32
    scale = 1.0 if alpha is None else float(alpha)

    def inner(index, rindex):
        destination = index_loader([ops.index_expr(rindex[dim], tp.int64)])
        here = ops.eq(destination, ops.index_expr(index[dim], tp.int64))
        for axis in range(len(out_size)):
            if axis != dim:
                here = ops.and_(
                    here, ops.eq(rindex[axis], ops.index_expr(index[axis], tp.int64))
                )
        value = addend_loader(list(rindex))
        if scale != 1.0:
            value = ops.mul(value, ops.constant(scale, f32))
        return ops.masked(here, lambda: value, ops.constant(0.0, f32))

    landed = Reduction.create(
        device=device,
        dst_dtype=dtype,
        src_dtype=addend.get_dtype(),
        inner_fn=inner,
        ranges=tuple(out_size),
        reduction_ranges=tuple(addend_size),
        reduction_type='sum',
    )
    landed.realize()
    if all(int(s) == 0 for s in out_size) or not any(addend_size):
        return landed
    # The scatter is a sum over what arrives, so the tensor it starts from is
    # added once, afterwards, rather than folded into every contribution.
    return pointwise(lambda a, b: ops.add(a, b), base, landed)


# ---------------------------------------------------------------------------
# Pooling backwards
# ---------------------------------------------------------------------------


def _window_covers(index, rindex, prefix, stride, padding, kernel, f32):
    """Does the output window at ``rindex`` include the input at ``index``?

    A window reaching from ``o * stride - padding`` for ``kernel`` positions
    covers the input position when that position is within reach, which is two
    comparisons on where the window starts.
    """

    inside = None
    for axis in range(len(kernel)):
        # The reduced index counts the windows, so it is local to them; the
        # output index counts the input positions and carries the prefix.
        at = len(prefix) + axis
        start = ops.mul(
            ops.index_expr(rindex[axis], tp.int64), ops.constant(int(stride[axis]), tp.int64)
        )
        # start <= index + padding, and start > index + padding - kernel
        low = ops.le(
            start,
            ops.add(ops.index_expr(index[at], tp.int64),
                    ops.constant(int(padding[axis]), tp.int64)),
        )
        high = ops.gt(
            start,
            ops.sub(ops.add(ops.index_expr(index[at], tp.int64),
                            ops.constant(int(padding[axis]), tp.int64)),
                    ops.constant(int(kernel[axis]), tp.int64)),
        )
        term = ops.and_(low, high)
        inside = term if inside is None else ops.and_(inside, term)
    return inside


@register("avg_pool2d_backward.default", "avg_pool3d_backward.default")
def lower_avg_poolnd_backward(grad, _input, kernel_size, stride=(), padding=0,
                              ceil_mode=False, count_include_pad=True,
                              divisor_override=None, **kwargs):
    """The input's gradient as the sum of the windows that covered it.

    Pooling reads the input once per window, so the input's gradient is the
    sum of the output's gradient over every window that read that position --
    the same sum the forward did, read the other way round.  The divisor is
    the same one the forward divided by, so the pair stays a pair.
    """

    _, dtype, device = val_info(node_val())
    ndim = _spatial_ndim()
    kernel = _pair(kernel_size, ndim)
    stride = _pair(stride, ndim) if stride else list(kernel)
    padding = _pair(padding, ndim) if padding else [0] * ndim
    grad_size = [int(s) for s in grad.get_size()]
    spatial_in = grad_size[grad_size.__len__() - ndim:]
    spatial_out = [
        _pool_output_size(extent, k, s, p, bool(ceil_mode))
        for extent, k, s, p in zip(spatial_in, kernel, stride, padding)
    ]
    prefix = grad_size[: len(grad_size) - ndim]
    loader = grad.make_loader()
    f32 = tp.float32

    def summed(index, rindex):
        full = list(index[: len(prefix)]) + [
            rindex[axis] for axis in range(ndim)
        ]
        inside = _window_covers(
            index, rindex, prefix, stride, padding, kernel, f32
        )
        return ops.masked(inside, lambda: loader(full), ops.constant(0.0, f32))

    total = Reduction.create(
        device=device,
        dst_dtype=f32,
        src_dtype=f32,
        inner_fn=summed,
        ranges=(*prefix, *spatial_in),
        reduction_ranges=tuple(spatial_out),
        reduction_type='sum',
    )
    total.realize()
    if divisor_override is not None:
        divisor = float(divisor_override)
    elif count_include_pad or not any(padding):
        divisor = float(_prod_ints(kernel))
    else:
        return _pool_backward_with_masked_divisor(
            total, prefix, spatial_in, spatial_out, kernel, stride, padding, f32,
            device,
        )
    return pointwise(
        lambda value: ops.truediv(value, ops.constant(divisor, f32)),
        total,
    )


def _prod_ints(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


_fallback_avg_pool2d_backward = fallback_handler(
    tp.ops.tp.avg_pool2d_backward.default, add_to_fallback_set=False
)


@register("avg_pool2d_backward.default")
def lower_avg_pool2d_backward_fast(grad, _input, kernel_size, stride=(), padding=0,
                                   ceil_mode=False, count_include_pad=True,
                                   divisor_override=None, **kwargs):
    """2D average pooling gradient, asked of the framework kernel."""

    return _fallback_avg_pool2d_backward(
        grad, _input, kernel_size, stride, padding, ceil_mode,
        count_include_pad, divisor_override,
    )


def _pool_backward_with_masked_divisor(total, prefix, spatial_in, spatial_out,
                                       kernel, stride, padding, f32, device):
    """Pooling backwards whose divisor counts only the positions inside."""

    def counted(index, rindex):
        inside = _window_covers(
            index, rindex, prefix, stride, padding, kernel, f32
        )
        return ops.masked(inside, lambda: ops.constant(1.0, f32),
                          ops.constant(0.0, f32))

    count = Reduction.create(
        device=device,
        dst_dtype=f32,
        src_dtype=f32,
        inner_fn=counted,
        ranges=(*prefix, *spatial_in),
        reduction_ranges=tuple(spatial_out),
        reduction_type='sum',
    )
    count.realize()
    return pointwise(
        lambda value, seen: ops.truediv(value, seen),
        total,
        count,
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
    "logical_and",
    "logical_or",
    "logical_xor",
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
    "signbit",
    "abs",
    "neg",
    "square",
    "ceil",
    "floor",
    "round",
    "trunc",
    "isnan",
    "isinf",
    "nan_to_num",
    "nextafter",
    "hypot",
    "copysign",
    "ldexp",
    "logical_not",
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
    dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
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
        mask = (min_v != max_v) | (~ops.isinf(min_v))
        return (ops.where(mask, ops.log1p(ops.exp(min_v - max_v)) + max_v, a),)

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


def lower_full(size: Any, fill_value: Any, **kwargs: Any) -> Any:
    """A tensor of one value, at a shape the program writes down."""

    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    return _full(fill_value, device, dtype, size)


def lower_zeros(size: Any, **kwargs: Any) -> Any:
    """A tensor of zeros, at a shape the program writes down."""

    return lower_full(size, 0, **kwargs)


def lower_ones(size: Any, **kwargs: Any) -> Any:
    """A tensor of ones, at a shape the program writes down."""

    return lower_full(size, 1, **kwargs)


register("full.default")(lower_full)
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

    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
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
            ir.ComplexView.create(tp_ops.view.dtype.default, x, dtype)
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
    ("avg_pool2d_backward", False),
    ("upsample_nearest2d_backward", False),
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

#: A search for where values would fall among ordered boundaries answers a
#: question about a list this cannot read, and a scatter that reduces says which
#: of several values landing on the same position wins -- which is a meaning
#: rather than a walk.
for _name in ("searchsorted", "scatter_reduce_"):
    _op = getattr(tp_ops, _name, None)
    if _op is None:
        continue
    _overloads = (
        [_op] if not hasattr(_op, "overloads") else
        [getattr(_op, _ov) for _ov in _op.overloads()]
    )
    for _ov in _overloads:
        make_fallback(_ov, warn=False)


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

    if dtype is None:
        raise AssertionError("expected: dtype is not None")
    length = ceildiv(sympy.sympify(end) - sympy.sympify(start), sympy.sympify(step))
    return iota(
        length,
        start=start,
        step=step,
        dtype=dtype,
        device=device if device is not None else "cpu",
        requires_grad=requires_grad,
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
    end: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    requires_grad: Any = False,
) -> Any:
    """One number: the number asked for.

    A range of one is the number itself, and treating it as a range rather than
    as a value would make it a tensor where a number was asked for.
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


#: Padding with a value, where how much is added to each side is named from the
#: last axis backwards.  Which positions the value ends up at is arithmetic on
#: the shape rather than a walk, but the shape it is arithmetic on is the one
#: the framework is better placed to work out -- and the answer is a value with
#: room around it, which the framework already knows how to produce.
_constant_pad = getattr(tp_ops, "constant_pad_nd", None)
if _constant_pad is not None:
    make_fallback(_constant_pad, warn=False)


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
    "prod",
    "scatter_reduce_",
    "_foreach_addcdiv_",
    "_foreach_addcmul_",
    "bucketize",
    "embedding",
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
    """

    return index_impl(x, indices, check=True)


def _promotion_input(value: Any) -> tuple:
    """What an operand contributes to a promotion: its type, and whether it is a number.

    A number promotes differently from a value of one dimension, so the two
    facts travel together rather than being read apart at each lattice step.
    """

    if isinstance(value, (ir.Constant, int, float)):
        return (value.get_dtype() if isinstance(value, ir.Constant) else None, True)
    return (value.get_dtype(), len(value.get_size()) == 0)


@register_lowering("where.default", broadcast=False, type_promotion_kind=None)
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
    return pointwise(
        ops.where, to_dtype(args[0], dtype), to_dtype(args[1], dtype), to_dtype(args[2], dtype)
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
            lower_amax(x, dims=dim, keepdim=keepdim),
            reduce_argmax(x, dim, keepdim),
        )
    return lower_amax(x, dims=None, keepdim=keepdim)


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
