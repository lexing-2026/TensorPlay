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

import functools
import itertools
import logging
import math
import operator
from numbers import Number
from typing import Any, Callable

import sympy

import tensorplay as tp

from .utils import register_op_dtype_propagation_rules

from tensorplay.primitives.common import ELEMENTWISE_TYPE_PROMOTION_KIND
from .dtype_propagation import get_promoted_dtype
from tensorplay.utils._pytree import tree_map

from .ir import (
    Buffer,
    Constant,
    IndexingConstant,
    DeviceCopy,
    ExpandView,
    FixedLayout,
    FallbackKernel,
    IRNode,
    PermuteView,
    Pointwise,
    Reduction,
    ReinterpretView,
    SliceView,
    StorageBox,
    TensorBox,
    View,
    has_free_unbacked_symbols,
    ops_wrapper,
    validate_ir,
)
from .loops import (
    V,
    as_index,
    contiguous_strides,
    dtype_name,
    floordiv,
    modular_indexing,
    ops,
    prod,
)

log = logging.getLogger(__name__)

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
    from tensorplay._decomp import decomposition_table

    return dict(decomposition_table)


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
            dtype = get_promoted_dtype(*promoting_args, type_promotion_kind=type_promotion_kind)
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
            elif isinstance(arg, ir_Constant):
                return ir_Constant(value=arg.value, dtype=dtype, device=device)
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
            if isinstance(args[i], ir_Constant):
                args[i] = ExpandView.create(args[i], size)
        for k in kwargs:
            if isinstance(kwargs[k], ir_Constant):
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
    val = node_val() if val is None else val
    size, dtype, device = val_info(val)
    loaders = [
        as_value_node(x, dtype, device).make_loader() for x in inputs
    ]

    def inner(index):
        return fn(*[load(index) for load in loaders])

    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner, ranges=size)


def cast_to(value, dtype):
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


@register("add.Tensor", "add.Scalar")
def lower_add(a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(ops.add, a, b)
    return pointwise(lambda x, y: ops.add(x, ops.mul(y, ops.constant(float(alpha), "float32"))), a, b)


@register("sub.Tensor", "sub.Scalar")
def lower_sub(a, b, *rest, **kwargs):
    alpha = _alpha((a, b, *rest), kwargs, 2)
    if alpha == 1:
        return pointwise(ops.sub, a, b)
    return pointwise(lambda x, y: ops.sub(x, ops.mul(y, ops.constant(float(alpha), "float32"))), a, b)


@register("rsub.Scalar", "rsub.Tensor")
def lower_rsub(a, b, *rest, **kwargs):
    return pointwise(lambda x, y: ops.sub(y, x), a, b)


@register("mul.Tensor", "mul.Scalar")
def lower_mul(a, b):
    return pointwise(ops.mul, a, b)


@register("div.Tensor", "div.Scalar")
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
}.items():
    LOWERINGS[_name] = _unary(_op)


@register("silu.default")
def lower_silu(x):
    # silu(x) = x * sigmoid(x)
    return pointwise(lambda v: ops.mul(v, ops.sigmoid(v)), x)


@register("silu_backward.default")
def lower_silu_backward(grad, x):
    # grad * s * (1 + x * (1 - s)),  s = sigmoid(x)
    def fn(g, v):
        s = ops.sigmoid(v)
        one = ops.constant(1.0, "float32")
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
    return View(data=_underlying(x), size=size, reindex=reindex)


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
            out.append(modular_indexing(flat, stride, int(extent)) if out else floordiv(flat, sympy.Integer(stride)))
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
    return lower_permute(node, grad, inverse)


@register("transpose.default", "transpose.int")
def lower_transpose(x, d0, d1):
    rank = len(x.get_size())
    dims = list(range(rank))
    a, b = normalize_dim(d0, rank), normalize_dim(d1, rank)
    dims[a], dims[b] = dims[b], dims[a]
    return lower_permute(node, x, dims)


@register("t.default")
def lower_t(x):
    return lower_permute(node, x, list(reversed(range(len(x.get_size())))))


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
        position = ops.index_expr(index[dim], "int64")
        value = None
        for k in range(len(inputs)):
            lo = starts[k]
            hi = lo + int(inputs[k].get_size()[dim])
            shifted = list(index)
            shifted[dim] = index[dim] - lo
            if k == 0:
                cond = ops.lt(position, ops.constant(hi, "int64"))
            elif k == len(inputs) - 1:
                cond = ops.ge(position, ops.constant(lo, "int64"))
            else:
                cond = ops.and_(
                    ops.ge(position, ops.constant(lo, "int64")),
                    ops.lt(position, ops.constant(hi, "int64")),
                )
            loaded = ops.masked(cond, lambda k=k, shifted=shifted: loaders[k](shifted), 0.0)
            value = loaded if value is None else ops.where(cond, loaded, value)
        return value

    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner, ranges=size)


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


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
    # A body is only ever recorded over a value that is settled: the loader
    # reads memory through the value's layout, and a layout that may still
    # change is not one to read through.  Realizing settles it.
    x.realize()
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
    box.realize()
    if keepdim:
        kept = [1 if d in dims else size[d] for d in range(rank)]

        def reindex(index):
            return [index[d] for d in range(rank) if d not in dims]

        return make_view(box, kept, reindex)
    return box


@register("sum.dim_IntList", "sum.default")
def lower_sum(x, dims=None, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val())
    if not dims:
        dims = list(range(len(x.get_size())))
    return make_reduction(x, dims, keepdim, dtype, device, "sum")


@register("mean.dim")
def lower_mean(x, dims, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val())
    count = prod(x.get_size()[normalize_dim(d, len(x.get_size()))] for d in dims)
    total = make_reduction(x, dims, keepdim, dtype, device, "sum")
    return pointwise(lambda v: ops.truediv(v, ops.constant(float(count), "float32")), total)


@register("amax.default")
def lower_amax(x, dims=None, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val())
    if not dims:
        dims = list(range(len(x.get_size())))
    return make_reduction(x, dims, keepdim, dtype, device, "max")


@register("amin.default")
def lower_amin(x, dims=None, keepdim=False, **kwargs):
    size, dtype, device = val_info(node_val())
    if not dims:
        dims = list(range(len(x.get_size())))
    return make_reduction(x, dims, keepdim, dtype, device, "min")


@register("conv2d_grad_bias.default", "conv_grad_bias.default")
def lower_conv_grad_bias(grad_out, *args):
    """The bias gradient is the output gradient summed over all but channels."""

    size, dtype, device = val_info(node_val())
    rank = len(grad_out.get_size())
    return make_reduction(grad_out, [0] + list(range(2, rank)), False, dtype, device, "sum")


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def _group_view(x: TensorBox, n, groups, row):
    """``x`` addressed as (N, G, C/G * spatial) rows."""

    return reshape(x, (n, groups, row))


@register("native_group_norm.default")
def lower_native_group_norm(x, weight, bias, n, c, hxw, groups, eps):
    out_val, mean_val, rstd_val = node_val()
    out_size, out_dtype, device = val_info(out_val)
    stat_dtype = mean_val.dtype
    cpg = c // groups
    row = cpg * hxw
    rows = _group_view(x, n, groups, row)
    rows_loader = rows.make_loader()

    def welford_inner(index, rindex):
        return ops.to_dtype(rows_loader([index[0], index[1], rindex[0]]), "float32")

    stats = Reduction.create(
        device=device,
        dst_dtype=stat_dtype,
        src_dtype=stat_dtype,
        inner_fn=welford_inner,
        ranges=(n, groups),
        reduction_ranges=(row,),
        reduction_type='welford',
    )
    mean_buf, m2_buf = V.graph.register_welford(_underlying(stats))
    mean_box = TensorBox(mean_buf)
    m2_loader = TensorBox(m2_buf).make_loader()
    mean_loader = mean_box.make_loader()

    def rstd_at(ng):
        var = ops.truediv(m2_loader(ng), ops.constant(float(row), "float32"))
        return ops.rsqrt(ops.add(var, ops.constant(float(eps), "float32")))

    rstd_box = Pointwise.create(
            device=device,
            dtype=stat_dtype,
            inner_fn=lambda idx: rstd_at(idx),
            ranges=(n, groups),
        )
    x_loader = x.make_loader()
    w_loader = weight.make_loader() if is_tensor_box(weight) else None
    b_loader = bias.make_loader() if is_tensor_box(bias) else None
    spatial = list(out_size[2:])

    def out_inner(index):
        channel = index[1]
        group = floordiv(as_index(channel), sympy.Integer(cpg))
        ng = [index[0], group]
        value = ops.to_dtype(x_loader(index), "float32")
        value = ops.mul(ops.sub(value, mean_loader(ng)), rstd_at(ng))
        if w_loader is not None:
            value = ops.mul(value, w_loader([channel]))
        if b_loader is not None:
            value = ops.add(value, b_loader([channel]))
        return value

    out = Pointwise.create(device=device, dtype=out_dtype, inner_fn=out_inner, ranges=out_size)
    return (out, mean_box, rstd_box)


@register("native_group_norm_backward.default")
def lower_native_group_norm_backward(grad_out, x, mean, rstd, gamma, n, c, hxw, groups, output_mask):
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
    f32 = "float32"

    # Per (n, c) sums over the spatial extent: ds = sum(dy * x), db = sum(dy).
    def full_index(index, rindex):
        s = rindex[0]
        return [index[0], index[1]] + _unflatten_index(s, spatial)

    def ds_inner(index, rindex):
        full = full_index(index, rindex)
        return ops.mul(ops.to_dtype(dy_loader(full), f32), ops.to_dtype(x_loader(full), f32))

    def db_inner(index, rindex):
        return ops.to_dtype(dy_loader(full_index(index, rindex)), f32)

    ds = Reduction.create(
        device=device,
        dtype='float32',
        inner_fn=ds_inner,
        ranges=(n, c),
        reduction_ranges=(hxw,),
        reduction_type='sum',
    )
    db = Reduction.create(
        device=device,
        dtype='float32',
        inner_fn=db_inner,
        ranges=(n, c),
        reduction_ranges=(hxw,),
        reduction_type='sum',
    )
    ds.realize()
    db.realize()
    ds_loader = ds.make_loader()
    db_loader = db.make_loader()

    def gamma_at(ch):
        return ops.to_dtype(gamma_loader([ch]), f32) if gamma_loader is not None else ops.constant(1.0, f32)

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
        dtype='float32',
        inner_fn=dsv_inner,
        ranges=(n, groups),
        reduction_ranges=(cpg,),
        reduction_type='sum',
    )
        db_val = Reduction.create(
        device=device,
        dtype='float32',
        inner_fn=dbv_inner,
        ranges=(n, groups),
        reduction_ranges=(cpg,),
        reduction_type='sum',
    )
        ds_val.realize()
        db_val.realize()
        dsv = ds_val.make_loader()
        dbv = db_val.make_loader()

        def c2_at(ng):
            r = ops.to_dtype(rstd_loader(ng), f32)
            m = ops.to_dtype(mean_loader(ng), f32)
            num = ops.sub(ops.mul(dbv(ng), m), dsv(ng))
            return ops.mul(ops.mul(ops.mul(ops.mul(num, r), r), r), ops.constant(s, f32))

        def c3_at(ng):
            r = ops.to_dtype(rstd_loader(ng), f32)
            m = ops.to_dtype(mean_loader(ng), f32)
            left = ops.mul(ops.neg(c2_at(ng)), m)
            right = ops.mul(ops.mul(dbv(ng), r), ops.constant(s, f32))
            return ops.sub(left, right)

        c2 = Pointwise.create(
                device=device, dtype="float32", inner_fn=c2_at, ranges=(n, groups)
        )
        c3 = Pointwise.create(
                device=device, dtype="float32", inner_fn=c3_at, ranges=(n, groups)
        )
        c2.realize()
        c3.realize()
        c2_loader = c2.make_loader()
        c3_loader = c3.make_loader()
        dx_size, dx_dtype, _ = val_info(vals[0])

        def dx_inner(index):
            ch = index[1]
            ng = [index[0], floordiv(as_index(ch), sympy.Integer(cpg))]
            c1 = ops.mul(ops.to_dtype(rstd_loader(ng), f32), gamma_at(ch))
            dy = ops.to_dtype(dy_loader(index), f32)
            xv = ops.to_dtype(x_loader(index), f32)
            return ops.add(ops.add(ops.mul(dy, c1), ops.mul(xv, c2_loader(ng))), c3_loader(ng))

        results[0] = Pointwise.create(device=device, dtype=dx_dtype, inner_fn=dx_inner, ranges=dx_size)
    if output_mask[1]:
        dg_size, dg_dtype, _ = val_info(vals[1])

        def dgamma_inner(index, rindex):
            ch = index[0]
            ng = [rindex[0], floordiv(as_index(ch), sympy.Integer(cpg))]
            m = ops.to_dtype(mean_loader(ng), f32)
            r = ops.to_dtype(rstd_loader(ng), f32)
            nc = [rindex[0], ch]
            return ops.mul(ops.sub(ds_loader(nc), ops.mul(db_loader(nc), m)), r)

        results[1] = Reduction.create(
        device=device,
        dst_dtype=dg_dtype,
        src_dtype=dg_dtype,
        inner_fn=dgamma_inner,
        ranges=(c,),
        reduction_ranges=(n,),
        reduction_type='sum',
    )
        results[1].realize()
    if output_mask[2]:
        db_size, db_dtype, _ = val_info(vals[2])

        def dbeta_inner(index, rindex):
            return db_loader([rindex[0], index[0]])

        results[2] = Reduction.create(
        device=device,
        dst_dtype=db_dtype,
        src_dtype=db_dtype,
        inner_fn=dbeta_inner,
        ranges=(c,),
        reduction_ranges=(n,),
        reduction_type='sum',
    )
        results[2].realize()
    return tuple(results)


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


def _pool_output_size(extent: int, kernel: int, stride: int, padding: int,
                      ceil_mode: bool) -> int:
    """Output extent of one pooling axis."""

    if ceil_mode:
        return -((-(extent + 2 * padding - kernel)) // stride) + 1
    return (extent + 2 * padding - kernel) // stride + 1


@register("upsample_nearest2d.default", "_upsample_nearest_exact2d.default",
          "upsample_nearest3d.default", "_upsample_nearest_exact3d.default")
def lower_upsample_nearestnd(x, output_size, scales_h=None, scales_w=None,
                             **kwargs):
    """Nearest upsampling as an index remap of the source.

    Each output element reads the input element the scale maps it to, so the
    operator is a view with a remapped address rather than a call: it fuses
    with whatever consumes it instead of standing on its own.
    """

    size, dtype, device = val_info(node_val())
    in_size = list(x.get_size())
    ndim = 3 if "3d" in target_name(node.target) else 2
    out_spatial = [int(s) for s in output_size][-ndim:]
    in_spatial = in_size[-ndim:]
    prefix = in_size[:-ndim]

    def reindex(index):
        # Nearest maps output position i to floor(i / scale) of the input.
        return [
            *index[: len(prefix)],
            *[
                floordiv(
                    as_index(index[len(prefix) + axis]) * i, sympy.Integer(o)
                )
                for axis, (i, o) in enumerate(zip(in_spatial, out_spatial))
            ],
        ]

    return make_view(x, size, reindex)


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
    ndim = 3 if "3d" in target_name(node.target) else 2
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
    f32 = "float32"

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
            position = ops.index_expr(full[len(prefix) + axis], "int64")
            low = ops.ge(position, ops.constant(0, "int64"))
            high = ops.lt(
                position, ops.constant(spatial_in[axis], "int64")
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
            node, total, prefix, spatial_out, kernel, stride, padding,
            spatial_in, f32, device,
        )
    return pointwise(
        node,
        lambda value: ops.truediv(value, ops.constant(divisor, f32)),
        total,
    )


def _pool_with_masked_divisor(node, total, prefix, spatial_out, kernel, stride,
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
            position = ops.index_expr(full[len(prefix) + axis], "int64")
            term = ops.and_(
                ops.ge(position, ops.constant(0, "int64")),
                ops.lt(position, ops.constant(spatial_in[axis], "int64")),
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
        node,
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
    f32 = "float32"
    scale = 1.0 if alpha is None else float(alpha)

    def inner(index, rindex):
        destination = index_loader([ops.index_expr(rindex[dim], "int64")])
        here = ops.eq(destination, ops.index_expr(index[dim], "int64"))
        for axis in range(len(out_size)):
            if axis != dim:
                here = ops.and_(
                    here, ops.eq(rindex[axis], ops.index_expr(index[axis], "int64"))
                )
        value = addend_loader(list(rindex))
        if scale != 1.0:
            value = ops.mul(value, ops.constant(scale, f32))
        return ops.masked(here, lambda: value, ops.constant(0.0, f32))

    landed = Reduction.create(
        device=device,
        dst_dtype=dtype,
        src_dtype=src_dtype,
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
            ops.index_expr(rindex[axis], "int64"), ops.constant(int(stride[axis]), "int64")
        )
        # start <= index + padding, and start > index + padding - kernel
        low = ops.le(
            start,
            ops.add(ops.index_expr(index[at], "int64"),
                    ops.constant(int(padding[axis]), "int64")),
        )
        high = ops.gt(
            start,
            ops.sub(ops.add(ops.index_expr(index[at], "int64"),
                            ops.constant(int(padding[axis]), "int64")),
                    ops.constant(int(kernel[axis]), "int64")),
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
    ndim = 3 if "3d" in target_name(node.target) else 2
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
    f32 = "float32"

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
            node, total, prefix, spatial_in, spatial_out, kernel, stride, padding, f32,
            device,
        )
    return pointwise(
        node,
        lambda value: ops.truediv(value, ops.constant(divisor, f32)),
        total,
    )


def _prod_ints(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _pool_backward_with_masked_divisor(node, total, prefix, spatial_in, spatial_out,
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
        node,
        lambda value, seen: ops.truediv(value, seen),
        total,
        count,
    )

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

