"""What type each operation produces, derived from the types it was given.

An operation is handed values rather than tensors, so it cannot ask one what
type it has.  This answers instead: the type of a result is the promotion of
its operands, and the handful of operations whose result is not the promotion
of anything -- a comparison, a cast, a constant, a read -- say what they
produce.

The rules are not written out one by one.  A rule is registered by the code
that defines the operation, and this builds a handler from the registry, so an
operation that needs a rule cannot be added without one and cannot be left
without one either: the handler's construction fails on an operation that has
no rule, which is where the mistake is cheapest to find.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from numbers import Number
from typing import Any, Optional, Protocol

import sympy

import tensorplay as tp

from tensorplay.graph.experimental.sympy_functions import OrderedSet
from tensorplay.primitives.common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    elementwise_dtypes,
    is_integer_dtype,
    type_to_dtype,
)
from .codegen.common import pointwise_overrides_data
from .loops import V
from .ops_handler import OP_NAMES
from .utils import (
    boolean_ops,
    op_dtype_propagation_rules,
    upcast_compute_type,
)

T = TypeVar = object  # the alias is only used in annotations below
_MISSING_SHAPE = object()
_UNSIGNED_INT_DTYPES = frozenset(
    (
        tp.uint8,
        tp.uint16,
        tp.uint32,
        tp.uint64,
    )
)


class DTypeVar(Protocol):
    """A value that knows its own type."""

    @property
    def dtype(self): ...


DTypeArg = Any


def promoted_dtype_of_values(
    *args: Any,
    type_promotion_kind=None,
    return_compute_dtype: bool = False,
):
    """The type the promotion of these values comes to.

    Asked of the values themselves rather than of a list of types, because a
    lowering is handed values and should not have to take them apart first: a
    value knows its own type and its own rank, and a number is already a type
    as far as the promotion is concerned.  The promotion is settled on a value
    of the right type and rank standing in for each, since what decides a
    promotion is the types involved and not how many elements there are.
    """

    def construct_input(inp):
        if isinstance(inp, (Number, sympy.Basic)):
            return inp
        dim = len(inp.get_size())
        return tp.zeros([1] * dim, dtype=inp.get_dtype())

    inps = [construct_input(arg) for arg in args]
    compute_dtype, result_dtype = elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return compute_dtype if return_compute_dtype else result_dtype


@functools.cache
def get_promoted_dtype(
    *args: Sequence,
    type_promotion_kind=None,
):
    """The type the promotion of these types comes to.

    The promotion is a lattice operation on types, and it is asked of the same
    place the region's own promotion is asked of, so that a value computed in a
    kernel and the same value computed outside it have the same type rather
    than two types that usually agree.
    """

    def construct_input(inp):
        if inp[1]:
            return tp.empty([], dtype=inp[0])
        else:
            return tp.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return dtype


def promote_types(args: Sequence, type_promotion_kind=None):
    """The type these operands promote to, as a rule rather than as a lattice step.

    A number among the operands carries the type of the number rather than a
    declared type, and a value that is a scalar promotes differently from one
    that is not, so each operand contributes a type and whether it is a scalar
    before the promotion is asked for.
    """

    dtype_prop_candidates = []

    for arg in args:
        if isinstance(arg, str):
            raise AssertionError(f"expected non-str arg, got {type(arg)}")
        value = getattr(arg, "value", arg)
        if not (isinstance(value, (int, float, complex, bool)) or hasattr(value, "dtype")):
            raise AssertionError("expected a number or a value with a type")

        if isinstance(value, (int, float, complex, bool)):
            dtype_prop_candidates.append((type_to_dtype(type(value)), True))
            continue

        dtype_prop_candidates.append((value.dtype, getattr(value, "is_scalar", False)))

    return get_promoted_dtype(
        *dtype_prop_candidates,
        type_promotion_kind=type_promotion_kind,
    )


def _unwrap_dtype_arg(arg):
    return getattr(arg, "value", arg)


def _is_scalar_dtype_arg(arg) -> bool:
    """Whether an operand is a scalar, which changes how it promotes.

    A scalar promotes as the number it is, and a value whose shape is not known
    is treated as a scalar because a shape that is not known cannot be told
    apart from one with no extents.
    """

    arg = _unwrap_dtype_arg(arg)
    if isinstance(arg, (int, float, complex, bool)):
        return True

    is_scalar = getattr(arg, "is_scalar", False)
    if callable(is_scalar):
        if is_scalar():
            return True
    elif is_scalar:
        return True

    shape = getattr(arg, "shape", _MISSING_SHAPE)
    if shape is _MISSING_SHAPE:
        return False

    if shape is None:
        return True
    if not isinstance(shape, Sequence):
        return False

    return len(shape) == 0


def _has_known_nonnegative_scalar_int_value(arg) -> bool:
    """Whether an integer scalar is known to be at least zero.

    An integer power of a negative exponent is a real number even when both
    operands are integers, so whether the exponent is known to be non-negative
    is what decides the type of the result.
    """

    arg = _unwrap_dtype_arg(arg)

    if isinstance(arg, bool):
        return True
    if isinstance(arg, int):
        return arg >= 0

    dtype = getattr(arg, "dtype", None)
    if dtype is None or not is_integer_dtype(dtype):
        return False
    if dtype in _UNSIGNED_INT_DTYPES:
        return True

    lower = getattr(getattr(arg, "bounds", None), "lower", None)
    if lower is None:
        return False
    if isinstance(lower, sympy.Expr):
        return lower.is_nonnegative is True
    return lower >= 0


class DtypePropagationOpsHandler:
    """The type each operation produces.

    The rules are built once and shared, because a compile asks for the type of
    an operation thousands of times and the rules do not change between two of
    those questions; the construction is repeated when the registry has grown,
    so a rule registered after the first use is not missed.
    """

    _instance: Optional["DtypePropagationOpsHandler"] = None
    _rules_key: Optional[tuple] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # The rules live where the operations are declared, so that a rule and
        # the operation it belongs to are declared in the same place and cannot
        # drift apart; importing it here is what makes this handler independent
        # of the order things happen to be imported in.
        import tensorplay.compiler.backends.stax.op_lowerings  # noqa: F401

        rules = op_dtype_propagation_rules
        key = tuple(rules.items())
        if key == DtypePropagationOpsHandler._rules_key:
            return

        for op, rule in rules.items():
            fn = (
                functools.partial(self.return_dtype, dtype=rule.override_return_dtype)
                if rule.override_return_dtype
                else functools.partial(
                    self.op_dtype_rule, type_promotion_kind=rule.type_promotion_kind
                )
            )
            setattr(self, op, fn)

        # The operations whose spelling is recorded per language also carry
        # their promotion, and one that has no rule of its own takes it from
        # there.
        for op, data in pointwise_overrides_data.items():
            if not hasattr(self, op):
                setattr(
                    self,
                    op,
                    functools.partial(
                        self.op_dtype_rule, type_promotion_kind=data.type_promotion_kind
                    ),
                )

        for op in boolean_ops():
            if not hasattr(self, op):
                setattr(self, op, functools.partial(self.return_dtype, dtype=tp.bool))

        unimplemented_ops = set(OP_NAMES) - set(dir(self))
        if unimplemented_ops:
            raise AssertionError(
                f"Unimplemented dtype rule for ops: {sorted(unimplemented_ops)}"
            )
        DtypePropagationOpsHandler._rules_key = key

    # The two rules every operation is built from.

    @staticmethod
    def op_dtype_rule(*args, type_promotion_kind):
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args, dtype):
        return dtype

    # The operations whose result is not the promotion of their operands.

    @staticmethod
    def constant(value, dtype):
        return upcast_compute_type(dtype)

    @staticmethod
    def load_seed(name: str, offset: int):
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def randint64(seed: int, offset: int, low: int, high: int):
        return tp.int64

    @staticmethod
    def masked(mask, body: Callable, other):
        """The type of a value read under a condition.

        With one value to choose from the answer is that value's type; with
        several, it is the type of the last one read, which is the one the
        region's own computation settled on.
        """

        loads = getattr(getattr(body, "graph", None), "find_nodes", None)
        found = loads(op="call_method", target="load") if loads else []
        if len(found) <= 1:
            return promote_types([other])
        return upcast_compute_type(V.graph.get_dtype(found[-1].args[1]))

    @staticmethod
    def where(a, b, c):
        """A value chosen between two others is their promotion, not a's.

        The condition does not contribute: it is a truth value, and a truth
        value does not raise the type of what it chooses.
        """

        return promote_types([b, c])

    @staticmethod
    def index_expr(expr, dtype):
        """An index is the type the kernel indexes by, not the type asked for.

        A kernel that indexes in 32 bits is handed 32-bit extents whatever the
        operation that produced the index asked for, because the index is
        about addressing rather than about arithmetic.  A use that has to
        honour the requested type is a value rather than an index.
        """

        if dtype not in (tp.int32, tp.int64) or not hasattr(V.kernel, "index_dtype"):
            return upcast_compute_type(dtype)

        index_dtype = V.kernel.index_dtype
        if index_dtype == "int64":
            return tp.int64
        if index_dtype == "int32":
            return tp.int32
        return upcast_compute_type(dtype)

    @staticmethod
    def value_expr(expr, dtype):
        """A value is the type asked for, with no promotion to a wider type.

        The expression is emitted as it stands, so a half-precision value stays
        half precision rather than being computed in single as an index would be.
        """

        return dtype

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        return upcast_compute_type(dtype) if use_compute_types else dtype

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        return upcast_compute_type(dtype)

    @staticmethod
    def gelu(x):
        return promote_types([x])

    @staticmethod
    def mul(a, b):
        return promote_types([a, b])

    @staticmethod
    def truediv(a, b):
        return promote_types([a, b])

    @staticmethod
    def div_rn(a, b):
        return promote_types([a, b])

    @staticmethod
    def pow(a, b):
        """An integer power of an integer base is an integer, unless the exponent
        may be negative.

        A negative exponent has no integer result, so a scalar integer power
        whose exponent is not known to be non-negative is a real number; a
        tensor of exponents goes down a different path, where the result stays
        integral or the operation is refused.
        """

        dtype = promote_types([a, b])
        if (
            is_integer_dtype(dtype)
            and _is_scalar_dtype_arg(a)
            and _is_scalar_dtype_arg(b)
            and not _has_known_nonnegative_scalar_int_value(b)
        ):
            return tp.float64
        return dtype

    @staticmethod
    def mod(a, b):
        return promote_types([a, b])

    @staticmethod
    def indirect_indexing(var, size, check: bool = True, wrap_neg: bool = True):
        return tp.int64

    @staticmethod
    def randn(seed: int, offset: int):
        return tp.float32

    @staticmethod
    def rand(seed: int, offset: int):
        return tp.float32

    @staticmethod
    def rand_eager(seed, offset, threads_per_round, tid, vec):
        return tp.float32

    @staticmethod
    def store_reduction(name: str, index, value) -> None:
        return None

    @staticmethod
    def reduction(dtype, src_dtype, reduction_type, value):
        return dtype

    @staticmethod
    def store(name: str, index, value, mode=None) -> None:
        return None

    @staticmethod
    def partial_accumulate(name, reduction_type, value, extra_meta) -> None:
        return None

    @staticmethod
    def load(name: str, index):
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def floor(x):
        return promote_types([x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

    @staticmethod
    def ceil_to_int(x, dtype):
        return dtype

    @staticmethod
    def int_truediv(x, y):
        return promote_types([x, y], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

    @staticmethod
    def scan(dtypes, combine_fn, values):
        return dtypes

    @staticmethod
    def fmod(x, y):
        return promote_types([x, y])

    @staticmethod
    def round_to_int(x, dtype):
        return dtype

    @staticmethod
    def identity(x):
        return promote_types([x])

    @staticmethod
    def frexp(x):
        """A value split into a fraction and an exponent, the exponent an integer."""

        return (promote_types([x]), tp.int32)

    @staticmethod
    def sort(dtypes, values, stable: bool, descending: bool):
        return dtypes

    @staticmethod
    def trunc(x):
        return promote_types([x])

    @staticmethod
    def bucketize(
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right: bool,
        sorter=None,
        sorter_indices=None,
    ):
        """The buckets are indices, so they are of the type the boundaries are indexed by."""

        return indexing_dtype

    @staticmethod
    def rshift(x, y):
        return promote_types([x])

    @staticmethod
    def round(x):
        return promote_types([x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

    @staticmethod
    def trunc_to_int(x, dtype):
        return dtype

    @staticmethod
    def floor_to_int(x, dtype):
        return dtype

    @staticmethod
    def truncdiv(x, y):
        return promote_types([x, y])

    @staticmethod
    def floordiv(x, y):
        return promote_types([x, y])

    @staticmethod
    def halide_clamp(value, size, check):
        return tp.int32

    @staticmethod
    def dot(x, y):
        """A product accumulates in single precision whatever it reads."""

        return tp.float32

    @staticmethod
    def inline_asm_elementwise(
        inputs, asm, constraints, dtype, is_pure, pack, input_dtypes
    ):
        return dtype

    @staticmethod
    def lshift(x, y):
        return promote_types([x, y])
