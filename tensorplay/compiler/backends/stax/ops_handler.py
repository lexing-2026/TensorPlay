"""The operators a region is written in, and the contract for handling them.

A region's body is a sequence of operations on values and nothing else: there
are no tensors here, only the operations and the values they produce.  This
module states what those operations are, so that a handler is written against
a list rather than against whatever the code happened to call, and so that a
handler that does not implement one of them says so at the point of the call
rather than failing somewhere later.

The operations are dtype-polymorphic -- the same one multiplies integers and
floats -- and they do not promote: an operation returns the type it was given.
Promotion is a decision about the region rather than about the operation, and
it has been made by the time an operation is reached.  They are all scalar,
meaning each is defined on one element at a time, and a handler that works on
a whole tile at once is free to apply one of them to the tile.

Many take the type of the value they are given so that a handler does not have
to work it out again; where an analysis has already established the type,
passing it is cheaper than deriving it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

import sympy

import tensorplay as tp
from tensorplay.utils import _pytree as pytree
from tensorplay.graph.experimental.sympy_functions import OrderedSet

from .utils import IndentedBuffer


def _arg_str(a) -> str:
    """One argument as it would be written back into the call that made it."""

    if isinstance(a, (int, float, bool)):
        return str(a)
    if isinstance(a, sympy.Expr):
        return str(a)
    return repr(a)


class OpsHandler:
    def constant(self, value: bool | float | int, dtype: dtype):
        raise NotImplementedError

    def load_seed(self, name: str, offset):
        raise NotImplementedError

    def rand(self, seed, offset):
        raise NotImplementedError

    def rand_eager(self, seed, base_offset, threads_per_round, tid, vec):
        raise NotImplementedError

    def randn(self, seed, offset):
        raise NotImplementedError

    def randint64(self, seed, offset, low, high):
        raise NotImplementedError

    def masked(self, mask, body: Callable[[], T], other):
        raise NotImplementedError

    def where(self, condition, input, other):
        raise NotImplementedError

    def index_expr(self, expr: expr, dtype: dtype):
        raise NotImplementedError

    def value_expr(self, expr: expr, dtype: dtype):
        raise NotImplementedError

    def to_dtype(self, x, dtype: dtype, src_dtype: dtype | None=None, use_compute_types: bool=True):
        raise NotImplementedError

    def trunc_to_int(self, x, dtype: dtype):
        raise NotImplementedError

    def ceil_to_int(self, x, dtype: dtype):
        raise NotImplementedError

    def floor_to_int(self, x, dtype: dtype):
        raise NotImplementedError

    def round_to_int(self, x, dtype: dtype):
        raise NotImplementedError

    def to_dtype_bitcast(self, x, dtype: dtype, src_dtype: dtype):
        raise NotImplementedError

    def identity(self, x):
        raise NotImplementedError

    def indirect_indexing(self, x, size: expr, check: bool=True, wrap_neg=True) -> "expr":
        raise NotImplementedError

    def load(self, name: str, index: expr):
        raise NotImplementedError

    def store(self, name: str, index: expr, value, mode: StoreMode=None) -> None:
        raise NotImplementedError

    def reduction(self, dtype: dtype, src_dtype: dtype, reduction_type: ReductionType, value) -> tuple[T, ...]:
        raise NotImplementedError

    def store_reduction(self, name: str, index: expr, value) -> None:
        raise NotImplementedError

    def scan(self, dtypes: tuple[dtype, ...], combine_fn: Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]], values: tuple[T, ...]) -> tuple[T, ...]:
        raise NotImplementedError

    def sort(self, dtypes: tuple[dtype, ...], values: tuple[T, ...], stable: bool, descending: bool) -> tuple[T, ...]:
        raise NotImplementedError

    def bucketize(self, values, boundaries: tuple[str, expr, expr, expr], boundary_indices, indexing_dtype: dtype, right: bool, sorter: tuple[str, expr] | None=None, sorter_indices: None=None):
        raise NotImplementedError

    def partial_accumulate(self, name: str, reduction_type: ReductionType, value, extra_meta: dict[str, Any]) -> None:
        raise NotImplementedError

    def abs(self, x0):
        raise NotImplementedError

    def exp(self, x0):
        raise NotImplementedError

    def exp2(self, x0):
        raise NotImplementedError

    def expm1(self, x0):
        raise NotImplementedError

    def sqrt(self, x0):
        raise NotImplementedError

    def relu(self, x0):
        raise NotImplementedError

    def minimum(self, x0, x1):
        raise NotImplementedError

    def maximum(self, x0, x1):
        raise NotImplementedError

    def fmaximum(self, x0, x1):
        raise NotImplementedError

    def cos(self, x0):
        raise NotImplementedError

    def sin(self, x0):
        raise NotImplementedError

    def lgamma(self, x0):
        raise NotImplementedError

    def erf(self, x0):
        raise NotImplementedError

    def cosh(self, x0):
        raise NotImplementedError

    def sinh(self, x0):
        raise NotImplementedError

    def acos(self, x0):
        raise NotImplementedError

    def acosh(self, x0):
        raise NotImplementedError

    def asin(self, x0):
        raise NotImplementedError

    def asinh(self, x0):
        raise NotImplementedError

    def atan2(self, x0, x1):
        raise NotImplementedError

    def atan(self, x0):
        raise NotImplementedError

    def atanh(self, x0):
        raise NotImplementedError

    def copysign(self, x0, x1):
        raise NotImplementedError

    def erfc(self, x0):
        raise NotImplementedError

    def erfinv(self, x0):
        raise NotImplementedError

    def frexp(self, x0):
        raise NotImplementedError

    def hypot(self, x0, x1):
        raise NotImplementedError

    def log10(self, x0):
        raise NotImplementedError

    def log2(self, x0):
        raise NotImplementedError

    def ldexp(self, x0, n):
        raise NotImplementedError

    def nextafter(self, x0, x1):
        raise NotImplementedError

    def logical_and(self, x0, x1):
        raise NotImplementedError

    def logical_not(self, x0):
        raise NotImplementedError

    def logical_or(self, x0, x1):
        raise NotImplementedError

    def logical_xor(self, x0, x1):
        raise NotImplementedError

    def bitwise_and(self, x0, x1):
        raise NotImplementedError

    def bitwise_not(self, x0):
        raise NotImplementedError

    def bitwise_or(self, x0, x1):
        raise NotImplementedError

    def bitwise_xor(self, x0, x1):
        raise NotImplementedError

    def bitwise_left_shift(self, x0, x1):
        raise NotImplementedError

    def bitwise_right_shift(self, x0, x1):
        raise NotImplementedError

    def rsqrt(self, x0):
        raise NotImplementedError

    def log1p(self, x0):
        raise NotImplementedError

    def tan(self, x0):
        raise NotImplementedError

    def tanh(self, x0):
        raise NotImplementedError

    def sigmoid(self, x0):
        raise NotImplementedError

    def signbit(self, x0):
        raise NotImplementedError

    def fmod(self, x0, x1):
        raise NotImplementedError

    def log(self, x0):
        raise NotImplementedError

    def isinf(self, x0):
        raise NotImplementedError

    def isnan(self, x0):
        raise NotImplementedError

    def round(self, x0):
        raise NotImplementedError

    def floor(self, x0):
        raise NotImplementedError

    def sign(self, x0):
        raise NotImplementedError

    def trunc(self, x0):
        raise NotImplementedError

    def ceil(self, x0):
        raise NotImplementedError

    def neg(self, x0):
        raise NotImplementedError

    def reciprocal(self, x0):
        raise NotImplementedError

    def eq(self, x0, x1):
        raise NotImplementedError

    def ne(self, x0, x1):
        raise NotImplementedError

    def lt(self, x0, x1):
        raise NotImplementedError

    def gt(self, x0, x1):
        raise NotImplementedError

    def le(self, x0, x1):
        raise NotImplementedError

    def ge(self, x0, x1):
        raise NotImplementedError

    def add(self, x0, x1):
        raise NotImplementedError

    def sub(self, x0, x1):
        raise NotImplementedError

    def mul(self, x0, x1):
        raise NotImplementedError

    def pow(self, x0, x1):
        raise NotImplementedError

    def and_(self, x0, x1):
        raise NotImplementedError

    def or_(self, x0, x1):
        raise NotImplementedError

    def xor(self, x0, x1):
        raise NotImplementedError

    def lshift(self, x0, x1):
        raise NotImplementedError

    def rshift(self, x0, x1):
        raise NotImplementedError

    def airy_ai(self, x):
        raise NotImplementedError

    def bessel_j0(self, x):
        raise NotImplementedError

    def bessel_j1(self, x):
        raise NotImplementedError

    def bessel_y0(self, x):
        raise NotImplementedError

    def bessel_y1(self, x):
        raise NotImplementedError

    def digamma(self, x):
        raise NotImplementedError

    def erfcx(self, x):
        raise NotImplementedError

    def fma(self, x, y, z):
        raise NotImplementedError

    def mul_rn(self, x, y):
        raise NotImplementedError

    def igamma(self, x, y):
        raise NotImplementedError

    def igammac(self, x, y):
        raise NotImplementedError

    def gammainc(self, x, y):
        raise NotImplementedError

    def gammaincc(self, x, y):
        raise NotImplementedError

    def i0(self, x):
        raise NotImplementedError

    def i0e(self, x):
        raise NotImplementedError

    def i1(self, x):
        raise NotImplementedError

    def i1e(self, x):
        raise NotImplementedError

    def log_ndtr(self, x):
        raise NotImplementedError

    def modified_bessel_i0(self, x):
        raise NotImplementedError

    def modified_bessel_i1(self, x):
        raise NotImplementedError

    def modified_bessel_k0(self, x):
        raise NotImplementedError

    def modified_bessel_k1(self, x):
        raise NotImplementedError

    def ndtr(self, x):
        raise NotImplementedError

    def ndtri(self, x):
        raise NotImplementedError

    def polygamma(self, x, y):
        raise NotImplementedError

    def scaled_modified_bessel_k0(self, x):
        raise NotImplementedError

    def scaled_modified_bessel_k1(self, x):
        raise NotImplementedError

    def spherical_bessel_j0(self, x):
        raise NotImplementedError

    def zeta(self, x, y):
        raise NotImplementedError

    def chebyshev_polynomial_t(self, x, y):
        raise NotImplementedError

    def chebyshev_polynomial_u(self, x, y):
        raise NotImplementedError

    def chebyshev_polynomial_v(self, x, y):
        raise NotImplementedError

    def chebyshev_polynomial_w(self, x, y):
        raise NotImplementedError

    def legendre_polynomial_p(self, x, y):
        raise NotImplementedError

    def shifted_chebyshev_polynomial_t(self, x, y):
        raise NotImplementedError

    def shifted_chebyshev_polynomial_u(self, x, y):
        raise NotImplementedError

    def shifted_chebyshev_polynomial_v(self, x, y):
        raise NotImplementedError

    def shifted_chebyshev_polynomial_w(self, x, y):
        raise NotImplementedError

    def hermite_polynomial_h(self, x, y):
        raise NotImplementedError

    def hermite_polynomial_he(self, x, y):
        raise NotImplementedError

    def laguerre_polynomial_l(self, x, y):
        raise NotImplementedError

    def truncdiv(self, x0, x1):
        raise NotImplementedError

    def floordiv(self, x0, x1):
        raise NotImplementedError

    def truediv(self, x0, x1):
        raise NotImplementedError

    def div_rn(self, x0, x1):
        raise NotImplementedError

    def int_truediv(self, x0, x1):
        raise NotImplementedError

    def mod(self, x0, x1):
        raise NotImplementedError

    def remainder(self, x0, x1):
        raise NotImplementedError

    def square(self, x0):
        raise NotImplementedError

    def check_bounds(self, expr: expr, size: expr, lower: bool, upper: bool) -> None:
        raise NotImplementedError

    def halide_clamp(self, value, size: expr, check: bool):
        raise NotImplementedError

    def dot(self, x, y):
        raise NotImplementedError

    def inline_asm_elementwise(self, *inputs, asm: str, constraints: str | None=None, dtype: dtype=tp.float32, is_pure: bool=True, pack: int=1, input_dtypes: tuple | None=None):
        raise NotImplementedError

    def output(self, *args) -> None:
        raise NotImplementedError

    def placeholder(self, index: int):
        raise NotImplementedError

    def device_assert_async(self, cond, msg: str):
        raise NotImplementedError

#: Every operation the handler contract names.  A handler is written against
#: this list, and the forwarding methods are built from it, so an operation
#: added to the contract is reachable without being added to the handler too.
OP_NAMES = [name for name in dir(OpsHandler) if not name.startswith("_")]


class NullHandler:
    """The handler that answers nothing, used where there is no handler at all.

    Two places want this: a region being analysed rather than emitted, and a
    piece of code that runs outside any region's context.  Both need an answer
    to "who is handling operations" that is not an error, and both need that
    answer to be a question rather than a value.
    """


class NullKernelHandler(NullHandler):
    """The kernel that is there when no kernel is being emitted.

    A name is looked up in the kernel's two tables whether or not a kernel is
    being written -- the wrapper is written with no kernel in context -- so the
    tables have to exist before one does.  Having them here rather than being
    looked up with a default means a misspelling is an error instead of an
    empty answer.
    """

    def __init__(self):
        super().__init__()
        from tensorplay.graph.experimental.sympy_functions import OrderedSet

        self.removed_buffers: OrderedSet = OrderedSet()
        self.inplaced_to_remove: OrderedSet = OrderedSet()
        self.index_dtype = "int64"

    def get_index_dtype_as_dtype(self):
        """The type an index is written as, as a type rather than as a name."""

        import tensorplay as tp

        if self.index_dtype == "int64":
            return tp.int64
        if self.index_dtype == "int32":
            return tp.int32
        raise ValueError(f"Unknown dtype: {self.index_dtype}")


class DefaultHandler(OpsHandler):
    """A handler that answers every operation the same way.

    A subclass says what the answer is by overriding one method and naming the
    operation in the call, which is what lets a handler that implements three
    operations be three lines long instead of three methods.  The forwarding
    methods themselves are generated rather than written, because there are a
    hundred and fifty of them and their only content is the operation's name.
    """

    def _default(self, name, args, kwargs):
        """The answer for one operation, given its name and its arguments."""

        raise NotImplementedError

    def __getattr__(self, name):
        """An operation this handler does not implement, answered the default way.

        Reaching here means the contract names an operation the handler has no
        answer for, which is a hole rather than a mistake in the call, so it is
        reported as one.
        """

        import warnings

        def fallback(*args, **kwargs):
            return self._default(name, args, kwargs)

        warnings.warn(
            f"undefined handler operation {name}, the handler has no answer for it",
            stacklevel=2,
        )
        return fallback

    @staticmethod
    def _call_default(target: str):
        """One forwarding method, for an operation whose signature is not plain."""

        def call_default(self, *args, **kwargs):
            return self._default(target, args, kwargs)

        call_default.__name__ = target
        return call_default

    @classmethod
    def _init_cls(cls):
        """Build a forwarding method for every operation, once per class.

        An operation whose parameters are all plain ones gets a method written
        with those parameters named, which is faster to call than packing them
        into a tuple first; one with a default or a rest parameter gets the
        general forwarding method, since its signature cannot be written out
        that way.  The methods are built by executing the text, which is what
        makes the parameters real ones rather than a generic argument list.
        """

        import inspect
        from io import StringIO

        code = StringIO()
        for target in OP_NAMES:
            sig = inspect.signature(getattr(OpsHandler, target))
            if all(
                p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and p.default is inspect.Parameter.empty
                for p in sig.parameters.values()
            ):
                self_arg, *args = sig.parameters.keys()
                if self_arg != "self":
                    raise AssertionError(
                        f"expected first parameter 'self', got {self_arg!r}"
                    )
                code.write(
                    f"""
                    def {target}(self, {", ".join(args)}):
                        return self._default({target!r}, ({", ".join(args)}, ), {{}})
                    """.strip()
                )
                code.write("\n\n")
            else:
                setattr(cls, target, cls._call_default(target))

        ctx: dict = {}
        exec(code.getvalue(), ctx)
        for target, impl in ctx.items():
            if target in OP_NAMES:
                setattr(cls, target, impl)


class NoopHandler(DefaultHandler):
    """A handler that answers nothing, for an operation with nothing to answer.

    The point is not to be right about the operation but to be the right shape:
    an analysis that only wants to know which operations a region uses can run
    with this and get a value for every one of them.
    """

    name = "NoopHandler"

    def _default(self, name, args, kwargs):
        return None

    @staticmethod
    def masked(mask, body, other) -> None:
        return None

    @staticmethod
    def frexp(x):
        return (None, None)

    @staticmethod
    def scan(dtypes, combine_fn, values):
        return (None,) * len(values)

    @staticmethod
    def sort(dtypes, values, stable, descending):
        return (None,) * len(values)

    @staticmethod
    def indirect_indexing(index_var, size, check=True, wrap_neg=True):
        return 0


DefaultHandler._init_cls()


class MockHandler(DefaultHandler):
    """A handler that writes operations down instead of doing them.

    Each operation comes back as the text of a call that would perform it, so
    that a body can be recorded as source and examined without running it.
    """

    name = "MockHandler"

    def _default(self, name: str, args: tuple, kwargs: dict):
        fargs = [*map(_arg_str, args)]
        for k, v in kwargs.items():
            fargs.append(f"{k}={_arg_str(v)}")
        return f"ops.{name}({', '.join(fargs)})"

    @staticmethod
    def masked(mask, body, other) -> str:
        return f"ops.masked({mask}, {body()}, {other})"

    @staticmethod
    def frexp(x):
        return (f"ops.frexp({x})[0]", f"ops.frexp({x})[1]")

    @staticmethod
    def scan(dtypes, combine_fn, values):
        return tuple(
            f"ops.scan({dtypes}, {combine_fn}, {values})[{i}]"
            for i in range(len(values))
        )

    @staticmethod
    def sort(dtypes, values, stable, descending):
        return tuple(
            f"ops.sort({dtypes}, {values}, stable={stable}, descending={descending})[{i}]"
            for i in range(len(values))
        )

    @staticmethod
    def indirect_indexing(index_var, size, check=True, wrap_neg=True):
        from .utils import sympy_index_symbol

        return sympy_index_symbol(str(index_var))


class KernelFormatterHandler(DefaultHandler):
    """A handler that writes a body out as the source of a function.

    Each operation is assigned to a name of its own and the names are what the
    body refers to, so that what a body does can be read rather than inferred
    from what it returned.
    """

    def __init__(self, parent_handler: OpsHandler):
        self.parent_handler = parent_handler
        self._output = IndentedBuffer(1)
        self.var_counter = itertools.count()

    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None) -> str:
        args = [index, rindex] if rindex is not None else [index]
        names = ["index", "rindex"] if rindex is not None else ["index"]
        formatter = KernelFormatterHandler(MockHandler())

        with formatter._output.indent(-1):
            formatter._output.writeline(f"def inner_fn({', '.join(names)}):")
        for name, arg in zip(names, args):
            if arg:
                lhs = ", ".join(
                    [
                        str("_" if isinstance(v, (int, sympy.Integer)) else v)
                        for v in arg
                    ]
                )
                formatter._output.writeline(f"{lhs} = {name}")

        from .loops import V

        with V.set_ops_handler(formatter):
            result = ir_fn(*args)
            return formatter.getvalue(result)

    def indirect_indexing(self, *args, **kwargs):
        return self.parent_handler.indirect_indexing(*args, **kwargs)

    def _write(self, line):
        varname = f"tmp{next(self.var_counter)}"
        self._output.writeline(f"{varname} = {line}")
        return varname

    def _default(self, name: str, args: tuple, kwargs: dict):
        def _map(v):
            return self._write(v) if isinstance(v, str) else v

        return _map(getattr(self.parent_handler, name)(*args, **kwargs))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        from .utils import reduction_num_outputs

        line = self.parent_handler.reduction(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        varnames = [f"tmp{next(self.var_counter)}" for _ in range(num_values)]
        self._output.writeline(f"{','.join(varnames)} = {line}")
        return tuple(varnames) if num_values > 1 else varnames[0]

    def getvalue(self, result):
        self._output.writeline(f"return {result}")
        return self._output.getvalue()


class WrapperHandler(DefaultHandler):
    """A handler that passes every operation on to the one it wraps."""

    def __init__(self, inner: OpsHandler):
        self._inner = inner

    def _default(self, name: str, args: tuple, kwargs: dict):
        return getattr(self._inner, name)(*args, **kwargs)


class SimpleCSEHandler(WrapperHandler):
    """A pass that computes each distinct computation once while tracing.

    Simplified compared to the pass that runs while code is being printed,
    because this one runs while a subgraph is being traced and there is nothing
    being stored: a body that only reads has nothing to invalidate, so caching on
    what was asked for is the whole of what can be shared.
    """

    def __init__(self, inner: OpsHandler):
        super().__init__(inner)
        self.cse_cache: dict = {}
        self.mock = MockHandler()

    def indirect_indexing(self, *args, **kwargs):
        return super().indirect_indexing(*args, **kwargs)

    def store(self, *args, **kwargs) -> None:
        raise NotImplementedError("store not implemented")

    def store_reduction(self, *args, **kwargs) -> None:
        raise NotImplementedError("store not implemented")

    def _default(self, name: str, args: tuple, kwargs: dict):
        # The key is what the computation is rather than how it was written, so
        # that two ways of asking for the same thing are recognised as one.
        key = getattr(self.mock, name)(*args, **kwargs)
        found = self.cse_cache.get(key)
        if found is not None:
            return found

        value = getattr(self._inner, name)(*args, **kwargs)
        self.cse_cache[key] = value
        return value

    def device_assert_async(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by "
            f"CSEProxy"
        )


class OpCountResult(NamedTuple):
    """What a body does, in enough detail to decide whether to realize it.

    The count is of the distinct values the body produces, which is not the
    number of operations: a value computed once and used twice is one value.
    Alongside it are which operations were used and which buffers were read, so
    that a decision about fusing can be made from what the body reaches rather
    than from how long it is.
    """

    num_ops: int
    used_ops: OrderedSet
    read_buffers: list
    nontrivial_read_count: int


class OpCounterCSE(DefaultHandler):
    """Counts what a body computes, giving each distinct value a name.

    The naming is what makes a repeated computation visible: the same value
    asked for twice comes back under the same name, so the count is of the
    values rather than of the requests.
    """

    def __init__(self, inner: "OpsHandler"):
        super().__init__()
        self.parent_handler = inner
        self.op_count = 0
        self.var_names: dict = {}
        self._used_ops: OrderedSet = OrderedSet()
        self._read_names: list = []
        self._nontrivial_read_count = 0

    def _default(self, name: str, args: tuple, kwargs: dict):
        self._used_ops.add(name)
        return pytree.tree_map(
            self._update_count, getattr(self.parent_handler, name)(*args, **kwargs)
        )

    def _update_count(self, val):
        varname = self.var_names.get(val)
        if not varname:
            varname = f"tmp{self.op_count}"
            self.op_count += 1
            self.var_names[val] = varname
        return varname

    def indirect_indexing(self, *args, **kwargs):
        self._used_ops.add("indirect_indexing")
        return self.parent_handler.indirect_indexing(*args, **kwargs)

    def load(self, name: str, index):
        val = self.parent_handler.load(name, index)
        if val not in self.var_names:
            self._used_ops.add("load")
            self._read_names.append(name)
            if not isinstance(index, (sympy.Integer, int)):
                # A read at a fixed place is cheaper than one that has to be
                # computed, so the two are counted apart.
                self._nontrivial_read_count += 1
        return self._update_count(val)

    def load_seed(self, name: str, offset):
        val = self.parent_handler.load_seed(name, offset)
        if val not in self.var_names:
            self._used_ops.add("load_seed")
            self._read_names.append(name)
        return self._update_count(val)

    def bucketize(
        self,
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right,
        sorter=None,
        sorter_indices=None,
    ):
        val = self.parent_handler.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )
        if val not in self.var_names:
            self._used_ops.add("bucketize")
            self._read_names.append(boundaries[0])
            if sorter is not None:
                self._read_names.append(sorter[0])
        return self._update_count(val)

    def getvalue(self):
        return OpCountResult(
            self.op_count, self._used_ops, self._read_names, self._nontrivial_read_count
        )


#: How a value was written into a place, when the way it was written decides
#: where it goes.  ``None`` is an ordinary store; a named mode is a store
#: through a mechanism that resolves the place itself -- an atomic update, or a
#: descriptor -- and a kernel handed one has to be written for that mechanism
#: rather than for the store.
AtomicMode = Literal[
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_cas",
    "atomic_xchg",
]
StoreMode = AtomicMode | Literal["tma"] | None

ReductionType = Literal['argmax', 'argmin', 'argmax_value', 'argmin_value', 'argmax_with_value', 'argmin_with_value', 'welford_reduce', 'welford_combine', 'any', 'fmax', 'max', 'min', 'prod', 'sum', 'dot', 'xor_sum', 'online_softmax_reduce']
