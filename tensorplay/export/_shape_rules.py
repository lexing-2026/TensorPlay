"""The extents of a computed value, derived from the extents it was computed from.

A capture that sizes some input dimensions symbolically has to know, for every
value the program computes, the extents that value takes for every size the
inputs may take -- not only for the example.  An extent that does not depend on
a symbolic dimension is the example's number; one that does is an expression
in the symbols.  Each rule here states, for one family of operations, how the
extents of the result follow from the extents and arguments of the call: the
same facts the operation's own size checks state, written over expressions
instead of numbers.

A rule that has to decide something about an extent -- whether two extents
broadcast, whether a dimension is 1, where a slice ends -- asks ``env``, which
answers from the example and keeps the answer as a guard unless the
declaration already settles it.  A rule that meets something it cannot size (a
result sized by tensor data, a size held in a tensor) raises
:class:`Undecidable`, and the result's extents are then left as fresh symbols
nothing is known about.

A rule receives the call's arguments as the program passed them, with every
tensor replaced by a :class:`Shaped` and every size by its expression, and
returns the result's extents: one tuple for a tensor, a list of them (``None``
for an entry that is not a tensor) for a sequence of results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import sympy

Extent = Any  # int | sympy.Expr


class Undecidable(Exception):
    """A rule cannot state the result's extents from what it was given."""


class DataDependent(Undecidable):
    """The result's extents depend on the values of a tensor, not on its extents."""


class Sizes(Protocol):
    """How a rule decides a condition on extents."""

    def holds(self, fact: Any) -> bool:
        """Decide ``fact``, keeping it as a guard unless the declaration settles it."""

    def obliviously(self, fact: Any) -> bool:
        """Decide ``fact`` taking every varying extent to be at least 2.

        Used where sizes 0 and 1 lead to the same program anyway -- whether a
        dimension broadcasts -- so that deciding it costs no guard.
        """

    def example(self, extent: Extent) -> int:
        """What ``extent`` was on the example."""


class Shaped:
    """A tensor as a rule sees it: its extents, and the example run's value.

    ``origins`` names the program inputs the tensor was computed from.
    """

    __slots__ = ("extents", "example", "origins")

    def __init__(
        self, extents: Sequence[Extent], example: Any, origins: frozenset[str] = frozenset()
    ) -> None:
        self.extents = tuple(extents)
        self.example = example
        self.origins = origins

    @property
    def rank(self) -> int:
        return len(self.extents)

    @property
    def dtype(self) -> Any:
        return self.example.dtype

    def __repr__(self) -> str:
        return f"Shaped{self.extents}"


class Data:
    """A Python value the program read out of a tensor's contents."""

    __slots__ = ("example",)

    def __init__(self, example: Any) -> None:
        self.example = example

    def __repr__(self) -> str:
        return f"Data({self.example!r})"


#: Operations whose result is sized by the contents of a tensor.
DATA_DEPENDENT = frozenset(
    {
        "nonzero",
        "argwhere",
        "unique",
        "unique_consecutive",
        "masked_select",
        "bincount",
        "histogram",
        "item",
        "tolist",
        "numpy",
        "allclose",
        "equal",
        "_local_scalar_dense",
    }
)

_RULES: dict[str, Callable[..., Any]] = {}


def rule(*names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the decorated function as the rule for every name in ``names``.

    A name prefixed ``nn.`` is the ``tensorplay.nn.functional`` function of that
    name, for the functions whose arguments differ from the tensor operation's.
    """

    def register(function: Callable[..., Any]) -> Callable[..., Any]:
        for name in names:
            _RULES[name] = function
        return function

    return register


_OWN_MODULES = ("tensorplay", "_operator", "operator", "builtins")


def operation(kind: str, target: Any) -> str | None:
    """The name a call is sized by, or ``None`` for a call no rule can cover."""

    if kind == "call_method":
        return target if isinstance(target, str) else None
    if kind != "call_function":
        return None
    name = getattr(target, "__name__", None)
    if not isinstance(name, str):
        return None
    module = getattr(target, "__module__", None) or ""
    qualname = getattr(target, "__qualname__", "") or ""
    if not module.startswith(_OWN_MODULES) and not qualname.startswith(
        ("Tensor.", "TensorBase.")
    ):
        return None
    if module.startswith("tensorplay.nn.functional") and f"nn.{name}" in _RULES:
        return f"nn.{name}"
    return name


def find(name: str) -> Callable[..., Any] | None:
    """The rule for ``name``.  An in-place variant without a rule of its own
    keeps the extents of the tensor it writes to."""

    found = _RULES.get(name)
    if found is not None:
        return found
    if name.endswith("_") and not name.endswith("__"):
        return _same
    return None


# -- arguments ------------------------------------------------------------------


def _extent(value: Any) -> Extent:
    if isinstance(value, bool):
        raise Undecidable("a bool is not an extent")
    if isinstance(value, int):
        return value
    if isinstance(value, sympy.Basic):
        return int(value) if value.is_Integer else value
    if isinstance(value, (Shaped, Data)):
        raise DataDependent("an extent held in a tensor")
    raise Undecidable(f"{type(value).__name__} is not an extent")


def _shape(values: Sequence[Any]) -> list[Extent]:
    """A shape given either spread out (``view(a, b)``) or as one sequence."""

    if len(values) == 1 and isinstance(values[0], (tuple, list)):
        values = values[0]
    return [_extent(value) for value in values]


def _int(value: Any) -> int:
    if isinstance(value, sympy.Integer):
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Undecidable(f"{value!r} is not a fixed integer")
    return value


def _axis(dim: Any, rank: int) -> int:
    dim = _int(dim)
    wrapped = max(rank, 1)
    if not -wrapped <= dim < wrapped:
        raise Undecidable(f"dimension {dim} out of range for rank {rank}")
    return dim % wrapped


def _axes(dims: Any, rank: int) -> list[int]:
    if isinstance(dims, (tuple, list)):
        return [_axis(dim, rank) for dim in dims]
    return [_axis(dims, rank)]


def _ntuple(value: Any, count: int) -> list[Any]:
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return list(value) * count
        if len(value) != count:
            raise Undecidable("a per-dimension argument of the wrong length")
        return list(value)
    return [value] * count


def _product(extents: Any) -> Extent:
    result: Extent = 1
    for extent in extents:
        result = result * extent
    return result


def _floordiv(numerator: Extent, denominator: Extent) -> Extent:
    if isinstance(numerator, int) and isinstance(denominator, int):
        return numerator // denominator
    return sympy.floor(sympy.sympify(numerator) / denominator)


def _ceildiv(numerator: Extent, denominator: Extent) -> Extent:
    if isinstance(numerator, int) and isinstance(denominator, int):
        return -(-numerator // denominator)
    return sympy.ceiling(sympy.sympify(numerator) / denominator)


def _rational(value: Any) -> Any:
    """A scale factor as an exact number, so ``floor(n * 1.5)`` stays exact."""

    if isinstance(value, float):
        return sympy.Rational(repr(value))
    return value


def _tensors(*values: Any) -> list[Shaped]:
    return [value for value in values if isinstance(value, Shaped)]


def _first(*values: Any) -> Shaped:
    for value in values:
        if isinstance(value, Shaped):
            return value
    raise Undecidable("no tensor to take the extents of")


# -- shared shape logic -----------------------------------------------------------


def broadcast(env: Sizes, *shapes: Sequence[Extent]) -> tuple[Extent, ...]:
    """The extents of a result broadcast from operands of these extents."""

    rank = max((len(shape) for shape in shapes), default=0)
    result: list[Extent] = []
    for position in range(rank):
        extent: Extent = 1
        for shape in shapes:
            index = len(shape) - rank + position
            if index < 0:
                continue
            other = shape[index]
            if env.obliviously(sympy.Eq(other, 1)):
                continue
            if env.obliviously(sympy.Eq(extent, 1)):
                extent = other
                continue
            if not env.holds(sympy.Eq(extent, other)):
                raise Undecidable("extents that do not broadcast")
            if not isinstance(extent, int) and isinstance(other, int):
                extent = other
        result.append(extent)
    return tuple(result)


def _reduced(extents: Sequence[Extent], dims: Any, keepdim: bool) -> tuple[Extent, ...]:
    rank = len(extents)
    if dims is None or (isinstance(dims, (tuple, list)) and not dims):
        axes = set(range(rank))
    else:
        axes = set(_axes(dims, rank)) if rank else set()
    if keepdim:
        return tuple(1 if axis in axes else extent for axis, extent in enumerate(extents))
    return tuple(extent for axis, extent in enumerate(extents) if axis not in axes)


def _infer(env: Sizes, extents: Sequence[Extent], shape: Sequence[Extent]) -> tuple[Extent, ...]:
    """``shape`` with its ``-1`` replaced by what the count of values implies."""

    unknown = [
        index for index, extent in enumerate(shape) if isinstance(extent, int) and extent == -1
    ]
    if len(unknown) > 1:
        raise Undecidable("more than one inferred extent")
    total = _product(extents)
    if not unknown:
        if not env.holds(sympy.Eq(total, _product(shape))):
            raise Undecidable("a reshape that changes the number of values")
        return tuple(shape)
    (at,) = unknown
    known = _product(extent for index, extent in enumerate(shape) if index != at)
    inferred = _floordiv(total, known)
    if not env.holds(sympy.Eq(total, inferred * known)):
        raise Undecidable("a reshape that does not divide the values evenly")
    result = list(shape)
    result[at] = _extent(sympy.simplify(inferred)) if isinstance(inferred, sympy.Basic) else inferred
    return tuple(result)


def _slice_length(env: Sizes, extent: Extent, item: slice) -> Extent:
    step = 1 if item.step is None else _int(item.step)
    if step <= 0:
        raise Undecidable("a slice step that is not positive")

    # Clamping a bound to the extent is decided taking the extent to be large,
    # so ``x[1:]`` costs no guard; only a bound near the example's size does.
    def bound(value: Any, default: Extent) -> Extent:
        if value is None:
            return default
        value = _extent(value)
        if env.holds(sympy.Lt(value, 0)):
            value = value + extent
            return 0 if env.obliviously(sympy.Lt(value, 0)) else value
        return extent if env.obliviously(sympy.Gt(value, extent)) else value

    start = bound(item.start, 0)
    stop = bound(item.stop, extent)
    if (not isinstance(start, int) or start != 0) and env.obliviously(sympy.Le(stop, start)):
        return 0
    return stop - start if step == 1 else _ceildiv(stop - start, step)


def _counted(env: Sizes, value: Extent) -> int:
    """How many results ``value`` makes, kept as a guard when it varies."""

    count = env.example(value)
    if not isinstance(value, int):
        env.holds(sympy.Eq(value, count))
    return count


# -- elementwise ------------------------------------------------------------------

_SAME_SHAPE = (
    # value-wise maps
    "abs absolute neg negative positive pos sign sgn signbit floor ceil round trunc fix "
    "frac exp exp2 expm1 log log2 log10 log1p sqrt rsqrt square reciprocal sin cos tan "
    "asin acos atan sinh cosh tanh asinh acosh atanh arcsin arccos arctan arcsinh "
    "arccosh arctanh sigmoid logit erf erfc erfinv lgamma digamma i0 sinc nan_to_num "
    "angle conj conj_physical isnan isinf isfinite isneginf isposinf invert logical_not "
    "bitwise_not deg2rad rad2deg polygamma mvlgamma "
    # activations
    "relu relu6 gelu silu mish elu selu celu leaky_relu rrelu hardtanh hardswish "
    "hardsigmoid hardshrink softshrink tanhshrink softsign softplus threshold logsigmoid "
    "prelu "
    # normalizations, and running reductions that keep the extents
    "softmax log_softmax softmin layer_norm group_norm batch_norm instance_norm rms_norm "
    "local_response_norm normalize cumsum cumprod logcumsumexp "
    "dropout dropout1d dropout2d dropout3d alpha_dropout feature_alpha_dropout "
    # copies, conversions and rearrangements of the values
    "clone contiguous detach to type_as float double half bfloat16 int long short char "
    "byte bool cpu cuda cfloat cdouble pin_memory fill masked_fill masked_scatter "
    "index_fill index_copy index_add index_put scatter scatter_add scatter_reduce put "
    "flip fliplr flipud roll tril triu argsort bucketize"
).split()


@rule(*_SAME_SHAPE)
def _same(env: Sizes, *args: Any, **kwargs: Any) -> tuple[Extent, ...]:
    return _first(*args, *kwargs.values()).extents


#: ``x += y`` and its kin write into ``x``, which keeps its extents.
rule(
    "iadd", "isub", "imul", "itruediv", "ifloordiv", "imod", "ipow",
    "iand", "ior", "ixor", "ilshift", "irshift",
)(_same)


@rule("nn.normalize", "nn.dropout")
def _nn_same(env: Sizes, input: Shaped, *args: Any, **kwargs: Any) -> tuple[Extent, ...]:
    return input.extents


_BROADCAST = (
    "add sub subtract mul multiply div divide true_divide floor_divide floordiv truediv "
    "remainder fmod mod pow float_power atan2 arctan2 hypot copysign xlogy nextafter "
    "ldexp logaddexp logaddexp2 heaviside maximum minimum fmax fmin rsub "
    "eq ne lt le gt ge greater less greater_equal less_equal not_equal "
    "logical_and logical_or logical_xor bitwise_and bitwise_or bitwise_xor "
    "bitwise_left_shift bitwise_right_shift and_ or_ xor lshift rshift "
    "lerp addcmul addcdiv clamp clip clamp_min clamp_max gcd lcm"
).split()


@rule(*_BROADCAST)
def _broadcasting(env: Sizes, *args: Any, **kwargs: Any) -> tuple[Extent, ...]:
    tensors = _tensors(*args, *kwargs.values())
    if not tensors:
        raise Undecidable("no tensor operand")
    return broadcast(env, *(tensor.extents for tensor in tensors))


@rule("where")
def _where(env: Sizes, condition: Shaped, *args: Any, **kwargs: Any) -> tuple[Extent, ...]:
    if not args and not kwargs:
        raise DataDependent("where with only a condition")
    tensors = _tensors(condition, *args, *kwargs.values())
    return broadcast(env, *(tensor.extents for tensor in tensors))


# -- reductions -------------------------------------------------------------------


@rule(
    "sum", "mean", "prod", "nansum", "nanmean", "amax", "amin", "all", "any",
    "logsumexp", "argmax", "argmin",
)
def _reduction(
    env: Sizes, input: Shaped, dim: Any = None, keepdim: bool = False, *args: Any, **kwargs: Any
) -> tuple[Extent, ...]:
    return _reduced(input.extents, dim, bool(keepdim))


@rule("count_nonzero")
def _count_nonzero(env: Sizes, input: Shaped, dim: Any = None) -> tuple[Extent, ...]:
    return _reduced(input.extents, dim, False)


def _deviation_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, bool]:
    """``dim`` and ``keepdim`` of ``std``/``var``: ``(dim, unbiased, keepdim)``,
    where a leading bool is ``unbiased`` with every dimension reduced."""

    rest = list(args)
    if rest and isinstance(rest[0], bool):
        return None, bool(kwargs.get("keepdim", False))
    dim = kwargs.get("dim", rest[0] if rest else None)
    keepdim = kwargs.get("keepdim", rest[2] if len(rest) > 2 else False)
    return dim, bool(keepdim)


@rule("std", "var")
def _deviation(env: Sizes, input: Shaped, *args: Any, **kwargs: Any) -> tuple[Extent, ...]:
    dim, keepdim = _deviation_arguments(args, kwargs)
    return _reduced(input.extents, dim, keepdim)


@rule("std_mean", "var_mean")
def _deviation_pair(env: Sizes, input: Shaped, *args: Any, **kwargs: Any) -> list[Any]:
    dim, keepdim = _deviation_arguments(args, kwargs)
    extents = _reduced(input.extents, dim, keepdim)
    return [extents, extents]


@rule("aminmax")
def _aminmax(env: Sizes, input: Shaped, *, dim: Any = None, keepdim: bool = False) -> list[Any]:
    extents = _reduced(input.extents, dim, keepdim)
    return [extents, extents]


@rule("max", "min")
def _max(env: Sizes, input: Shaped, dim: Any = None, keepdim: bool = False, **kwargs: Any) -> Any:
    other = kwargs.get("other", dim)
    if isinstance(other, Shaped):
        return broadcast(env, input.extents, other.extents)
    if dim is None:
        return ()
    extents = _reduced(input.extents, dim, bool(keepdim))
    return [extents, extents]


@rule("median", "nanmedian")
def _median(env: Sizes, input: Shaped, dim: Any = None, keepdim: bool = False) -> Any:
    if dim is None:
        return ()
    extents = _reduced(input.extents, dim, bool(keepdim))
    return [extents, extents]


@rule("mode")
def _mode(env: Sizes, input: Shaped, dim: Any = -1, keepdim: bool = False) -> Any:
    extents = _reduced(input.extents, dim, bool(keepdim))
    return [extents, extents]


@rule("kthvalue")
def _kthvalue(env: Sizes, input: Shaped, k: Any, dim: Any = -1, keepdim: bool = False) -> Any:
    extents = _reduced(input.extents, dim, bool(keepdim))
    return [extents, extents]


@rule("cummax", "cummin", "sort")
def _pair_same(env: Sizes, input: Shaped, *args: Any, **kwargs: Any) -> Any:
    return [input.extents, input.extents]


@rule("topk")
def _topk(env: Sizes, input: Shaped, k: Any, dim: Any = -1, *args: Any, **kwargs: Any) -> Any:
    extents = list(input.extents)
    if extents:
        extents[_axis(dim, len(extents))] = _extent(k)
    return [tuple(extents), tuple(extents)]


@rule("norm")
def _norm(env: Sizes, input: Shaped, p: Any = "fro", dim: Any = None, keepdim: bool = False, **kwargs: Any) -> Any:
    return _reduced(input.extents, dim, bool(keepdim))


# -- products -----------------------------------------------------------------------


@rule("matmul", "mm", "bmm", "mv", "dot", "vdot", "inner")
def matmul(env: Sizes, input: Shaped, other: Shaped, **kwargs: Any) -> tuple[Extent, ...]:
    left, right = list(input.extents), list(other.extents)
    if not left or not right:
        raise Undecidable("a product of a 0-d tensor")
    vector_left = len(left) == 1
    vector_right = len(right) == 1
    if vector_left:
        left = [1, *left]
    if vector_right:
        right = [*right, 1]
    env.holds(sympy.Eq(left[-1], right[-2]))
    result = list(broadcast(env, left[:-2], right[:-2]))
    if not vector_left:
        result.append(left[-2])
    if not vector_right:
        result.append(right[-1])
    return tuple(result)


@rule("outer", "ger")
def _outer(env: Sizes, input: Shaped, other: Shaped) -> tuple[Extent, ...]:
    return (input.extents[0], other.extents[0])


@rule("linear", "nn.linear")
def _linear(env: Sizes, input: Shaped, weight: Shaped, bias: Any = None) -> tuple[Extent, ...]:
    if weight.rank == 1:
        return input.extents[:-1]
    env.holds(sympy.Eq(input.extents[-1], weight.extents[-1]))
    return (*input.extents[:-1], weight.extents[0])


@rule("bilinear", "nn.bilinear")
def _bilinear(env: Sizes, input1: Shaped, input2: Shaped, weight: Shaped, bias: Any = None) -> Any:
    return (*input1.extents[:-1], weight.extents[0])


@rule("addmm", "baddbmm")
def _addmm(env: Sizes, input: Shaped, left: Shaped, right: Shaped, **kwargs: Any) -> Any:
    return broadcast(env, input.extents, matmul(env, left, right))


@rule("addmv")
def _addmv(env: Sizes, input: Shaped, matrix: Shaped, vector: Shaped, **kwargs: Any) -> Any:
    return broadcast(env, input.extents, matmul(env, matrix, vector))


@rule("addbmm")
def _addbmm(env: Sizes, input: Shaped, left: Shaped, right: Shaped, **kwargs: Any) -> Any:
    return broadcast(env, input.extents, matmul(env, left, right)[1:])


@rule("addr")
def _addr(env: Sizes, input: Shaped, vector1: Shaped, vector2: Shaped, **kwargs: Any) -> Any:
    return broadcast(env, input.extents, _outer(env, vector1, vector2))


@rule("einsum")
def _einsum(env: Sizes, equation: str, *operands: Any, **kwargs: Any) -> tuple[Extent, ...]:
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = tuple(operands[0])
    equation = equation.replace(" ", "")
    inputs, arrow, output = equation.partition("->")
    terms = inputs.split(",")
    if len(terms) != len(operands) or not all(isinstance(op, Shaped) for op in operands):
        raise Undecidable("einsum operands")
    letters: dict[str, Extent] = {}
    ellipses: list[Sequence[Extent]] = []
    for term, operand in zip(terms, operands):
        extents = list(operand.extents)
        head, dots, tail = term.partition("...")
        if not dots and len(head) != len(extents):
            raise Undecidable("an einsum term of the wrong rank")
        labelled = list(zip(head, extents[: len(head)]))
        if dots:
            ellipses.append(extents[len(head) : len(extents) - len(tail)])
            labelled += list(zip(tail, extents[len(extents) - len(tail) :]))
        for letter, extent in labelled:
            known = letters.get(letter)
            if known is None or env.obliviously(sympy.Eq(known, 1)):
                letters[letter] = extent
            elif not env.obliviously(sympy.Eq(extent, 1)):
                env.holds(sympy.Eq(known, extent))
    batch = broadcast(env, *ellipses) if ellipses else ()
    if not arrow:
        once = sorted(letter for letter in letters if inputs.count(letter) == 1)
        output = ("..." if ellipses else "") + "".join(once)
    head, dots, tail = output.partition("...")
    return (
        *(letters[letter] for letter in head),
        *(batch if dots else ()),
        *(letters[letter] for letter in tail),
    )


@rule("cdist")
def _cdist(env: Sizes, x1: Shaped, x2: Shaped, *args: Any, **kwargs: Any) -> Any:
    batch = broadcast(env, x1.extents[:-2], x2.extents[:-2])
    return (*batch, x1.extents[-2], x2.extents[-2])


@rule("cosine_similarity", "nn.cosine_similarity")
def _cosine_similarity(env: Sizes, x1: Shaped, x2: Shaped, dim: Any = 1, eps: Any = None) -> Any:
    return _reduced(broadcast(env, x1.extents, x2.extents), dim, False)


@rule("pairwise_distance", "nn.pairwise_distance")
def _pairwise_distance(
    env: Sizes, x1: Shaped, x2: Shaped, p: Any = 2.0, eps: Any = 1e-6, keepdim: bool = False
) -> Any:
    extents = broadcast(env, x1.extents, x2.extents)
    return _reduced(extents, -1, bool(keepdim)) if extents else ()


@rule("cross")
def _cross(env: Sizes, input: Shaped, other: Shaped, *args: Any, **kwargs: Any) -> Any:
    return broadcast(env, input.extents, other.extents)


# -- views and rearrangements -----------------------------------------------------------


@rule("view", "reshape")
def _reshape(env: Sizes, input: Shaped, *shape: Any, **kwargs: Any) -> tuple[Extent, ...]:
    if "shape" in kwargs:
        shape = (kwargs["shape"],)
    if len(shape) == 1 and not isinstance(shape[0], (tuple, list, int, sympy.Basic)):
        # ``view(dtype)`` reinterprets the values rather than rearranging them.
        raise Undecidable("a view as another dtype")
    return _infer(env, input.extents, _shape(shape))


@rule("view_as", "reshape_as", "expand_as")
def _as(env: Sizes, input: Shaped, other: Shaped) -> tuple[Extent, ...]:
    return other.extents


@rule("flatten")
def _flatten(env: Sizes, input: Shaped, start_dim: Any = 0, end_dim: Any = -1) -> tuple[Extent, ...]:
    extents = input.extents
    if not extents:
        return (1,)
    start, end = _axis(start_dim, len(extents)), _axis(end_dim, len(extents))
    return (*extents[:start], _product(extents[start : end + 1]), *extents[end + 1 :])


@rule("unflatten")
def _unflatten(env: Sizes, input: Shaped, dim: Any, sizes: Any) -> tuple[Extent, ...]:
    extents = input.extents
    axis = _axis(dim, len(extents))
    inner = _infer(env, (extents[axis],), _shape((sizes,)))
    return (*extents[:axis], *inner, *extents[axis + 1 :])


@rule("squeeze")
def _squeeze(env: Sizes, input: Shaped, dim: Any = None) -> tuple[Extent, ...]:
    extents = input.extents
    axes = range(len(extents)) if dim is None else _axes(dim, len(extents))
    drop = {axis for axis in axes if extents and env.holds(sympy.Eq(extents[axis], 1))}
    return tuple(extent for axis, extent in enumerate(extents) if axis not in drop)


@rule("unsqueeze")
def _unsqueeze(env: Sizes, input: Shaped, dim: Any) -> tuple[Extent, ...]:
    extents = list(input.extents)
    extents.insert(_axis(dim, len(extents) + 1), 1)
    return tuple(extents)


@rule("permute")
def _permute(env: Sizes, input: Shaped, *dims: Any) -> tuple[Extent, ...]:
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    return tuple(input.extents[_axis(dim, input.rank)] for dim in dims)


@rule("transpose", "swapaxes", "swapdims")
def _transpose(env: Sizes, input: Shaped, dim0: Any, dim1: Any) -> tuple[Extent, ...]:
    extents = list(input.extents)
    if not extents:
        return ()
    first, second = _axis(dim0, len(extents)), _axis(dim1, len(extents))
    extents[first], extents[second] = extents[second], extents[first]
    return tuple(extents)


@rule("t")
def _t(env: Sizes, input: Shaped) -> tuple[Extent, ...]:
    return input.extents[::-1] if input.rank == 2 else input.extents


@rule("movedim", "moveaxis")
def _movedim(env: Sizes, input: Shaped, source: Any, destination: Any) -> tuple[Extent, ...]:
    sources = _axes(source, input.rank)
    targets = _axes(destination, input.rank)
    order: list[int | None] = [None] * input.rank
    for src, dst in zip(sources, targets):
        order[dst] = src
    rest = iter(axis for axis in range(input.rank) if axis not in sources)
    return tuple(input.extents[next(rest) if axis is None else axis] for axis in order)


@rule("expand")
def _expand(env: Sizes, input: Shaped, *shape: Any, **kwargs: Any) -> tuple[Extent, ...]:
    if "size" in kwargs:
        shape = (kwargs["size"],)
    target = _shape(shape)
    offset = len(target) - input.rank
    if offset < 0:
        raise Undecidable("expand to fewer dimensions")
    result: list[Extent] = []
    for index, extent in enumerate(target):
        if isinstance(extent, int) and extent == -1:
            if index < offset:
                raise Undecidable("-1 for a new dimension")
            extent = input.extents[index - offset]
        result.append(extent)
    return tuple(result)


@rule("broadcast_to")
def _broadcast_to(env: Sizes, input: Shaped, size: Any) -> tuple[Extent, ...]:
    return _expand(env, input, size)


@rule("repeat")
def _repeat(env: Sizes, input: Shaped, *repeats: Any) -> tuple[Extent, ...]:
    counts = _shape(repeats)
    extents = [1] * (len(counts) - input.rank) + list(input.extents)
    if len(extents) != len(counts):
        raise Undecidable("fewer repeats than dimensions")
    return tuple(extent * count for extent, count in zip(extents, counts))


@rule("tile")
def _tile(env: Sizes, input: Shaped, *dims: Any) -> tuple[Extent, ...]:
    counts = _shape(dims)
    counts = [1] * (input.rank - len(counts)) + counts
    extents = [1] * (len(counts) - input.rank) + list(input.extents)
    return tuple(extent * count for extent, count in zip(extents, counts))


@rule("narrow")
def _narrow(env: Sizes, input: Shaped, dim: Any, start: Any, length: Any) -> tuple[Extent, ...]:
    extents = list(input.extents)
    extents[_axis(dim, len(extents))] = _extent(length)
    return tuple(extents)


@rule("select")
def _select(env: Sizes, input: Shaped, dim: Any, index: Any) -> tuple[Extent, ...]:
    extents = list(input.extents)
    del extents[_axis(dim, len(extents))]
    return tuple(extents)


@rule("diagonal")
def _diagonal(env: Sizes, input: Shaped, offset: Any = 0, dim1: Any = 0, dim2: Any = 1) -> Any:
    first, second = _axis(dim1, input.rank), _axis(dim2, input.rank)
    offset = _int(offset)
    rows = input.extents[first] + min(offset, 0)
    columns = input.extents[second] - max(offset, 0)
    length = rows if env.holds(sympy.Le(rows, columns)) else columns
    if env.holds(sympy.Lt(length, 0)):
        length = 0
    rest = [extent for axis, extent in enumerate(input.extents) if axis not in (first, second)]
    return (*rest, length)


@rule("unfold")
def _unfold(env: Sizes, input: Shaped, dimension: Any, size: Any, step: Any) -> Any:
    extents = list(input.extents)
    if not extents:
        return (_extent(size),)
    axis = _axis(dimension, len(extents))
    extents[axis] = _floordiv(extents[axis] - _extent(size), _extent(step)) + 1
    return (*extents, _extent(size))


@rule("pixel_shuffle", "nn.pixel_shuffle")
def _pixel_shuffle(env: Sizes, input: Shaped, upscale_factor: Any) -> Any:
    factor = _int(upscale_factor)
    *batch, channels, height, width = input.extents
    return (*batch, _floordiv(channels, factor * factor), height * factor, width * factor)


@rule("pixel_unshuffle", "nn.pixel_unshuffle")
def _pixel_unshuffle(env: Sizes, input: Shaped, downscale_factor: Any) -> Any:
    factor = _int(downscale_factor)
    *batch, channels, height, width = input.extents
    return (
        *batch,
        channels * factor * factor,
        _floordiv(height, factor),
        _floordiv(width, factor),
    )


@rule("rot90")
def _rot90(env: Sizes, input: Shaped, k: Any = 1, dims: Any = (0, 1)) -> Any:
    if _int(k) % 2 == 0:
        return input.extents
    first, second = _axes(dims, input.rank)
    return _transpose(env, input, first, second)


@rule("diff")
def _diff(
    env: Sizes, input: Shaped, n: Any = 1, dim: Any = -1, prepend: Any = None, append: Any = None
) -> Any:
    extents = list(input.extents)
    axis = _axis(dim, len(extents))
    for extra in (prepend, append):
        if isinstance(extra, Shaped):
            extents[axis] = extents[axis] + (extra.extents[axis] if extra.rank else 1)
    extents[axis] = extents[axis] - _int(n)
    if env.holds(sympy.Lt(extents[axis], 0)):
        extents[axis] = 0
    return tuple(extents)


def _at_least(rank: int) -> Callable[..., Any]:
    def at_least(env: Sizes, *tensors: Any) -> Any:
        if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
            tensors = tuple(tensors[0])
        results = [_raised(tensor.extents, rank) for tensor in tensors]
        return results[0] if len(results) == 1 else results

    return at_least


def _raised(extents: Sequence[Extent], rank: int) -> tuple[Extent, ...]:
    """``extents`` as ``atleast_{rank}d`` leaves them."""

    extents = tuple(extents)
    if len(extents) >= rank:
        return extents
    if rank == 3 and len(extents) == 2:
        return (*extents, 1)
    if rank == 3 and len(extents) == 1:
        return (1, *extents, 1)
    return (1,) * (rank - len(extents)) + extents


for _rank in (1, 2, 3):
    rule(f"atleast_{_rank}d")(_at_least(_rank))


@rule("getattr")
def _getattr(env: Sizes, input: Shaped, name: str, *default: Any) -> Any:
    if name in ("T", "H"):
        return input.extents[::-1]
    if name in ("mT", "mH"):
        return (*input.extents[:-2], input.extents[-1], input.extents[-2])
    if name in ("real", "imag", "data", "grad"):
        return input.extents
    raise Undecidable(f"the attribute {name!r}")


# -- indexing ---------------------------------------------------------------------------


def _is_mask(value: Shaped) -> bool:
    return str(value.dtype).endswith(("bool", "uint8"))


@rule("getitem")
def _getitem(env: Sizes, input: Shaped, key: Any) -> tuple[Extent, ...]:
    items = list(key) if isinstance(key, tuple) else [key]
    for item in items:
        if isinstance(item, Shaped) and _is_mask(item):
            raise DataDependent("indexing with a mask")
    consumed = sum(1 for item in items if item is not None and item is not Ellipsis)
    fill = [slice(None)] * (input.rank - consumed)
    if Ellipsis in items:
        at = items.index(Ellipsis)
        items[at : at + 1] = fill
    else:
        items.extend(fill)

    result: list[Extent] = []
    indices: list[Sequence[Extent]] = []
    positions: list[int] = []
    axis = 0
    for item in items:
        if item is None:
            result.append(1)
            continue
        if isinstance(item, slice):
            result.append(_slice_length(env, input.extents[axis], item))
        elif isinstance(item, Shaped):
            indices.append(item.extents)
            positions.append(len(result))
        elif isinstance(item, (list, tuple)):
            indices.append((len(item),))
            positions.append(len(result))
        else:
            _extent(item)
        axis += 1
    if indices:
        indexed = broadcast(env, *indices)
        # Indices next to each other put their extents where they stand;
        # indices apart put them first.
        position = positions[0] if len(set(positions)) == 1 else 0
        result[position:position] = indexed
    return tuple(result)


@rule("index_select")
def _index_select(env: Sizes, input: Shaped, dim: Any, index: Shaped) -> Any:
    extents = list(input.extents)
    if extents:
        extents[_axis(dim, len(extents))] = index.extents[0] if index.rank else 1
    return tuple(extents)


@rule("gather")
def _gather(env: Sizes, input: Shaped, dim: Any, index: Shaped, **kwargs: Any) -> Any:
    return index.extents


@rule("take_along_dim")
def _take_along_dim(env: Sizes, input: Shaped, indices: Shaped, dim: Any = None) -> Any:
    if dim is None:
        return (_product(indices.extents),)
    axis = _axis(dim, input.rank)
    left, right = list(input.extents), list(indices.extents)
    left[axis] = right[axis] = 1
    result = list(broadcast(env, left, right))
    result[axis] = indices.extents[axis]
    return tuple(result)


@rule("take")
def _take(env: Sizes, input: Shaped, index: Shaped) -> Any:
    return index.extents


@rule("embedding")
def _embedding(env: Sizes, weight: Shaped, indices: Shaped, *args: Any, **kwargs: Any) -> Any:
    return (*indices.extents, weight.extents[-1])


@rule("nn.embedding")
def _nn_embedding(env: Sizes, input: Shaped, weight: Shaped, *args: Any, **kwargs: Any) -> Any:
    return (*input.extents, weight.extents[-1])


@rule("one_hot", "nn.one_hot")
def _one_hot(env: Sizes, input: Shaped, num_classes: Any = -1) -> Any:
    if isinstance(num_classes, int) and num_classes < 0:
        raise DataDependent("one_hot sized by the largest class")
    return (*input.extents, _extent(num_classes))


@rule("repeat_interleave")
def _repeat_interleave(
    env: Sizes, input: Any, repeats: Any = None, dim: Any = None, *, output_size: Any = None
) -> Any:
    if not isinstance(input, Shaped) or repeats is None:
        raise DataDependent("repeat_interleave with counts held in a tensor")
    if isinstance(repeats, Shaped):
        if output_size is None or dim is None:
            raise DataDependent("repeat_interleave with counts held in a tensor")
        count = None
    else:
        count = _extent(repeats)
    if dim is None:
        return (_product(input.extents) * count,)
    extents = list(input.extents)
    axis = _axis(dim, len(extents))
    extents[axis] = _extent(output_size) if count is None else extents[axis] * count
    return tuple(extents)


@rule("searchsorted")
def _searchsorted(env: Sizes, sorted_sequence: Shaped, input: Any, *args: Any, **kwargs: Any) -> Any:
    return input.extents if isinstance(input, Shaped) else ()


@rule("histc")
def _histc(env: Sizes, input: Shaped, bins: Any = 100, *args: Any, **kwargs: Any) -> Any:
    return (_extent(bins),)


# -- splitting and joining --------------------------------------------------------------


def _pieces(input: Shaped, axis: int, lengths: Sequence[Extent]) -> list[tuple[Extent, ...]]:
    pieces = []
    for length in lengths:
        piece = list(input.extents)
        piece[axis] = length
        pieces.append(tuple(piece))
    return pieces


def _even_pieces(env: Sizes, input: Shaped, axis: int, size: Extent) -> list[tuple[Extent, ...]]:
    total = input.extents[axis]
    count = _counted(env, _ceildiv(total, size))
    if not count:
        return []
    return _pieces(input, axis, [size] * (count - 1) + [total - size * (count - 1)])


@rule("split")
def _split(env: Sizes, input: Shaped, split_size_or_sections: Any = None, dim: Any = 0, **kwargs: Any) -> Any:
    split = kwargs.get("split_size", split_size_or_sections)
    axis = _axis(dim, input.rank)
    if isinstance(split, (tuple, list)):
        return _pieces(input, axis, _shape((split,)))
    return _even_pieces(env, input, axis, _extent(split))


@rule("split_with_sizes")
def _split_with_sizes(env: Sizes, input: Shaped, split_sizes: Any, dim: Any = 0) -> Any:
    return _pieces(input, _axis(dim, input.rank), _shape((split_sizes,)))


@rule("chunk")
def _chunk(env: Sizes, input: Shaped, chunks: Any, dim: Any = 0) -> Any:
    axis = _axis(dim, input.rank)
    return _even_pieces(env, input, axis, _ceildiv(input.extents[axis], _int(chunks)))


@rule("unbind")
def _unbind(env: Sizes, input: Shaped, dim: Any = 0) -> Any:
    extents = list(input.extents)
    axis = _axis(dim, len(extents))
    count = _counted(env, extents[axis])
    del extents[axis]
    return [tuple(extents)] * count


def _joined(tensors: Any, axis_of: Callable[[int], int], *, append: bool) -> tuple[Extent, ...]:
    shaped = list(tensors)
    if not shaped or not all(isinstance(tensor, Shaped) for tensor in shaped):
        raise Undecidable("joining something that is not a tensor")
    # One-dimensional empty tensors are skipped by cat, whatever the rank.
    kept = [
        tensor
        for tensor in shaped
        if not (tensor.rank == 1 and isinstance(tensor.extents[0], int) and tensor.extents[0] == 0)
    ] or shaped[:1]
    first = list(kept[0].extents)
    axis = axis_of(len(first))
    for tensor in kept[1:]:
        for index, extent in enumerate(tensor.extents):
            if index != axis and not isinstance(first[index], int) and isinstance(extent, int):
                first[index] = extent
    if append:
        first[axis] = sum((tensor.extents[axis] for tensor in kept), start=0)
    else:
        first.insert(axis, len(shaped))
    return tuple(first)


@rule("cat", "concat", "concatenate")
def _cat(env: Sizes, tensors: Any, dim: Any = 0, **kwargs: Any) -> Any:
    dim = kwargs.get("axis", dim)
    return _joined(tensors, lambda rank: _axis(dim, rank), append=True)


@rule("stack")
def _stack(env: Sizes, tensors: Any, dim: Any = 0, **kwargs: Any) -> Any:
    return _joined(tensors, lambda rank: _axis(dim, rank + 1), append=False)


def _stacking(rank: int, axis: Callable[[list[Shaped]], int]) -> Callable[..., Any]:
    def stacking(env: Sizes, tensors: Any, **kwargs: Any) -> Any:
        raised = [Shaped(_raised(tensor.extents, rank), tensor.example) for tensor in tensors]
        return _joined(raised, lambda _rank: axis(raised), append=True)

    return stacking


rule("vstack", "row_stack")(_stacking(2, lambda tensors: 0))
rule("dstack")(_stacking(3, lambda tensors: 2))
rule("hstack")(_stacking(1, lambda tensors: 0 if all(t.rank == 1 for t in tensors) else 1))


@rule("column_stack")
def _column_stack(env: Sizes, tensors: Any) -> Any:
    columns = [
        Shaped((tensor.extents[0] if tensor.rank else 1, 1), tensor.example)
        if tensor.rank <= 1
        else tensor
        for tensor in tensors
    ]
    return _joined(columns, lambda rank: 1, append=True)


@rule("meshgrid")
def _meshgrid(env: Sizes, *tensors: Any, indexing: str = "ij") -> Any:
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        tensors = tuple(tensors[0])
    lengths = [tensor.extents[0] if tensor.rank else 1 for tensor in tensors]
    if indexing == "xy" and len(lengths) >= 2:
        lengths[0], lengths[1] = lengths[1], lengths[0]
    return [tuple(lengths)] * len(tensors)


@rule("broadcast_tensors")
def _broadcast_tensors(env: Sizes, *tensors: Any) -> Any:
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        tensors = tuple(tensors[0])
    extents = broadcast(env, *(tensor.extents for tensor in tensors))
    return [extents] * len(tensors)


# -- convolution, pooling, resampling -----------------------------------------------


def _window(
    env: Sizes,
    extent: Extent,
    kernel: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    ceil_mode: bool = False,
) -> Extent:
    """How many windows fit along an extent: the length of a sliding output."""

    kernel, stride, padding, dilation = (
        _extent(value) for value in (kernel, stride, padding, dilation)
    )
    span = extent + 2 * padding - dilation * (kernel - 1) - 1
    if not ceil_mode:
        return _floordiv(span, stride) + 1
    count = _ceildiv(span, stride) + 1
    # The last window has to start inside the input or its leading padding.
    if env.holds(sympy.Ge((count - 1) * stride, extent + padding)):
        count = count - 1
    return count


def _split_spatial(input: Shaped, count: int) -> tuple[list[Extent], list[Extent]]:
    extents = list(input.extents)
    return extents[: len(extents) - count], extents[len(extents) - count :]


def _convolution(
    env: Sizes,
    input: Shaped,
    weight: Shaped,
    stride: Any,
    padding: Any,
    dilation: Any,
    *,
    transposed: bool = False,
    output_padding: Any = 0,
    groups: Any = 1,
) -> tuple[Extent, ...]:
    count = weight.rank - 2
    leading, spatial = _split_spatial(input, count)
    kernels = list(weight.extents[2:])
    strides = _ntuple(stride, count)
    dilations = _ntuple(dilation, count)
    if isinstance(padding, str):
        if padding == "same":
            spatial = list(spatial)
        elif padding == "valid":
            spatial = [
                _window(env, extent, kernel, step, 0, dil)
                for extent, kernel, step, dil in zip(spatial, kernels, strides, dilations)
            ]
        else:
            raise Undecidable(f"padding {padding!r}")
    elif transposed:
        pads = _ntuple(padding, count)
        extras = _ntuple(output_padding, count)
        spatial = [
            (extent - 1) * _extent(step)
            - 2 * _extent(pad)
            + _extent(dil) * (kernel - 1)
            + _extent(extra)
            + 1
            for extent, kernel, step, pad, dil, extra in zip(
                spatial, kernels, strides, pads, dilations, extras
            )
        ]
    else:
        pads = _ntuple(padding, count)
        spatial = [
            _window(env, extent, kernel, step, pad, dil)
            for extent, kernel, step, pad, dil in zip(spatial, kernels, strides, pads, dilations)
        ]
    channels = weight.extents[1] * _extent(groups) if transposed else weight.extents[0]
    return (*leading[:-1], channels, *spatial)


@rule("nn.conv1d", "nn.conv2d", "nn.conv3d", "conv1d", "conv2d", "conv3d")
def _conv(
    env: Sizes,
    input: Shaped,
    weight: Shaped,
    bias: Any = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: Any = 1,
) -> Any:
    return _convolution(env, input, weight, stride, padding, dilation)


@rule(
    "nn.conv_transpose1d", "nn.conv_transpose2d", "nn.conv_transpose3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
)
def _conv_transpose(
    env: Sizes,
    input: Shaped,
    weight: Shaped,
    bias: Any = None,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    groups: Any = 1,
    dilation: Any = 1,
) -> Any:
    return _convolution(
        env, input, weight, stride, padding, dilation,
        transposed=True, output_padding=output_padding, groups=groups,
    )


def _pool(count: int, *, dilated: bool) -> Callable[..., Any]:
    """Max pooling takes ``dilation, ceil_mode, return_indices`` after the
    padding; average pooling takes ``ceil_mode`` and options that keep the size."""

    def pool(
        env: Sizes,
        input: Shaped,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        *rest: Any,
        **kwargs: Any,
    ) -> Any:
        rest = list(rest)
        dilation = kwargs.get("dilation", rest.pop(0) if dilated and rest else 1)
        ceil_mode = bool(kwargs.get("ceil_mode", rest.pop(0) if rest else False))
        indices = dilated and bool(kwargs.get("return_indices", rest.pop(0) if rest else False))
        leading, spatial = _split_spatial(input, count)
        kernels = _ntuple(kernel_size, count)
        strides = kernels if stride is None or stride in ((), []) else _ntuple(stride, count)
        extents = (
            *leading,
            *(
                _window(env, extent, kernel, step, pad, dil, ceil_mode)
                for extent, kernel, step, pad, dil in zip(
                    spatial, kernels, strides, _ntuple(padding, count), _ntuple(dilation, count)
                )
            ),
        )
        return [extents, extents] if indices else extents

    return pool


def _adaptive(count: int) -> Callable[..., Any]:
    def pool(env: Sizes, input: Shaped, output_size: Any, return_indices: bool = False) -> Any:
        leading, spatial = _split_spatial(input, count)
        extents = (
            *leading,
            *(
                extent if target is None else _extent(target)
                for extent, target in zip(spatial, _ntuple(output_size, count))
            ),
        )
        return [extents, extents] if return_indices else extents

    return pool


for _count in (1, 2, 3):
    rule(
        f"nn.max_pool{_count}d", f"max_pool{_count}d",
        f"nn.max_pool{_count}d_with_indices", f"max_pool{_count}d_with_indices",
    )(_pool(_count, dilated=True))
    rule(f"nn.avg_pool{_count}d", f"avg_pool{_count}d")(_pool(_count, dilated=False))
    rule(
        f"nn.adaptive_avg_pool{_count}d", f"adaptive_avg_pool{_count}d",
        f"nn.adaptive_max_pool{_count}d", f"adaptive_max_pool{_count}d",
    )(_adaptive(_count))


def _floor(value: Any) -> Extent:
    if isinstance(value, int):
        return value
    value = sympy.floor(value)
    return int(value) if value.is_Integer else value


@rule("nn.interpolate", "interpolate", "nn.upsample", "nn.upsample_nearest", "nn.upsample_bilinear")
def _interpolate(
    env: Sizes, input: Shaped, size: Any = None, scale_factor: Any = None, *args: Any, **kwargs: Any
) -> Any:
    count = input.rank - 2
    leading, spatial = _split_spatial(input, count)
    if size is not None:
        targets = [_extent(value) for value in _ntuple(size, count)]
    elif scale_factor is not None:
        targets = [
            _floor(extent * _rational(scale))
            for extent, scale in zip(spatial, _ntuple(scale_factor, count))
        ]
    else:
        raise Undecidable("interpolate without a size")
    return (*leading, *targets)


@rule("nn.pad", "pad", "constant_pad_nd")
def _pad(env: Sizes, input: Shaped, pad: Any, *args: Any, **kwargs: Any) -> Any:
    amounts = _shape((pad,))
    extents = list(input.extents)
    for pair in range(len(amounts) // 2):
        axis = len(extents) - 1 - pair
        extents[axis] = extents[axis] + amounts[2 * pair] + amounts[2 * pair + 1]
    return tuple(extents)


@rule("nn.unfold")
def _im2col(
    env: Sizes, input: Shaped, kernel_size: Any, dilation: Any = 1, padding: Any = 0, stride: Any = 1
) -> Any:
    *batch, channels, height, width = input.extents
    kernels = _ntuple(kernel_size, 2)
    blocks = _product(
        _window(env, extent, kernel, step, pad, dil)
        for extent, kernel, step, pad, dil in zip(
            (height, width), kernels, _ntuple(stride, 2), _ntuple(padding, 2), _ntuple(dilation, 2)
        )
    )
    return (*batch, channels * _product(_extent(kernel) for kernel in kernels), blocks)


@rule("nn.fold")
def _col2im(env: Sizes, input: Shaped, output_size: Any, kernel_size: Any, *args: Any, **kwargs: Any) -> Any:
    *batch, columns, _blocks = input.extents
    channels = _floordiv(columns, _product(_extent(kernel) for kernel in _ntuple(kernel_size, 2)))
    return (*batch, channels, *(_extent(value) for value in _ntuple(output_size, 2)))


@rule("nn.grid_sample", "grid_sampler")
def _grid_sample(env: Sizes, input: Shaped, grid: Shaped, *args: Any, **kwargs: Any) -> Any:
    return (input.extents[0], input.extents[1], *grid.extents[1:-1])


@rule("nn.affine_grid", "affine_grid_generator")
def _affine_grid(env: Sizes, theta: Shaped, size: Any, *args: Any, **kwargs: Any) -> Any:
    target = _shape((size,))
    return (target[0], *target[2:], len(target) - 2)


@rule("nn.glu", "glu")
def _glu(env: Sizes, input: Shaped, dim: Any = -1) -> Any:
    extents = list(input.extents)
    axis = _axis(dim, len(extents))
    extents[axis] = _floordiv(extents[axis], 2)
    return tuple(extents)


# -- attention ----------------------------------------------------------------------


@rule("nn.scaled_dot_product_attention", "scaled_dot_product_attention")
def _attention(env: Sizes, query: Shaped, key: Shaped, value: Shaped, *args: Any, **kwargs: Any) -> Any:
    return (*query.extents[:-1], value.extents[-1])


_ATTENTION_OPTIONS = (
    "in_proj_weight in_proj_bias bias_k bias_v add_zero_attn dropout_p out_proj_weight "
    "out_proj_bias training key_padding_mask need_weights attn_mask "
    "use_separate_proj_weight q_proj_weight k_proj_weight v_proj_weight static_k "
    "static_v average_attn_weights is_causal"
).split()


@rule("nn.multi_head_attention_forward")
def _multi_head_attention(
    env: Sizes,
    query: Shaped,
    key: Shaped,
    value: Shaped,
    embed_dim_to_check: Any,
    num_heads: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    options = dict(zip(_ATTENTION_OPTIONS, args))
    options.update(kwargs)
    target_length = query.extents[0]
    source_length = key.extents[0]
    if isinstance(options.get("static_k"), Shaped):
        source_length = options["static_k"].extents[1]
    if options.get("bias_k") is not None:
        source_length = source_length + 1
    if options.get("add_zero_attn"):
        source_length = source_length + 1
    if not options.get("need_weights", True):
        return [query.extents, None]
    if options.get("average_attn_weights", True):
        weights: tuple[Extent, ...] = (target_length, source_length)
    else:
        weights = (_extent(num_heads), target_length, source_length)
    if query.rank == 3:
        weights = (query.extents[1], *weights)
    return [query.extents, weights]


# -- losses -------------------------------------------------------------------------


def _loss(*, classes: bool) -> Callable[..., Any]:
    """A loss: a 0-d result unless ``reduction='none'``, which keeps one value
    per element (per sample, for a loss over ``classes``)."""

    def loss(env: Sizes, input: Shaped, target: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("size_average") is not None or kwargs.get("reduce") is not None:
            raise Undecidable("a legacy reduction argument")
        reduction = kwargs.get("reduction")
        if reduction is None:
            reduction = next((value for value in args if isinstance(value, str)), "mean")
        if reduction != "none":
            return ()
        if not classes:
            return broadcast(env, input.extents, target.extents)
        if isinstance(target, Shaped) and target.rank == input.rank - 1:
            return target.extents
        return (input.extents[0], *input.extents[2:]) if input.rank > 1 else ()

    return loss


for _name in (
    "mse_loss l1_loss smooth_l1_loss huber_loss binary_cross_entropy "
    "binary_cross_entropy_with_logits kl_div soft_margin_loss poisson_nll_loss"
).split():
    rule(f"nn.{_name}", _name)(_loss(classes=False))
for _name in ("cross_entropy", "nll_loss"):
    rule(f"nn.{_name}", _name)(_loss(classes=True))


# -- factories ----------------------------------------------------------------------


@rule("zeros", "ones", "empty", "rand", "randn")
def _filled(env: Sizes, *size: Any, **kwargs: Any) -> Any:
    if "size" in kwargs:
        size = (kwargs["size"],)
    return tuple(_shape(size))


@rule("full")
def _full(env: Sizes, size: Any, fill_value: Any = None, **kwargs: Any) -> Any:
    return tuple(_shape((size,)))


@rule("randint")
def _randint(env: Sizes, *args: Any, **kwargs: Any) -> Any:
    size = kwargs.get("size")
    if size is None:
        size = next((value for value in args if isinstance(value, (tuple, list))), None)
    if size is None:
        raise Undecidable("randint without a size")
    return tuple(_shape((size,)))


@rule(
    "zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like",
    "randint_like", "bernoulli", "poisson",
)
def _like(env: Sizes, input: Shaped, *args: Any, **kwargs: Any) -> Any:
    return input.extents


@rule("new_zeros", "new_ones", "new_empty")
def _new_filled(env: Sizes, input: Shaped, *size: Any, **kwargs: Any) -> Any:
    if "size" in kwargs:
        size = (kwargs["size"],)
    return tuple(_shape(size))


@rule("new_full")
def _new_full(env: Sizes, input: Shaped, size: Any, fill_value: Any = None, **kwargs: Any) -> Any:
    return tuple(_shape((size,)))


@rule("arange")
def _arange(env: Sizes, *args: Any, **kwargs: Any) -> Any:
    bounds = list(args[:3])
    for key in ("start", "end", "step"):
        if key in kwargs:
            bounds.append(kwargs[key])
    if any(isinstance(value, (Shaped, Data)) for value in bounds):
        raise DataDependent("arange bounded by a value held in a tensor")
    if len(bounds) == 1:
        start, end, step = 0, bounds[0], 1
    elif len(bounds) == 2:
        (start, end), step = bounds, 1
    else:
        start, end, step = bounds
    start, end, step = (_rational(value) for value in (start, end, step))
    length = sympy.ceiling(sympy.sympify(end - start) / step)
    if length.is_Integer:
        return (max(int(length), 0),)
    return (length,)


@rule("linspace", "logspace")
def _linspace(env: Sizes, start: Any, end: Any, steps: Any, *args: Any, **kwargs: Any) -> Any:
    return (_extent(steps),)


@rule("eye")
def _eye(env: Sizes, n: Any, m: Any = None, **kwargs: Any) -> Any:
    rows = _extent(n)
    return (rows, rows if m is None else _extent(m))


del _name, _count, _rank

__all__ = [
    "DATA_DEPENDENT",
    "Data",
    "DataDependent",
    "Shaped",
    "Sizes",
    "Undecidable",
    "broadcast",
    "find",
    "operation",
]
