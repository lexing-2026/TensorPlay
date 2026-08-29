"""Numeric comparison engine for the testing utilities.

The public entrypoints are :func:`assert_close` and :func:`assert_allclose`.
Both dispatch on the category of their inputs (tensors, python scalars,
booleans, ``None`` or arbitrary objects) and raise :class:`AssertionError`
with a structured mismatch report on failure.
"""

import collections.abc
import math
import warnings
from typing import Any, Callable, Sequence, Tuple, Union

import tensorplay as tp
from tensorplay import Tensor

try:
    import numpy as np
except ImportError:
    np = None
HAS_NUMPY = np is not None

__all__ = [
    "assert_close",
    "assert_allclose",
    "default_tolerances",
    "get_tolerances",
]

# {dtype: (rtol, atol)}: default per-dtype testing tolerances.
_DTYPE_PRECISIONS = {
    tp.float16: (0.001, 1e-5),
    tp.bfloat16: (0.016, 1e-5),
    tp.float32: (1.3e-6, 1e-5),
    tp.float64: (1e-7, 1e-7),
    tp.complex64: (1.3e-6, 1e-5),
    tp.complex128: (1e-7, 1e-7),
}

_INTEGRAL_TYPES = [
    tp.uint8,
    tp.int8,
    tp.int16,
    tp.int32,
    tp.int64,
    tp.uint16,
    tp.uint32,
    tp.uint64,
]
_FLOATING_TYPES = [tp.float16, tp.bfloat16, tp.float32, tp.float64]
_COMPLEX_TYPES = [tp.complex64, tp.complex128]
_BOOLEAN_OR_INTEGRAL_TYPES = [tp.bool, *_INTEGRAL_TYPES]
_FLOATING_OR_COMPLEX_TYPES = [*_FLOATING_TYPES, *_COMPLEX_TYPES]


class UnsupportedInputs(Exception):
    """Raised during the construction of a :class:`Pair` when it cannot handle the inputs."""


class ErrorMeta(Exception):
    """Internal exception carrying the eventual error type and message."""

    def __init__(
        self, type: type[Exception], msg: str, *, id: tuple[Any, ...] = ()
    ) -> None:
        super().__init__(msg)
        self.type = type
        self.msg = msg
        self.id = id

    def to_error(self, msg: str | Callable[[str], str] | None = None) -> Exception:
        if not isinstance(msg, str):
            generated_msg = self.msg
            if self.id:
                generated_msg += (
                    f"\n\nThe failure occurred for item {''.join(str([item]) for item in self.id)}"
                )
            msg = msg(generated_msg) if callable(msg) else generated_msg
        return self.type(msg)


def default_tolerances(
    *inputs: Union[Tensor, tp.dtype],
    dtype_precisions: dict[tp.dtype, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Returns the default absolute and relative testing tolerances for a set of inputs based on the dtype.

    Returns:
        (Tuple[float, float]): Loosest tolerances of all input dtypes.
    """
    dtypes = []
    for input in inputs:
        if isinstance(input, Tensor):
            dtypes.append(input.dtype)
        elif isinstance(input, tp.dtype):
            dtypes.append(input)
        else:
            raise TypeError(
                f"Expected a tensor or a dtype, but got {type(input)} instead."
            )
    dtype_precisions = dtype_precisions or _DTYPE_PRECISIONS
    rtols, atols = zip(
        *[dtype_precisions.get(dtype, (0.0, 0.0)) for dtype in dtypes]
    )
    return max(rtols), max(atols)


def get_tolerances(
    *inputs: Union[Tensor, tp.dtype],
    rtol: float | None,
    atol: float | None,
    id: tuple[Any, ...] = (),
) -> tuple[float, float]:
    """Gets absolute and relative tolerances to be used for numeric comparisons.

    If both ``rtol`` and ``atol`` are specified, this is a no-op. If neither is
    specified, :func:`default_tolerances` is used. Specifying only one raises a
    :class:`ValueError`, since a single tolerance might lead to surprising
    results.
    """
    if (rtol is None) ^ (atol is None):
        raise ErrorMeta(
            ValueError,
            f"Both 'rtol' and 'atol' must be either specified or omitted, "
            f"but got no {'rtol' if rtol is None else 'atol'}.",
            id=id,
        ).to_error()
    elif rtol is not None and atol is not None:
        return rtol, atol
    else:
        return default_tolerances(*inputs)


def _make_mismatch_msg(
    *,
    default_identifier: str,
    identifier: str | Callable[[str], str] | None = None,
    extra: str | None = None,
    abs_diff: float,
    abs_diff_idx: int | tuple[int, ...] | None = None,
    atol: float,
    rel_diff: float,
    rel_diff_idx: int | tuple[int, ...] | None = None,
    rtol: float,
) -> str:
    """Makes a mismatch error message for numeric values."""
    equality = rtol == 0 and atol == 0

    def make_diff_msg(
        *, type: str, diff: float, idx: int | tuple[int, ...] | None, tol: float
    ) -> str:
        if idx is None:
            msg = f"{type.title()} difference: {diff}"
        else:
            msg = f"Greatest {type} difference: {diff} at index {idx}"
        if not equality:
            msg += f" (up to {tol} allowed)"
        return msg + "\n"

    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not {'equal' if equality else 'close'}!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"

    msg += make_diff_msg(type="absolute", diff=abs_diff, idx=abs_diff_idx, tol=atol)
    msg += make_diff_msg(type="relative", diff=rel_diff, idx=rel_diff_idx, tol=rtol)

    return msg.strip()


def _make_same_value_mismatch_msg(
    *,
    default_identifier: str,
    identifier: str | Callable[[str], str] | None = None,
    extra: str | None = None,
    first_mismatch_idx: tuple[int, ...] | None = None,
) -> str:
    """Makes a mismatch error message for values compared by equality."""
    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not 'equal'!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"
    if first_mismatch_idx is not None:
        msg += f"The first mismatched element is at index {first_mismatch_idx}.\n"
    return msg.strip()


def make_scalar_mismatch_msg(
    actual: bool | int | float | complex,
    expected: bool | int | float | complex,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = None,
) -> str:
    abs_diff = abs(actual - expected)
    rel_diff = float("inf") if expected == 0 else abs_diff / abs(expected)
    return _make_mismatch_msg(
        default_identifier="Scalars",
        identifier=identifier,
        extra=f"Expected {expected} but got {actual}.",
        abs_diff=abs_diff,
        atol=atol,
        rel_diff=rel_diff,
        rtol=rtol,
    )


def make_tensor_mismatch_msg(
    actual: Tensor,
    expected: Tensor,
    matches: Tensor,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = None,
) -> str:
    """Makes a mismatch error message for tensors.

    ``matches`` is a boolean mask of the same shape as the inputs indicating
    the locations that satisfy the tolerance.
    """

    def unravel_flat_index(flat_index: int) -> tuple[int, ...]:
        if not matches.shape:
            return ()

        inverse_index = []
        index = flat_index
        for size in matches.shape[::-1]:
            div, mod = divmod(index, size)
            index = div
            inverse_index.append(mod)

        return tuple(inverse_index[::-1])

    number_of_elements = matches.numel()
    total_mismatches = number_of_elements - int(matches.sum().item())
    extra = (
        f"Mismatched elements: {total_mismatches} / {number_of_elements} "
        f"({total_mismatches / number_of_elements:.1%})"
    )

    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    matches_flat = matches.flatten()

    if not _is_floating_dtype(actual_flat.dtype) and not _is_complex_dtype(
        actual_flat.dtype
    ):
        actual_flat = actual_flat.to(tp.int64)
        expected_flat = expected_flat.to(tp.int64)

    abs_diff = (actual_flat - expected_flat).abs()
    # Only mismatches contribute to the reported maxima.
    abs_diff = abs_diff.masked_fill(matches_flat, 0)
    max_abs = abs_diff.max(0)

    rel_diff = abs_diff / expected_flat.abs()
    rel_diff = rel_diff.masked_fill(matches_flat, 0)
    max_rel = rel_diff.max(0)

    return _make_mismatch_msg(
        default_identifier="Tensor-likes",
        identifier=identifier,
        extra=extra,
        abs_diff=max_abs.values.item(),
        abs_diff_idx=unravel_flat_index(int(max_abs.indices.item())),
        atol=atol,
        rel_diff=max_rel.values.item(),
        rel_diff_idx=unravel_flat_index(int(max_rel.indices.item())),
        rtol=rtol,
    )


def _is_floating_dtype(dtype: tp.dtype) -> bool:
    return dtype in _FLOATING_TYPES


def _is_complex_dtype(dtype: tp.dtype) -> bool:
    return dtype in _COMPLEX_TYPES


class Pair:
    """Base class for all comparison pairs."""

    def __init__(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...] = (), **unknown_parameters: Any
    ) -> None:
        self.actual = actual
        self.expected = expected
        self.id = id
        self._unknown_parameters = unknown_parameters

    @staticmethod
    def _inputs_not_supported() -> None:
        raise UnsupportedInputs

    @staticmethod
    def _check_inputs_isinstance(*inputs: Any, cls: type | tuple[type, ...]) -> None:
        if not all(isinstance(input, cls) for input in inputs):
            Pair._inputs_not_supported()

    def _fail(self, type: type[Exception], msg: str) -> None:
        raise ErrorMeta(type, msg, id=self.id)

    def compare(self) -> None:
        raise NotImplementedError

    def extra_repr(self) -> Sequence[str | tuple[str, Any]]:
        return ()

    def _preamble(self) -> str:
        preamble = f"Comparing {self.actual!r} and {self.expected!r}"
        extra_repr = [
            f"{name if isinstance(name, str) else name[0]}: {self.actual if isinstance(name, str) else name[1]}"
            for name in self.extra_repr()
        ]
        if extra_repr:
            preamble += "\n" + "\n".join(extra_repr)
        return preamble

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._preamble()})"


class ObjectPair(Pair):
    def compare(self) -> None:
        if self.actual != self.expected:
            self._fail(
                AssertionError,
                f"Objects are not equal:\n\n{self.actual} != {self.expected}",
            )


class NonePair(Pair):
    """Pair for ``None`` inputs."""

    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None:
        if not (actual is None or expected is None):
            self._inputs_not_supported()

        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> None:
        if not (self.actual is None and self.expected is None):
            self._fail(
                AssertionError, f"None mismatch: {self.actual} is not {self.expected}"
            )


class BooleanPair(Pair):
    """Pair for :class:`bool` inputs.

    .. note::

        If ``numpy`` is available, also handles :class:`numpy.bool_` inputs.
    """

    def __init__(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...] = (), **other_parameters: Any
    ) -> None:
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, **other_parameters)

    @property
    def _supported_types(self) -> tuple[type, ...]:
        cls: list[type] = [bool]
        if HAS_NUMPY:
            cls.append(np.bool_)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...]
    ) -> tuple[bool, bool]:
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        actual, expected = (
            self._to_bool(bool_like) for bool_like in (actual, expected)
        )
        return actual, expected

    def _to_bool(self, bool_like: Any) -> bool:
        if isinstance(bool_like, bool):
            return bool_like
        elif HAS_NUMPY and isinstance(bool_like, np.bool_):
            return bool_like.item()
        else:
            raise ErrorMeta(TypeError, f"Unknown boolean type {type(bool_like)}.", id=self.id)

    def compare(self) -> None:
        if self.actual is not self.expected:
            self._fail(
                AssertionError,
                f"Booleans mismatch: {self.actual} is not {self.expected}",
            )


class NumberPair(Pair):
    """Pair for python number (:class:`int`, :class:`float`, and :class:`complex`) inputs.

    .. note::

        If ``numpy`` is available, also handles :class:`numpy.number` inputs.

    The following table displays the correspondence between the python number
    type and the dtype the tolerances are derived from:

    +------------------+-----------------------+
    | ``type``         | corresponding dtype   |
    +==================+=======================+
    | :class:`int`     | ``int64``             |
    +------------------+-----------------------+
    | :class:`float`   | ``float64``           |
    +------------------+-----------------------+
    | :class:`complex` | ``complex128``        |
    +------------------+-----------------------+
    """

    _TYPE_TO_DTYPE = {
        int: tp.int64,
        float: tp.float64,
        complex: tp.complex128,
    }
    _NUMBER_TYPES = tuple(_TYPE_TO_DTYPE.keys())

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        check_dtype: bool = False,
        **other_parameters: Any,
    ) -> None:
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, id=id, **other_parameters)

        self.rtol, self.atol = get_tolerances(
            *[self._TYPE_TO_DTYPE[type(input)] for input in (actual, expected)],
            rtol=rtol,
            atol=atol,
            id=id,
        )
        self.equal_nan = equal_nan
        self.check_dtype = check_dtype

    @property
    def _supported_types(self) -> tuple[type, ...]:
        cls = list(self._NUMBER_TYPES)
        if HAS_NUMPY:
            cls.append(np.number)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...]
    ) -> tuple[int | float | complex, int | float | complex]:
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        actual, expected = (
            self._to_number(number_like, id=id) for number_like in (actual, expected)
        )
        return actual, expected

    def _to_number(
        self, number_like: Any, *, id: tuple[Any, ...]
    ) -> int | float | complex:
        if HAS_NUMPY and isinstance(number_like, np.number):
            return number_like.item()
        elif isinstance(number_like, self._NUMBER_TYPES):
            if isinstance(number_like, bool):
                # Booleans route to BooleanPair; letting them through here
                # would hit an unresolvable dtype lookup.
                self._inputs_not_supported()
            return number_like
        else:
            raise ErrorMeta(
                TypeError, f"Unknown number type {type(number_like)}.", id=id
            )

    def compare(self) -> None:
        if self.check_dtype and type(self.actual) is not type(self.expected):
            self._fail(
                AssertionError,
                f"The (d)types do not match: {type(self.actual)} != {type(self.expected)}.",
            )

        actual = self.actual
        expected = self.expected

        if actual == expected:
            return

        if self.equal_nan and _isnan(actual) and _isnan(expected):
            return

        abs_diff = abs(actual - expected)
        tolerance = self.atol + self.rtol * abs(expected)

        if math.isfinite(abs_diff) and abs_diff <= tolerance:
            return

        self._fail(
            AssertionError,
            make_scalar_mismatch_msg(
                actual, expected, rtol=self.rtol, atol=self.atol
            ),
        )


class TensorLikePair(Pair):
    """Pair for tensor-like inputs.

    Kwargs:
        allow_subclasses (bool): If ``True`` (default), subclasses of tensors and
            python types are allowed.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified.
        equal_nan (bool): If ``True``, two ``NaN`` values are considered equal. Defaults to ``False``.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            device. If disabled, tensors on different devices are moved to the CPU before comparison.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same
            dtype. If disabled, tensors with different dtypes are promoted to a common dtype before
            comparison.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same
            layout.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they
            have the same stride.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        allow_subclasses: bool = True,
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        check_device: bool = True,
        check_dtype: bool = True,
        check_layout: bool = True,
        check_stride: bool = False,
        **other_parameters: Any,
    ) -> None:
        actual, expected = self._process_inputs(
            actual, expected, id=id, allow_subclasses=allow_subclasses
        )
        super().__init__(actual, expected, id=id, **other_parameters)

        self.rtol, self.atol = get_tolerances(
            actual, expected, rtol=rtol, atol=atol, id=self.id
        )
        self.equal_nan = equal_nan
        self.check_device = check_device
        self.check_dtype = check_dtype
        self.check_layout = check_layout
        self.check_stride = check_stride

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...], allow_subclasses: bool
    ) -> tuple[Tensor, Tensor]:
        directly_related = isinstance(actual, type(expected)) or isinstance(
            expected, type(actual)
        )
        tensor_like = (Tensor, np.ndarray) if HAS_NUMPY else (Tensor,)
        if not directly_related and not (
            isinstance(actual, tensor_like) and isinstance(expected, tensor_like)
        ):
            self._inputs_not_supported()

        if not allow_subclasses and type(actual) is not type(expected):
            self._inputs_not_supported()

        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        return actual, expected

    def _to_tensor(self, tensor_like: Any) -> Tensor:
        if isinstance(tensor_like, Tensor):
            return tensor_like

        try:
            return tp.as_tensor(tensor_like)
        except Exception:
            self._inputs_not_supported()

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)

        actual, expected = self._equalize_attributes(actual, expected)
        self._compare_values(actual, expected)

    def _compare_attributes(self, actual: Tensor, expected: Tensor) -> None:
        """Checks if the attributes of two tensors match.

        The shape is always checked. Layout, stride, device, and dtype checks
        are optional and can be disabled through the corresponding ``check_*``
        flag during construction of the pair.
        """

        def raise_mismatch_error(
            attribute_name: str, actual_value: Any, expected_value: Any
        ) -> None:
            self._fail(
                AssertionError,
                f"The values for attribute '{attribute_name}' do not match: "
                f"{actual_value} != {expected_value}.",
            )

        if actual.shape != expected.shape:
            raise_mismatch_error("shape", actual.shape, expected.shape)

        if actual.layout != expected.layout:
            if self.check_layout:
                raise_mismatch_error("layout", actual.layout, expected.layout)
        elif (
            actual.layout == tp.strided
            and self.check_stride
            and actual.stride() != expected.stride()
        ):
            raise_mismatch_error("stride()", actual.stride(), expected.stride())

        if self.check_device and actual.device != expected.device:
            raise_mismatch_error("device", actual.device, expected.device)

        if self.check_dtype and actual.dtype != expected.dtype:
            raise_mismatch_error("dtype", actual.dtype, expected.dtype)

    def _equalize_attributes(self, actual: Tensor, expected: Tensor) -> tuple[Tensor, Tensor]:
        """Equalizes some attributes of two tensors for value comparison.

        Tensors on different devices are moved to CPU memory, and tensors of
        different dtypes are promoted to a common dtype.
        """
        if actual.device != expected.device:
            actual = actual.cpu()
            expected = expected.cpu()

        if actual.dtype != expected.dtype:
            actual_dtype = actual.dtype
            expected_dtype = expected.dtype
            # Unsigned dtypes above 8 bits do not promote soundly in general,
            # but for testing purposes confusion with large values is
            # unlikely.
            if actual_dtype in [tp.uint64, tp.uint32, tp.uint16]:
                actual_dtype = tp.int64
            if expected_dtype in [tp.uint64, tp.uint32, tp.uint16]:
                expected_dtype = tp.int64
            dtype = tp.promote_types(actual_dtype, expected_dtype)
            actual = actual.to(dtype)
            expected = expected.to(dtype)

        return actual, expected

    def _compare_values(self, actual: Tensor, expected: Tensor) -> None:
        if actual.numel() == 0:
            return

        if _is_bool_dtype(actual.dtype):
            return self._compare_regular_values_equal(
                actual,
                expected,
                identifier="Tensor-likes",
            )
        elif _is_integral_dtype(actual.dtype):
            return self._compare_regular_values_equal(
                actual,
                expected,
                identifier="Tensor-likes",
            )
        else:
            return self._compare_regular_values_close(
                actual,
                expected,
                rtol=self.rtol,
                atol=self.atol,
                equal_nan=self.equal_nan,
            )

    def _compare_regular_values_equal(
        self,
        actual: Tensor,
        expected: Tensor,
        *,
        identifier: str | Callable[[str], str] | None = None,
    ) -> None:
        matches = actual == expected
        if bool(matches.all().item()):
            return

        if actual.shape == ():
            self._fail(
                AssertionError,
                f"Scalars are not 'equal'!\n\nExpected {expected.item()} but got {actual.item()}.",
            )
        else:
            msg = _make_same_value_mismatch_msg(
                default_identifier=identifier
                if isinstance(identifier, str)
                else "Tensor-likes",
                identifier=identifier if not isinstance(identifier, str) else None,
                extra=None,
                first_mismatch_idx=_first_mismatch_index(matches),
            )
            self._fail(AssertionError, msg)

    def _compare_regular_values_close(
        self,
        actual: Tensor,
        expected: Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
        identifier: str | Callable[[str], str] | None = None,
    ) -> None:
        matches = _isclose_tensor(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        if bool(matches.all().item()):
            return

        if actual.shape == ():
            msg = make_scalar_mismatch_msg(
                actual.item(),
                expected.item(),
                rtol=rtol,
                atol=atol,
                identifier=identifier,
            )
        else:
            msg = make_tensor_mismatch_msg(
                actual, expected, matches, rtol=rtol, atol=atol, identifier=identifier
            )
        self._fail(AssertionError, msg)

    def extra_repr(self) -> Sequence[str | tuple[str, Any]]:
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_device",
            "check_dtype",
            "check_layout",
            "check_stride",
        )


def _is_bool_dtype(dtype: tp.dtype) -> bool:
    return dtype == tp.bool


def _is_integral_dtype(dtype: tp.dtype) -> bool:
    return dtype in _INTEGRAL_TYPES


def _isnan(value: Any) -> bool:
    if isinstance(value, complex):
        return math.isnan(value.real) and math.isnan(value.imag)
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _is_integral_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_floating_number(value: Any) -> bool:
    return isinstance(value, float)


def _is_complex_number(value: Any) -> bool:
    return isinstance(value, complex)


def _first_mismatch_index(matches: Tensor) -> tuple[int, ...] | None:
    flat = matches.flatten()
    for i in range(flat.numel()):
        if not flat[i].item():
            return _unravel(i, tuple(matches.shape))
    return None


def _unravel(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    inverse_index = []
    index = flat_index
    for size in shape[::-1]:
        div, mod = divmod(index, size)
        index = div
        inverse_index.append(mod)
    return tuple(inverse_index[::-1])


def _isclose(
    actual: float | complex,
    expected: float | complex,
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> bool:
    if math.isnan(actual) and math.isnan(expected):
        return equal_nan
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    return abs(actual - expected) <= atol + rtol * abs(expected)


def _isclose_tensor(
    actual: Tensor, expected: Tensor, *, rtol: float, atol: float, equal_nan: bool
) -> Tensor:
    matches = tp.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return matches


def originate_pairs(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]],
    sequence_types: tuple[type, ...] = (collections.abc.Sequence,),
    mapping_types: tuple[type, ...] = (collections.abc.Mapping,),
    id: tuple[Any, ...] = (),
    **options: Any,
) -> list[Pair]:
    """Originates pairs from the individual inputs.

    ``actual`` and ``expected`` can be possibly nested sequences or mappings,
    in which case the pairs are originated by recursing through them.
    """
    # TODO: the order of the sequence_types and mapping_types is not significant

    actual_type = type(actual)
    expected_type = type(expected)

    if (
        _issubclass(actual_type, sequence_types)
        and _issubclass(expected_type, sequence_types)
        and not isinstance(actual, str)
        and not isinstance(expected, str)
    ):
        if len(actual) != len(expected):
            raise ErrorMeta(
                ValueError,
                f"The length of the sequences do not match: {len(actual)} != {len(expected)}",
                id=id,
            )
        return [
            pair
            for idx, (actual_exp, expected_exp) in enumerate(zip(actual, expected))
            for pair in originate_pairs(
                actual_exp,
                expected_exp,
                pair_types=pair_types,
                sequence_types=sequence_types,
                mapping_types=mapping_types,
                id=(*id, idx),
                **options,
            )
        ]
    elif _issubclass(actual_type, mapping_types) and _issubclass(
        expected_type, mapping_types
    ):
        if actual.keys() != expected.keys():
            raise ErrorMeta(
                ValueError,
                f"The keys of the mappings do not match:\n{actual.keys()} != {expected.keys()}",
                id=id,
            )
        return [
            pair
            for key in actual.keys()
            for pair in originate_pairs(
                actual[key],
                expected[key],
                pair_types=pair_types,
                sequence_types=sequence_types,
                mapping_types=mapping_types,
                id=(*id, key),
                **options,
            )
        ]
    else:
        for pair_type in pair_types:
            try:
                pair = pair_type(actual, expected, id=id, **options)
            except UnsupportedInputs:
                continue
            else:
                return [pair]

        raise ErrorMeta(
            TypeError,
            f"No comparison pair was able to handle inputs of type {actual_type} and {expected_type}.",
            id=id,
        )


def _issubclass(type: type, classinfo: tuple[type, ...]) -> bool:
    return any(issubclass(type, cls) for cls in classinfo)


def assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    msg: str | Callable[[str], str] | None = None,
):
    """Asserts that ``actual`` and ``expected`` are close.

    If ``actual`` and ``expected`` are strided and finite, they are considered
    close if

    .. math::

        \\lvert \\text{actual} - \\text{expected} \\rvert \\le \\texttt{atol} + \\texttt{rtol} \\cdot \\lvert \\text{expected} \\rvert

    Non-finite values (``-inf`` and ``inf``) are only considered close if and
    only if they are equal. ``NaN``'s are only considered equal to each other
    if ``equal_nan`` is ``True``.

    In addition, they are only considered close if they have the same

    - device (if ``check_device`` is ``True``),
    - dtype (if ``check_dtype`` is ``True``),
    - layout (if ``check_layout`` is ``True``), and
    - stride (if ``check_stride`` is ``True``).

    If either ``actual`` or ``expected`` is a scalar or a nested python
    container, the other side is converted to a tensor-like value before the
    comparison.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default) and other than exact type match, inputs that are
            subclasses of each other are considered close.
        rtol (Optional[float]): Relative tolerance. If specified :attr:`atol` must also be specified. If omitted,
            default values based on the :attr:`~tensorplay.Tensor.dtype` are selected. See below for details.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol` must also be specified. If omitted,
            default values based on the :attr:`~tensorplay.Tensor.dtype` are selected. See below for details.
        equal_nan (bool): If ``True``, two ``NaN`` values are considered equal. Defaults to ``False``.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same device.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same dtype.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same layout.
        check_stride (bool): If ``True``, asserts that corresponding strided tensors have the same stride.
        msg (Optional[Union[str, Callable[[str], str]]]): Optional error message to use in case of failure.

    Raises:
        ValueError: If only :attr:`rtol` or only :attr:`atol` is specified.
        AssertionError: If corresponding values are not close.

    Default tolerances by dtype:

    ================  ==========  ==========
    ``dtype``         ``rtol``    ``atol``
    ================  ==========  ==========
    ``float16``       ``1e-3``    ``1e-5``
    ``bfloat16``      ``1.6e-2``  ``1e-5``
    ``float32``       ``1.3e-6``  ``1e-5``
    ``float64``       ``1e-7``    ``1e-7``
    ``complex64``     ``1.3e-6``  ``1e-5``
    ``complex128``    ``1e-7``    ``1e-7``
    ================  ==========  ==========

    .. note::

        Tensors are compared elementwise, allowing for a relative and an
        absolute tolerance per element. If both tolerances are omitted, the
        loosest tolerance of the involved dtypes is selected.
    """
    pair_types = [NonePair, BooleanPair, NumberPair, TensorLikePair]

    error_meta = None
    pairs = []
    try:
        pairs = originate_pairs(
            actual,
            expected,
            pair_types=pair_types,
            allow_subclasses=allow_subclasses,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            check_device=check_device,
            check_dtype=check_dtype,
            check_layout=check_layout,
            check_stride=check_stride,
        )
    except ErrorMeta as error:
        error_meta = error

    if error_meta is not None:
        raise error_meta.to_error(msg)

    for pair in pairs:
        try:
            pair.compare()
        except ErrorMeta as error:
            raise error.to_error(msg) from None


def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = True,
    msg: str = "",
) -> None:
    """Legacy alias of :func:`assert_close` with positional tolerances."""
    if rtol is None and atol is None:
        rtol, atol = default_tolerances(
            *(
                input if isinstance(input, (Tensor, tp.dtype)) else tp.as_tensor(input)
                for input in (actual, expected)
            ),
            dtype_precisions={
                tp.float16: (1e-3, 1e-3),
                tp.float32: (1e-4, 1e-5),
                tp.float64: (1e-5, 1e-8),
            },
        )

    assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=True,
        check_dtype=False,
        check_stride=False,
        msg=msg or None,
    )
