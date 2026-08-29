"""Shared helpers for the test suite: environment flags, an assertion-rich
``TestCase``, and the standard test entrypoint.
"""

import contextlib
import enum
import math
import os
import platform
import sys
import unittest
import warnings
from functools import wraps
from typing import Any

import tensorplay as tp
from tensorplay import Tensor

import tensorplay.testing._comparison as _cmp
from tensorplay.testing._creation import make_tensor
from tensorplay.testing._comparison import (
    NonePair,
    NumberPair,
    BooleanPair,
    TensorLikePair,
    Pair,
    ErrorMeta,
    UnsupportedInputs,
    originate_pairs,
    make_scalar_mismatch_msg,
    _is_bool_dtype,
)

try:
    import numpy as np
except ImportError:
    np = None
HAS_NUMPY = np is not None

__all__ = [
    "IS_WINDOWS",
    "IS_LINUX",
    "IS_MACOS",
    "TEST_CUDA",
    "TEST_NUM_GPUS",
    "TEST_MULTIGPU",
    "TEST_NUMPY",
    "TEST_SCIPY",
    "TEST_MKL",
    "TEST_WITH_SLOW",
    "TestCase",
    "run_tests",
    "freeze_rng_state",
    "set_rng_seed",
    "get_rng_seed",
    "suppress_warnings",
    "slowTest",
    "subtest",
    "lazy_skip_if",
    "skipIfNoLapack",
    "skipIfNoSciPy",
    "skipIfNoNumPy",
    "noncontiguous_like",
    "dtype_name",
    "make_tensor",
]

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

TEST_CUDA = tp.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and tp.cuda.device_count() > 1
TEST_NUM_GPUS = tp.cuda.device_count() if TEST_CUDA else 0

TEST_WITH_SLOW = os.getenv("TP_TEST_WITH_SLOW", "0") == "1"

TEST_NUMPY = HAS_NUMPY
try:
    import scipy  # noqa: F401

    TEST_SCIPY = True
except ImportError:
    TEST_SCIPY = False
TEST_MKL = tp._C.has_mkl()


def dtype_name(dtype) -> str:
    """Returns the pretty name of the dtype (e.g. ``int64``)."""
    return str(dtype).split(".")[-1]


def lazy_skip_if(condition, reason):
    """Skips the decorated test when ``condition`` is true at call time."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if condition():
                raise unittest.SkipTest(reason)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _has_lapack() -> bool:
    try:
        tp.linalg.cholesky(tp.eye(2))
        return True
    except Exception:
        return False


def skipIfNoLapack(fn):
    return lazy_skip_if(lambda: not _has_lapack(), "compiled without Lapack")(fn)


def skipIfNoSciPy(fn):
    return lazy_skip_if(lambda: not TEST_SCIPY, "test requires SciPy, but SciPy not found")(fn)


def skipIfNoNumPy(fn):
    return lazy_skip_if(lambda: not TEST_NUMPY, "test requires NumPy, but NumPy not found")(fn)


def subtest(**kwargs):
    """Marks a parameter set as a subtest for use with ``product``-style sweeps."""
    return (kwargs,)


def noncontiguous_like(t: Tensor) -> Tensor:
    """Returns a non-contiguous tensor with the same values as ``t``."""
    if not t.is_contiguous():
        return t

    # Choose a "weird" value that will not be accessed.
    if t.dtype in (tp.float16, tp.bfloat16, tp.float32, tp.float64,
                   tp.complex64, tp.complex128):
        value = math.nan
    elif t.dtype == tp.bool:
        value = True
    else:
        value = 12

    result = tp.empty(tuple(t.shape) + (2,), dtype=t.dtype, device=t.device)
    result[..., 0] = value
    result[..., 1] = t.detach()
    result = result[..., 1]
    result.requires_grad = t.requires_grad
    return result


def set_rng_seed(seed: int) -> None:
    """Seeds the default generator on every available device."""
    tp.manual_seed(seed)


def get_rng_seed() -> int:
    return tp.initial_seed()


@contextlib.contextmanager
def freeze_rng_state():
    """Runs the wrapped block without advancing the visible random state."""
    cpu_state = tp.get_rng_state()
    cuda_states = []
    if TEST_CUDA:
        for i in range(tp.cuda.device_count()):
            cuda_states.append(tp.cuda.get_rng_state(i))
    try:
        yield
    finally:
        tp.set_rng_state(cpu_state)
        if TEST_CUDA:
            for i, state in enumerate(cuda_states):
                tp.cuda.set_rng_state(state, i)


@contextlib.contextmanager
def suppress_warnings():
    """Temporarily silences all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def slowTest(fn):
    """Marks a test as slow; skipped unless slow tests are enabled."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest(
                "slow tests are skipped; set TP_TEST_WITH_SLOW=1 to run them"
            )
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Tensor-aware assertions
# ---------------------------------------------------------------------------

_TENSOR_OR_ARRAY_TYPES: tuple[type, ...] = (
    (Tensor, np.ndarray) if HAS_NUMPY else (Tensor,)
)


def _numel(x) -> int:
    return x.numel() if isinstance(x, Tensor) else x.size


def _numpy_dtype_to_tp(dtype) -> "tp.dtype | None":
    mapping = {
        np.bool_: tp.bool,
        np.uint8: tp.uint8,
        np.int8: tp.int8,
        np.int16: tp.int16,
        np.int32: tp.int32,
        np.int64: tp.int64,
        np.float16: tp.float16,
        np.float32: tp.float32,
        np.float64: tp.float64,
        np.complex64: tp.complex64,
        np.complex128: tp.complex128,
    }
    return mapping.get(dtype)


class RelaxedBooleanPair(BooleanPair):
    """Pair for boolean-like inputs.

    In contrast to :class:`BooleanPair`, only one of the inputs has to be a
    boolean; the other may also be a number or a single-element tensor-like.
    """

    _supported_number_types = NumberPair(0, 0)._supported_types

    def _process_inputs(self, actual: Any, expected: Any, *, id: tuple) -> tuple[bool, bool]:
        if not (
            (isinstance(actual, self._supported_types)
             and isinstance(expected, (*self._supported_types, *self._supported_number_types, *_TENSOR_OR_ARRAY_TYPES)))
            or (isinstance(expected, self._supported_types)
                and isinstance(actual, (*self._supported_types, *self._supported_number_types, *_TENSOR_OR_ARRAY_TYPES)))
        ):
            self._inputs_not_supported()

        return (
            self._to_bool(input, id=id) for input in (actual, expected)
        )

    def _to_bool(self, bool_like: Any, *, id: tuple) -> bool:
        if HAS_NUMPY and isinstance(bool_like, np.number):
            return bool(bool_like.item())
        elif type(bool_like) in self._supported_number_types:
            return bool(bool_like)
        elif isinstance(bool_like, _TENSOR_OR_ARRAY_TYPES):
            if _numel(bool_like) > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a boolean. "
                    f"Got {_numel(bool_like)} elements instead.",
                    id=id,
                )
            return bool(bool_like.item())
        else:
            raise UnsupportedInputs


class RelaxedNumberPair(NumberPair):
    """Pair for number-like inputs.

    In contrast to :class:`NumberPair`, only one of the inputs has to be a
    number; the other may also be a single-element tensor-like or an
    :class:`enum.Enum`. (D)type checks are always disabled, so comparing
    ``1`` to ``1.0`` succeeds. Floating scalars use the looser ``float32``
    tolerances rather than ``float64``.
    """

    _TYPE_TO_DTYPE = {
        int: tp.int64,
        float: tp.float32,
        complex: tp.complex64,
    }

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple = (),
        rtol: float | None = None,
        atol: float | None = None,
        rtol_override: float = 0.0,
        atol_override: float = 0.0,
        equal_nan: bool = False,
        check_dtype: bool | None = None,
        **other_parameters: Any,
    ) -> None:
        # (D)type checks are always disabled for the relaxed pair: comparing
        # 1 to 1.0 must succeed even when check_dtype is requested.
        super().__init__(
            actual,
            expected,
            id=id,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            check_dtype=False,
            **other_parameters,
        )
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple
    ) -> tuple[int | float | complex, int | float | complex]:
        tensor_or_array_types = _TENSOR_OR_ARRAY_TYPES
        other_supported_types = (*self._supported_types, *tensor_or_array_types)
        if not (
            (isinstance(actual, self._supported_types)
             and isinstance(expected, other_supported_types))
            or (isinstance(expected, self._supported_types)
                and isinstance(actual, other_supported_types))
        ):
            self._inputs_not_supported()

        return (
            self._to_number(input, id=id) for input in (actual, expected)
        )

    def _to_number(self, number_like: Any, *, id: tuple) -> int | float | complex:
        if isinstance(number_like, _TENSOR_OR_ARRAY_TYPES):
            if _numel(number_like) > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a number. "
                    f"Got {_numel(number_like)} elements instead.",
                    id=id,
                )
            number = number_like.item()
            if isinstance(number, bool):
                number = int(number)
            return number
        elif isinstance(number_like, enum.Enum):
            return int(number_like)
        else:
            number = super()._to_number(number_like, id=id)
            if type(number) not in self._TYPE_TO_DTYPE:
                self._inputs_not_supported()
            return number

    def compare(self) -> None:
        actual = self.actual
        expected = self.expected

        if actual == expected:
            return

        if self.equal_nan and _scalar_is_nan(actual) and _scalar_is_nan(expected):
            return

        abs_diff = abs(actual - expected)
        tolerance = self.atol + self.rtol * abs(expected)

        if math.isfinite(abs_diff) and abs_diff <= tolerance:
            return

        self._fail(
            AssertionError,
            make_scalar_mismatch_msg(actual, expected, rtol=self.rtol, atol=self.atol),
        )


class TensorOrArrayPair(TensorLikePair):
    """Pair strictly for ``Tensor`` and numpy ``ndarray`` inputs.

    Unlike :class:`TensorLikePair`, inputs must be actual tensor or array
    instances; unrelated types are not converted automatically.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple = (),
        rtol_override: float = 0.0,
        atol_override: float = 0.0,
        **other_parameters: Any,
    ) -> None:
        super().__init__(actual, expected, id=id, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple, allow_subclasses: bool
    ) -> tuple[Tensor, Tensor]:
        self._check_inputs_isinstance(actual, expected, cls=_TENSOR_OR_ARRAY_TYPES)

        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        return actual, expected


class UnittestPair(Pair):
    """Fallback pair that delegates to :meth:`unittest.TestCase.assertEqual`."""

    CLS: type | tuple[type, ...]
    TYPE_NAME: str | None = None

    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None:
        self._check_inputs_isinstance(actual, expected, cls=self.CLS)
        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> None:
        test_case = unittest.TestCase()

        try:
            return test_case.assertEqual(self.actual, self.expected)
        except test_case.failureException as error:
            msg = str(error)

        type_name = self.TYPE_NAME or (
            self.CLS if isinstance(self.CLS, type) else self.CLS[0]
        ).__name__
        self._fail(AssertionError, f"{type_name.title()} comparison failed: {msg}")


class StringPair(UnittestPair):
    CLS = (str, bytes)
    TYPE_NAME = "string"


class SetPair(UnittestPair):
    CLS = set


class TypePair(UnittestPair):
    CLS = type


class ObjectPair(UnittestPair):
    CLS = object


def _scalar_is_nan(value) -> bool:
    if isinstance(value, complex):
        return math.isnan(value.real) and math.isnan(value.imag)
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _not_close_error_metas(actual, expected, *, pair_types, **options):
    """Originates and compares pairs, returning the collected error metas."""
    try:
        pairs = originate_pairs(
            actual,
            expected,
            pair_types=pair_types,
            **options,
        )
    except ErrorMeta as error_meta:
        return [error_meta]

    error_metas = []
    for pair in pairs:
        try:
            pair.compare()
        except ErrorMeta as error_meta:
            error_metas.append(error_meta)
    return error_metas


def _sequence_types() -> tuple[type, ...]:
    import collections.abc

    types: list[type] = [collections.abc.Sequence]
    try:
        from tensorplay.nn import ModuleList, ParameterList, Sequential

        types += [Sequential, ModuleList, ParameterList]
    except ImportError:
        pass
    return tuple(types)


def _mapping_types() -> tuple[type, ...]:
    import collections.abc

    types: list[type] = [collections.abc.Mapping]
    try:
        from tensorplay.nn import ModuleDict, ParameterDict

        types += [ModuleDict, ParameterDict]
    except ImportError:
        pass
    return tuple(types)


class TestCase(unittest.TestCase):
    """A ``unittest.TestCase`` with tensor-aware assertions.

    :meth:`assertEqual` dispatches on the input category: tensors (and numpy
    arrays) are compared with :func:`tensorplay.testing.assert_close` using
    per-dtype default tolerances, scalars are compared with relaxed rules
    that also accept single-element tensors, and remaining categories fall
    back to :mod:`unittest` semantics.

    The class attributes :attr:`precision` and :attr:`rel_tol` raise the
    minimum ``atol``/``rtol`` used by all comparisons of a test class.
    """

    # Minimum atol/rtol values for comparisons; overridable per class or test.
    _precision: float = 0
    _rel_tol: float = 0

    _diffThreshold = sys.maxsize
    maxDiff = None

    exact_dtype = True

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, prec: float) -> None:
        self._precision = prec

    @property
    def rel_tol(self) -> float:
        return self._rel_tol

    @rel_tol.setter
    def rel_tol(self, prec: float) -> None:
        self._rel_tol = prec

    def assertEqual(
        self,
        x,
        y,
        msg: str | None = None,
        *,
        atol: float | None = None,
        rtol: float | None = None,
        equal_nan: bool = True,
        exact_dtype: bool | None = None,
        exact_device: bool = False,
        exact_layout: bool = False,
        exact_stride: bool = False,
    ) -> None:
        if exact_dtype is None:
            exact_dtype = self.exact_dtype

        # A sequence next to a tensor is converted so the comparison runs on
        # equal footing instead of recursing elementwise into the sequence.
        if isinstance(x, Tensor) and isinstance(y, (list, tuple)):
            y = tp.as_tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(x, (list, tuple)) and isinstance(y, Tensor):
            x = tp.as_tensor(x, dtype=y.dtype, device=y.device)

        error_metas = _not_close_error_metas(
            x,
            y,
            pair_types=(
                NonePair,
                RelaxedBooleanPair,
                RelaxedNumberPair,
                TensorOrArrayPair,
                StringPair,
                SetPair,
                TypePair,
                ObjectPair,
            ),
            sequence_types=_sequence_types(),
            mapping_types=_mapping_types(),
            rtol=rtol,
            rtol_override=self.rel_tol,
            atol=atol,
            atol_override=self.precision,
            equal_nan=equal_nan,
            check_dtype=exact_dtype,
            check_device=exact_device,
            check_layout=exact_layout,
            check_stride=exact_stride,
        )

        if error_metas:
            # Emulates unittest.TestCase behavior with longMessage (default
            # True): a custom string message is appended to the generated one.
            error = error_metas[0]
            raise error.to_error(
                (lambda generated: f"{generated}\n{msg}")
                if isinstance(msg, str) and self.longMessage
                else msg
            )

    def assertNotEqual(self, x, y, msg=None, *, atol=None, rtol=None, **kwargs) -> None:
        with self.assertRaises(AssertionError, msg=msg):
            self.assertEqual(x, y, msg, atol=atol, rtol=rtol, **kwargs)

    def assertEqualIgnoreType(self, *args, **kwargs) -> None:
        # If you are seeing this function used, that means the test is written
        # loosely with respect to dtypes and deserves detailed investigation.
        return self.assertEqual(*args, exact_dtype=False, **kwargs)

    def assertEqualBroadcasting(self, x, y, *args, **kwargs) -> None:
        """Tests if tensor x equals y, with y broadcast to x.shape."""
        if not isinstance(y, Tensor) and not isinstance(y, (list, tuple)):
            y = tp.ones_like(x) * y
        if not isinstance(y, Tensor):
            y = tp.ones_like(x) * tp.tensor(y)
        return self.assertEqual(x, y, *args, **kwargs)

    def assertEqualTypeString(self, x, y) -> None:
        self.assertEqual(x.device, y.device)
        self.assertEqual(x.dtype, y.dtype)

    def assertObjectIn(self, obj: Any, iterable) -> None:
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    def assertClose(self, actual, expected, *, atol=None, rtol=None, msg=None) -> None:
        """Asserts closeness with per-dtype default tolerances."""
        self.assertEqual(actual, expected, msg=msg, atol=atol, rtol=rtol)

    def assertNotClose(self, actual, expected, *, atol=None, rtol=None, msg=None) -> None:
        """Asserts that two tensor-like values are NOT close."""
        from tensorplay.testing import assert_close, default_tolerances

        if atol is None and rtol is None:
            rtol, atol = default_tolerances(
                *(
                    input if isinstance(input, (Tensor, tp.dtype)) else tp.as_tensor(input)
                    for input in (actual, expected)
                )
            )
        try:
            assert_close(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
                check_dtype=False,
                check_stride=False,
                msg=msg,
            )
        except AssertionError:
            return
        raise AssertionError(
            "Tensor-likes are close!\n\nThe values were expected to differ."
            if msg is None
            else msg
        )


def run_tests(argv=None) -> None:
    """Runs all tests in the calling module using ``unittest.main`` semantics."""
    if argv is None:
        argv = sys.argv[1:]
    unittest.main(argv=[sys.argv[0], *argv], verbosity=2, exit=True)
