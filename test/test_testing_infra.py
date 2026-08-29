"""Self-checks for the testing infrastructure itself:
``tensorplay.testing`` (assert_close / make_tensor), the shared ``TestCase``
and the device-type parametrization helpers.
"""

import unittest

import numpy as np
import tensorplay as tp

from tensorplay.testing import (
    assert_close,
    assert_allclose,
    default_tolerances,
    get_tolerances,
    make_tensor,
)
from tensorplay.testing._internal.common_utils import (
    TestCase,
    freeze_rng_state,
    run_tests,
    set_rng_seed,
    get_rng_seed,
    make_tensor as make_tensor_reexport,
    subtest,
    noncontiguous_like,
    dtype_name,
    TEST_NUMPY,
    TEST_SCIPY,
    TEST_MKL,
    skipIfNoLapack,
    skipIfNoSciPy,
    skipIfNoNumPy,
)
from tensorplay.testing._internal.common_dtype import (
    get_all_dtypes,
    get_all_fp_dtypes,
    get_all_int_dtypes,
    get_all_complex_dtypes,
    get_all_math_dtypes,
    highest_precision_float,
)
from tensorplay.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCPU,
    onlyCPU,
    onlyCUDA,
    onlyOn,
    skipIf,
    skipCPUIf,
    skipCPUIfNoLapack,
    skipGPUIf,
    skipMeta,
    expectedFailure,
    expectedFailureCPU,
    expectedFailureCUDA,
    precisionOverride,
    toleranceOverride,
    largeTensorTest,
    dtype_name as device_dtype_name,
    get_all_device_types,
    get_all_dtypes as device_get_all_dtypes,
    instantiate_device_type_tests,
)


class TestTestingSurface(TestCase):
    """Asserts the inventory of testing utilities exposed by the package."""

    def test_public_api(self):
        import tensorplay.testing as tt

        for name in ("assert_close", "assert_allclose", "make_tensor",
                     "default_tolerances", "get_tolerances"):
            self.assertTrue(hasattr(tt, name), name)

    def test_common_utils_api(self):
        import tensorplay.testing._internal.common_utils as cu

        for name in (
            "IS_WINDOWS", "IS_LINUX", "IS_MACOS", "TEST_CUDA", "TEST_NUM_GPUS",
            "TEST_MULTIGPU", "TEST_NUMPY", "TEST_SCIPY", "TEST_MKL",
            "TEST_WITH_SLOW", "TestCase", "run_tests", "freeze_rng_state",
            "set_rng_seed", "get_rng_seed", "suppress_warnings", "slowTest",
            "subtest", "lazy_skip_if", "skipIfNoLapack", "skipIfNoSciPy",
            "skipIfNoNumPy", "noncontiguous_like", "dtype_name", "make_tensor",
        ):
            self.assertTrue(hasattr(cu, name), name)

    def test_common_dtype_api(self):
        import tensorplay.testing._internal.common_dtype as cd

        for name in ("get_all_dtypes", "get_all_math_dtypes",
                     "get_all_complex_dtypes", "get_all_int_dtypes",
                     "get_all_fp_dtypes", "highest_precision_float"):
            self.assertTrue(hasattr(cd, name), name)

    def test_common_device_type_api(self):
        import tensorplay.testing._internal.common_device_type as cdt

        for name in (
            "deviceCountAtLeast", "onlyCUDA", "onlyCPU", "onlyNativeDeviceTypes",
            "onlyOn", "skipIf", "skipCPUIf", "skipCUDAIf", "skipCPUIfNoLapack",
            "skipGPUIf", "skipMeta", "expectedFailure", "expectedFailureCPU",
            "expectedFailureCUDA", "precisionOverride", "toleranceOverride",
            "largeTensorTest", "dtypes", "dtypesIfCPU", "dtypesIfCUDA",
            "dtype_name", "get_all_device_types", "get_all_dtypes",
            "instantiate_device_type_tests",
        ):
            self.assertTrue(hasattr(cdt, name), name)


class TestDtypeCollections(TestCase):
    def test_get_all_dtypes(self):
        self.assertIn(tp.float32, get_all_dtypes())
        self.assertIn(tp.float16, get_all_dtypes())
        self.assertNotIn(tp.float16, get_all_dtypes(include_half=False))
        self.assertNotIn(tp.bfloat16, get_all_dtypes(include_bfloat16=False))
        self.assertNotIn(tp.bool, get_all_dtypes(include_bool=False))
        self.assertNotIn(tp.complex64, get_all_dtypes(include_complex=False))

    def test_get_all_fp_int_complex(self):
        self.assertEqual(get_all_int_dtypes(),
                         [tp.uint8, tp.int8, tp.int16, tp.int32, tp.int64])
        self.assertIn(tp.bfloat16, get_all_fp_dtypes())
        self.assertNotIn(tp.bfloat16, get_all_fp_dtypes(include_bfloat16=False))
        self.assertEqual(get_all_complex_dtypes(), [tp.complex64, tp.complex128])
        self.assertIn(tp.complex32, get_all_complex_dtypes(include_complex32=True))

    def test_get_all_math_dtypes(self):
        dtypes = get_all_math_dtypes("cpu")
        self.assertIn(tp.float32, dtypes)
        self.assertNotIn(tp.bfloat16, dtypes)  # math sweep excludes bf16 on cpu
        self.assertIn(tp.complex64, dtypes)

    def test_highest_precision_float(self):
        self.assertEqual(highest_precision_float("cpu"), tp.float64)


class TestCommonUtilsExtras(TestCase):
    def test_dtype_name(self):
        self.assertEqual(dtype_name(tp.int64), "int64")
        self.assertEqual(device_dtype_name(tp.float32), "float32")

    def test_noncontiguous_like(self):
        t = tp.arange(6.0).reshape(2, 3)
        nc = noncontiguous_like(t)
        self.assertFalse(nc.is_contiguous())
        self.assertEqual(nc.shape, t.shape)
        self.assertEqual(nc, t)

    def test_flags_present(self):
        self.assertTrue(TEST_NUMPY)
        self.assertIsInstance(TEST_SCIPY, bool)
        self.assertIsInstance(TEST_MKL, bool)

    def test_skip_decorators(self):
        @skipIfNoSciPy
        def needs_scipy():
            pass

        if TEST_SCIPY:
            needs_scipy()
        else:
            with self.assertRaises(unittest.SkipTest):
                needs_scipy()

        @skipIfNoLapack
        def needs_lapack():
            pass

        needs_lapack()

        @skipIfNoNumPy
        def needs_numpy():
            pass

        needs_numpy()

    def test_subtest_marker(self):
        marker = subtest(dtype=tp.float32)
        self.assertEqual(marker, (({"dtype": tp.float32}),))


class TestDeviceTypeExtras(TestCase):
    def test_class_level_tolerance_override(self):
        holder = {}

        @precisionOverride(1e-2)
        class Sample(TestCase):
            def test_within_loose_precision(self, device):
                self.assertEqual(tp.tensor([1.0]), tp.tensor([1.005]))

        instantiate_device_type_tests(Sample, holder)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(holder["Sample"])
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 1)

    def test_skip_if_and_expected_failure(self):
        holder = {}

        class Sample(TestCase):
            @skipIf(True, "generic skip")
            def test_skipped(self, device):
                pass

            @expectedFailureCPU
            def test_xfail(self, device):
                self.assertEqual(tp.tensor([1.0]), tp.tensor([2.0]))

            @largeTensorTest(2**30)
            def test_too_large(self, device):
                pass

        instantiate_device_type_tests(Sample, holder)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(holder["Sample"])
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(result.skipped), 2)
        self.assertEqual(len(result.expectedFailures), 1)

    def test_only_cuda_generates_nothing_without_gpu(self):
        holder = {}

        class Sample(TestCase):
            @onlyCUDA
            def test_gpu_only(self, device):
                pass

        instantiate_device_type_tests(Sample, holder)
        self.assertEqual(
            [n for n in holder["Sample"].__dict__ if n.startswith("test_")],
            [],
        )


class TestFileCheck(TestCase):
    """Behavioral checks for the native FileCheck utility."""

    def _fc(self):
        from tensorplay.testing import FileCheck

        return FileCheck()

    def test_basic_and_chaining(self):
        self._fc().check("foo").run("foo bar")
        self._fc().check("a").check("b").run("a\nb")
        with self.assertRaisesRegex(RuntimeError, 'Expected to find "foo"'):
            self._fc().check("foo").run("bar")
        # Checks apply sequentially: each continues after the previous match.
        with self.assertRaisesRegex(RuntimeError, 'Expected to find "a"'):
            self._fc().check("b").check("a").run("a\nb")

    def test_check_next(self):
        self._fc().check("a").check_next("b").run("a\nb")
        with self.assertRaisesRegex(RuntimeError, 'Expected to not find "\\n"'):
            self._fc().check("a").check_next("b").run("a\nx\nb")

    def test_check_same(self):
        self._fc().check("a").check_same("b").run("a b")
        with self.assertRaisesRegex(RuntimeError, 'Expected to not find "\\n"'):
            self._fc().check("a").check_same("b").run("a\nb")

    def test_check_not(self):
        self._fc().check("a").check_not("b").run("a c")
        with self.assertRaisesRegex(RuntimeError, 'Expected to not find "b"'):
            self._fc().check("a").check_not("b").run("a b")
        # A NOT group covers the region up to the next match only.
        self._fc().check("a").check("c").check_not("b").run("a\nb\nc")

    def test_check_count(self):
        self._fc().check_count("x", 2, exactly=True).run("x\ny\nx")
        self._fc().check_count("x", 2, exactly=False).run("x\ny\nx\nx")
        with self.assertRaisesRegex(RuntimeError, 'Expected to find "x"'):
            self._fc().check_count("x", 3, exactly=True).run("x\ny\nx")

    def test_check_dag(self):
        self._fc().check_dag("b").check_dag("a").run("b\na")
        with self.assertRaisesRegex(RuntimeError, 'Expected to find "b"'):
            self._fc().check_dag("b").check_dag("a").run("a\nc")

    def test_check_source_highlighted(self):
        self._fc().check_source_highlighted("foo").run("some foo here\n     ~~~\nnext")
        with self.assertRaisesRegex(RuntimeError, "highlighted but it is not"):
            self._fc().check_source_highlighted("foo").run("some foo here\n    ~~~~~~\nnext")

    def test_check_regex(self):
        self._fc().check_regex("[0-9]+ items").run("there are 42 items")
        with self.assertRaisesRegex(RuntimeError, 'Expected to find regex'):
            self._fc().check_regex("[0-9]+ items").run("no numbers")

    def test_error_message_format(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._fc().check("foo").run("bar baz")
        message = str(ctx.exception)
        self.assertIn('Expected to find "foo" but did not find it', message)
        self.assertIn("Searched string:", message)
        self.assertIn("From CHECK: foo", message)

    def test_error_from_real_exception_text(self):
        # FileCheck applied to actual runtime error output.
        t = tp.empty([2, 1, 4]).expand(2, 3, 4)
        try:
            t.uniform_(0, 1)
            raise AssertionError("expected uniform_ to reject overlapping input")
        except RuntimeError as e:
            self._fc().check("more than one element").check(
                "Please clone() the tensor"
            ).run(str(e))


class TestAssertClose(TestCase):
    def test_tensor_pass(self):
        assert_close(tp.tensor([1.0, 2.0]), tp.tensor([1.0, 2.0]))
        assert_close(
            tp.tensor([1.0, 2.0]),
            tp.tensor([1.0, 2.0 + 1e-7]),
            rtol=0,
            atol=1e-6,
        )
        t = tp.rand(4, 4)
        assert_close(t, t.clone())

    def test_attribute_mismatch(self):
        with self.assertRaisesRegex(
            AssertionError, r"The values for attribute 'shape' do not match"
        ):
            assert_close(tp.zeros(2, 3), tp.zeros(3, 2))
        with self.assertRaisesRegex(
            AssertionError, r"The values for attribute 'dtype' do not match"
        ):
            assert_close(tp.zeros(2, dtype=tp.float32), tp.zeros(2, dtype=tp.float64))
        with self.assertRaisesRegex(
            AssertionError, r"The values for attribute 'stride\(\)' do not match"
        ):
            assert_close(
                tp.zeros(2, 3),
                tp.zeros(2, 6)[:, ::2],
                check_stride=True,
            )
        # stride check is off by default
        assert_close(tp.zeros(2, 3), tp.zeros(2, 6)[:, ::2])

    def test_dtype_promotion_when_unchecked(self):
        assert_close(
            tp.zeros(2, dtype=tp.float32),
            tp.zeros(2, dtype=tp.float64),
            check_dtype=False,
        )

    def test_value_mismatch_message(self):
        with self.assertRaises(AssertionError) as ctx:
            assert_close(tp.tensor([1.0, 2.0]), tp.tensor([1.0, 2.5]))
        message = str(ctx.exception)
        self.assertIn("Tensor-likes are not close!", message)
        self.assertIn("Mismatched elements: 1 / 2 (50.0%)", message)
        self.assertIn("Greatest absolute difference: 0.5 at index (1,)", message)
        self.assertIn("Greatest relative difference: 0.2", message)

    def test_strict_equality_message(self):
        with self.assertRaises(AssertionError) as ctx:
            assert_close(tp.tensor([1, 2]), tp.tensor([1, 3]), rtol=0, atol=0)
        message = str(ctx.exception)
        self.assertIn("Tensor-likes are not 'equal'!", message)
        self.assertIn("The first mismatched element is at index (1,)", message)

    def test_tolerances_both_or_neither(self):
        with self.assertRaisesRegex(ValueError, "Both 'rtol' and 'atol'"):
            assert_close(tp.zeros(2), tp.zeros(2), rtol=1e-3)

    def test_default_tolerances_by_dtype(self):
        rtol, atol = default_tolerances(tp.float32)
        self.assertEqual(rtol, 1.3e-6)
        self.assertEqual(atol, 1e-5)
        rtol, atol = default_tolerances(tp.float16)
        self.assertEqual((rtol, atol), (0.001, 1e-5))
        rtol, atol = default_tolerances(tp.bfloat16)
        self.assertEqual((rtol, atol), (0.016, 1e-5))
        rtol, atol = default_tolerances(tp.float64)
        self.assertEqual((rtol, atol), (1e-7, 1e-7))

    def test_get_tolerances_loosest(self):
        rtol, atol = get_tolerances(tp.float32, tp.float64, rtol=None, atol=None)
        self.assertEqual((rtol, atol), (1.3e-6, 1e-5))
        rtol, atol = get_tolerances(tp.float32, rtol=1e-2, atol=1e-3)
        self.assertEqual((rtol, atol), (1e-2, 1e-3))

    def test_nan_handling(self):
        nan = float("nan")
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([nan]), tp.tensor([nan]))
        assert_close(tp.tensor([nan]), tp.tensor([nan]), equal_nan=True)
        # NaN vs non-NaN never passes
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([nan]), tp.tensor([1.0]), equal_nan=True)

    def test_inf_handling(self):
        inf = float("inf")
        assert_close(tp.tensor([inf]), tp.tensor([inf]))
        assert_close(tp.tensor([-inf]), tp.tensor([-inf]))
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([inf]), tp.tensor([-inf]))
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([inf]), tp.tensor([1e30]))

    def test_integral_equality(self):
        assert_close(tp.tensor([1, 2]), tp.tensor([1, 2]))
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([1, 2]), tp.tensor([1, 3]))
        # integral dtypes compare exactly even with loose tolerances
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([1, 2]), tp.tensor([1, 3]), rtol=1e-3, atol=1e-3)

    def test_complex_components(self):
        assert_close(
            tp.tensor([1 + 2j]), tp.tensor([1 + 2j]),
        )
        with self.assertRaises(AssertionError):
            assert_close(tp.tensor([1 + 2j]), tp.tensor([1 + 3j]))

    def test_nested_containers(self):
        assert_close([1.0, 2.0], [1.0, 2.0])
        assert_close([1.0, [2.0, 3.0]], [1.0, [2.0, 3.0]])
        assert_close({"x": 1.0, "y": (2.0, 3.0)}, {"x": 1.0, "y": (2.0, 3.0)})
        with self.assertRaisesRegex(ValueError, "length"):
            assert_close([1.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "keys"):
            assert_close({"a": 1.0}, {"b": 1.0})

    def test_scalars(self):
        assert_close(1, 1)
        assert_close(3.14, 3.14)
        assert_close(2.0, 2.0 + 1e-10)
        with self.assertRaises(AssertionError):
            assert_close(1, 2)
        with self.assertRaises(AssertionError):
            assert_close(True, False)

    def test_msg_override(self):
        # A string message replaces the generated report; a callable wraps it.
        with self.assertRaisesRegex(AssertionError, "^custom message$"):
            assert_close(tp.zeros(2), tp.ones(2), msg="custom message")

        def custom(generated):
            return f"wrapped: {generated}"

        with self.assertRaisesRegex(AssertionError, "wrapped:") as ctx:
            assert_close(tp.zeros(2), tp.ones(2), msg=custom)
        self.assertIn("Tensor-likes are not close", str(ctx.exception))

    def test_assert_allclose_legacy(self):
        assert_allclose([1.0, 2.0], [1.0, 2.0])
        assert_allclose([1.0], [1.0 + 1e-4], rtol=0, atol=1e-3)
        with self.assertRaises(AssertionError):
            assert_allclose([1.0], [2.0])


class TestMakeTensor(TestCase):
    def test_floating_range(self):
        t = make_tensor(1000, dtype=tp.float32, device="cpu")
        values = t.tolist()
        self.assertTrue(all(-9 <= v < 9 for v in values))

    def test_explicit_range(self):
        t = make_tensor(100, dtype=tp.float64, device="cpu", low=-2.5, high=7.5)
        values = t.tolist()
        self.assertTrue(all(-2.5 <= v < 7.5 for v in values))

    def test_integral_range(self):
        t = make_tensor(500, dtype=tp.int64, device="cpu")
        self.assertTrue(all(-9 <= v < 10 for v in t.tolist()))
        t = make_tensor(500, dtype=tp.uint8, device="cpu", low=100, high=200)
        self.assertTrue(all(100 <= v < 200 for v in t.tolist()))

    def test_bool(self):
        t = make_tensor(100, dtype=tp.bool, device="cpu")
        self.assertEqual(set(t.tolist()), {True, False})

    def test_complex(self):
        t = make_tensor(100, dtype=tp.complex64, device="cpu")
        values = t.tolist()
        self.assertTrue(all(-9 <= v.real < 9 and -9 <= v.imag < 9 for v in values))

    def test_low_precision_dtypes(self):
        for dtype in (tp.float16, tp.bfloat16):
            t = make_tensor(100, dtype=dtype, device="cpu")
            self.assertEqual(t.dtype, dtype)

    def test_exclude_zero(self):
        t = make_tensor(2000, dtype=tp.float32, device="cpu", exclude_zero=True)
        self.assertEqual(t.eq(0).sum().item(), 0)
        t = make_tensor(100, dtype=tp.int32, device="cpu", exclude_zero=True)
        self.assertEqual(t.eq(0).sum().item(), 0)

    def test_noncontiguous(self):
        t = make_tensor(4, 8, dtype=tp.float32, device="cpu", noncontiguous=True)
        self.assertFalse(t.is_contiguous())
        self.assertEqual(t.shape, (4, 8))
        self.assertTrue(all(-9 <= v < 9 for v in t.flatten().tolist()))

    def test_requires_grad(self):
        t = make_tensor(3, dtype=tp.float32, device="cpu", requires_grad=True)
        self.assertTrue(t.requires_grad)
        with self.assertRaisesRegex(ValueError, "requires_grad"):
            make_tensor(3, dtype=tp.int64, device="cpu", requires_grad=True)

    def test_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "must be less than"):
            make_tensor(3, dtype=tp.float32, device="cpu", low=5, high=1)
        # Non-intersecting intervals are rejected; intersecting ones clamp.
        with self.assertRaisesRegex(ValueError, "value interval"):
            make_tensor(3, dtype=tp.float16, device="cpu", low=1e6, high=2e6)
        clamped = make_tensor(3, dtype=tp.float16, device="cpu", low=0, high=1e6)
        self.assertTrue(all(0 <= v <= 65504 for v in clamped.tolist()))
        with self.assertRaisesRegex(ValueError, "cannot be NaN"):
            make_tensor(3, dtype=tp.float32, device="cpu", low=float("nan"), high=1)

    def test_unsupported_dtype(self):
        with self.assertRaisesRegex(TypeError, "not supported"):
            make_tensor(3, dtype=tp.undefined, device="cpu")


class TestRngHelpers(TestCase):
    def test_freeze_rng_state(self):
        set_rng_seed(1234)
        expected = tp.rand(16)
        with freeze_rng_state():
            tp.rand(16)
            tp.rand(16)
        set_rng_seed(1234)
        # a fresh seed reproduces the same draw after the frozen block
        actual = tp.rand(16)
        assert_close(expected, actual, rtol=0, atol=0)

    def test_seed_roundtrip(self):
        seed = 0xDEADBEEF
        set_rng_seed(seed)
        self.assertEqual(get_rng_seed(), seed)


class TestDeviceTypeInstantiation(TestCase):
    def test_generated_method_names(self):
        holder = {}

        class Sample(TestCase):
            def test_basic(self, device):
                self.assertEqual(device, "cpu")

            @dtypes(tp.float32, tp.int64)
            def test_typed(self, device, dtype):
                self.assertEqual(device, "cpu")
                self.assertIn(str(dtype), ("tensorplay.float32", "tensorplay.int64"))

            @onlyCPU
            def test_cpu_only(self, device):
                pass

            @skipCPUIf(True, "always skipped")
            def test_skipped(self, device):
                pass

        instantiate_device_type_tests(Sample, holder)
        names = {name for name, _ in holder["Sample"].__dict__.items()}
        self.assertIn("test_basic_cpu", names)
        self.assertIn("test_typed_cpu_float32", names)
        self.assertIn("test_typed_cpu_int64", names)
        self.assertIn("test_cpu_only_cpu", names)
        self.assertIn("test_skipped_cpu", names)
        self.assertNotIn("test_basic", names)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(holder["Sample"])
        by_name = {test._testMethodName: test for test in suite}
        by_name["test_basic_cpu"].debug()
        by_name["test_typed_cpu_float32"].debug()
        by_name["test_typed_cpu_int64"].debug()
        by_name["test_cpu_only_cpu"].debug()
        result = unittest.TestResult()
        by_name["test_skipped_cpu"].run(result)
        self.assertEqual(len(result.skipped), 1)

    def test_only_on_and_cuda_filter(self):
        holder = {}

        class Sample(TestCase):
            @onlyOn("cpu")
            def test_on_cpu(self, device):
                self.assertEqual(device, "cpu")

            @onlyCUDA
            def test_cuda_only(self, device):
                pass

        instantiate_device_type_tests(Sample, holder)
        names = set(holder["Sample"].__dict__)
        self.assertIn("test_on_cpu_cpu", names)
        # No CUDA device in this environment, so no CUDA variant is generated
        self.assertNotIn("test_cuda_only_cuda", names)


class _DeviceParametrized(TestCase):
    """Runs through the generated device methods inside this module itself."""

    def test_rand_in_bounds(self, device):
        t = tp.rand(64, device=device)
        values = t.tolist()
        self.assertTrue(all(0 <= v < 1 for v in values))

    @dtypes(tp.float32, tp.float64)
    def test_full_range(self, device, dtype):
        t = tp.full([10], 3.0, device=device, dtype=dtype)
        self.assertEqual(str(t.dtype), str(dtype))
        assert_close(t, tp.full([10], 3.0, device=device, dtype=dtype))


instantiate_device_type_tests(_DeviceParametrized, globals())


if __name__ == "__main__":
    run_tests()
