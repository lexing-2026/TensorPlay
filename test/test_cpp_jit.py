"""Tests for the tp-side JIT frontend (tensorplay.utils.cpp_jit).

The frontend wraps the compile engine behind stable entry points and
must (a) forward to it faithfully, (b) produce an actionable error when
the engine is missing, and (c) keep the zero-copy DLPack contract for
TensorPlay tensors.
"""

import shutil
import unittest

import tensorplay as tp
from tensorplay.utils import cpp_jit

_SCALE_SOURCE = r"""
void scale_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  for (int64_t i = 0; i < x.size(0); ++i) {
    static_cast<float*>(y.data_ptr())[i] =
        static_cast<float*>(x.data_ptr())[i] * 3.0f;
  }
}
"""

# File-based sources are compiled as-is: the export macro must live in
# the file itself.
_FILE_SOURCE = r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

void scale_file_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  for (int64_t i = 0; i < x.size(0); ++i) {
    static_cast<float*>(y.data_ptr())[i] =
        static_cast<float*>(x.data_ptr())[i] * 5.0f;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(scale_file_cpu, scale_file_cpu);
"""

# The authoring contract: typed signature enforcement at the boundary,
# scalar crossing, the kernel-side dtype responsibility and error
# translation. The string entry points prepend the common headers, so
# only <string> is included explicitly.
_AUTHORING_SOURCE = r"""
#include <string>

double weighted_sum(tvm::ffi::TensorView x, double factor, int64_t offset,
                    const std::string& tag) {
  double s = static_cast<double>(offset);
  for (int64_t i = 0; i < x.numel(); ++i)
    s += static_cast<float*>(x.data_ptr())[i];
  return s * factor + tag.size();
}

void require_f32(tvm::ffi::TensorView x) {
  if (x.dtype().code != kDLFloat || x.dtype().bits != 32) {
    TVM_FFI_THROW(TypeError) << "require_f32: expected float32, got dtype with "
        << static_cast<int>(x.dtype().bits) << " bits";
  }
}

void boom(int64_t code) {
  TVM_FFI_THROW(RuntimeError) << "boom with code " << code;
}
"""


class AvailabilityTest(unittest.TestCase):
    def test_is_available_matches_engine(self):
        try:
            import tvm_ffi  # noqa: F401

            installed = True
        except ImportError:
            installed = False
        self.assertEqual(cpp_jit.is_available(), installed)

    def test_missing_engine_error_is_actionable(self):
        if cpp_jit.is_available():
            self.skipTest("compile engine installed; error path unexercised")

        with self.assertRaises(RuntimeError) as ctx:
            cpp_jit.load_inline("never", cpp_sources="", functions=[])
        message = str(ctx.exception)
        self.assertIn("tvm-ffi", message)
        self.assertIn("install", message.lower())


@unittest.skipUnless(cpp_jit.is_available(), "compile engine is not installed")
class CppJitFrontendTest(unittest.TestCase):
    def test_load_inline_forwards_and_runs_zero_copy(self):
        mod = cpp_jit.load_inline(
            name="tp_cppjit_jit",
            cpp_sources=_SCALE_SOURCE,
            functions=["scale_cpu"],
        )
        x = tp.tensor([1.0, 2.0, 3.0])
        y = tp.empty_like(x)
        mod.scale_cpu(x, y)
        self.assertEqual(y.tolist(), [3.0, 6.0, 9.0])

    def test_functions_mapping_form(self):
        mod = cpp_jit.load_inline(
            name="tp_cppjit_doc",
            cpp_sources=_SCALE_SOURCE,
            functions={"scale_cpu": "multiply by three"},
        )
        x = tp.tensor([1.0])
        y = tp.empty_like(x)
        mod.scale_cpu(x, y)
        self.assertEqual(y.tolist(), [3.0])

    def test_aot_build_then_load_roundtrip(self):
        out_dir = "/tmp/opencode/tp_cppjit_aot"
        lib_path = cpp_jit.build_inline(
            name="tp_cppjit_aot",
            cpp_sources=_SCALE_SOURCE,
            functions=["scale_cpu"],
            build_directory=out_dir,
        )
        self.assertTrue(lib_path.endswith(".so"))
        mod = cpp_jit.load_module(lib_path)
        x = tp.tensor([2.0])
        y = tp.empty_like(x)
        mod.scale_cpu(x, y)
        self.assertEqual(y.tolist(), [6.0])
        if lib_path.startswith("/tmp/opencode"):
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_file_based_load_and_aot_roundtrip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as workdir:
            src = Path(workdir) / "scale_file.cpp"
            src.write_text(_FILE_SOURCE)

            mod = cpp_jit.load(
                name="tp_cppjit_file_jit", cpp_files=[str(src)])
            x, y = tp.tensor([1.0, 2.0]), tp.empty(2)
            mod.scale_file_cpu(x, y)
            self.assertEqual(y.tolist(), [5.0, 10.0])

            lib_path = cpp_jit.build(
                name="tp_cppjit_file_aot",
                cpp_files=[str(src)],
                build_directory=str(Path(workdir) / "out"),
            )
            self.assertTrue(lib_path.endswith(".so"))
            mod2 = cpp_jit.load_module(lib_path)
            y2 = tp.empty(2)
            mod2.scale_file_cpu(x, y2)
            self.assertEqual(y2.tolist(), [5.0, 10.0])

    def test_system_lib_returns_library_module(self):
        lib = cpp_jit.system_lib()
        self.assertTrue(callable(lib.implements_function))

    def test_composes_with_custom_op(self):
        from tensorplay import library

        mod = cpp_jit.load_inline(
            name="tp_cppjit_op",
            cpp_sources=_SCALE_SOURCE,
            functions=["scale_cpu"],
        )

        @library.custom_op(
            "cppjit::triple",
            mutates_args=(),
            schema="cppjit::triple(Tensor self) -> Tensor",
        )
        def triple(x):
            y = tp.empty_like(x)
            mod.scale_cpu(x, y)
            return y

        @triple.register_fake
        def _(x):
            return tp.empty_like(x)

        triple.register_autograd(lambda ctx, grad: (grad * 3.0,))

        x = tp.tensor([1.0], requires_grad=True)
        y = triple(x)
        y.sum().backward()
        self.assertEqual(y.tolist(), [3.0])
        self.assertEqual([float(g) for g in x.grad.tolist()], [3.0])

        compiled = tp.compile(lambda a: tp.mul(triple(a), 2.0))
        self.assertEqual(compiled(tp.tensor([1.0, 2.0])).tolist(), [6.0, 12.0])


@unittest.skipUnless(cpp_jit.is_available(), "compile engine is not installed")
class CppJitAuthoringContractTest(unittest.TestCase):
    """The kernel authoring contract documented in the extending note."""

    @classmethod
    def setUpClass(cls):
        cls._mod = cpp_jit.load_inline(
            name="tp_cppjit_authoring",
            cpp_sources=_AUTHORING_SOURCE,
            functions=["weighted_sum", "require_f32", "boom"],
        )

    def test_scalar_arguments_and_return_value(self):
        x = tp.tensor([1.0, 2.0, 3.0])
        # (5 + 1+2+3) * 10 + len("abcd") == 114.0
        self.assertEqual(
            self._mod.weighted_sum(x, 10.0, 5, "abcd"), 114.0)

    def test_void_kernel_returns_none(self):
        self.assertIsNone(self._mod.require_f32(tp.tensor([1.0])))

    def test_dtype_guard_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            self._mod.require_f32(tp.zeros(3, dtype=tp.float64))
        self.assertIn("float32", str(ctx.exception))

    def test_throw_translates_to_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._mod.boom(42)
        self.assertIn("42", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
