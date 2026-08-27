"""Custom operators backed by raw C++/CUDA via Apache TVM's ``tvm_ffi``.

torch has no first-party TVM integration: TVM ships its own Python API
(``tvm_ffi``) whose exported functions accept any DLPack-compatible tensor
directly (``tvm::ffi::TensorView``).  TensorPlay implements the DLPack
protocol on :class:`tensorplay.Tensor`, so TP tensors pass through
zero-copy -- no adapter layer exists or is needed:

- JIT:  ``tvm_ffi.cpp.load_inline(name, cpp_sources=..., functions=...)``
- AOT:  ``tvm_ffi.cpp.build_inline(...) -> .so`` then ``tvm_ffi.load_module(path)``

The tests below exercise both paths plus the full ``tensorplay.library``
story wrapped around an ffi kernel: device dispatch, fake metadata,
autograd, ``opcheck``, and the opaque capture barrier under
``tensorplay.compile``.
"""

import shutil
import unittest

import tensorplay as tp
from tensorplay import library

try:
    import tvm_ffi
    import tvm_ffi.cpp

    _HAS_TVM_FFI = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_TVM_FFI = False

_SCALE_SOURCE = r"""
void scale_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  for (int64_t i = 0; i < x.size(0); ++i) {
    static_cast<float*>(y.data_ptr())[i] =
        static_cast<float*>(x.data_ptr())[i] * 3.0f;
  }
}
"""


@unittest.skipUnless(_HAS_TVM_FFI, "tvm_ffi is not installed")
class TvmFfiInteropTest(unittest.TestCase):
    """Raw tvm-ffi usage: exactly the upstream API, TP tensors in."""

    def test_jit_load_inline_accepts_tp_tensors(self):
        mod = tvm_ffi.cpp.load_inline(
            name="tp_tvmffi_jit",
            cpp_sources=_SCALE_SOURCE,
            functions=["scale_cpu"],
        )
        x = tp.tensor([1.0, 2.0, 3.0])
        y = tp.empty_like(x)
        mod.scale_cpu(x, y)
        self.assertEqual(y.tolist(), [3.0, 6.0, 9.0])

    def test_aot_build_then_load_roundtrip(self):
        out_dir = "/tmp/opencode/tp_aot_test"
        lib_path = tvm_ffi.cpp.build_inline(
            name="tp_tvmffi_aot",
            cpp_sources=_SCALE_SOURCE,
            functions=["scale_cpu"],
            build_directory=out_dir,
        )
        self.assertTrue(lib_path.endswith(".so"))
        mod = tvm_ffi.load_module(lib_path)
        x = tp.tensor([2.0])
        y = tp.empty_like(x)
        mod.scale_cpu(x, y)
        self.assertEqual(y.tolist(), [6.0])
        if lib_path.startswith("/tmp/opencode"):
            shutil.rmtree(out_dir, ignore_errors=True)


@unittest.skipUnless(_HAS_TVM_FFI, "tvm_ffi is not installed")
class TvmFfiCustomOpTest(unittest.TestCase):
    """The same ffi kernel behind a full tensorplay.library operator."""

    @classmethod
    def setUpClass(cls):
        cls._mod = tvm_ffi.cpp.load_inline(
            name="tp_tvmffi_op",
            cpp_sources=_SCALE_SOURCE,
            functions=["scale_cpu"],
        )

        @library.custom_op(
            "ffins::triple",
            mutates_args=(),
            schema="ffins::triple(Tensor self) -> Tensor",
        )
        def triple(x):
            y = tp.empty_like(x)
            TvmFfiCustomOpTest._mod.scale_cpu(x, y)
            return y

        @triple.register_fake
        def _(x):
            return tp.empty_like(x)

        def backward(ctx, grad_out):
            return (grad_out * 3.0,)  # d/dx [3x]

        triple.register_autograd(backward)
        cls.triple = triple

    def test_eager_matches_kernel(self):
        x = tp.tensor([1.0, 4.0])
        self.assertEqual(self.triple(x).tolist(), [3.0, 12.0])

    def test_autograd_flows_through_ffi_boundary(self):
        x = tp.tensor([1.0], requires_grad=True)
        y = self.triple(x)
        y.sum().backward()
        self.assertEqual([float(g) for g in x.grad.tolist()], [3.0])

    def test_opcheck_passes_all(self):
        failures = library.opcheck(
            self.triple,
            (tp.tensor([1.5]),),
            raise_exception=False,
        )
        self.assertEqual(failures, {})

    def test_compiled_barrier_preserves_semantics(self):
        x = tp.tensor([1.0, 2.0])
        compiled = tp.compile(lambda a: tp.mul(self.triple(a), 2.0))
        self.assertEqual(compiled(x).tolist(), [6.0, 12.0])


if __name__ == "__main__":
    unittest.main()
