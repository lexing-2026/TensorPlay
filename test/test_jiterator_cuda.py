"""End-to-end checks for the jiterator CUDA pipeline.

Covers the eager Python API: single- and multi-output functors, extra
runtime arguments (float/int/bool), multiple functors in one source
string, dtype coverage including half precision, broadcasting, and
non-contiguous inputs normalized before launch.
"""
import unittest

import tensorplay as tp
from tensorplay.cuda import jiterator as jit

if not tp.cuda.is_available():
    raise unittest.SkipTest("CUDA runtime is not available")


def _allclose(a, b, rtol=1e-5, atol=1e-6):
    return tp.allclose(a, b, rtol=rtol, atol=atol)


class TestJiterator(unittest.TestCase):
    """Jiterator-compiled kernels must match equivalent TP ops."""

    def test_basic_add(self):
        code_string = "template <typename T> T my_kernel(T x, T y) { return x + y; }"
        jitted_fn = jit._create_jiterator_fn(code_string)
        a = tp.rand(3, device="cuda")
        b = tp.rand(3, device="cuda")
        result = jitted_fn(a, b)
        self.assertTrue(_allclose(result, a + b))

    def test_tail_and_vector_paths(self):
        # 513 elements: the first block is vectorized, the last is a tail
        # handled by the scalar path of the same kernel.
        code_string = "template <typename T> T my_kernel(T x) { return x * x; }"
        jitted_fn = jit._create_jiterator_fn(code_string)
        x = tp.rand(513, device="cuda")
        result = jitted_fn(x)
        self.assertTrue(_allclose(result, x * x, atol=1e-6))

    def test_all_dtype_contiguous(self):
        code_string = "template <typename T> T my_kernel(T x) { return -x + x; }"
        jitted_fn = jit._create_jiterator_fn(code_string)
        for dtype in (tp.float32, tp.float64, tp.int32, tp.int64):
            x = tp.randint(-10, 10, (5, 3), dtype=dtype, device="cuda")
            self.assertTrue(_allclose(jitted_fn(x), -x + x))

    def test_half_dtype(self):
        code_string = (
            "template <typename T> T my_kernel(T x, T y) { return x + y; }"
        )
        jitted_fn = jit._create_jiterator_fn(code_string)
        a = tp.rand(7, device="cuda", dtype=tp.float16)
        b = tp.rand(7, device="cuda", dtype=tp.float16)
        result = jitted_fn(a, b)
        self.assertTrue(_allclose(result, a + b, atol=1e-3))

    def test_noncontiguous_input(self):
        code_string = "template <typename T> T my_kernel(T x) { return x * 2; }"
        jitted_fn = jit._create_jiterator_fn(code_string)
        x = tp.rand(4, 4, device="cuda")
        view = x[:, 1]  # strided view, normalized to contiguous upstream
        result = jitted_fn(view)
        self.assertTrue(_allclose(result, view * 2))

    def test_broadcasting(self):
        code_string = "template <typename T> T my_kernel(T x, T y) { return x * y; }"
        jitted_fn = jit._create_jiterator_fn(code_string)
        a = tp.rand(3, 1, device="cuda")
        b = tp.rand(1, 4, device="cuda")
        result = jitted_fn(a, b)
        expected = tp.broadcast_to(a, (3, 4)) * tp.broadcast_to(b, (3, 4))
        self.assertTrue(_allclose(result, expected))

    def test_extra_args(self):
        code_string = (
            "template <typename T> T my_kernel(T x, T y, T alpha, T beta) "
            "{ return -x + alpha * y + beta; }"
        )
        jitted_fn = jit._create_jiterator_fn(
            code_string, alpha=1.0, beta=0.5
        )
        a = tp.rand(3, device="cuda") * 10
        b = tp.rand(3, device="cuda") * 10
        result = jitted_fn(a, b)  # defaults
        self.assertTrue(_allclose(result, -a + 1.0 * b + 0.5))
        # overrides
        result2 = jitted_fn(a, b, alpha=2.0, beta=0.0)
        self.assertTrue(_allclose(result2, -a + 2.0 * b))

    def test_int_extra_args(self):
        code_string = (
            "template <typename T> T my_kernel(T x, long long count) "
            "{ return x + count; }"
        )
        jitted_fn = jit._create_jiterator_fn(code_string, count=3)
        x = tp.zeros(5, device="cuda", dtype=tp.float32)
        result = jitted_fn(x)
        self.assertTrue(
            _allclose(result, tp.full((5,), 3, device="cuda", dtype=tp.float32))
        )

    def test_bool_extra_args(self):
        code_string = (
            "template <typename T> T conditional(T x, T mask, bool is_train) "
            "{ return is_train ? x * mask : x; }"
        )
        jitted_fn = jit._create_jiterator_fn(code_string, is_train=False)
        a = tp.rand(3, device="cuda")
        b = tp.rand(3, device="cuda")
        result = jitted_fn(a, b, is_train=True)
        self.assertTrue(_allclose(result, a * b))
        result2 = jitted_fn(a, b, is_train=False)
        self.assertTrue(_allclose(result2, a))

    def test_multiple_functors(self):
        code_string = """
        template <typename T> T fn(T x, T mask) { return x * mask; }
        template <typename T> T main_fn(T x, T mask, T y) { return fn(x, mask) + y; }
        """
        jitted_fn = jit._create_jiterator_fn(code_string)
        a = tp.rand(3, device="cuda")
        b = tp.rand(3, device="cuda")
        c = tp.rand(3, device="cuda")
        result = jitted_fn(a, b, c)
        self.assertTrue(_allclose(result, a * b + c))

    def test_various_num_inputs(self):
        for num_inputs in (1, 5, 8):
            inputs = [
                tp.rand(3, device="cuda") * 10 for _ in range(num_inputs)
            ]
            input_string = ",".join([f"T i{i}" for i in range(num_inputs)])
            function_body = "+".join([f"i{i}" for i in range(num_inputs)])
            code_string = (
                f"template <typename T> T my_kernel({input_string}) "
                f"{{ return {function_body}; }}"
            )
            jitted_fn = jit._create_jiterator_fn(code_string)
            result = jitted_fn(*inputs)
            expected = inputs[0]
            for extra in inputs[1:]:
                expected = expected + extra
            self.assertTrue(_allclose(result, expected))

    def test_various_num_outputs(self):
        for num_outputs in (1, 4, 8):
            input_ = tp.rand(3, device="cuda")
            output_string = ", ".join(
                [f"T& out{i}" for i in range(num_outputs)]
            )
            function_body = ""
            for i in range(num_outputs):
                function_body += f"out{i} = input + {i};\n"
            code_string = (
                f"template <typename T> void my_kernel(T input, {output_string}) "
                f"{{ {function_body} }}"
            )
            jitted_fn = jit._create_multi_output_jiterator_fn(
                code_string, num_outputs
            )
            result = jitted_fn(input_)
            if num_outputs == 1:
                # single output comes back as a bare tensor, not a tuple
                result = (result,)
            self.assertIsInstance(result, tuple)
            for i in range(num_outputs):
                self.assertTrue(_allclose(result[i], input_ + i))

    def test_invalid_function_name(self):
        for code_string in (
            "template <typename T> T my _kernel(T x) { return x; }",
            "template <typename T> Tmy_kernel(T x) { return x; }",
        ):
            with self.assertRaises(RuntimeError):
                jit._create_jiterator_fn(code_string)


if __name__ == "__main__":
    unittest.main()
