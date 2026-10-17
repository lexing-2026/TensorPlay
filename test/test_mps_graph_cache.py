"""Correctness of the two ways the MPS backend reuses work between calls.

Binary ops and reductions build an MPSGraph once per call site, shape, dtype
and scalar, then feed later calls through it; a broadcast add, sub or mul may
instead address both operands straight out of the storage they already have.
Both make one call behave differently depending on what ran before it, so these
checks are about identity: a cached graph must never serve a call it was not
built for, and a stride-addressed read must return what the copy it replaced
would have.

Most cases compare an MPS result against the same call on CPU tensors, which
needs no second runtime. Where the backend does not fix a tie-break, the
assertion is the weaker invariant that holds for any legal answer.

Every test is skipped when the runner has no Metal device (Linux CI, Intel
macs without a supported GPU), so the file is safe to include in the macOS CPU
wheel test matrix.
"""

import math
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp


def mps_available():
    return tp.mps.is_available()


def _range_on_both(shape, dtype=tp.float32):
    """The same 0..n-1 sequence reshaped to `shape`, on CPU and on MPS."""
    count = math.prod(shape)
    cpu = tp.reshape(tp.arange(count, dtype=dtype, device="cpu"), list(shape))
    mps = tp.reshape(tp.arange(count, dtype=dtype, device="mps"), list(shape))
    return cpu, mps


def _is_exact(dtype):
    return dtype in (tp.bool, tp.int8, tp.int16, tp.int32, tp.int64)


class MPSCase(unittest.TestCase):
    """Shared assertions: an MPS result has to be the CPU result."""

    def assert_same_result(self, mps_result, cpu_result):
        got = mps_result.cpu()
        self.assertEqual(tuple(got.shape), tuple(cpu_result.shape))
        self.assertEqual(got.dtype, cpu_result.dtype)
        if _is_exact(cpu_result.dtype):
            self.assertTrue(tp.all(got == cpu_result).item())
        else:
            self.assertTrue(tp.allclose(got, cpu_result, rtol=1e-3,
                                        atol=1e-4).item())

    def assert_pair_op(self, lhs_shape, rhs_shape, fn, dtype=tp.float32):
        """`fn(a, b)` over a pair of reshaped ranges, MPS against CPU."""
        lhs_cpu, lhs_mps = _range_on_both(lhs_shape, dtype)
        rhs_cpu, rhs_mps = _range_on_both(rhs_shape, dtype)
        self.assert_same_result(fn(lhs_mps, rhs_mps), fn(lhs_cpu, rhs_cpu))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSBroadcastOperands(MPSCase):
    """A broadcast operand read through strides instead of an expanded copy."""

    def test_stretched_dimension(self):
        # Each side has a size-1 dimension the output stretches, so each needs a
        # stride of 0 along that axis to re-read one element per step.
        for name, fn in (("add", tp.add), ("sub", tp.sub), ("mul", tp.mul)):
            with self.subTest(op=name):
                self.assert_pair_op([32, 1, 64], [1, 16, 64], fn)

    def test_operand_of_lower_rank(self):
        # A shorter operand lines up with the trailing dimensions, so the
        # leading output dimensions contribute nothing to its offset.
        for shape in ([64], [16, 64], [4]):
            with self.subTest(rhs=shape):
                self.assert_pair_op([4, 16, 64], shape, tp.add)
        for shape in ([64], [16, 64]):
            with self.subTest(lhs=shape):
                self.assert_pair_op(shape, [4, 16, 64], tp.mul)

    def test_one_element_and_zero_dimensional(self):
        cpu, mps = _range_on_both([4, 3])
        self.assert_same_result(mps + tp.ones([1], device="mps"),
                                cpu + tp.ones([1], device="cpu"))
        scalar_cpu = tp.full((), 2.5, device="cpu")
        scalar_mps = tp.full((), 2.5, device="mps")
        self.assert_same_result(scalar_mps * mps, scalar_cpu * cpu)
        self.assert_same_result(mps - scalar_mps, cpu - scalar_cpu)

    def test_integral_storage(self):
        for dtype in (tp.int64, tp.int32):
            for name, fn in (("add", tp.add), ("mul", tp.mul),
                             ("sub", tp.sub)):
                with self.subTest(dtype=dtype, op=name):
                    self.assert_pair_op([5, 1, 7], [1, 4, 7], fn, dtype)

    def test_empty_result(self):
        # Nothing to write, but the shape still has to come out right.
        lhs_cpu, lhs_mps = _range_on_both([0, 3])
        rhs_cpu, rhs_mps = _range_on_both([1, 3])
        self.assert_same_result(tp.add(lhs_mps, rhs_mps),
                                tp.add(lhs_cpu, rhs_cpu))

    def test_views_of_one_storage(self):
        """Permuted, offset and stretched views address the same buffer."""
        base_cpu, base_mps = _range_on_both([6, 4])
        cases = (("transposed", lambda t: tp.transpose(t, 0, 1), [1, 6]),
                 ("column slice", lambda t: t[:, 1:3], [1, 2]),
                 ("row slice", lambda t: t[1:4], [1, 4]),
                 ("stretched view", lambda t: t[:1], [6, 4]))
        for name, pick, rhs_shape in cases:
            with self.subTest(view=name):
                lhs_cpu, lhs_mps = pick(base_cpu), pick(base_mps)
                rhs_cpu, rhs_mps = _range_on_both(rhs_shape)
                self.assert_same_result(tp.mul(lhs_mps, rhs_mps),
                                        tp.mul(lhs_cpu, rhs_cpu))

    def test_declined_calls_agree_with_cpu(self):
        """Combinations a stride read declines, which a graph serves."""
        f_cpu, f_mps = _range_on_both([8, 1, 5])
        g_cpu, g_mps = _range_on_both([1, 6, 5])
        i32_cpu, i32_mps = _range_on_both([8, 1, 5], tp.int32)
        i64_cpu, i64_mps = _range_on_both([1, 6, 5], tp.int64)
        h_cpu, h_mps = _range_on_both([8, 1, 5], tp.float16)
        # Different storage dtypes are promoted, which a graph builds as a cast.
        self.assert_same_result(tp.add(i32_mps, i64_mps),
                                tp.add(i32_cpu, i64_cpu))
        # Half keeps its own operation math type.
        self.assert_same_result(tp.mul(h_mps, g_mps), tp.mul(h_cpu, g_cpu))
        # A scaled operand is a multiply the shader does not do.
        self.assert_same_result(tp.add(f_mps, g_mps, alpha=2),
                                tp.add(f_cpu, g_cpu, alpha=2))
        # A backwards view would need an offset base the shader has no slot for.
        self.assert_same_result(tp.sub(tp.flip(f_mps, [0]), g_mps),
                                tp.sub(tp.flip(f_cpu, [0]), g_cpu))

    def test_inplace_keeps_the_arithmetic(self):
        rhs_cpu, rhs_mps = _range_on_both([1, 5])
        acc_cpu = tp.zeros([3, 5], device="cpu")
        acc_mps = tp.zeros([3, 5], device="mps")
        for _ in range(2):
            acc_cpu.add_(rhs_cpu)
            acc_mps.add_(rhs_mps)
        self.assert_same_result(acc_mps, acc_cpu)

    def test_other_binary_ops_unaffected(self):
        lhs_cpu, lhs_mps = _range_on_both([4, 1, 6])
        rhs_cpu, rhs_mps = _range_on_both([1, 3, 6])
        for name, fn in (("div", tp.div), ("maximum", tp.maximum),
                         ("eq", tp.eq), ("pow", tp.pow)):
            with self.subTest(op=name):
                self.assert_same_result(fn(lhs_mps, rhs_mps + 1.0),
                                        fn(lhs_cpu, rhs_cpu + 1.0))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSGraphCacheKeys(MPSCase):
    """One call site behind which several different graphs must sit."""

    def test_alternating_shapes(self):
        shapes = [[8, 8], [3, 1, 5], [16, 4], [2, 2, 2, 2]]
        for rep in range(3):
            for shape in shapes:
                with self.subTest(rep=rep, shape=shape):
                    self.assert_pair_op(shape, shape, tp.add)

    def test_alternating_dtypes(self):
        pairs = []
        for dtype in (tp.float32, tp.int64, tp.float16, tp.int32):
            cpu, mps = _range_on_both([4, 1, 3], dtype)
            pairs.append((cpu + 1, mps + 1, dtype))
        for rep in range(2):
            for cpu_out, mps_out, dtype in pairs:
                with self.subTest(rep=rep, dtype=dtype):
                    self.assert_same_result(mps_out, cpu_out)

    def test_alternating_alphas(self):
        lhs_cpu, lhs_mps = _range_on_both([5, 1, 4])
        rhs_cpu, rhs_mps = _range_on_both([1, 3, 4])
        for rep in range(2):
            for alpha in (1, 2, 0, -3):
                with self.subTest(rep=rep, alpha=alpha):
                    self.assert_same_result(
                        tp.add(lhs_mps, rhs_mps, alpha=alpha),
                        tp.add(lhs_cpu, rhs_cpu, alpha=alpha))

    def test_alternating_ops_on_one_pair(self):
        lhs_cpu, lhs_mps = _range_on_both([6, 1, 2])
        rhs_cpu, rhs_mps = _range_on_both([1, 4, 2])
        divisor_cpu, divisor_mps = rhs_cpu + 1.0, rhs_mps + 1.0
        for rep in range(2):
            for name, fn in (("add", tp.add), ("mul", tp.mul), ("sub", tp.sub),
                             ("div", tp.div)):
                with self.subTest(rep=rep, op=name):
                    if name == "div":
                        self.assert_same_result(fn(lhs_mps, divisor_mps),
                                                fn(lhs_cpu, divisor_cpu))
                    else:
                        self.assert_same_result(fn(lhs_mps, rhs_mps),
                                                fn(lhs_cpu, rhs_cpu))

    def test_alternating_reductions(self):
        x_cpu, x_mps = _range_on_both([3, 4, 5])
        for rep in range(2):
            for axes in ([0], [2], [1], [0, 2], [1]):
                with self.subTest(rep=rep, axes=axes):
                    self.assert_same_result(tp.sum(x_mps, dim=axes),
                                            tp.sum(x_cpu, dim=axes))
                    self.assert_same_result(tp.mean(x_mps, dim=axes),
                                            tp.mean(x_cpu, dim=axes))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSAddmmAddend(MPSCase):
    """The addend of addmm: scaled, broadcast, one-dimensional, scalar."""

    @staticmethod
    def _product(rows, inner, cols, device):
        a = tp.reshape(tp.arange(rows * inner, dtype=tp.float32, device=device),
                       [rows, inner])
        b = tp.reshape(tp.arange(inner * cols, dtype=tp.float32, device=device),
                       [inner, cols])
        return a, b

    def test_beta_and_alpha(self):
        rows, inner, cols = 2, 3, 4
        addend_cpu, addend_mps = _range_on_both([rows, cols])
        a_cpu, b_cpu = self._product(rows, inner, cols, "cpu")
        a_mps, b_mps = self._product(rows, inner, cols, "mps")
        for beta in (0, 1, 2, -1):
            for alpha in (1, 0.5, 0):
                with self.subTest(beta=beta, alpha=alpha):
                    self.assert_same_result(
                        tp.addmm(addend_mps, a_mps, b_mps, beta=beta,
                                 alpha=alpha),
                        tp.addmm(addend_cpu, a_cpu, b_cpu, beta=beta,
                                 alpha=alpha))

    def test_one_dimensional_bias(self):
        rows, inner, cols = 2, 3, 4
        a_cpu, b_cpu = self._product(rows, inner, cols, "cpu")
        a_mps, b_mps = self._product(rows, inner, cols, "mps")
        bias_cpu = tp.arange(cols, dtype=tp.float32, device="cpu")
        bias_mps = tp.arange(cols, dtype=tp.float32, device="mps")
        for beta in (1, 2, 0):
            with self.subTest(beta=beta):
                self.assert_same_result(tp.addmm(bias_mps, a_mps, b_mps,
                                                 beta=beta),
                                        tp.addmm(bias_cpu, a_cpu, b_cpu,
                                                 beta=beta))

    def test_scalar_addend(self):
        # A 0-d addend holds one element behind an empty shape; the expansion
        # over the result has to describe that element as broadcastable first.
        rows, inner, cols = 2, 3, 4
        a_cpu, b_cpu = self._product(rows, inner, cols, "cpu")
        a_mps, b_mps = self._product(rows, inner, cols, "mps")
        for value in (3.0, -1.5):
            for beta in (1, 0, 2):
                with self.subTest(value=value, beta=beta):
                    self.assert_same_result(
                        tp.addmm(tp.full((), value, device="mps"), a_mps, b_mps,
                                 beta=beta),
                        tp.addmm(tp.full((), value, device="cpu"), a_cpu, b_cpu,
                                 beta=beta))

    def test_column_addend(self):
        rows, inner, cols = 2, 3, 4
        a_cpu, b_cpu = self._product(rows, inner, cols, "cpu")
        a_mps, b_mps = self._product(rows, inner, cols, "mps")
        col_cpu = tp.reshape(tp.arange(rows, dtype=tp.float32, device="cpu"),
                             [rows, 1])
        col_mps = tp.reshape(tp.arange(rows, dtype=tp.float32, device="mps"),
                             [rows, 1])
        self.assert_same_result(tp.addmm(col_mps, a_mps, b_mps),
                                tp.addmm(col_cpu, a_cpu, b_cpu))

    def test_alternating_result_shapes(self):
        for rep in range(2):
            for rows, inner, cols in ((2, 3, 4), (3, 2, 5), (8, 4, 8)):
                with self.subTest(rep=rep, shape=(rows, cols)):
                    results = []
                    for device in ("cpu", "mps"):
                        a, b = self._product(rows, inner, cols, device)
                        bias = tp.arange(cols, dtype=tp.float32,
                                         device=device)
                        results.append(tp.addmm(bias, a, b))
                    self.assert_same_result(results[1], results[0])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSValueIndexReduce(MPSCase):
    """max.dim and min.dim, which write values and indices from one dispatch."""

    def test_dimensions_and_keepdim(self):
        x_cpu, x_mps = _range_on_both([2, 3, 4])
        for name, fn in (("max", tp.max), ("min", tp.min)):
            for dim in (0, 1, 2, -1, -2, -3):
                for keepdim in (False, True):
                    with self.subTest(op=name, dim=dim, keepdim=keepdim):
                        values, indices = fn(x_mps, dim=dim, keepdim=keepdim)
                        ref_values, ref_indices = fn(x_cpu, dim=dim,
                                                     keepdim=keepdim)
                        self.assert_same_result(values, ref_values)
                        self.assertEqual(indices.cpu().tolist(),
                                         ref_indices.cpu().tolist())

    def test_alternating_dim_at_one_call_site(self):
        x_cpu, x_mps = _range_on_both([3, 4, 5])
        for rep in range(2):
            for dim in (0, 2, 1, 2, 0):
                with self.subTest(rep=rep, dim=dim):
                    values, indices = tp.max(x_mps, dim=dim)
                    ref_values, ref_indices = tp.max(x_cpu, dim=dim)
                    self.assert_same_result(values, ref_values)
                    self.assertEqual(indices.cpu().tolist(),
                                     ref_indices.cpu().tolist())

    def test_other_storage_types(self):
        for dtype in (tp.int64, tp.int32, tp.float16):
            x_cpu, x_mps = _range_on_both([3, 4], dtype)
            for name, fn in (("max", tp.max), ("min", tp.min)):
                with self.subTest(dtype=dtype, op=name):
                    values, indices = fn(x_mps, dim=1)
                    ref_values, _ = fn(x_cpu, dim=1)
                    self.assert_same_result(values, ref_values)
                    self.assertEqual(indices.dtype, tp.int64)

    def test_ties_name_a_matching_element(self):
        # Which of several equal extrema an index names is not fixed, so the
        # check is that it names one of them.
        x = tp.tensor([[2.0, 5.0, 5.0, 1.0], [7.0, 7.0, 0.0, 7.0]],
                      dtype=tp.float32, device="mps")
        rows = x.cpu().tolist()
        for name, fn in (("max", tp.max), ("min", tp.min)):
            with self.subTest(op=name):
                values, indices = fn(x, dim=1)
                for row, (value, index) in enumerate(
                        zip(values.cpu().tolist(), indices.cpu().tolist())):
                    self.assertEqual(rows[row][index], value)

    def test_bool_input(self):
        # Tie-free in both directions, so an index has one right answer.
        x_cpu = tp.tensor([[True, False], [False, True]], dtype=tp.bool,
                          device="cpu")
        x_mps = tp.tensor([[True, False], [False, True]], dtype=tp.bool,
                          device="mps")
        for name, fn in (("argmax", tp.argmax), ("argmin", tp.argmin)):
            for dim in (0, 1, -1, -2):
                for keepdim in (False, True):
                    with self.subTest(op=name, dim=dim, keepdim=keepdim):
                        self.assert_same_result(
                            fn(x_mps, dim=dim, keepdim=keepdim),
                            fn(x_cpu, dim=dim, keepdim=keepdim))

    def test_flattened_extrema(self):
        x_cpu, x_mps = _range_on_both([5, 6])
        for name, fn in (("argmax", tp.argmax), ("argmin", tp.argmin),
                         ("max", tp.max), ("min", tp.min)):
            with self.subTest(op=name):
                self.assert_same_result(fn(x_mps), fn(x_cpu))


if __name__ == "__main__":
    unittest.main()
