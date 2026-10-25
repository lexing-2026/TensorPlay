"""Correctness of the three ways the MPS backend serves a call from what it
already has.

Binary ops and reductions build an MPSGraph once per operation, shape, dtype
and scalar, then feed later calls through it; a broadcast add, sub or mul may
instead address both operands straight out of the storage they already have, and
a contiguous float operand of a unary math op may be read by a single shader
dispatch that builds no graph at all. Each of these makes one call behave
differently depending on what ran before it and on how its input is laid out, so
these checks are about identity: a cached graph must never serve a call it was
not built for, and a stride-addressed or single-dispatch read must return what
the copy or the graph it replaced would have.

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


def _repeating_on_both(shape):
    """Small integers in an order that repeats along the last axis.

    A row of such an operand holds its largest and smallest element several
    times over, away from either end, so a reduction has to answer with one of
    the positions rather than with whichever edge it happened to reach first.
    """
    count = math.prod(shape)
    values = [float((7 * i * i + 11 * i) % 13 - 6) for i in range(count)]
    cpu = tp.reshape(tp.tensor(values, dtype=tp.float32, device="cpu"),
                     list(shape))
    mps = tp.reshape(tp.tensor(values, dtype=tp.float32, device="mps"),
                     list(shape))
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
            self.assertTrue(tp.allclose(got, cpu_result, rtol=1e-3, atol=1e-4))

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
        for shape in ([64], [16, 64], [1]):
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
        # Two operands that already have the shape the result has: a stride read
        # would avoid no copy, so the call belongs with the graph.
        e_cpu, e_mps = f_cpu * 2, f_mps * 2
        self.assert_same_result(tp.sub(f_mps, e_mps), tp.sub(f_cpu, e_cpu))

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


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSRowValueAndPredicate(MPSCase):
    """The last axis of a float or bool operand, answered by one row dispatch.

    A reduction over the innermost axis gives each row one answer, so one SIMD
    group walks the row and the answer leaves the group: the extremum where
    amax, amin and the value and index of max and min ask for it, and one byte
    where all and any ask whether the row holds a nonzero element. A tie keeps
    the lowest position and a NaN counts as the extremum of its row and as a
    nonzero element, which is what a scan of the row in the order it lies in
    answers with -- and what the operand's own CPU reduction answers with.
    """

    SHAPES = ([4, 9], [37, 5], [2, 3, 64], [3, 33], [128, 1], [8, 6, 130])

    def test_extrema_along_the_last_axis(self):
        for shape in self.SHAPES:
            x_cpu, x_mps = _repeating_on_both(shape)
            for name, fn in (("amax", tp.amax), ("amin", tp.amin)):
                for keepdim in (False, True):
                    with self.subTest(shape=shape, op=name, keepdim=keepdim):
                        self.assert_same_result(
                            fn(x_mps, dim=[-1], keepdim=keepdim),
                            fn(x_cpu, dim=[-1], keepdim=keepdim))

    def test_value_and_index_describe_one_element(self):
        # Both halves of the pair come out of the same pass over the row, so an
        # index names the element the value was read from -- and it is the same
        # element its own row's CPU reduction names.
        for shape in self.SHAPES:
            x_cpu, x_mps = _repeating_on_both(shape)
            rows_cpu = tp.reshape(x_cpu, [-1, shape[-1]]).cpu().tolist()
            for name, fn in (("max", tp.max), ("min", tp.min)):
                for keepdim in (False, True):
                    with self.subTest(shape=shape, op=name, keepdim=keepdim):
                        values, indices = fn(x_mps, dim=-1, keepdim=keepdim)
                        ref_values, ref_indices = fn(x_cpu, dim=-1,
                                                     keepdim=keepdim)
                        self.assert_same_result(values, ref_values)
                        self.assertEqual(indices.cpu().tolist(),
                                         ref_indices.cpu().tolist())
                        got = tp.reshape(values, [-1, 1]).cpu().tolist()
                        held = tp.reshape(indices, [-1, 1]).cpu().tolist()
                        for row, (one_value, one_index) in enumerate(
                                zip(got, held)):
                            self.assertEqual(one_value[0],
                                             rows_cpu[row][one_index[0]])

    def test_ties_answer_with_the_lowest_position(self):
        # One right answer per row in each direction, so a dispatch and a scan
        # cannot both be correct without agreeing on the position.
        rows = [[3.0, 5.0, 5.0, 1.0], [7.0, 7.0, 0.0, 7.0],
                [-4.0, -4.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0],
                [6.0, -6.0, 6.0, -6.0]]
        cpu = tp.tensor(rows, dtype=tp.float32, device="cpu")
        mps = tp.tensor(rows, dtype=tp.float32, device="mps")
        for name, fn, want_values, want_indices in (
                ("max", tp.max, [5.0, 7.0, 2.0, 0.0, 6.0], [1, 0, 2, 0, 0]),
                ("min", tp.min, [1.0, 0.0, -4.0, 0.0, -6.0], [3, 2, 0, 0, 1])):
            with self.subTest(op=name):
                values, indices = fn(mps, dim=-1)
                self.assertEqual(values.cpu().tolist(), want_values)
                self.assertEqual(indices.cpu().tolist(), want_indices)
                self.assertEqual(fn(cpu, dim=-1)[1].cpu().tolist(),
                                 want_indices)
        for name, fn, want in (("amax", tp.amax, [5.0, 7.0, 2.0, 0.0, 6.0]),
                               ("amin", tp.amin, [1.0, 0.0, -4.0, 0.0, -6.0])):
            with self.subTest(op=name):
                self.assertEqual(fn(mps, dim=[-1]).cpu().tolist(), want)

    def test_a_nan_is_the_extremum_of_its_row(self):
        # A NaN displaces every value and nothing displaces it, so the first of
        # a row's NaNs answers in both directions. A row without one is
        # untouched by that rule, so its own extremum still answers.
        rows = [[1.0, float("nan"), 9.0, float("nan"), 4.0],
                [-2.0, 7.0, 7.0, 3.0, 8.0],
                [float("nan"), 5.0, 5.0, 6.0, 0.0],
                [2.0, 2.0, float("nan"), 1.0, 1.0]]
        cpu = tp.tensor(rows, dtype=tp.float32, device="cpu")
        mps = tp.tensor(rows, dtype=tp.float32, device="mps")
        nan_rows = (0, 2, 3)
        for name, fn, want_indices, want_finite in (
                ("max", tp.max, [1, 4, 0, 2], [8.0]),
                ("min", tp.min, [1, 0, 0, 2], [-2.0])):
            for keepdim in (False, True):
                with self.subTest(op=name, keepdim=keepdim):
                    values, indices = fn(mps, dim=-1, keepdim=keepdim)
                    flat = tp.reshape(values, [-1]).cpu().tolist()
                    for row, one in enumerate(flat):
                        if row in nan_rows:
                            self.assertTrue(math.isnan(one),
                                            msg=f"row {row}: {one}")
                        else:
                            self.assertEqual(one, want_finite[0])
                    self.assertEqual(tp.reshape(indices, [-1]).cpu().tolist(),
                                     want_indices)
                    ref_values, ref_indices = fn(cpu, dim=-1, keepdim=keepdim)
                    self.assertEqual(
                        tp.reshape(ref_indices, [-1]).cpu().tolist(),
                        want_indices)
                    self.assertEqual(
                        [math.isnan(one) for one in
                         tp.reshape(ref_values, [-1]).cpu().tolist()],
                        [row in nan_rows for row in range(len(rows))])
        for name, fn in (("amax", tp.amax), ("amin", tp.amin)):
            with self.subTest(op=name):
                flat = tp.reshape(fn(mps, dim=[-1]), [-1]).cpu().tolist()
                self.assertEqual([math.isnan(one) for one in flat],
                                 [row in nan_rows for row in range(len(rows))])

    def test_predicates_along_the_last_axis(self):
        for shape in self.SHAPES:
            x_cpu, x_mps = _repeating_on_both(shape)
            # A row of small integers holds zeros and nonzeros, so both answers
            # are reached within one operand rather than only one of them.
            for name, fn in (("all", tp.all), ("any", tp.any)):
                for keepdim in (False, True):
                    with self.subTest(shape=shape, op=name, keepdim=keepdim):
                        self.assert_same_result(
                            fn(x_mps, dim=-1, keepdim=keepdim),
                            fn(x_cpu, dim=-1, keepdim=keepdim))
            for name, fn in (("all", tp.all), ("any", tp.any)):
                with self.subTest(op=name, operand="bool"):
                    self.assert_same_result(fn(x_mps > 0.0, dim=-1),
                                            fn(x_cpu > 0.0, dim=-1))

    def test_a_nonzero_test_is_of_the_operand_s_type(self):
        # Against zero of its own type: a negative zero is not nonzero and a NaN
        # is, so a row of nothing but zeros and negative zeros answers false to
        # both questions.
        rows = [[0.0, -0.0, 0.0], [1.0, -0.0, float("nan")], [0.0, 2.0, 0.0],
                [3.0, float("nan"), -1.0], [-0.0, -0.0, -0.0]]
        cpu = tp.tensor(rows, dtype=tp.float32, device="cpu")
        mps = tp.tensor(rows, dtype=tp.float32, device="mps")
        for name, fn, want in (("all", tp.all, [False, False, False, True,
                                                False]),
                               ("any", tp.any, [False, True, True, True,
                                                False])):
            with self.subTest(op=name):
                self.assertEqual(fn(mps, dim=-1).cpu().tolist(), want)
                self.assertEqual(fn(cpu, dim=-1).cpu().tolist(), want)

    def test_shader_and_graph_alternate_at_one_call_site(self):
        # An axis the row dispatch serves sits between axes it declines, so a
        # call site sees both ways of reducing and has to answer the same way
        # both times.
        x_cpu, x_mps = _repeating_on_both([3, 4, 5])
        b_cpu, b_mps = x_cpu > 0.0, x_mps > 0.0
        for rep in range(2):
            for dim in (0, -1, 1, -1, 2, 0, -1):
                with self.subTest(rep=rep, dim=dim):
                    for name, fn in (("max", tp.max), ("min", tp.min)):
                        values, indices = fn(x_mps, dim=dim)
                        ref_values, ref_indices = fn(x_cpu, dim=dim)
                        self.assert_same_result(values, ref_values)
                        self.assertEqual(indices.cpu().tolist(),
                                         ref_indices.cpu().tolist())
                    for name, fn in (("all", tp.all), ("any", tp.any)):
                        self.assert_same_result(fn(x_mps, dim=dim),
                                                fn(x_cpu, dim=dim))
                        self.assert_same_result(fn(b_mps, dim=dim),
                                                fn(b_cpu, dim=dim))

    def test_declined_inputs_agree_with_cpu(self):
        """The last-axis reductions a single row dispatch hands to the graph."""
        wide_cpu, wide_mps = _repeating_on_both([6, 8])
        base_cpu, base_mps = _repeating_on_both([4, 8])
        # A rank-1 operand leaves no shape for the write to be described with.
        line_cpu = tp.reshape(wide_cpu[0], [8])
        line_mps = tp.reshape(wide_mps[0], [8])
        for name, fn in (("max", tp.max), ("min", tp.min), ("all", tp.all),
                         ("any", tp.any)):
            with self.subTest(op=name, operand="rank-1"):
                if name in ("max", "min"):
                    values, indices = fn(line_mps, dim=-1)
                    ref_values, ref_indices = fn(line_cpu, dim=-1)
                    self.assert_same_result(values, ref_values)
                    self.assertEqual(indices.cpu().tolist(),
                                     ref_indices.cpu().tolist())
                else:
                    self.assert_same_result(fn(line_mps, dim=-1),
                                            fn(line_cpu, dim=-1))
        # A permuted view does not run through its storage in the order the rows
        # are read in, and a slice that starts inside its storage has no base
        # offset to give.
        for name, fn in (("amax", tp.amax), ("amin", tp.amin), ("all", tp.all),
                         ("any", tp.any)):
            axis = [-1] if name in ("amax", "amin") else -1
            with self.subTest(op=name, operand="transposed"):
                self.assert_same_result(
                    fn(tp.transpose(base_mps, 0, 1), axis),
                    fn(tp.transpose(base_cpu, 0, 1), axis))
            with self.subTest(op=name, operand="offset slice"):
                self.assert_same_result(fn(base_mps[1:4], axis),
                                        fn(base_cpu[1:4], axis))
        # An operand of a type no row kernel is instantiated for reduces in a
        # type of its own, which the graph builds around the reduction.
        for dtype in (tp.int64, tp.int32, tp.float16):
            x_cpu, x_mps = _range_on_both([5, 7], dtype)
            for name, fn in (("max", tp.max), ("min", tp.min), ("all", tp.all),
                             ("any", tp.any)):
                with self.subTest(dtype=dtype, op=name):
                    if name in ("max", "min"):
                        values, indices = fn(x_mps, dim=-1)
                        ref_values, ref_indices = fn(x_cpu, dim=-1)
                        self.assert_same_result(values, ref_values)
                        self.assertEqual(indices.cpu().tolist(),
                                         ref_indices.cpu().tolist())
                    else:
                        self.assert_same_result(fn(x_mps, dim=-1),
                                                fn(x_cpu, dim=-1))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSUnaryOperandReads(MPSCase):
    """A unary float operand read by one shader dispatch, no graph built."""

    def test_each_operation_matches_cpu(self):
        base_cpu, base_mps = _range_on_both([3, 17])
        pos_cpu, pos_mps = base_cpu + 1.0, base_mps + 1.0
        mid_cpu, mid_mps = (base_cpu - 8.0) / 4.0, (base_mps - 8.0) / 4.0
        signed_cpu, signed_mps = base_cpu - 25.0, base_mps - 25.0
        for name, arg_cpu, arg_mps, fn in (
                ("abs", signed_cpu, signed_mps, tp.abs),
                ("exp", mid_cpu, mid_mps, tp.exp),
                ("sqrt", pos_cpu, pos_mps, tp.sqrt),
                ("rsqrt", pos_cpu, pos_mps, tp.rsqrt),
                ("relu", signed_cpu, signed_mps, tp.relu)):
            with self.subTest(op=name):
                self.assert_same_result(fn(arg_mps), fn(arg_cpu))

    def test_powers_of_four_are_exact(self):
        # These square roots and reciprocal square roots are representable in
        # binary floating point, so a shader and a graph have to agree on every
        # bit -- and a zero-dimensional operand is read by the same dispatch.
        values = [1.0, 4.0, 16.0, 64.0, 0.25]
        cpu = tp.tensor(values, dtype=tp.float32, device="cpu")
        mps = tp.tensor(values, dtype=tp.float32, device="mps")
        self.assertEqual(tp.sqrt(mps).cpu().tolist(),
                         [1.0, 2.0, 4.0, 8.0, 0.5])
        self.assertEqual(tp.rsqrt(mps).cpu().tolist(),
                         [1.0, 0.5, 0.25, 0.125, 2.0])
        self.assertEqual(tp.abs(-mps).cpu().tolist(), values)
        # A zero-dimensional operand is read by the same dispatch, and a scalar
        # tensor reads back as one number rather than a one-element list.
        scalar = tp.full((), 4.0, device="mps")
        self.assertEqual(tp.rsqrt(scalar).cpu().tolist(), 0.5)
        self.assertEqual(tp.exp(tp.zeros((2, 2), device="mps")).cpu().tolist(),
                         [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(tp.relu(tp.tensor([-4.0, 0.0, 0.25, 4.0],
                                           dtype=tp.float32,
                                           device="mps")).cpu().tolist(),
                         [0.0, 0.0, 0.25, 4.0])

    def test_special_values_match_cpu(self):
        # Reading the operand as it lies means handing back the infinities and
        # the NaNs that a graph hands back, rather than smoothing them over.
        values = [float("nan"), float("inf"), float("-inf"),
                  -4.0, -0.0, 0.0, 4.0]
        cpu = tp.tensor(values, dtype=tp.float32, device="cpu")
        mps = tp.tensor(values, dtype=tp.float32, device="mps")
        # exp runs on the hardware approximation, whose contract admits a few
        # units in the last place on an ordinary value; the infinities, the
        # zeros and the NaNs still have to come back exactly.
        ulps = {"exp": 4}
        for name, fn in (("abs", tp.abs), ("exp", tp.exp), ("relu", tp.relu),
                         ("sqrt", tp.sqrt), ("rsqrt", tp.rsqrt)):
            with self.subTest(op=name):
                got, want = fn(mps).cpu().tolist(), fn(cpu).cpu().tolist()
                self.assertEqual(len(got), len(want))
                for index, (one, other) in enumerate(zip(got, want)):
                    if math.isnan(other):
                        self.assertTrue(math.isnan(one),
                                        msg=f"{name}[{index}]: {one}, want NaN")
                    elif name in ulps and math.isfinite(other) and other != 0.0:
                        ulp = math.ulp(other) * 2.0 ** 29  # float32 spacing
                        self.assertLessEqual(abs(one - other), ulps[name] * ulp,
                                             msg=f"{name}[{index}]: {one} vs {other}")
                    else:
                        self.assertEqual(one, other, msg=f"{name}[{index}]")

    def test_alternating_shapes_at_one_call_site(self):
        shapes = [[2, 3], [24], [1, 5, 11], [2, 2, 2, 3]]
        for rep in range(2):
            for shape in shapes:
                with self.subTest(rep=rep, shape=shape):
                    cpu, mps = _range_on_both(shape)
                    self.assert_same_result(tp.exp(mps - 3.0),
                                            tp.exp(cpu - 3.0))

    def test_alternating_dtypes_at_one_call_site(self):
        # Half and int32 reach the graph, which caches a different dtype beside
        # the float it serves between them.
        for rep in range(2):
            for dtype in (tp.float32, tp.float16, tp.int32, tp.float32):
                with self.subTest(rep=rep, dtype=dtype):
                    cpu, mps = _range_on_both([5, 4], dtype)
                    self.assert_same_result(tp.abs(mps - 6), tp.abs(cpu - 6))

    def test_declined_inputs_agree_with_cpu(self):
        """The reads a single dispatch declines, which a graph serves."""
        base_cpu, base_mps = _range_on_both([4, 8])
        # A permuted view is not laid out in the order its result is written in.
        self.assert_same_result(tp.exp(tp.transpose(base_mps, 0, 1)),
                                tp.exp(tp.transpose(base_cpu, 0, 1)))
        # A slice that starts inside its storage has no base offset to give.
        wide_cpu, wide_mps = _range_on_both([16, 8])
        self.assert_same_result(tp.abs(wide_mps[3:11] - 60.0),
                                tp.abs(wide_cpu[3:11] - 60.0))
        # Nothing to read for an empty operand.
        empty_cpu, empty_mps = _range_on_both([0, 3])
        self.assert_same_result(tp.sqrt(empty_mps), tp.sqrt(empty_cpu))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSGraphCacheParameters(MPSCase):
    """Operations whose graph body is built from the numbers of the call.

    A clamp bound, an activation slope or a softmax axis is a constant inside
    the graph rather than a value fed through it, so calls that differ only
    there must not be served the entry built for the first one.
    """

    def test_alternating_clamp_bounds(self):
        cpu, mps = _range_on_both([3, 5])
        for rep in range(2):
            for lo, hi in ((-2.0, None), (None, 7.0), (-1.0, 6.0), (0.0, 4.0),
                           (-2.0, None), (3.0, 3.5)):
                with self.subTest(rep=rep, bounds=(lo, hi)):
                    self.assert_same_result(tp.clamp(mps, lo, hi),
                                            tp.clamp(cpu, lo, hi))

    def test_alternating_clamp_min_and_max(self):
        cpu, mps = _range_on_both([4, 5])
        for rep in range(2):
            for value in (0.0, 3.0, -1.0, 6.5):
                with self.subTest(rep=rep, value=value):
                    self.assert_same_result(tp.clamp_min(mps, value),
                                            tp.clamp_min(cpu, value))
                    self.assert_same_result(tp.clamp_max(mps, value),
                                            tp.clamp_max(cpu, value))

    def test_alternating_activation_parameters(self):
        cpu, mps = _range_on_both([2, 9])
        shifted_cpu, shifted_mps = cpu - 4.0, mps - 4.0
        cases = (
            ("leaky_relu",
             lambda x, value: tp.leaky_relu(x, negative_slope=value),
             (0.01, 0.5, 0.0, 1.0)),
            ("elu", lambda x, value: tp.elu(x, alpha=value), (1.0, 0.5, 2.0)),
            ("softplus", lambda x, value: tp.softplus(x, beta=value),
             (1.0, 2.0, 0.5)),
            ("softplus_threshold",
             lambda x, value: tp.softplus(x, beta=1.0, threshold=value),
             (1.0, 4.0, 20.0)),
            ("hardtanh",
             lambda x, value: tp.hardtanh(x, min_val=-value, max_val=value),
             (1.0, 2.0, 0.5)),
            ("gelu", lambda x, value: tp.gelu(x, approximate=value),
             ("none", "tanh", "none")),
        )
        for name, fn, values in cases:
            with self.subTest(op=name):
                for value in values:
                    self.assert_same_result(fn(shifted_mps, value),
                                            fn(shifted_cpu, value))

    def test_alternating_softmax_axis(self):
        # A square operand keeps the result shape the same while the axis
        # changes, so the axis has to be part of what names the graph.
        for shape in ([4, 4], [3, 3, 3]):
            count = math.prod(shape)
            cpu = tp.reshape(tp.arange(count, dtype=tp.float32,
                                       device="cpu"), list(shape))
            mps = tp.reshape(tp.arange(count, dtype=tp.float32,
                                       device="mps"), list(shape))
            cpu = (cpu - count / 2) / 8.0
            mps = (mps - count / 2) / 8.0
            for rep in range(2):
                for dim in range(len(shape)):
                    order = [dim, (dim + 1) % len(shape)]
                    for axis in order:
                        with self.subTest(shape=shape, rep=rep, dim=axis):
                            self.assert_same_result(tp.softmax(mps, axis),
                                                    tp.softmax(cpu, axis))
                            self.assert_same_result(
                                tp.log_softmax(mps, axis),
                                tp.log_softmax(cpu, axis))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSQueuedChains(MPSCase):
    """A chain of calls the stream commits, and reads, only once.

    A call queues into the command buffer its stream is already building and
    leaves that buffer open, so nothing reaches the GPU until something needs
    what the chain has produced: a copy to the host, a graph launch, an explicit
    synchronize. The whole chain then runs in the order it was queued. Each check
    below builds a chain and reads it back exactly once, so a link that read
    storage a later link had already been given -- or a link the single commit
    left behind -- would answer wrong.
    """

    def test_graph_and_shader_links_in_one_chain(self):
        # The three matrix links are served by a graph and the three math links
        # by a single shader dispatch, so the buffer changes hands twice.
        def scaled(shape, device):
            # A spread of magnitudes rather than one near-constant answer, so a
            # link that read the wrong storage cannot hide inside the tolerance.
            count = math.prod(shape)
            values = tp.reshape(tp.arange(count, dtype=tp.float32, device=device),
                                list(shape))
            return (values - count / 2) / count

        def chain(x, w, v, bias):
            first = tp.mm(x, w)
            second = tp.addmm(bias, first, v)
            operand = tp.reshape(second, [4, 2, 2])
            third = tp.bmm(operand, operand)
            return tp.relu(tp.exp(third) - 0.5)

        shapes = ([4, 8], [8, 4], [4, 4], [4])
        cpu = [scaled(shape, "cpu") for shape in shapes]
        mps = [scaled(shape, "mps") for shape in shapes]
        self.assert_same_result(chain(*mps), chain(*cpu))

    def test_dropped_intermediates_are_rewritten(self):
        # Each round queues a graph link and two shader links and frees every
        # intermediate as soon as the link after it has been queued, so blocks
        # return to the pool while the chain is still open and a later link is
        # routinely handed the storage an earlier one is queued to read.
        def scaled(rows, device):
            values = tp.reshape(tp.arange(rows * rows, dtype=tp.float32,
                                          device=device), [rows, rows])
            return (values - rows * rows / 2) / (rows * rows)

        shifts = (0.25, -0.5, 1.0, -0.75, 0.5, -0.25, 0.75, -1.0)
        for device in ("cpu", "mps"):
            x = scaled(8, device)
            w = scaled(8, device)
            acc = None
            for value in shifts:
                product = tp.mm(x, w)
                bounded = tp.tanh(product)
                del product
                shifted = tp.add(bounded, tp.full([8, 8], value, device=device))
                del bounded
                acc = shifted if acc is None else tp.add(acc, shifted)
            if device == "cpu":
                on_cpu = acc
            else:
                on_mps = acc
        self.assert_same_result(on_mps, on_cpu)


if __name__ == "__main__":
    unittest.main()
