"""Coverage sweep for previously untested ops.

Each check pins a golden value (or shape/dtype contract for nondeterministic
ops) captured from the CPU kernels; the values are independent of any other
framework. Ops whose results depend on uninitialized memory (resize_ growth,
empty_like/rand_like allocations) are checked for shape/dtype only.
"""
import pytest

import tensorplay as tp
from tensorplay.testing._internal.common_utils import TestCase, run_tests


class TestOpCoverage(TestCase):
    # ------------------------------------------------------------- reductions
    def test_cummax(self):
        vals, idxs = tp.cummax(tp.tensor([[1.0, 3.0], [2.0, 1.0]]), 0)
        self.assertEqual(vals.tolist(), [[1.0, 3.0], [2.0, 3.0]])
        self.assertEqual(idxs.tolist(), [[0, 0], [1, 0]])

    def test_cummin(self):
        vals, idxs = tp.cummin(tp.tensor([[1.0, 3.0], [2.0, 1.0]]), 1)
        self.assertEqual(vals.tolist(), [[1.0, 1.0], [2.0, 1.0]])
        self.assertEqual(idxs.tolist(), [[0, 0], [0, 1]])

    def test_cumprod(self):
        self.assertEqual(tp.cumprod(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), 1).tolist(),
                         [[1.0, 2.0], [3.0, 12.0]])

    def test_cumsum(self):
        self.assertEqual(tp.cumsum(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), 0).tolist(),
                         [[1.0, 2.0], [4.0, 6.0]])

    def test_nan_reductions(self):
        nan = float("nan")
        self.assertEqual(tp.nansum(tp.tensor([1.0, nan, 2.0])).item(), 3.0)
        self.assertEqual(tp.nanmean(tp.tensor([1.0, nan, 3.0])).item(), 2.0)
        self.assertEqual(tp.nanmedian(tp.tensor([3.0, nan, 1.0])).item(), 1.0)

    def test_std_mean_var_mean_full(self):
        # No dim argument: reduce to scalars.
        std, mean = tp.std_mean(tp.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(std.shape, ())
        self.assertEqual(mean.shape, ())
        self.assertAlmostEqual(mean.item(), 2.0)
        self.assertAlmostEqual(std.item(), 1.0)
        var, mean = tp.var_mean(tp.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(var.shape, ())
        self.assertAlmostEqual(var.item(), 1.0)
        self.assertAlmostEqual(mean.item(), 2.0)

    def test_std_mean_var_mean_dim(self):
        std, mean = tp.std_mean(tp.tensor([1.0, 2.0, 3.0]), 0)
        self.assertEqual(std.shape, ())
        self.assertEqual(mean.shape, ())
        var, mean = tp.var_mean(tp.tensor([1.0, 2.0, 3.0]), 0)
        self.assertEqual(var.shape, ())
        self.assertEqual(mean.shape, ())

    def test_logsumexp_logcumsumexp(self):
        self.assertEqual(tp.logsumexp(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), 0).tolist(),
                         [3.1269280910491943, 4.126927852630615])
        self.assertEqual(tp.logcumsumexp(tp.tensor([0.0, 1.0, 2.0]), 0).tolist(),
                         [0.0, 1.3132617473602295, 2.4076061248779297])

    def test_aminmax_kthvalue_topk(self):
        lo, hi = tp.aminmax(tp.tensor([3.0, 1.0, 2.0]))
        self.assertEqual(lo.item(), 1.0)
        self.assertEqual(hi.item(), 3.0)
        val, idx = tp.kthvalue(tp.tensor([4.0, 1.0, 3.0, 2.0]), 2)
        self.assertEqual(val.item(), 2.0)
        self.assertEqual(idx.item(), 3)
        vals, idxs = tp.topk(tp.tensor([1.0, 3.0, 2.0, 4.0]), 2)
        self.assertEqual(vals.tolist(), [4.0, 3.0])
        self.assertEqual(idxs.tolist(), [3, 1])

    def test_count_nonzero(self):
        self.assertEqual(tp.count_nonzero(tp.tensor([[1, 0], [0, 2]])).item(), 2)

    def test_cov_corrcoef(self):
        self.assertEqual(tp.cov(tp.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])).tolist(),
                         [[1.0, -1.0], [-1.0, 1.0]])
        self.assertEqual(tp.corrcoef(tp.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])).tolist(),
                         [[1.0, -1.0], [-1.0, 1.0]])

    def test_histc_bincount(self):
        self.assertEqual(tp.histc(tp.tensor([1.0, 2.0, 1.0]), bins=3, min=0, max=2).tolist(),
                         [0.0, 2.0, 1.0])
        self.assertEqual(tp.bincount(tp.tensor([1, 1, 2, 2, 2])).tolist(), [0, 2, 3])

    def test_argmin_argsort_argwhere(self):
        self.assertEqual(tp.argmin(tp.tensor([[3.0, 1.0], [2.0, 2.0]]), dim=1).tolist(), [1, 0])
        self.assertEqual(tp.argsort(tp.tensor([[3.0, 1.0], [2.0, 2.0]]), dim=1).tolist(),
                         [[1, 0], [0, 1]])
        self.assertEqual(tp.argwhere(tp.tensor([0, 1, 0, 2])).tolist(), [[1], [3]])

    # -------------------------------------------------- predicates / logical
    def test_isfinite_family(self):
        inf, nan = float("inf"), float("nan")
        self.assertEqual(tp.isfinite(tp.tensor([1.0, inf, nan])).tolist(), [True, False, False])
        self.assertEqual(tp.isinf(tp.tensor([1.0, inf])).tolist(), [False, True])
        self.assertEqual(tp.isnan(tp.tensor([1.0, nan])).tolist(), [False, True])
        self.assertEqual(tp.isneginf(tp.tensor([1.0, -inf])).tolist(), [False, True])
        self.assertEqual(tp.isposinf(tp.tensor([1.0, inf])).tolist(), [False, True])
        self.assertEqual(tp.isreal(tp.tensor([1.0 + 0j, 1.0 + 1j])).tolist(), [True, False])

    def test_signbit(self):
        self.assertEqual(tp.signbit(tp.tensor([-1.0, 1.0, 0.0])).tolist(), [True, False, False])

    def test_logical_ops(self):
        a = tp.tensor([True, False, True])
        b = tp.tensor([True, True, False])
        self.assertEqual(tp.logical_and(a, b).tolist(), [True, False, False])
        self.assertEqual(tp.logical_or(a, b).tolist(), [True, True, True])
        self.assertEqual(tp.logical_not(a).tolist(), [False, True, False])
        self.assertEqual(tp.logical_xor(a, b).tolist(), [False, True, True])

    def test_isin(self):
        self.assertEqual(tp.isin(tp.tensor([1, 2, 3]), tp.tensor([2, 3, 4])).tolist(),
                         [False, True, True])

    # -------------------------------------------------------- pointwise math
    def test_erfc_erfinv_exp2(self):
        self.assertEqual(tp.erfc(tp.tensor([0.0, 1.0])).tolist(), [1.0, 0.15729920566082])
        self.assertEqual(tp.erfinv(tp.tensor([0.0, 0.5])).tolist(), [0.0, 0.4769362807273865])
        self.assertEqual(tp.exp2(tp.tensor([1.0, 3.0])).tolist(), [2.0, 8.0])

    def test_bessel_family(self):
        self.assertEqual(tp.i0(tp.tensor([0.0, 1.0])).tolist(), [1.0, 1.2660658359527588])

    def test_gamma_family(self):
        self.assertEqual(tp.lgamma(tp.tensor([1.0, 5.0])).tolist(), [0.0, 3.178053855895996])
        self.assertEqual(tp.digamma(tp.tensor([1.0, 2.0])).tolist(),
                         [-0.5772156715393066, 0.42278432846069336])
        # trigamma(1) = pi^2/6
        self.assertAlmostEqual(tp.polygamma(1, tp.tensor([1.0])).item(), 1.644934058189392)
        self.assertEqual(tp.mvlgamma(tp.tensor([[1.5, 2.5]]), 2).tolist(),
                         [[0.4515827000141144, 0.8570477962493896]])
        self.assertEqual(tp.igamma(tp.tensor([1.0]), tp.tensor([1.0])).tolist(),
                         [0.6321205496788025])
        self.assertEqual(tp.igammac(tp.tensor([1.0]), tp.tensor([1.0])).tolist(),
                         [0.3678794503211975])

    def test_gcd_lcm(self):
        self.assertEqual(tp.gcd(tp.tensor([12, 18]), tp.tensor([8, 6])).tolist(), [4, 6])
        self.assertEqual(tp.lcm(tp.tensor([4, 6]), tp.tensor([6, 8])).tolist(), [12, 24])

    def test_hypot_ldexp(self):
        self.assertEqual(tp.hypot(tp.tensor([3.0, 5.0]), tp.tensor([4.0, 12.0])).tolist(),
                         [5.0, 13.0])
        self.assertEqual(tp.ldexp(tp.tensor([1.0, 2.0]), tp.tensor([3, 4])).tolist(),
                         [8.0, 32.0])

    def test_frexp_float_power(self):
        mant, exp = tp.frexp(tp.tensor([8.0, 0.5]))
        self.assertEqual(mant.tolist(), [0.5, 0.5])
        self.assertEqual(exp.tolist(), [4, 0])
        self.assertEqual(tp.float_power(tp.tensor([2.0]), tp.tensor([10])).tolist(), [1024.0])

    def test_xlogy_heaviside(self):
        self.assertEqual(tp.xlogy(tp.tensor([0.0, 2.0]), tp.tensor([3.0, 4.0])).tolist(),
                         [0.0, 2.7725887298583984])
        self.assertEqual(tp.heaviside(tp.tensor([-1.0, 0.0, 2.0]), tp.tensor([0.5])).tolist(),
                         [0.0, 0.5, 1.0])

    def test_logaddexp_family(self):
        self.assertEqual(tp.logaddexp(tp.tensor([0.0]), tp.tensor([0.0])).tolist(),
                         [0.6931471824645996])
        self.assertEqual(tp.logaddexp2(tp.tensor([1.0]), tp.tensor([2.0])).tolist(),
                         [2.5849626064300537])

    def test_sinc_sgn_signbit_copysign(self):
        self.assertEqual(tp.sinc(tp.tensor([0.0])).item(), 1.0)
        self.assertEqual(tp.sgn(tp.tensor([-3.0, 0.0, 2.0])).tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(tp.copysign(tp.tensor([3.0, -3.0]), tp.tensor([-1.0, 1.0])).tolist(),
                         [-3.0, 3.0])

    def test_deg_rad(self):
        self.assertEqual(tp.deg2rad(tp.tensor([0.0, 180.0])).tolist(),
                         [0.0, 3.1415927410125732])
        self.assertEqual(tp.rad2deg(tp.tensor([0.0, 3.141592653589793])).tolist(),
                         [0.0, 180.0])

    def test_atan2(self):
        self.assertAlmostEqual(tp.atan2(tp.tensor([1.0]), tp.tensor([1.0])).item(),
                               0.7853981852531433)

    def test_bitwise_ops(self):
        a = tp.tensor([1, 2, 3])
        b = tp.tensor([3, 2, 1])
        self.assertEqual(tp.bitwise_and(a, b).tolist(), [1, 2, 1])
        self.assertEqual(tp.bitwise_or(a, b).tolist(), [3, 2, 3])
        self.assertEqual(tp.bitwise_xor(a, b).tolist(), [2, 0, 2])
        self.assertEqual(tp.bitwise_not(a).tolist(), [-2, -3, -4])
        self.assertEqual(tp.bitwise_left_shift(tp.tensor([1, 2]), tp.tensor([2, 3])).tolist(),
                         [4, 16])
        self.assertEqual(tp.bitwise_right_shift(tp.tensor([4, 8]), tp.tensor([1, 2])).tolist(),
                         [2, 2])

    def test_fmax_fmin(self):
        nan = float("nan")
        self.assertEqual(tp.fmax(tp.tensor([1.0, nan]), tp.tensor([nan, 2.0])).tolist(),
                         [1.0, 2.0])
        self.assertEqual(tp.fmin(tp.tensor([1.0, nan]), tp.tensor([nan, 2.0])).tolist(),
                         [1.0, 2.0])

    def test_nextafter(self):
        # The step happens in the element dtype, not double (which would
        # round-trip back to the input for Float32).
        self.assertEqual(tp.nextafter(tp.tensor([1.0]), tp.tensor([2.0])).tolist(),
                         [1.0000001192092896])
        self.assertEqual(tp.nextafter(tp.tensor([0.0]), tp.tensor([1.0])).tolist(),
                         [1.401298464324817e-45])

    def test_nan_to_num(self):
        self.assertEqual(
            tp.nan_to_num(tp.tensor([1.0, float("nan"), float("inf"), float("-inf")]),
                          nan=0.0, posinf=9.0, neginf=-9.0).tolist(),
            [1.0, 0.0, 9.0, -9.0])

    def test_msort_fliplr_flipud(self):
        self.assertEqual(tp.msort(tp.tensor([[2.0, 1.0], [3.0, 0.0]])).tolist(),
                         [[2.0, 0.0], [3.0, 1.0]])
        self.assertEqual(tp.fliplr(tp.tensor([[1.0, 2.0], [3.0, 4.0]])).tolist(),
                         [[2.0, 1.0], [4.0, 3.0]])
        self.assertEqual(tp.flipud(tp.tensor([[1.0, 2.0], [3.0, 4.0]])).tolist(),
                         [[3.0, 4.0], [1.0, 2.0]])

    def test_tril_triu(self):
        ones = tp.ones(2, 2)
        self.assertEqual(tp.tril(ones, 1).tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(tp.triu(ones, 1).tolist(), [[0.0, 1.0], [0.0, 0.0]])

    def test_isclose(self):
        self.assertEqual(tp.isclose(tp.tensor([1.0, 1.0]), tp.tensor([1.0, 2.0])).tolist(),
                         [True, False])
        self.assertEqual(tp.isclose(tp.tensor([1.0, 1.0]), tp.tensor([1.1, 1.0]),
                                    rtol=0.1, atol=0.0).tolist(),
                         [True, True])
        # Non-finite values only match themselves.
        self.assertEqual(tp.isclose(tp.tensor([float("inf"), float("nan")]),
                                    tp.tensor([float("inf"), float("nan")])).tolist(),
                         [True, False])

    def test_isclose_dtype_and_tolerance_checks(self):
        # Operands must share the input dtype; mixed kinds are rejected.
        with self.assertRaises(RuntimeError):
            tp.isclose(tp.tensor([1.0]), tp.tensor([1.0], dtype=tp.float64))
        with self.assertRaises(RuntimeError):
            tp.isclose(tp.tensor([1.0]), tp.tensor([1], dtype=tp.int64))
        with self.assertRaises(RuntimeError):
            tp.isclose(tp.tensor([1.0]), tp.tensor([1.0]), rtol=-0.1)

    def test_compare_le_ne_rsub(self):
        self.assertEqual(tp.le(tp.tensor([1.0, 2.0]), 1.5).tolist(), [True, False])
        self.assertEqual(tp.ne(tp.tensor([1, 2]), 1).tolist(), [False, True])
        self.assertEqual(tp.rsub(tp.tensor([1.0, 2.0]), 1).tolist(), [0.0, -1.0])

    def test_clamp_max(self):
        self.assertEqual(tp.clamp_max(tp.tensor([1.0, 5.0]), 3.0).tolist(), [1.0, 3.0])

    def test_softmax(self):
        self.assertEqual(tp.softmax(tp.tensor([1.0, 2.0, 3.0]), 0).tolist(),
                         [0.09003057330846786, 0.2447284758090973, 0.6652409434318542])

    # ------------------------------------------------------ shape manipulation
    def test_meshgrid(self):
        g1, g2 = tp.meshgrid([tp.tensor([1, 2]), tp.tensor([3, 4, 5])])
        self.assertEqual(g1.tolist(), [[1, 1, 1], [2, 2, 2]])
        self.assertEqual(g2.tolist(), [[3, 4, 5], [3, 4, 5]])

    def test_movedim_moveaxis(self):
        self.assertEqual(tp.movedim(tp.tensor([[[1.0, 2.0]]]), 0, 2).tolist(),
                         [[[1.0], [2.0]]])
        self.assertEqual(tp.moveaxis(tp.tensor([[[1.0, 2.0]]]), 0, 2).tolist(),
                         [[[1.0], [2.0]]])

    def test_roll_rot90(self):
        self.assertEqual(tp.roll(tp.tensor([1.0, 2.0, 3.0, 4.0]), 1).tolist(),
                         [4.0, 1.0, 2.0, 3.0])
        self.assertEqual(tp.rot90(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), 1, [0, 1]).tolist(),
                         [[2.0, 4.0], [1.0, 3.0]])

    def test_atleast_nd(self):
        self.assertEqual(tp.atleast_1d(tp.tensor(3.0)).tolist(), [3.0])
        self.assertEqual(tp.atleast_2d(tp.tensor([1.0, 2.0])).tolist(), [[1.0, 2.0]])
        self.assertEqual(tp.atleast_3d(tp.tensor([1.0, 2.0])).tolist(), [[[1.0], [2.0]]])

    def test_broadcast_ops(self):
        b1, b2 = tp.broadcast_tensors(tp.tensor([[1.0], [2.0]]), tp.tensor([[1.0, 2.0]]))
        self.assertEqual(b1.tolist(), [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(b2.tolist(), [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(tp.broadcast_to(tp.tensor([1.0, 2.0]), (2, 2)).tolist(),
                         [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(tp.tensor([1.0]).expand_as(tp.zeros(2, 2)).tolist(),
                         [[1.0, 1.0], [1.0, 1.0]])

    def test_stack_family(self):
        self.assertEqual(tp.column_stack((tp.tensor([1.0, 2.0]), tp.tensor([3.0, 4.0]))).tolist(),
                         [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(tp.dstack((tp.tensor([1.0, 2.0]), tp.tensor([3.0, 4.0]))).tolist(),
                         [[[1.0, 3.0], [2.0, 4.0]]])
        self.assertEqual(tp.hstack((tp.tensor([[1.0]]), tp.tensor([[2.0]]))).tolist(),
                         [[1.0, 2.0]])
        self.assertEqual(tp.vstack((tp.tensor([1.0, 2.0]), tp.tensor([3.0, 4.0]))).tolist(),
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_split_family(self):
        self.assertEqual([p.tolist() for p in tp.hsplit(tp.tensor([[1.0, 2.0, 3.0, 4.0]]), 2)],
                         [[[1.0, 2.0]], [[3.0, 4.0]]])
        self.assertEqual([p.tolist() for p in tp.vsplit(tp.ones(4, 2), 2)],
                         [[[1.0, 1.0], [1.0, 1.0]]] * 2)
        self.assertEqual([p.tolist() for p in tp.dsplit(tp.ones(2, 2, 4), 2)],
                         [tp.ones(2, 2, 2).tolist()] * 2)
        self.assertEqual([p.tolist() for p in tp.tensor_split(tp.arange(6.0), 2)],
                         [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual([p.tolist() for p in tp.split_with_sizes(tp.arange(4.0), (1, 3))],
                         [[0.0], [1.0, 2.0, 3.0]])
        self.assertEqual([p.tolist() for p in tp.split_with_sizes_copy(tp.arange(4.0), (1, 3))],
                         [[0.0], [1.0, 2.0, 3.0]])
        self.assertEqual([p.tolist() for p in tp.unsafe_chunk(tp.arange(4.0), 2)],
                         [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual([p.tolist() for p in tp.unsafe_split(tp.arange(4.0), 2)],
                         [[0.0, 1.0], [2.0, 3.0]])

    def test_unbind_unflatten_ravel(self):
        self.assertEqual([p.tolist() for p in tp.unbind(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), 0)],
                         [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(tp.unflatten(tp.tensor([[1.0, 2.0, 3.0, 4.0]]), 1, (2, 2)).tolist(),
                         [[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(tp.ravel(tp.tensor([[1.0, 2.0], [3.0, 4.0]])).tolist(),
                         [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(tp.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape_as(tp.zeros(4)).tolist(),
                         [1.0, 2.0, 3.0, 4.0])

    def test_repeat_family(self):
        self.assertEqual(tp.tensor([1.0, 2.0]).repeat(2).tolist(), [1.0, 2.0, 1.0, 2.0])
        self.assertEqual(tp.tensor([1.0, 2.0]).repeat_interleave(2).tolist(),
                         [1.0, 1.0, 2.0, 2.0])

    def test_resize_(self):
        t = tp.tensor([1.0, 2.0])
        t.resize_((4,))
        self.assertEqual(tuple(t.shape), (4,))
        self.assertEqual(t[:2].tolist(), [1.0, 2.0])

    def test_block_diag_cartesian_prod(self):
        self.assertEqual(tp.block_diag(tp.tensor([[1.0, 2.0]]), tp.tensor([[3.0]])).tolist(),
                         [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        self.assertEqual(tp.cartesian_prod(tp.tensor([1, 2]), tp.tensor([3, 4])).tolist(),
                         [[1, 3], [1, 4], [2, 3], [2, 4]])

    def test_diagflat_diag_embed(self):
        self.assertEqual(tp.diagflat(tp.tensor([1.0, 2.0])).tolist(), [[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(tp.diag_embed(tp.tensor([1.0, 2.0])).tolist(),
                         [[1.0, 0.0], [0.0, 2.0]])

    def test_combinations_kron(self):
        self.assertEqual(tp.combinations(tp.tensor([1, 2, 3]), 2).tolist(),
                         [[1, 2], [1, 3], [2, 3]])
        self.assertEqual(tp.kron(tp.tensor([[1.0, 2.0]]), tp.tensor([[3.0]])).tolist(),
                         [[3.0, 6.0]])

    def test_view_as_real_resolve_neg(self):
        self.assertEqual(tp.view_as_real(tp.tensor([1.0 + 2j])).tolist(), [[1.0, 2.0]])
        self.assertEqual(tp.resolve_neg(tp.tensor([1.0 + 2j]).neg()).tolist(), [(-1.0 - 2j)])
        self.assertEqual(tp.conj_physical(tp.tensor([1.0 + 2j])).tolist(), [(1.0 - 2j)])

    def test_to_sparse_roundtrip(self):
        dense = tp.tensor([[1.0, 0.0]])
        self.assertEqual(tp.to_sparse(dense).to_dense().tolist(), [[1.0, 0.0]])

    # ------------------------------------------------------- indexing/scatter
    def test_index_copy_fill(self):
        self.assertEqual(tp.index_copy(tp.zeros(2, 2), 0, tp.tensor([1]),
                                       tp.tensor([[5.0, 6.0]])).tolist(),
                         [[0.0, 0.0], [5.0, 6.0]])
        self.assertEqual(tp.index_fill(tp.ones(2, 2), 0, tp.tensor([0]), 2.0).tolist(),
                         [[2.0, 2.0], [1.0, 1.0]])

    def test_index_put(self):
        want = [0.0, 0.0, 9.0]
        self.assertEqual(tp.index_put(tp.zeros(3), (tp.tensor([2]),), tp.tensor([9.0])).tolist(),
                         want)
        self.assertEqual(tp.zeros(3).index_put((tp.tensor([2]),), tp.tensor([9.0])).tolist(),
                         want)

    def test_index_reduce(self):
        # prod with a zero base row stays zero.
        self.assertEqual(tp.index_reduce(tp.zeros(2, 2), 0, tp.tensor([0]),
                                         tp.ones(1, 2), "prod").tolist(),
                         [[0.0, 0.0], [0.0, 0.0]])

    def test_index_select(self):
        self.assertEqual(tp.index_select(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), 0,
                                         tp.tensor([1])).tolist(), [[3.0, 4.0]])

    def test_masked_fill_scatter(self):
        self.assertEqual(tp.masked_fill(tp.ones(2), tp.tensor([True, False]), 0.0).tolist(),
                         [0.0, 1.0])
        self.assertEqual(tp.masked_scatter(tp.zeros(3), tp.tensor([True, False, True]),
                                           tp.tensor([7.0, 8.0])).tolist(),
                         [7.0, 0.0, 8.0])

    def test_scatter_add(self):
        self.assertEqual(tp.scatter_add(tp.zeros(3), 0, tp.tensor([0, 0]),
                                        tp.tensor([1.0, 2.0])).tolist(),
                         [3.0, 0.0, 0.0])

    def test_select_slice_diagonal_scatter(self):
        self.assertEqual(tp.select_scatter(tp.zeros(2, 2), tp.tensor([5.0, 6.0]), 0, 1).tolist(),
                         [[0.0, 0.0], [5.0, 6.0]])
        self.assertEqual(tp.slice_scatter(tp.zeros(4), tp.tensor([7.0, 8.0]), start=1, end=3).tolist(),
                         [0.0, 7.0, 8.0, 0.0])
        self.assertEqual(tp.diagonal_scatter(tp.zeros(2, 2), tp.tensor([1.0, 2.0])).tolist(),
                         [[1.0, 0.0], [0.0, 2.0]])

    def test_take_take_along_dim(self):
        self.assertEqual(tp.take(tp.tensor([[1.0, 2.0], [3.0, 4.0]]), tp.tensor([0, 3])).tolist(),
                         [1.0, 4.0])
        self.assertEqual(tp.take_along_dim(tp.tensor([[1.0, 2.0], [3.0, 4.0]]),
                                           tp.tensor([[1], [0]]), 0).tolist(),
                         [[3.0, 4.0], [1.0, 2.0]])

    def test_put(self):
        self.assertEqual(tp.put(tp.zeros(3), tp.tensor([2]), tp.tensor([9.0])).tolist(),
                         [0.0, 0.0, 9.0])

    def test_bucketize(self):
        self.assertEqual(tp.bucketize(tp.tensor([1.0, 3.0]), tp.tensor([2.0, 4.0])).tolist(),
                         [0, 1])

    def test_tril_triu_indices(self):
        self.assertEqual(tp.tril_indices(2, 2).tolist(), [[0, 1, 1], [0, 0, 1]])
        self.assertEqual(tp.triu_indices(2, 2).tolist(), [[0, 0, 1], [0, 1, 1]])

    def test_nonzero_static(self):
        self.assertEqual(tp.nonzero_static(tp.tensor([0, 1, 0, 2]), size=3).tolist(),
                         [[1], [3], [-1]])

    # ---------------------------------------------------------------- linalg
    def test_cholesky(self):
        self.assertEqual(tp.cholesky(tp.tensor([[4.0, 2.0], [2.0, 3.0]])).tolist(),
                         [[2.0, 0.0], [1.0, 1.4142135381698608]])

    def test_cholesky_inverse(self):
        self.assertEqual(tp.cholesky_inverse(tp.cholesky(tp.tensor([[4.0, 2.0], [2.0, 3.0]]))).tolist(),
                         [[0.375, -0.25], [-0.25, 0.5]])

    def test_cholesky_solve(self):
        L = tp.cholesky(tp.tensor([[4.0, 2.0], [2.0, 3.0]]))
        x = tp.cholesky_solve(tp.tensor([[1.0], [2.0]]), L)
        # (L L^T)^-1 B = [[-0.125], [0.75]]
        self.assertAlmostEqual(x[0][0].item(), -0.125, places=6)
        self.assertAlmostEqual(x[1][0].item(), 0.75, places=6)

    def test_triangular_solve(self):
        x, A = tp.triangular_solve(tp.tensor([[1.0], [2.0]]),
                                   tp.tensor([[2.0, 0.0], [1.0, 3.0]]), upper=False)
        self.assertEqual(x.tolist(), [[0.5], [0.5]])
        self.assertEqual(A.tolist(), [[2.0, 0.0], [1.0, 3.0]])

    def test_svd(self):
        A = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        U, S, V = tp.svd(A)
        self.assertEqual(tuple(S.shape), (2,))
        # Singular values are in descending order and pair with the U / V columns.
        self.assertAlmostEqual(S[0].item(), 5.4649858474731445, places=5)
        self.assertAlmostEqual(S[1].item(), 0.36596620082855225, places=5)
        recon = (U @ tp.diag(S) @ V.T).tolist()
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(recon[i][j], A.tolist()[i][j], places=4)

    def test_trapz(self):
        self.assertEqual(tp.trapz(tp.tensor([1.0, 2.0, 3.0]), dx=1.0).item(), 4.0)

    def test_stft(self):
        spec = tp.stft(tp.ones(64), 16, 8)
        self.assertEqual(tuple(spec.shape), (9, 9))
        self.assertAlmostEqual(spec[0][0].item(), 16.0)

    # ------------------------------------------------------ factories/random
    def test_like_factories(self):
        base = tp.ones(2, 3)
        self.assertEqual(tp.ones_like(base).tolist(), tp.ones(2, 3).tolist())
        self.assertEqual(tp.zeros_like(base).tolist(), tp.zeros(2, 3).tolist())
        self.assertEqual(tp.full_like(base, 5).tolist(), tp.full((2, 3), 5).tolist())

    def test_randint_deterministic_range(self):
        t = tp.randint(2, 3, (2, 2))
        self.assertEqual(tuple(t.shape), (2, 2))
        self.assertEqual(t.tolist(), [[2, 2], [2, 2]])
        like = tp.randint_like(tp.zeros(2, 2), 2, 3)
        self.assertEqual(like.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_linspace_logspace_scalar_tensor(self):
        self.assertEqual(tp.linspace(0.0, 1.0, 5).tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(tp.logspace(0.0, 2.0, 3).tolist(), [1.0, 10.0, 100.0])
        self.assertEqual(tp.scalar_tensor(3.5).item(), 3.5)

    def test_multinomial_bernoulli_deterministic(self):
        self.assertEqual(tp.multinomial(tp.tensor([1.0, 0.0, 0.0]), 1).tolist(), [0])
        self.assertEqual(tp.bernoulli(tp.zeros(2, 2)).tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_empty_allocations_shape_only(self):
        self.assertEqual(tuple(tp.empty_like(tp.ones(2, 2)).shape), (2, 2))
        self.assertEqual(tuple(tp.empty_strided((2, 2), (2, 1)).shape), (2, 2))
        self.assertEqual(tuple(tp.rand_like(tp.zeros(2, 2)).shape), (2, 2))
        self.assertEqual(tuple(tp.randn_like(tp.zeros(2, 2)).shape), (2, 2))

    def test_fill(self):
        self.assertEqual(tp.fill(tp.ones(2), 3).tolist(), [3.0, 3.0])
        self.assertEqual(tp.ones(2).fill_(3).tolist(), [3.0, 3.0])
        self.assertEqual(tp.ones(2).zero_().tolist(), [0.0, 0.0])

    # ------------------------------------------------- shaped pools & misc ops
    def test_renorm(self):
        self.assertEqual(tp.renorm(tp.tensor([[3.0, 4.0]]), 2, 0, 1.0).tolist(),
                         [[0.6000000238418579, 0.800000011920929]])

    def test_polar(self):
        out = tp.polar(tp.tensor([1.0]), tp.tensor([0.5235987755982988]))
        self.assertAlmostEqual(out[0].item().real, 0.8660253882408142, places=6)
        self.assertAlmostEqual(out[0].item().imag, 0.5, places=6)

    def test_cdist(self):
        self.assertEqual(tp.cdist(tp.tensor([[0.0, 0.0]]), tp.tensor([[3.0, 4.0]])).tolist(),
                         [[5.0]])

    def test_unique_consecutive(self):
        self.assertEqual(tp.unique_consecutive(tp.tensor([1, 1, 2, 3, 3])).tolist(),
                         [1, 2, 3])

    def test_constant_pad_nd(self):
        self.assertEqual(tp.constant_pad_nd(tp.ones(2, 2), (1, 1), 0.0).tolist(),
                         [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

    def test_vdot(self):
        self.assertEqual(tp.vdot(tp.tensor([1.0 + 2j]), tp.tensor([3.0 + 4j])).item(),
                         11.0 - 2j)

    def test_max_pool2d_with_indices_backward(self):
        inp = tp.ones(1, 1, 4, 4)
        out, idx = tp.max_pool2d_with_indices(inp, 2)
        grad = tp.max_pool2d_with_indices_backward(out, inp, (2,), (2,), (0,), (1,),
                                                   False, indices=idx)
        self.assertEqual(grad.flatten().tolist(),
                         [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_max_pool2d_with_indices_backward_shape_mismatch_raises(self):
        inp = tp.ones(1, 1, 4, 4)
        out, idx = tp.max_pool2d_with_indices(inp, 2)
        # grad_output larger than the pooled map must raise, not corrupt memory.
        with self.assertRaises(RuntimeError):
            tp.max_pool2d_with_indices_backward(tp.ones(1, 1, 4, 4), inp, (2,), (2,),
                                                (0,), (1,), False, indices=idx)

    def test_native_dropout_backward(self):
        grad = tp.native_dropout_backward(tp.ones(2, 2), tp.ones(2, 2).bool(), 0.5)
        self.assertEqual(grad.tolist(), [[0.5, 0.5], [0.5, 0.5]])

    # ------------------------------------------------- reflected dunders
    def test_reflected_dunders(self):
        t = tp.tensor([2, 4])
        self.assertEqual((1 + t).tolist(), [3, 5])
        self.assertEqual((9 - t).tolist(), [7, 5])
        self.assertEqual((3 * t).tolist(), [6, 12])
        self.assertEqual((8 / t).tolist(), [4.0, 2.0])
        self.assertEqual((6 & t).tolist(), [2, 4])
        self.assertEqual((1 | t).tolist(), [3, 5])
        self.assertEqual((3 ^ t).tolist(), [1, 7])
        self.assertEqual((9 % t).tolist(), [1, 1])
        self.assertEqual((2 ** t).tolist(), [4, 16])


if __name__ == "__main__":
    run_tests()
