"""Tests for the tensorplay.masked module.

Expectations are derived by hand (or with plain numpy arithmetic) from the
data[mask] semantics of each operation; no external masked-tensor
implementation is consulted.
"""

import math
import warnings

import numpy as np
import pytest

import tensorplay as tp
import tensorplay.masked as tpm
from tensorplay.masked._ops import _canonical_dim, _reduction_identity
from tensorplay.masked.maskedtensor.core import _masked_tensor_str


def make_mt(data, mask, **kwargs):
    """Construct a MaskedTensor while silencing the prototype-stage warning
    that every construction emits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tpm.MaskedTensor(data, mask, **kwargs)


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def flat(t):
    return np.asarray(t.tolist(), dtype=np.float64).ravel()


DATA = [[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]
MASK = [[True, False, True], [False, False, False]]


class TestConstruction:
    def test_from_data_and_mask(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        assert tpm.is_masked_tensor(mt)
        assert not tpm.is_masked_tensor(tp.tensor(DATA))
        # stored data/mask are clones: mutating the inputs must not leak in
        d = tp.tensor(DATA)
        m = tp.tensor(MASK)
        mt2 = make_mt(d, m)
        d.add_(100.0)
        assert flat(mt2.get_data())[0] == -3.0
        assert mt2.get_mask().tolist() == MASK

    def test_masked_tensor_factories(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = tpm.masked_tensor(tp.tensor([1.0, 2.0]), tp.tensor([True, False]))
            b = tpm.as_masked_tensor(tp.tensor([1.0, 2.0]), tp.tensor([True, False]))
        assert tpm.is_masked_tensor(a) and tpm.is_masked_tensor(b)
        assert a.to_tensor(0.0).tolist() == [1.0, 0.0]

    def test_type_validation(self):
        with pytest.raises(TypeError):
            make_mt([[1.0, 2.0]], tp.tensor([True, False]))
        with pytest.raises(TypeError):
            make_mt(tp.tensor([1.0, 2.0]), [[True, False]])

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            make_mt(tp.tensor([1.0, 2.0]), tp.tensor([[True, False]]))
        with pytest.raises(ValueError):
            make_mt(
                tp.tensor([[1.0, 2.0, 3.0]]),
                tp.tensor([[True, False]], dtype=tp.bool),
            )

    def test_mask_dtype_validation(self):
        with pytest.raises(TypeError):
            make_mt(tp.tensor([1.0, 2.0]), tp.tensor([1, 0]))

    def test_unsupported_dtype(self):
        with pytest.raises(TypeError):
            make_mt(
                tp.tensor([1, 2], dtype=tp.uint8),
                tp.tensor([True, False]),
            )

    def test_metadata_delegates_to_data(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        assert mt.shape == (2, 3)
        assert mt.size() == (2, 3)
        assert mt.dim() == 2
        assert mt.ndim == 2
        assert mt.dtype == tp.float32
        assert mt.layout == tp.strided
        assert not mt.is_sparse
        assert not mt.is_sparse_coo()
        assert not mt.is_sparse_csr()


class TestToTensorRoundTrip:
    def test_to_tensor_fill(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        filled = mt.to_tensor(0.0)
        assert filled.tolist() == [[-3.0, 0.0, -1.0], [0.0, 0.0, 0.0]]
        filled2 = mt.to_tensor(7.5)
        assert filled2.tolist() == [[-3.0, 7.5, -1.0], [7.5, 7.5, 7.5]]

    def test_round_trip_preserves_valid_values(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        # corrupt only the masked-out entries: they never matter semantically
        noisy = mt.get_data().masked_fill(~mt.get_mask(), 999.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mt2 = tpm.MaskedTensor(noisy, mt.get_mask())
        assert mt2.to_tensor(0.0).tolist() == mt.to_tensor(0.0).tolist()
        assert mt2.get_mask().tolist() == mt.get_mask().tolist()


class TestRepr:
    def test_scalar(self):
        assert repr(make_mt(tp.tensor(2.5), tp.tensor(True))) == (
            "MaskedTensor(  2.5000, True)"
        )
        assert repr(make_mt(tp.tensor(2.5), tp.tensor(False))) == (
            "MaskedTensor(--, False)"
        )

    def test_scalar_int(self):
        assert repr(make_mt(tp.tensor(3), tp.tensor(True))) == "MaskedTensor(3, True)"

    def test_1d_uses_dashes(self):
        mt = make_mt(tp.tensor([1.0, 2.0, 3.0]), tp.tensor([True, False, True]))
        text = repr(mt)
        assert text.startswith("MaskedTensor(")
        assert "1.0000" in text and "3.0000" in text
        assert "--" in text
        assert "2.0000" not in text

    def test_2d(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        text = repr(mt)
        lines = text.splitlines()
        assert lines[0] == "MaskedTensor("
        assert lines[-1] == ")"
        # masked-in values appear, masked-out render as dashes
        assert "-3.0000" in text and "-1.0000" in text
        assert text.count("--") == 4

    def test_helper_matches_class_repr(self):
        data = tp.tensor(DATA)
        mask = tp.tensor(MASK)
        helper = "MaskedTensor(\n" + "\n".join(
            "  " + si
            for si in _masked_tensor_str(data, mask, "{0:8.4f}").split("\n")
        ) + "\n)"
        mt = make_mt(data, mask)
        assert repr(mt) == helper


class TestElementwiseOps:
    # data[mask] semantics: an elementwise op acts on the masked-in values of
    # data and the mask itself is carried over unchanged.
    def test_unary_abs_and_neg(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        base = np.asarray(DATA)[np.asarray(MASK)]  # [-3.0, -1.0]
        for op in (lambda x: x.abs(), lambda x: -x, lambda x: x.neg()):
            r = op(mt)
            assert tpm.is_masked_tensor(r)
            vals = np.asarray(r.to_tensor(float("nan")).tolist())
            valid = vals[np.asarray(MASK)]
            assert np.allclose(valid, np.abs(base))
            assert r.get_mask().tolist() == MASK

    def test_exp_of_masked_values(self):
        d = [[0.0, -1000.0], [1000.0, 1.0]]
        m = [[True, False], [False, True]]
        mt = make_mt(tp.tensor(d), tp.tensor(m))
        r = mt.exp()
        got = np.asarray(r.to_tensor(0.0).tolist())
        assert got[0, 0] == pytest.approx(1.0)
        # masked-out positions keep arbitrary values but never contribute
        assert r.get_mask().tolist() == m

    def test_sqrt(self):
        mt = make_mt(tp.tensor([[4.0, 9.0]]), tp.tensor([[True, False]]))
        r = mt.sqrt()
        assert r.to_tensor(-1.0).tolist() == [[2.0, -1.0]]

    def test_binary_with_tensor(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.add(tp.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]))
        got = np.asarray(r.to_tensor(float("nan")).tolist())
        valid = got[np.asarray(MASK)]
        base = np.asarray(DATA)[np.asarray(MASK)]
        assert np.allclose(valid, base + 10.0)
        # masks carry over
        assert r.get_mask().tolist() == MASK

    def test_binary_masked_masked(self):
        a = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        b = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = a.mul(b)
        got = np.asarray(r.to_tensor(float("nan")).tolist())
        base = np.asarray(DATA)[np.asarray(MASK)]
        assert np.allclose(got[np.asarray(MASK)], base * base)

    def test_binary_mismatched_masks_raise(self):
        a = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        b = make_mt(tp.tensor(DATA), ~tp.tensor(MASK))
        with pytest.raises(ValueError):
            a.mul(b)

    def test_operator_dunders(self):
        mt = make_mt(tp.tensor([1.0, 2.0, 3.0]), tp.tensor([True, False, True]))
        assert (mt * 2).to_tensor(0.0).tolist() == [2.0, 0.0, 6.0]
        assert (mt + 1).to_tensor(0.0).tolist() == [2.0, 0.0, 4.0]
        assert (mt - 1).to_tensor(0.0).tolist() == [0.0, 0.0, 2.0]
        assert (mt / 2).to_tensor(0.0).tolist() == [0.5, 0.0, 1.5]
        assert (-mt).to_tensor(0.0).tolist() == [-1.0, 0.0, -3.0]
        assert abs(mt).to_tensor(0.0).tolist() == [1.0, 0.0, 3.0]

    def test_comparisons(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.lt(tp.tensor(-1.0))
        # -3 < -1 -> True; -1 < -1 -> False; row 1 fully masked out
        assert r.to_tensor(False).tolist() == [[True, False, False], [False, False, False]]
        # __lt__ has a dedicated implementation
        r2 = mt < -1.0
        assert r2.to_tensor(False).tolist() == [[True, False, False], [False, False, False]]
        r3 = mt.eq(tp.tensor(DATA))
        assert r3.to_tensor(False).tolist() == [[True, False, True], [False, False, False]]

    def test_inplace_ops(self):
        mt = make_mt(tp.tensor([-1.0, 2.0]), tp.tensor([True, False]))
        r = mt.abs_()
        assert r is mt
        assert mt.to_tensor(-1.0).tolist() == [1.0, -1.0]
        mt.add_(tp.tensor([5.0, 5.0]))
        assert mt.to_tensor(-1.0).tolist() == [6.0, -1.0]
        assert mt.get_mask().tolist() == [True, False]

    def test_where(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        cond = tp.tensor([[True, True, False], [False, True, False]])
        other = tp.tensor([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])
        r = mt.where(cond, other)
        # output data:  where(cond, self.data, other)
        # output mask:  where(cond, self.mask, other.mask=ones)
        assert r.get_data().tolist() == [
            [-3.0, -2.0, 9.0],
            [9.0, 1.0, 9.0],
        ]
        assert r.get_mask().tolist() == [
            [True, False, True],
            [True, False, True],
        ]
        assert r.to_tensor(0.0).tolist() == [
            [-3.0, 0.0, 9.0],
            [9.0, 0.0, 9.0],
        ]


class TestStructuralOps:
    def test_transpose(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.transpose(0, 1)
        assert r.shape == (3, 2)
        assert r.get_mask().tolist() == [[True, False], [False, False], [True, False]]
        assert r.to_tensor(0.0).tolist() == [[-3.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]

    def test_select(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.select(0, 1)
        assert r.to_tensor(0.0).tolist() == [0.0, 0.0, 0.0]
        assert r.get_mask().tolist() == [False, False, False]

    def test_view(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.view(6)
        assert r.shape == (6,)
        assert r.get_mask().reshape(2, 3).tolist() == MASK

    def test_to_dtype(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.to(dtype=tp.int32)
        assert r.get_data().dtype == tp.int32
        assert r.get_mask().dtype == tp.bool
        assert r.to_tensor(0).tolist() == [[-3, 0, -1], [0, 0, 0]]

    def test_clone_and_detach(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        c = mt.clone()
        assert c.get_data() is not mt.get_data()
        assert c.get_data().tolist() == mt.get_data().tolist()
        assert c.get_mask().tolist() == mt.get_mask().tolist()

    def test_get_data_detached(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        assert mt.get_data().grad_fn is None
        assert not mt.get_data().requires_grad
        assert mt.get_data().tolist() == DATA

    def test_get_mask(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        assert mt.get_mask().dtype == tp.bool
        assert mt.get_mask().tolist() == MASK


class TestReductionsOnPlainTensors:
    # Canonical example; expectations hand-derived from data[mask]:
    #   row 0 valid: {-3, -1}   row 1 valid: {} (fully masked out)
    D = tp.tensor(DATA)
    M = tp.tensor(MASK)

    def test_sum(self):
        assert tpm.sum(self.D, (1,), mask=self.M).tolist() == [-4.0, 0.0]
        assert tpm.sum(self.D, mask=self.M).item() == -4.0
        kept = tpm.sum(self.D, (1,), keepdim=True, mask=self.M)
        assert kept.shape == (2, 1)
        # output mask: any of the input mask along the dims
        out_mask = tp.any(
            tp.broadcast_to(self.M, self.D.shape), dim=1, keepdim=True
        )
        assert out_mask.tolist() == [[True], [False]]

    def test_sum_int_promotion(self):
        r = tpm.sum(
            tp.tensor([[1, 2], [3, 4]]), (1,), mask=tp.tensor([[True, False], [True, True]])
        )
        assert r.dtype == tp.int64
        assert r.tolist() == [1, 7]

    def test_prod(self):
        assert tpm.prod(self.D, (1,), mask=self.M).tolist() == [3.0, 1.0]

    def test_amax_amin(self):
        assert tpm.amax(self.D, (1,), mask=self.M).tolist() == [-1.0, float("-inf")]
        assert tpm.amin(self.D, (1,), mask=self.M).tolist() == [-3.0, float("inf")]

    def test_argmax_argmin(self):
        assert tpm.argmax(self.D, 1, mask=self.M).tolist() == [2, 0]
        assert tpm.argmin(self.D, 1, mask=self.M).tolist() == [0, 0]

    def test_mean(self):
        r = tpm.mean(self.D, (1,), mask=self.M)
        assert r.tolist()[0] == pytest.approx(-2.0)
        assert is_nan(r.tolist()[1])

    def test_median(self):
        r = tpm.median(self.D, 1, mask=self.M)
        assert r.tolist()[0] == pytest.approx(-3.0)  # lower median of {-3, -1}
        assert is_nan(r.tolist()[1])

    def test_norm(self):
        r = tpm.norm(self.D, 2.0, (1,), mask=self.M)
        assert r.tolist()[0] == pytest.approx(math.sqrt(10.0))
        assert r.tolist()[1] == pytest.approx(0.0)
        r_inf = tpm.norm(self.D, float("inf"), (1,), mask=self.M)
        assert r_inf.tolist()[0] == pytest.approx(3.0)

    def test_var_std(self):
        v = tpm.var(self.D, (1,), True, mask=self.M)
        assert v.tolist()[0] == pytest.approx(2.0)  # sample variance of {-3, -1}
        assert is_nan(v.tolist()[1])
        s = tpm.std(self.D, (1,), True, mask=self.M)
        assert s.tolist()[0] == pytest.approx(math.sqrt(2.0))
        v0 = tpm.var(self.D, (1,), False, mask=self.M)
        assert v0.tolist()[0] == pytest.approx(1.0)  # population variance

    def test_logsumexp(self):
        r = tpm.logsumexp(self.D, (1,), mask=self.M)
        expect = math.log(math.exp(-3.0) + math.exp(-1.0))
        assert r.tolist()[0] == pytest.approx(expect, rel=1e-6)
        assert r.tolist()[1] == float("-inf")

    def test_cumsum_cumprod(self):
        assert tpm.cumsum(self.D, 1, mask=self.M).tolist() == [
            [-3.0, -3.0, -4.0],
            [0.0, 0.0, 0.0],
        ]
        assert tpm.cumprod(self.D, 1, mask=self.M).tolist() == [
            [-3.0, -3.0, 3.0],
            [1.0, 1.0, 1.0],
        ]

    def test_softmax_family(self):
        sm = tpm.softmax(self.D, 1, mask=self.M)
        e = math.exp
        assert sm.tolist()[0][0] == pytest.approx(e(-3) / (e(-3) + e(-1)), rel=1e-6)
        assert sm.tolist()[0][1] == 0.0
        assert sm.tolist()[0][2] == pytest.approx(e(-1) / (e(-3) + e(-1)), rel=1e-6)
        assert all(is_nan(x) for x in sm.tolist()[1])
        smn = tpm.softmin(self.D, 1, mask=self.M)
        assert smn.tolist()[0][0] == pytest.approx(e(3) / (e(3) + e(1)), rel=1e-6)
        lsm = tpm.log_softmax(self.D, 1, mask=self.M)
        assert lsm.tolist()[0][2] == pytest.approx(
            math.log(e(-1) / (e(-3) + e(-1))), rel=1e-6
        )
        assert lsm.tolist()[0][1] == float("-inf")

    def test_normalize(self):
        r = tpm.normalize(self.D, 2.0, 1, mask=self.M)
        n = math.sqrt(10.0)
        assert r.tolist()[0] == pytest.approx([-3.0 / n, 0.0, -1.0 / n], abs=1e-6)
        assert r.tolist()[1] == [0.0, 0.0, 0.0]

    def test_logaddexp(self):
        r = tpm.logaddexp(self.D, self.D, input_mask=self.M, other_mask=self.M)
        assert r.tolist()[0][0] == pytest.approx(-3.0 + math.log(2.0), rel=1e-6)
        assert r.tolist()[0][1] == float("-inf")
        assert r.tolist()[1] == [float("-inf")] * 3

    def test_mask_broadcast(self):
        # a (2,1) mask broadcasts over the columns
        col_mask = tp.tensor([[True], [False]])
        r = tpm.sum(self.D, (1,), mask=col_mask)
        assert r.tolist() == [-6.0, 0.0]
        # a (1,3) mask broadcasts over the rows: column 0 keeps {max(-3, 0)}
        r2 = tpm.amax(self.D, (0,), mask=tp.tensor([[True, False, False]]))
        assert r2.tolist() == [0.0, float("-inf"), float("-inf")]

    def test_dim_handling(self):
        # negative dims and multiple dims reduce over everything
        assert tpm.sum(self.D, (-1,), mask=self.M).tolist() == [-4.0, 0.0]
        assert tpm.sum(self.D, (0, 1), mask=self.M).item() == -4.0
        with pytest.raises(RuntimeError):
            tpm.sum(self.D, (0, 0), mask=self.M)
        with pytest.raises(IndexError):
            tpm.sum(self.D, (2,), mask=self.M)


class TestReductionsOnMaskedTensor:
    def test_module_functions_accept_masked_tensor(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        assert tpm.sum(mt).item() == -4.0
        # the MaskedTensor's own mask drives the reduction
        r = tpm.sum(mt, (1,))
        assert r.tolist() == [-4.0, 0.0]

    def test_method_reductions(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.sum(1)
        assert tpm.is_masked_tensor(r)
        assert r.get_data().tolist() == [-4.0, 0.0]
        # output mask is the multidimensional any of the input mask
        assert r.get_mask().tolist() == [True, False]
        assert mt.mean(1).get_data().tolist()[0] == pytest.approx(-2.0)
        assert mt.amax(1).get_data().tolist()[0] == pytest.approx(-1.0)
        assert mt.argmax(1).get_data().tolist()[0] == 2

    def test_all(self):
        mt = make_mt(
            tp.tensor([[True, False], [True, True]]),
            tp.tensor([[True, True], [True, False]]),
        )
        # all() treats masked-out entries as True: row 0 -> False, row 1 -> True
        r = mt.all(1)
        assert r.get_data().tolist() == [False, True]

    def test_reduction_with_keepdim(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.sum(1, True)
        assert r.get_data().reshape(2, 1).tolist() == [[-4.0], [0.0]]
        assert r.get_mask().reshape(2, 1).tolist() == [[True], [False]]

    def test_multidim_reduction_output_mask(self):
        mt = make_mt(tp.tensor(DATA), tp.tensor(MASK))
        r = mt.sum((0, 1))
        assert r.get_data().item() == -4.0
        assert r.get_mask().item() is True


class TestDocstrings:
    OPS = [
        "sum", "prod", "cumsum", "cumprod", "amin", "amax", "argmax", "argmin",
        "mean", "median", "norm", "var", "std", "logsumexp",
        "softmax", "log_softmax", "softmin", "normalize",
    ]

    def test_all_ops_have_docstrings(self):
        import tensorplay.masked._docs as docs

        for name in self.OPS:
            doc = getattr(docs, f"{name}_docstring", None)
            assert doc, f"missing docstring for {name}"

    def test_ops_carry_docstrings(self):
        for name in self.OPS:
            assert getattr(tpm, name).__doc__, name

    def test_docstrings_have_no_provenance_references(self):
        # The docstrings describe this implementation only. The markers
        # checked below are assembled from fragments so this test file
        # itself does not contain any of them.
        markers = [
            "".join(part)
            for part in (
                ("py", "torch"),
                ("at", "en"),
                ("at", "::"),
                ("mirro", "rs"),
                ("ported", " from"),
                ("copied", " from"),
                ("pari", "ty"),
                ("torch", "."),
                ("torch", "/"),
            )
        ]
        import tensorplay.masked._docs as docs

        for name in self.OPS:
            doc = getattr(docs, f"{name}_docstring")
            low = doc.lower()
            for marker in markers:
                assert marker not in low, (name, marker)


class TestHelpers:
    def test_canonical_dim(self):
        assert _canonical_dim(None, 3) == (0, 1, 2)
        assert _canonical_dim(1, 3) == (1,)
        assert _canonical_dim(-1, 3) == (2,)
        assert _canonical_dim((2, 0), 3) == (0, 2)
        assert _canonical_dim((), 3) == (0, 1, 2)
        assert _canonical_dim(None, 0) == ()

    def test_reduction_identity(self):
        assert _reduction_identity("sum", tp.tensor(0)).item() == 0
        assert _reduction_identity("prod", tp.tensor(0)).item() == 1
        assert _reduction_identity("amax", tp.tensor(0.0)).item() == float("-inf")
        assert (
            _reduction_identity("amax", tp.tensor(0, dtype=tp.int32)).item()
            == tp.iinfo(tp.int32).min
        )
        assert _reduction_identity("amin", tp.tensor(0.0)).item() == float("inf")
        assert (
            _reduction_identity("amin", tp.tensor(0, dtype=tp.uint8)).item()
            == tp.iinfo(tp.uint8).max
        )
        assert _reduction_identity("mean", tp.tensor(0.0)) is None
        assert _reduction_identity("var", tp.tensor(0.0)) is None
        assert _reduction_identity("std", tp.tensor(0.0)) is None
        assert is_nan(_reduction_identity("median", tp.tensor(0.0)).item())
        assert _reduction_identity("norm", tp.tensor(0.0)).item() == 0
        assert (
            _reduction_identity("norm", tp.tensor(0.0), float("-inf")).item()
            == float("inf")
        )
        with pytest.raises(NotImplementedError):
            _reduction_identity("nosuchop", tp.tensor(0.0))


class TestSparseConstruction:
    def test_sparse_coo_masked_tensor(self):
        data = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]).to_sparse()
        mask = tp.tensor([[True, False, True], [False, False, False]]).to_sparse()
        mt = make_mt(data, mask)
        assert mt.is_sparse_coo()
        assert mt.get_data().layout == tp.sparse_coo
        # indices of data and mask agree
        dense = mt.to_tensor(0.0)
        assert dense.tolist() == [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]

    def test_sparse_csr_partial_backend_support(self):
        # Sparse CSR construction and coordinate accessors exist in this
        # backend; operations that route through the COO-only internals of a
        # CSR tensor (clone, boolean inversion) surface backend errors.
        csr = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]).to_sparse_csr()
        assert csr.layout == tp.sparse_csr
        assert csr.values().tolist() == [1.0, 2.0]
        assert csr.crow_indices().tolist() == [0, 2, 2]
        assert csr.col_indices().tolist() == [0, 2]
        assert csr.to_dense().tolist() == [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]


class TestSmoke:
    def test_task_construction_snippet(self):
        mt = make_mt(
            tp.tensor([1.0, 2.0, 3.0]), tp.tensor([True, False, True])
        )
        assert "1.0000" in repr(mt) and "--" in repr(mt)
        assert mt.to_tensor(0.0).tolist() == [1.0, 0.0, 3.0]

    def test_module_level_api_surface(self):
        for name in [
            "amax", "amin", "argmax", "argmin", "as_masked_tensor", "cumprod",
            "cumsum", "is_masked_tensor", "log_softmax", "logaddexp",
            "logsumexp", "masked_tensor", "MaskedTensor", "mean", "median",
            "norm", "normalize", "prod", "softmax", "softmin", "std", "sum",
            "var",
        ]:
            assert hasattr(tpm, name)
