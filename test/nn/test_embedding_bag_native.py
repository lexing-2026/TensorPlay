"""Native bagged-embedding operators.

Covers the four `_embedding_bag*` primitives -- the reduction itself, its
inference-only twin, and the two backward kernels -- plus the
``F.embedding_bag`` surface that now sits directly on top of them.  Reference
values come from the reference framework installed in the test environment.
"""

import numpy as np
import pytest
import torch

import tensorplay as tp
import tensorplay.nn.functional as F
import tensorplay._C as _C


SUM, MEAN, MAX = 0, 1, 2
MODES = [("sum", SUM), ("mean", MEAN), ("max", MAX)]


def _mk(array, dtype=None):
    t = tp.tensor(np.ascontiguousarray(array))
    return t.to(dtype) if dtype is not None else t


def _np(t):
    return np.asarray(t.tolist(), dtype=np.float64)


def _close(actual, expected, rtol=1e-9, atol=1e-11, msg=""):
    got = _np(actual)
    want = np.asarray(expected, dtype=np.float64)
    assert got.shape == want.shape, f"{msg}: shape {got.shape} != {want.shape}"
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=msg)


def _weight(rows=10, dim=4, seed=0):
    return np.random.RandomState(seed).randn(rows, dim).astype(np.float64)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestEmbeddingBagForward:
    @pytest.mark.parametrize("name,mode", MODES)
    def test_offsets_match_reference(self, name, mode):
        w = _weight(seed=1)
        idx = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
        offsets = np.array([0, 3, 3, 6], dtype=np.int64)

        out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, mode,
                                False, None, False, -1)[0]
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode=name)
        _close(out, want.detach().numpy(), msg=f"offsets {name}")

    @pytest.mark.parametrize("name,mode", MODES)
    def test_include_last_offset(self, name, mode):
        w = _weight(seed=2)
        idx = np.array([0, 7, 7, 2, 8, 8, 1], dtype=np.int64)
        offsets = np.array([0, 2, 5, 7], dtype=np.int64)

        out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, mode,
                                False, None, True, -1)[0]
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode=name, include_last_offset=True)
        _close(out, want.detach().numpy(), msg=f"include_last_offset {name}")

    @pytest.mark.parametrize("name,mode", MODES)
    def test_padding_idx_excluded(self, name, mode):
        w = _weight(seed=3)
        idx = np.array([4, 4, 2, 4, 6, 4], dtype=np.int64)
        offsets = np.array([0, 2, 4], dtype=np.int64)

        out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, mode,
                                False, None, False, 4)[0]
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode=name, padding_idx=4)
        _close(out, want.detach().numpy(), msg=f"padding_idx {name}")

    def test_per_sample_weights(self):
        w = _weight(seed=4)
        idx = np.array([1, 3, 5, 7, 9], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        psw = np.array([0.5, -1.5, 2.0, 0.25, 3.0], dtype=np.float64)

        out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, SUM,
                                False, _mk(psw), False, -1)[0]
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode="sum", per_sample_weights=torch.tensor(psw))
        _close(out, want.detach().numpy(), msg="per_sample_weights")

    @pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
    def test_dtypes(self, dtype):
        w = _weight(seed=5).astype(np.float32 if dtype is tp.float32 else np.float64)
        idx = np.array([0, 2, 4, 6], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)

        out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, MEAN,
                                False, None, False, -1)[0]
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode="mean")
        tol = 1e-6 if dtype is tp.float32 else 1e-11
        _close(out, want.detach().numpy(), rtol=1e-6, atol=tol, msg=str(dtype))

    def test_int32_indices(self):
        w = _weight(seed=6)
        idx = np.array([1, 2, 3, 4], dtype=np.int32)
        offsets = np.array([0, 2], dtype=np.int32)

        out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, SUM,
                                False, None, False, -1)[0]
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode="sum")
        _close(out, want.detach().numpy(), msg="int32 indices")

    def test_empty_bags_are_zero(self):
        w = _weight(seed=7)
        idx = np.array([2, 5], dtype=np.int64)
        # Bags 0 and 2 are empty; bag 1 owns both indices.
        offsets = np.array([0, 0, 2], dtype=np.int64)

        for name, mode in MODES:
            out = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, mode,
                                    False, None, False, -1)[0]
            want = torch.nn.functional.embedding_bag(
                torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
                mode=name)
            _close(out, want.detach().numpy(), msg=f"empty bags {name}")

    def test_forward_only_matches_full(self):
        w = _weight(seed=8)
        idx = np.array([9, 0, 3, 3, 7], dtype=np.int64)
        offsets = np.array([0, 1, 3], dtype=np.int64)

        for _, mode in MODES:
            full = _C._embedding_bag(_mk(w), _mk(idx), _mk(offsets), False, mode,
                                     False, None, False, -1)
            only = _C._embedding_bag_forward_only(
                _mk(w), _mk(idx), _mk(offsets), False, mode, False, None, False, -1)
            for a, b in zip(full, only):
                assert a.tolist() == b.tolist()


class TestEmbeddingBagBookkeeping:
    def test_offset2bag_and_bag_size(self):
        w = _weight(seed=9)
        idx = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        offsets = np.array([0, 2, 2], dtype=np.int64)

        _, offset2bag, bag_size, max_indices = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, SUM, False, None, False, -1)
        assert offset2bag.tolist() == [0, 0, 2, 2, 2]
        assert bag_size.tolist() == [2, 0, 3]
        assert max_indices.tolist() == [0, 0, 0]

    def test_bag_size_skips_padding(self):
        w = _weight(seed=10)
        idx = np.array([4, 1, 4, 4], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)

        _, _, bag_size, _ = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, SUM, False, None, False, 4)
        assert bag_size.tolist() == [1, 0]

    def test_max_indices_report_selected_rows(self):
        w = np.array([[0.0, 5.0], [1.0, 2.0], [4.0, 0.0]], dtype=np.float64)
        idx = np.array([0, 1, 2], dtype=np.int64)
        offsets = np.array([0], dtype=np.int64)

        out, _, _, max_indices = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, MAX, False, None, False, -1)
        assert out.tolist() == [[4.0, 5.0]]
        assert max_indices.tolist() == [[2, 0]]

    def test_uncovered_positions_are_marked(self):
        w = _weight(seed=11)
        idx = np.array([1, 2, 3, 4], dtype=np.int64)
        # include_last_offset stops the coverage at position 3.
        offsets = np.array([0, 2, 3], dtype=np.int64)

        _, offset2bag, bag_size, _ = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, SUM, False, None, True, -1)
        assert offset2bag.tolist() == [0, 0, 1, -1]
        assert bag_size.tolist() == [2, 1]


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

def _reference_grads(w, idx, offsets, mode, padding_idx=-1, psw=None,
                     scale_grad_by_freq=False, include_last_offset=False):
    tw = torch.tensor(w, requires_grad=True)
    tpsw = None if psw is None else torch.tensor(psw, requires_grad=True)
    out = torch.nn.functional.embedding_bag(
        torch.tensor(idx), tw, offsets=torch.tensor(offsets), mode=mode,
        padding_idx=None if padding_idx < 0 else padding_idx,
        per_sample_weights=tpsw, scale_grad_by_freq=scale_grad_by_freq,
        include_last_offset=include_last_offset)
    out.sum().backward()
    return tw.grad.numpy(), (None if tpsw is None else tpsw.grad.numpy())


class TestEmbeddingBagBackward:
    @pytest.mark.parametrize("name,mode", MODES)
    def test_dense_backward(self, name, mode):
        w = _weight(seed=12)
        idx = np.array([2, 2, 5, 0, 9, 5, 5], dtype=np.int64)
        offsets = np.array([0, 3, 5], dtype=np.int64)

        out, offset2bag, bag_size, max_indices = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, mode, False, None, False, -1)
        grad = tp.ones_like(out)
        got = _C._embedding_bag_dense_backward(
            grad, _mk(idx), offset2bag, bag_size, max_indices, w.shape[0],
            False, mode, None, -1)
        want, _ = _reference_grads(w, idx, offsets, name)
        _close(got, want, msg=f"dense backward {name}")

    def test_dense_backward_scale_by_freq(self):
        # scale_grad_by_freq divides each row's accumulated gradient by that
        # row's own occurrence count in `indices` -- the same inverse-frequency
        # convention the non-bagged `embedding` backward uses -- so a row that
        # appears n times with a unit upstream gradient lands on exactly 1.
        w = _weight(seed=13)
        idx = np.array([1, 1, 1, 4, 4, 7], dtype=np.int64)
        offsets = np.array([0, 3], dtype=np.int64)

        out, offset2bag, bag_size, max_indices = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), True, SUM, False, None, False, -1)
        got = _C._embedding_bag_dense_backward(
            tp.ones_like(out), _mk(idx), offset2bag, bag_size, max_indices,
            w.shape[0], True, SUM, None, -1)

        want = np.zeros_like(w)
        for row in (1, 4, 7):
            want[row] = 1.0
        _close(got, want, msg="dense backward scale_grad_by_freq")

        # Without the scaling the same rows accumulate their raw multiplicity.
        plain = _C._embedding_bag_dense_backward(
            tp.ones_like(out), _mk(idx), offset2bag, bag_size, max_indices,
            w.shape[0], False, SUM, None, -1)
        raw = np.zeros_like(w)
        for row, n in ((1, 3.0), (4, 2.0), (7, 1.0)):
            raw[row] = n
        _close(plain, raw, msg="dense backward without frequency scaling")

    def test_dense_backward_padding_idx(self):
        w = _weight(seed=14)
        idx = np.array([3, 6, 3, 1], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)

        out, offset2bag, bag_size, max_indices = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, MEAN, False, None, False, 3)
        got = _C._embedding_bag_dense_backward(
            tp.ones_like(out), _mk(idx), offset2bag, bag_size, max_indices,
            w.shape[0], False, MEAN, None, 3)
        want, _ = _reference_grads(w, idx, offsets, "mean", padding_idx=3)
        _close(got, want, msg="dense backward padding_idx")

    def test_per_sample_weights_backward(self):
        w = _weight(seed=15)
        idx = np.array([0, 4, 8, 2, 6], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        psw = np.array([1.0, -0.5, 0.25, 2.0, 0.75], dtype=np.float64)

        out, offset2bag, _, _ = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, SUM, False, _mk(psw), False, -1)
        got = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(out), _mk(w), _mk(idx), _mk(offsets), offset2bag, SUM, -1)
        _, want = _reference_grads(w, idx, offsets, "sum", psw=psw)
        _close(got, want, msg="per_sample_weights backward")

    def test_per_sample_weights_backward_rebuilds_offset2bag(self):
        w = _weight(seed=16)
        idx = np.array([1, 5, 3], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        psw = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        out, offset2bag, _, _ = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, SUM, False, _mk(psw), False, -1)
        empty = tp.zeros([0], dtype=tp.int64)
        from_empty = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(out), _mk(w), _mk(idx), _mk(offsets), empty, SUM, -1)
        from_map = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(out), _mk(w), _mk(idx), _mk(offsets), offset2bag, SUM, -1)
        _close(from_empty, _np(from_map), msg="rebuilt offset2bag")

    def test_per_sample_weights_backward_rejects_non_sum(self):
        w = _weight(seed=17)
        idx = np.array([1, 2], dtype=np.int64)
        offsets = np.array([0], dtype=np.int64)
        out, offset2bag, _, _ = _C._embedding_bag(
            _mk(w), _mk(idx), _mk(offsets), False, MEAN, False, None, False, -1)
        with pytest.raises(Exception):
            _C._embedding_bag_per_sample_weights_backward(
                tp.ones_like(out), _mk(w), _mk(idx), _mk(offsets), offset2bag,
                MEAN, -1)


class TestEmbeddingBagAutograd:
    @pytest.mark.parametrize("name,mode", MODES)
    def test_weight_grad_through_autograd(self, name, mode):
        w = _weight(seed=18)
        idx = np.array([0, 3, 3, 8, 1], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)

        tw = _mk(w)
        tw.requires_grad_(True)
        out = _C._embedding_bag(tw, _mk(idx), _mk(offsets), False, mode, False,
                                None, False, -1)[0]
        out.sum().backward()
        want, _ = _reference_grads(w, idx, offsets, name)
        _close(tw.grad, want, msg=f"autograd weight grad {name}")

    def test_per_sample_weights_grad_through_autograd(self):
        w = _weight(seed=19)
        idx = np.array([2, 4, 6, 8], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        psw = np.array([0.5, 1.5, -2.0, 3.0], dtype=np.float64)

        tw = _mk(w)
        tw.requires_grad_(True)
        tpsw = _mk(psw)
        tpsw.requires_grad_(True)
        out = _C._embedding_bag(tw, _mk(idx), _mk(offsets), False, SUM, False,
                                tpsw, False, -1)[0]
        out.sum().backward()
        want_w, want_psw = _reference_grads(w, idx, offsets, "sum", psw=psw)
        _close(tw.grad, want_w, msg="autograd weight grad with psw")
        _close(tpsw.grad, want_psw, msg="autograd per_sample_weights grad")


# ---------------------------------------------------------------------------
# Functional surface
# ---------------------------------------------------------------------------

class TestFunctionalEmbeddingBag:
    @pytest.mark.parametrize("name,_mode", MODES)
    def test_2d_input(self, name, _mode):
        w = _weight(seed=20)
        idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

        got = F.embedding_bag(_mk(idx), _mk(w), mode=name)
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), mode=name)
        _close(got, want.detach().numpy(), msg=f"2D {name}")

    @pytest.mark.parametrize("name,_mode", MODES)
    def test_negative_padding_idx(self, name, _mode):
        w = _weight(seed=21)
        idx = np.array([9, 1, 9, 2], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)

        got = F.embedding_bag(_mk(idx), _mk(w), offsets=_mk(offsets), mode=name,
                              padding_idx=-1)
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode=name, padding_idx=-1)
        _close(got, want.detach().numpy(), msg=f"negative padding_idx {name}")

    def test_rejects_out_of_range_padding_idx(self):
        w = _weight(seed=22)
        idx = np.array([1, 2], dtype=np.int64)
        offsets = np.array([0], dtype=np.int64)
        with pytest.raises(ValueError):
            F.embedding_bag(_mk(idx), _mk(w), offsets=_mk(offsets),
                            padding_idx=10)

    def test_rejects_offsets_with_2d_input(self):
        w = _weight(seed=23)
        idx = np.array([[1, 2], [3, 4]], dtype=np.int64)
        with pytest.raises(ValueError):
            F.embedding_bag(_mk(idx), _mk(w), offsets=_mk(np.array([0], dtype=np.int64)))

    def test_module_matches_reference(self):
        w = _weight(rows=8, dim=3, seed=24)
        idx = np.array([0, 5, 7, 2], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)

        for name in ("sum", "mean", "max"):
            mod = tp.nn.EmbeddingBag.from_pretrained(_mk(w), mode=name, freeze=True)
            got = mod(_mk(idx), _mk(offsets))
            want = torch.nn.EmbeddingBag.from_pretrained(
                torch.tensor(w), mode=name, freeze=True)(
                    torch.tensor(idx), torch.tensor(offsets))
            _close(got, want.detach().numpy(), msg=f"module {name}")
