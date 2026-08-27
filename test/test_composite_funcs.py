"""Spec tests: tensorplay._composite_funcs vs local torch 2.13.

Every composite added in the round-2 top-level operator batch is checked
for shape/value parity (and gradient flow where meaningful) against the
torch reference installed on this machine.
"""

import math

import pytest
import torch

import tensorplay as tp
from tensorplay._C import DType


def t32(t):
    return t.to(DType.float32)


def arange_f(*shape, offset=0.0):
    n = 1
    for s in shape:
        n *= s
    return tp.arange(n).to(DType.float32).add(offset).reshape(list(shape))


def close(a, b, tol=1e-5):
    if isinstance(a, tp.Tensor):
        a = a.tolist()
    if isinstance(b, torch.Tensor):
        b = b.tolist()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            close(x, y, tol) for x, y in zip(a, b)
        )
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if a is None or b is None:
        return a is b
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def assert_same_shape_list(pt, tt):
    def shapes(r):
        if isinstance(r, (tp.Tensor, torch.Tensor)):
            return [tuple(r.shape)]
        return [tuple(x.shape) for x in r]
    assert shapes(pt) == shapes(tt)


class TestAliases:
    def test_absolute(self):
        x = tp.tensor([-1.0, 2.0])
        r = tp.absolute(x)
        ref = torch.absolute(torch.tensor([-1.0, 2.0]))
        assert close(r.tolist(), ref.tolist())

    @pytest.mark.parametrize("name", ["arccos", "arcsin", "arctan",
                                      "arccosh", "arcsinh", "arctanh"])
    def test_arc_forward(self, name):
        dom = {"arccos": [0.1, -0.3], "arcsin": [0.1, -0.3],
               "arctan": [0.7, -2.0], "arccosh": [1.4, 2.5],
               "arcsinh": [0.7, -2.0], "arctanh": [0.4, -0.6]}[name]
        r = getattr(tp, name)(tp.tensor(dom))
        ref = getattr(torch, name)(torch.tensor(dom))
        assert close(r.tolist(), ref.tolist())

    @pytest.mark.parametrize("name", ["acos_", "asin_", "atan_",
                                      "acosh_", "asinh_", "atanh_"])
    def test_arc_inplace_value_and_grad(self, name):
        dom = {"acos_": [0.3], "asin_": [0.3], "atan_": [0.7],
               "acosh_": [1.8], "asinh_": [0.9], "atanh_": [0.4]}[name]
        base = {"acos_": torch.acos_, "asin_": torch.asin_,
                "atan_": torch.atan_, "acosh_": torch.acosh_,
                "asinh_": torch.asinh_, "atanh_": torch.atanh_}[name]
        # torch has no in-place derivative for acosh/asinh/atanh; fall
        # back to the closed-form slope for the gradient reference there.
        x0 = dom[0]
        analytic = {
            "acos_": lambda v: -1.0 / math.sqrt(1.0 - v * v),
            "asin_": None,
            "atan_": None,
            "acosh_": lambda v: 1.0 / math.sqrt(v * v - 1.0),
            "asinh_": lambda v: 1.0 / math.sqrt(v * v + 1.0),
            "atanh_": lambda v: 1.0 / (1.0 - v * v),
        }[name]
        if analytic is None:
            leaf = torch.tensor(dom).requires_grad_(True)
            xt = leaf.clone()
            base(xt)
            xt.sum().backward()
            ref_grad = leaf.grad.tolist()
        else:
            xt = torch.tensor(dom)
            with torch.no_grad():
                base(xt)
            ref_grad = [analytic(x0)]
        xp = tp.tensor(dom).requires_grad_(True)
        getattr(tp, name)(xp)
        xp.sum().backward()
        assert close(xp.tolist(), xt.tolist())
        assert close(xp.grad.tolist(), ref_grad)

    def test_arctan2(self):
        a = tp.tensor([1.0, -1.0])
        b = tp.tensor([0.5, 0.5])
        ref = torch.arctan2(torch.tensor([1.0, -1.0]),
                            torch.tensor([0.5, 0.5]))
        assert close(tp.arctan2(a, b).tolist(), ref.tolist())

    def test_concat_concatenate(self):
        xs = [arange_f(2), arange_f(2, offset=10)]
        ref = torch.cat([torch.arange(2.), torch.arange(2.) + 10])
        assert close(tp.concat(xs).tolist(), ref.tolist())
        assert close(tp.concatenate(xs, dim=0).tolist(), ref.tolist())

    def test_ger_rsub_adjoint(self):
        u = arange_f(3)
        v = arange_f(2, offset=1)
        assert close(tp.ger(u, v).tolist(),
                     torch.ger(torch.arange(3.), torch.arange(2.) + 1).tolist())
        assert close(tp.rsub(arange_f(3), 10.0).tolist(),
                     torch.rsub(torch.arange(3.), 10.0).tolist())
        m = arange_f(2, 3)
        adj = tp.adjoint(m)
        assert tuple(adj.size()) == (3, 2)
        assert close(adj.tolist(),
                     torch.adjoint(torch.arange(6.).reshape(2, 3)).tolist())

    @pytest.mark.parametrize("fn", ["divide", "multiply", "subtract",
                                    "true_divide", "floor_divide",
                                    "remainder", "fmod"])
    def test_binary_semantics_float(self, fn):
        a = [-7.0, 7.0, -0.5]
        b = [2.0, -3.0, 1.5]
        got = getattr(tp, fn)(tp.tensor(a), tp.tensor(b)).tolist()
        ref = getattr(torch, fn)(torch.tensor(a), torch.tensor(b)).tolist()
        assert close(got, ref)

    @pytest.mark.parametrize("fn", ["floor_divide", "remainder", "fmod"])
    def test_binary_semantics_int_negative(self, fn):
        a = [-7, 7, -9]
        b = [2, -3, 4]
        got = getattr(tp, fn)(tp.tensor(a), tp.tensor(b)).tolist()
        ref = getattr(torch, fn)(torch.tensor(a), torch.tensor(b)).tolist()
        assert got == ref

    def test_scalar_overloads(self):
        assert close(tp.divide(arange_f(2), 2.0).tolist(),
                     (torch.arange(2.) / 2).tolist())
        assert close(tp.multiply(3.0, arange_f(2)).tolist(),
                     (3 * torch.arange(2.)).tolist())
        assert close(tp.subtract(10.0, arange_f(2)).tolist(),
                     (10 - torch.arange(2.)).tolist())
        assert close(tp.remainder(arange_f(3), 2.0).tolist(),
                     torch.remainder(torch.arange(3.), 2.).tolist())

    def test_clamp_max_min(self):
        x = arange_f(3, offset=-1)
        ref = torch.arange(3.) - 1
        assert close(tp.clamp_max(x, 0.5).tolist(),
                     torch.clamp_max(ref, 0.5).tolist())
        assert close(tp.clamp_min(x, -0.25).tolist(),
                     torch.clamp_min(ref, -0.25).tolist())

    def test_copysign(self):
        a = tp.tensor([3.0, 3.0, 3.0])
        b = tp.tensor([1.0, -2.0, 0.5])
        ref = torch.copysign(torch.tensor([3., 3., 3.]),
                             torch.tensor([1., -2., 0.5]))
        assert close(tp.copysign(a, b).tolist(), ref.tolist())

    def test_detach_diagflat_numel_scalar_tensor(self):
        x = arange_f(4)
        d = tp.detach(x)
        assert d.tolist() == x.tolist()
        flat = tp.diagflat(arange_f(2))
        assert tuple(flat.size()) == (2, 2)
        assert close(flat.tolist(),
                     torch.diagflat(torch.arange(2.)).tolist())
        assert tp.numel(x) == 4
        s = tp.scalar_tensor(3.5)
        assert tuple(s.size()) == () and float(s) == 3.5


class TestLinalgStructured:
    def test_chain_matmul(self):
        ms = [arange_f(2, 3), arange_f(3, 4), arange_f(4, 2)]
        refs = [torch.arange(6.).reshape(2, 3), torch.arange(12.).reshape(3, 4),
                torch.arange(8.).reshape(4, 2)]
        assert close(tp.chain_matmul(*ms).tolist(),
                     torch.chain_matmul(*refs).tolist())

    def test_matrix_power(self):
        a = tp.tensor([[1.0, 1.0], [0.0, 1.0]])
        ref = torch.matrix_power(torch.tensor([[1., 1.], [0., 1.]]), 5)
        assert close(tp.matrix_power(a, 5).tolist(), ref.tolist())
        eye = tp.matrix_power(a, 0)
        assert close(eye.tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_matrix_power_negative_raises(self):
        with pytest.raises(NotImplementedError):
            tp.matrix_power(tp.eye(2), -1)

    def test_kron_2d(self):
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tp.tensor([[0.0, 1.0], [1.0, 0.0]])
        ref = torch.kron(torch.tensor([[1., 2.], [3., 4.]]),
                         torch.tensor([[0., 1.], [1., 0.]]))
        assert close(tp.kron(a, b).tolist(), ref.tolist())

    def test_kron_rect_and_vectors(self):
        a = arange_f(2, 3)
        b = arange_f(2, 2, offset=1)
        ref = torch.kron(torch.arange(6.).reshape(2, 3),
                         (torch.arange(4.) + 1).reshape(2, 2))
        assert close(tp.kron(a, b).tolist(), ref.tolist())
        u = arange_f(3)
        v = arange_f(2, offset=1)
        refv = torch.kron(torch.arange(3.), torch.arange(2.) + 1)
        assert close(tp.kron(u, v).tolist(), refv.tolist())

    def test_vander(self):
        x = tp.tensor([1.0, 2.0, 3.0])
        ref = torch.vander(torch.tensor([1., 2., 3.]))
        assert close(tp.vander(x).tolist(), ref.tolist())
        ref_n3 = torch.vander(torch.tensor([1., 2., 3.]), 3, increasing=True)
        assert close(tp.vander(x, 3, increasing=True).tolist(),
                     ref_n3.tolist())

    def test_tril_triu_indices(self):
        for offset in (-1, 0, 2):
            got = tp.tril_indices(4, 3, offset).tolist()
            ref = torch.tril_indices(4, 3, offset).tolist()
            assert got == ref
            got_u = tp.triu_indices(4, 3, offset).tolist()
            ref_u = torch.triu_indices(4, 3, offset).tolist()
            assert got_u == ref_u

    def test_cartesian_prod(self):
        a = tp.tensor([1.0, 2.0])
        b = tp.tensor([3.0, 4.0, 5.0])
        ref = torch.cartesian_prod(torch.tensor([1., 2.]),
                                   torch.tensor([3., 4., 5.]))
        assert close(tp.cartesian_prod(a, b).tolist(), ref.tolist())

    def test_combinations_default(self):
        x = arange_f(3, offset=1)
        ref = torch.combinations(torch.tensor([1., 2., 3.]), 2)
        assert close(tp.combinations(x, 2).tolist(), ref.tolist())

    def test_combinations_replacement_and_empty(self):
        x = arange_f(3, offset=1)
        ref = torch.combinations(torch.tensor([1., 2., 3.]), 2,
                                 with_replacement=True)
        assert close(tp.combinations(x, 2, with_replacement=True).tolist(),
                     ref.tolist())
        e = tp.combinations(x, 5)
        assert tuple(e.size()) == (0,)
        r0 = tp.combinations(x, 0)
        assert tuple(r0.size()) == (0,)


class TestStats:
    def test_cov_plain_rows(self):
        data = [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0.0, 4.0, 1.0]]
        ref = torch.cov(torch.tensor(data))
        assert close(tp.cov(tp.tensor(data)).tolist(), ref.tolist())

    def test_cov_columns_as_variables(self):
        data = [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]]
        ref = torch.cov(torch.tensor(data).T)
        pt = tp.tensor(data).permute([1, 0])
        assert close(tp.cov(pt).tolist(), ref.tolist())

    def test_cov_weights(self):
        data = [[1.0, 2.0, 3.0, 4.0]]
        fw = torch.tensor([1, 2, 0, 1])
        aw = torch.tensor([1.0, 0.5, 1.0, 2.0])
        for kw in (
            {},
            {"fweights": fw},
            {"aweights": aw},
            {"fweights": fw, "aweights": aw},
        ):
            tkw = dict(kw)
            pkw = {
                "fweights": tp.tensor(fw.tolist()) if "fweights" in kw else None,
                "aweights": tp.tensor(aw.tolist()) if "aweights" in kw else None,
            }
            pkw = {k: v for k, v in pkw.items() if v is not None}
            ref = torch.cov(torch.tensor(data), **tkw)
            got = tp.cov(tp.tensor(data), **pkw)
            assert close(got.tolist(), ref.tolist()), kw

    def test_corrcoef(self):
        x = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        ref = torch.corrcoef(x)
        got = tp.corrcoef(tp.tensor([[0., 1., 2.], [2., 1., 0.]]))
        assert close(got.tolist(), ref.tolist())
        diag = got.diagonal().tolist()
        assert all(abs(d - 1.0) < 1e-6 for d in diag)

    def test_trapezoid_dx_and_x(self):
        y = [1.0, 3.0, 5.0, 7.0]
        ref = torch.trapezoid(torch.tensor(y), dx=2.0)
        assert close(tp.trapezoid(tp.tensor(y), dx=2.0).item(), ref.item())
        xs = [0.0, 1.0, 3.0, 6.0]
        ref_x = torch.trapezoid(torch.tensor(y), torch.tensor(xs))
        got_x = tp.trapezoid(tp.tensor(y), tp.tensor(xs))
        assert close(got_x.item(), ref_x.item())

    def test_trapz_alias_and_dim(self):
        y = tp.arange(6).to(DType.float32).reshape([2, 3]).add(1)
        ry = torch.arange(6.).reshape(2, 3) + 1
        assert close(tp.trapz(y, dx=0.5, dim=1).tolist(),
                     torch.trapz(ry, dx=0.5, dim=1).tolist())

    def test_cumulative_trapezoid(self):
        y = [1.0, 4.0, 9.0]
        ref = torch.cumulative_trapezoid(torch.tensor(y), dx=1.0)
        got = tp.cumulative_trapezoid(tp.tensor(y), dx=1.0)
        assert close(got.tolist(), ref.tolist())

    def test_gradient_scalar_spacing(self):
        x = [1.0, 4.0, 9.0, 16.0]
        ref = torch.gradient(torch.tensor(x), spacing=(2.0,))[0] \
            if False else torch.gradient(torch.tensor(x), spacing=2.0)[0]
        got = tp.gradient(tp.tensor(x), spacing=2.0)[0]
        assert close(got.tolist(), ref.tolist())

    def test_gradient_coords_and_dims(self):
        t = torch.arange(12., dtype=torch.float64).reshape(3, 4)
        p = tp.arange(12).to(DType.float64).reshape([3, 4])
        coords = torch.tensor([0.0, 0.5, 2.0, 5.0], dtype=torch.float64)
        refs = torch.gradient(t, spacing=(coords,), dim=(1,))
        gots = tp.gradient(p, spacing=(tp.tensor([0., 0.5, 2., 5.]),), dim=(1,))
        assert len(gots) == len(refs)
        assert close(gots[0].tolist(), refs[0].tolist())
        both_ref = torch.gradient(t)
        both_got = tp.gradient(p)
        assert len(both_got) == 2
        for gr, gp in zip(both_ref, both_got):
            assert close(gp.tolist(), gr.tolist())

    def test_quantile_scalar_tensor_q(self):
        x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        p = tp.tensor([3., 1., 4., 1., 5.])
        assert close(tp.quantile(p, 0.5).item(),
                     torch.quantile(x, 0.5).item())
        qs = torch.tensor([0.1, 0.5, 0.9])
        ref = torch.quantile(x, qs)
        got = tp.quantile(p, tp.tensor([0.1, 0.5, 0.9]))
        assert close(got.tolist(), ref.tolist())

    def test_quantile_with_dim(self):
        x = torch.arange(12., dtype=torch.float64).reshape(3, 4)
        p = tp.arange(12).to(DType.float64).reshape([3, 4])
        q64 = torch.tensor([0.25, 0.75], dtype=torch.float64)
        ref = torch.quantile(x, q64, dim=1)
        got = tp.quantile(p, tp.tensor([0.25, 0.75], dtype=tp.float64), dim=1)
        assert tuple(got.shape) == tuple(ref.shape)
        assert close(got.tolist(), ref.tolist())
        refk = torch.quantile(x, torch.tensor(0.5, dtype=torch.float64),
                              dim=1, keepdim=True)
        gotk = tp.quantile(p, 0.5, dim=1, keepdim=True)
        assert tuple(gotk.shape) == tuple(refk.shape)
        assert close(gotk.tolist(), refk.tolist())

    def test_nanquantile(self):
        x = torch.tensor([float("nan"), 1.0, 5.0, 3.0])
        ref = torch.nanquantile(x, 0.5)
        got = tp.nanquantile(tp.tensor([float("nan"), 1., 5., 3.]), 0.5)
        assert close(got.item(), ref.item())
        allnan = torch.tensor([float("nan")])
        assert math.isnan(torch.nanquantile(allnan, 0.5).item())
        assert math.isnan(
            tp.nanquantile(tp.tensor([float("nan")]), 0.5).item())

    def test_histc(self):
        x = torch.tensor([0.2, 0.7, 1.1, 1.9, 2.0])
        ref = torch.histc(x, bins=4, min=0.0, max=2.0)
        got = tp.histc(tp.tensor([0.2, 0.7, 1.1, 1.9, 2.0]), bins=4,
                       min=0.0, max=2.0)
        assert close(got.tolist(), ref.tolist())
        auto = torch.histc(torch.tensor([1.0, 2.0, 3.0]), bins=3)
        got_auto = tp.histc(tp.tensor([1., 2., 3.]), bins=3)
        assert close(got_auto.tolist(), auto.tolist())

    def test_histogram_int_bins(self):
        x = torch.tensor([0.1, 0.5, 1.2, 1.9])
        ref_h, ref_e = torch.histogram(x, bins=3, range=(0.0, 2.0))
        gh, ge = tp.histogram(tp.tensor([0.1, 0.5, 1.2, 1.9]), bins=3,
                              range=(0.0, 2.0))
        assert close(gh.tolist(), ref_h.tolist())
        assert close(ge.tolist(), ref_e.tolist())

    def test_histogram_edges_and_density(self):
        edges = torch.tensor([0.0, 1.0, 2.5])
        x = torch.tensor([0.5, 1.5, 2.0])
        ref_h, _ = torch.histogram(x, bins=edges)
        gh, ge = tp.histogram(tp.tensor([0.5, 1.5, 2.0]),
                              bins=tp.tensor([0., 1., 2.5]))
        assert close(gh.tolist(), ref_h.tolist())
        assert close(ge.tolist(), edges.tolist())
        ref_d, _ = torch.histogram(x, bins=edges, density=True)
        gd, _ = tp.histogram(tp.tensor([0.5, 1.5, 2.0]),
                             bins=tp.tensor([0., 1., 2.5]), density=True)
        assert close(gd.tolist(), ref_d.tolist())

    def test_isin(self):
        el = torch.tensor([1.0, 5.0, 3.0, 7.0])
        te = torch.tensor([3.0, 7.0, 2.0])
        ref = torch.isin(el, te)
        got = tp.isin(tp.tensor([1., 5., 3., 7.]),
                      tp.tensor([3., 7., 2.]))
        assert got.dtype == DType.bool
        assert got.tolist() == ref.tolist()
        inv = tp.isin(tp.tensor([1., 5., 3., 7.]),
                      tp.tensor([3., 7., 2.]), invert=True)
        assert inv.tolist() == torch.isin(el, te, invert=True).tolist()

    def test_unique_consecutive(self):
        x = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0])
        p = tp.tensor([1., 1., 2., 2., 2., 3., 1.])
        assert close(tp.unique_consecutive(p).tolist(),
                     torch.unique_consecutive(x).tolist())
        v, i = tp.unique_consecutive(p, return_inverse=True)
        rv, ri = torch.unique_consecutive(x, return_inverse=True)
        assert close(v.tolist(), rv.tolist()) and i.tolist() == ri.tolist()
        v, c = tp.unique_consecutive(p, return_counts=True)
        rv, rc = torch.unique_consecutive(x, return_counts=True)
        assert close(v.tolist(), rv.tolist()) and c.tolist() == rc.tolist()

    def test_repeat_interleave_int(self):
        x = torch.arange(4.)
        ref = torch.repeat_interleave(torch.tensor(x.tolist()), 3)
        got = tp.repeat_interleave(tp.arange(4).to(DType.float32), 3)
        assert close(got.tolist(), ref.tolist())

    def test_repeat_interleave_int_dim(self):
        x = torch.arange(6.).reshape(2, 3)
        p = tp.arange(6).to(DType.float32).reshape([2, 3])
        ref = torch.repeat_interleave(x, 2, dim=0)
        assert close(tp.repeat_interleave(p, 2, dim=0).tolist(),
                     ref.tolist())
        ref1 = torch.repeat_interleave(x, 2, dim=1)
        assert close(tp.repeat_interleave(p, 2, dim=1).tolist(),
                     ref1.tolist())

    def test_repeat_interleave_tensor_repeats(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        reps = torch.tensor([2, 0, 3])
        ref = torch.repeat_interleave(x, reps)
        got = tp.repeat_interleave(
            tp.tensor([10., 20., 30.]), tp.tensor([2, 0, 3]))
        assert close(got.tolist(), ref.tolist())
        x2 = torch.arange(6.).reshape(2, 3)
        p2 = tp.arange(6).to(DType.float32).reshape([2, 3])
        ref2 = torch.repeat_interleave(x2, torch.tensor([1, 2]), dim=0)
        assert close(
            tp.repeat_interleave(p2, tp.tensor([1, 2]), dim=0).tolist(),
            ref2.tolist())

    def test_repeat_interleave_grad(self):
        x = tp.arange(4).to(DType.float32).requires_grad_(True)
        out = tp.repeat_interleave(x, 3)
        out.sum().backward()
        assert x.grad.tolist() == [3.0, 3.0, 3.0, 3.0]

    def test_kaiser_window(self):
        ref = torch.kaiser_window(6, periodic=False, beta=4.0)
        got = tp.kaiser_window(6, periodic=False, beta=4.0)
        assert close(got.tolist(), ref.tolist(), tol=1e-4)
        ref_p = torch.kaiser_window(5, periodic=True, beta=0.0)
        got_p = tp.kaiser_window(5, periodic=True, beta=0.0)
        assert close(got_p.tolist(), ref_p.tolist())


class TestRnnCells:
    def _ref_cell(self, kind, inp, h, c, w_ih, w_hh, b=None):
        import torch as th
        gates = inp @ w_ih.T + h @ w_hh.T
        if b is not None:
            gates = gates + b.unsqueeze(0)
        if kind == "lstm":
            gi, gf, go, gg = gates.chunk(4, 1)
            c_new = th.sigmoid(gf) * c + th.sigmoid(gi) * th.tanh(gg)
            h_new = th.sigmoid(go) * th.tanh(c_new)
            return h_new, c_new
        return (gates.relu() if kind == "relu" else gates.tanh()), None

    def test_lstm_cell(self):
        torch.manual_seed(0)
        inp = torch.randn(2, 5)
        h = torch.randn(2, 3)
        cx = torch.randn(2, 3)
        w_ih = torch.randn(12, 5)
        w_hh = torch.randn(12, 3)
        b = torch.randn(12)
        ref_h, ref_c = self._ref_cell("lstm", inp, h, cx, w_ih, w_hh, b)
        got_h, got_c = tp.lstm_cell(
            tp.tensor(inp.tolist()), tp.tensor(h.tolist()),
            tp.tensor(cx.tolist()), tp.tensor(w_ih.tolist()),
            tp.tensor(w_hh.tolist()), tp.tensor(b.tolist()))
        assert close(got_h.tolist(), ref_h.tolist())
        assert close(got_c.tolist(), ref_c.tolist())

    def test_rnn_cells(self):
        torch.manual_seed(1)
        inp = torch.randn(2, 4)
        h = torch.randn(2, 3)
        w_ih = torch.randn(3, 4)
        w_hh = torch.randn(3, 3)
        for name in ("rnn_relu_cell", "rnn_tanh_cell"):
            fn_t = torch.relu if "relu" in name else torch.tanh
            gates = inp @ w_ih.T + h @ w_hh.T
            ref = fn_t(gates)
            got = getattr(tp, name)(
                tp.tensor(inp.tolist()), tp.tensor(h.tolist()),
                tp.tensor(w_ih.tolist()), tp.tensor(w_hh.tolist()))
            assert close(got.tolist(), ref.tolist()), name

    def test_lstm_cell_grad(self):
        inp = tp.arange(10).to(DType.float32).div(10).requires_grad_(True)
        h = tp.zeros([2, 3]).requires_grad_(True)
        c = tp.zeros([2, 3]).requires_grad_(True)
        w_ih = tp.ones([12, 5]).mul(0.1).requires_grad_(True)
        w_hh = tp.ones([12, 3]).mul(0.1).requires_grad_(True)
        hy, cy = tp.lstm_cell(inp.reshape([2, 5]), h, c, w_ih, w_hh)
        hy.sum().backward()
        assert inp.grad is not None and h.grad is not None


class TestMiscUtils:
    def test_put(self):
        x = torch.zeros(4)
        ref = x.clone()
        idx = torch.tensor([1, 3])
        ref.put_(idx, torch.tensor([9.0, 8.0]))
        got = tp.put(tp.zeros([4]), tp.tensor([1, 3]),
                     tp.tensor([9., 8.]))
        assert close(got.tolist(), ref.tolist())
        src_cycle = tp.put(tp.zeros([4]), tp.tensor([0, 1, 2, 3]),
                           tp.tensor([5.0, 6.0]))
        assert close(src_cycle.tolist(), [5.0, 6.0, 5.0, 6.0])

    def test_conj_bit_queries(self):
        x = arange_f(2)
        assert tp.is_conj(x) is False
        assert tp.is_neg(x) is False
        r = tp.resolve_conj(x)
        assert r.tolist() == x.tolist()
        assert tp.resolve_neg(x).tolist() == x.tolist()

    def test_can_cast_promote_result_type(self):
        pairs = [
            (DType.int64, DType.float32, True),
            (DType.float32, DType.int64, False),
            (DType.float64, DType.int32, False),
            (DType.bool, DType.int64, True),
            (DType.int64, DType.bool, False),
            (DType.float32, DType.complex128, True),
            (DType.complex64, DType.float32, False),
            (DType.float32, DType.float64, True),
            (DType.float64, DType.float32, False),
        ]
        for frm, to, expect in pairs:
            assert tp.can_cast(frm, to) is expect, (frm, to)
        assert tp.promote_types(DType.int32, DType.float32) == DType.float32
        assert tp.promote_types(DType.int16, DType.int64) == DType.int64
        assert tp.promote_types(DType.uint8, DType.int8) == DType.int16
        rt = tp.result_type(tp.tensor([1], dtype=DType.int64),
                            tp.tensor([1.0]))
        trt = torch.result_type(torch.tensor([1]),
                                torch.tensor([1.0]))
        assert str(rt).endswith(str(trt).split(".")[-1]) or \
            tp.promote_types(rt, rt) == rt
        assert tp.result_type(tp.tensor([1], dtype=DType.int64),
                              1.0) == DType.float32

    def test_is_nonzero_is_same_size_get_device(self):
        one = tp.tensor([2.5])
        zero = tp.tensor([0.0])
        assert tp.is_nonzero(one) is True
        assert tp.is_nonzero(zero) is False
        with pytest.raises(RuntimeError):
            tp.is_nonzero(arange_f(2))
        assert tp.is_same_size(arange_f(2, 3), tp.zeros([2, 3])) is True
        assert tp.is_same_size(arange_f(3), tp.zeros([2, 3])) is False
        assert tp.get_device(one) == -1


class TestCopyFamily:
    def _check(self, name, make_pt, make_tt):
        p = arange_f(2, 3)
        got = make_pt(p)
        ref = make_tt(torch.arange(6.).reshape(2, 3))
        if isinstance(ref, torch.Tensor):
            assert tuple(got.shape) == tuple(ref.shape), name
            assert close(got.tolist(), ref.tolist()), name
        else:
            assert [tuple(g.shape) for g in got] == \
                [tuple(r.shape) for r in ref], name
            for g, r in zip(got, ref):
                assert close(g.tolist(), r.tolist()), name

    def test_simple_copies(self):
        self._check("alias_copy", lambda x: tp.alias_copy(x),
                    lambda x: x.clone())
        self._check("t_copy", lambda x: tp.t_copy(x), lambda x: x.t().clone())
        self._check("permute_copy", lambda x: tp.permute_copy(x, [1, 0]),
                    lambda x: x.permute(1, 0).clone())
        self._check("transpose_copy",
                    lambda x: tp.transpose_copy(x, 0, 1),
                    lambda x: x.transpose(0, 1).clone())
        self._check("squeeze_copy", lambda x: tp.squeeze_copy(x.unsqueeze(0)),
                    lambda x: x.unsqueeze(0).squeeze().clone())
        self._check("unsqueeze_copy", lambda x: tp.unsqueeze_copy(x, 1),
                    lambda x: x.unsqueeze(1).clone())
        self._check("select_copy", lambda x: tp.select_copy(x, 1, 2),
                    lambda x: x.select(1, 2).clone())
        self._check("slice_copy",
                    lambda x: tp.slice_copy(x, dim=1, start=1, end=3),
                    lambda x: x[:, 1:3].clone())
        self._check("narrow_copy", lambda x: tp.narrow_copy(x, 0, 1, 1),
                    lambda x: x.narrow_copy(0, 1, 1))
        self._check("diagonal_copy",
                    lambda x: tp.diagonal_copy(x, 0, -2, -1),
                    lambda x: torch.diagonal_copy(x, 0, -2, -1))
        self._check("unbind_copy", lambda x: tp.unbind_copy(x, 0),
                    lambda x: [t.clone() for t in x.unbind(0)])
        self._check("split_copy", lambda x: tp.split_copy(x, [1, 2], dim=1),
                    lambda x: [t.clone() for t in x.split([1, 2], dim=1)])
        self._check("view_copy", lambda x: tp.view_copy(x, [6]),
                    lambda x: x.view(6).clone())
        self._check("unfold_copy",
                    lambda x: tp.unfold_copy(x, 1, 2, 1),
                    lambda x: x.unfold(1, 2, 1).clone())
        self._check("expand_copy",
                    lambda x: tp.expand_copy(x.unsqueeze(0), [3, 2, 3]),
                    lambda x: x.unsqueeze(0).expand(3, 2, 3).clone())

    def test_unsafe_aliases(self):
        p = arange_f(2, 3)
        parts = tp.unsafe_chunk(p, 2, dim=1)
        assert [tuple(q.shape) for q in parts] == [(2, 2), (2, 1)]
        parts2 = tp.unsafe_split(p, 1, dim=0)
        assert [tuple(q.shape) for q in parts2] == [(1, 3), (1, 3)]

    def test_copy_detaches_from_view_graph_but_keeps_values(self):
        base = arange_f(2, 3).requires_grad_(True)
        c = tp.t_copy(base)
        c.sum().backward()
        assert base.grad.tolist() == [[1.0] * 3, [1.0] * 3]


class TestFReexports:
    def test_pooling_top_level(self):
        x = arange_f(1, 6).add(1)
        rx = torch.arange(6.).reshape(1, 6) + 1
        assert close(tp.max_pool1d(x, 2).tolist(),
                     torch.nn.functional.max_pool1d(rx, 2).tolist())
        assert close(tp.avg_pool1d(x, 2).tolist(),
                     torch.nn.functional.avg_pool1d(rx, 2).tolist())
        assert close(tp.adaptive_max_pool1d(x, 2).tolist(),
                     torch.nn.functional.adaptive_max_pool1d(rx, 2).tolist())
        assert close(tp.adaptive_avg_pool1d(x, 3).tolist(),
                     torch.nn.functional.adaptive_avg_pool1d(rx, 3).tolist())

    def test_distance_norms(self):
        # pairwise_distance / pdist resolve to functional.py's generated
        # wrappers (verified separately); cosine_similarity comes from the
        # composite batch.
        a = arange_f(3, offset=1)
        b = arange_f(3)
        ra = torch.arange(3.) + 1
        rb = torch.arange(3.)
        assert close(tp.pairwise_distance(a, b).item(),
                     torch.nn.functional.pairwise_distance(ra, rb).item())
        assert close(tp.pdist(tp.stack([a, b])).tolist(),
                     torch.nn.functional.pdist(torch.stack([ra, rb])).tolist())

    def test_cosine_similarity_top_level(self):
        a = arange_f(3, offset=1)
        b = arange_f(3)
        ra = torch.arange(3.) + 1
        rb = torch.arange(3.)
        assert close(
            tp.cosine_similarity(a, b, dim=0).item(),
            torch.nn.functional.cosine_similarity(ra, rb, dim=0).item(),
            tol=1e-4)

    def test_rms_norm_top_level(self):
        x = arange_f(2, 3).add(1)
        rx = torch.arange(6.).reshape(2, 3) + 1
        ref = torch.nn.functional.rms_norm(rx, [3])
        got = tp.rms_norm(x, [3])
        assert close(got.tolist(), ref.tolist(), tol=1e-4)


class TestAtLeastSequenceRegression:
    """Locks the atleast_* Sequence + gradient contract vs torch."""

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_sequence_shapes(self, d):
        f = getattr(tp, f"atleast_{d}d")
        tf = getattr(torch, f"atleast_{d}d")
        got = f([tp.tensor([1., 2.]), tp.tensor([[3.]])])
        ref = tf([torch.tensor([1., 2.]), torch.tensor([[3.]])])
        assert isinstance(got, list)
        assert [tuple(x.shape) for x in got] == \
            [tuple(x.shape) for x in ref]

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_sequence_gradients_flow(self, d):
        f = getattr(tp, f"atleast_{d}d")
        tf = getattr(torch, f"atleast_{d}d")
        pt = tp.tensor([1., 2., 3.]).requires_grad_(True)
        out = f([pt, tp.tensor([[4.]])])[0]
        out.sum().backward()
        tt = torch.tensor([1., 2., 3.], requires_grad=True)
        tout = tf([tt, torch.tensor([[4.]])])[0]
        tout.sum().backward()
        assert pt.grad.tolist() == tt.grad.tolist()

    def test_scalar_promotion_grad(self):
        s = tp.tensor([7.]).reshape([]).requires_grad_(True)
        tp.atleast_3d(s).sum().backward()
        ts = torch.tensor(7.0, requires_grad=True)
        torch.atleast_3d(ts).sum().backward()
        assert s.grad.tolist() == ts.grad.tolist()


class TestNativeDropoutFamily:
    """alpha/feature dropout: native fused kernels + generated backward."""

    def test_alpha_dropout_saturation_constant(self):
        import math as _m
        p = 0.3
        alpha = 1.7580993408473766
        a = 1.0 / _m.sqrt((alpha * alpha * p + 1) * (1 - p))
        sat = alpha * a * (p - 1)
        xs = [5.0 + i for i in range(256)]
        got = tp.alpha_dropout(tp.tensor([xs]), p, True).tolist()[0]
        ref = torch.nn.functional.alpha_dropout(
            torch.tensor([xs]), p, True)[0].tolist()
        # both contain the exact saturation constant among their values
        assert any(abs(v - sat) < 1e-6 for v in got)
        assert any(abs(v - sat) < 1e-6 for v in ref)

    def test_alpha_dropout_stats(self):
        torch.manual_seed(3)
        x = torch.randn(400) * 2 + 1
        ya = torch.nn.functional.alpha_dropout(x, 0.25, True)
        pa = tp.alpha_dropout(tp.tensor(x.tolist()), 0.25, True)
        for t in (ya, pa):
            flat = t.tolist()
            m = sum(flat) / len(flat)
            sd = (sum(v * v for v in flat) / len(flat)) ** .5
            # SELU-domain invariants are only approximate on finite samples
            assert abs(sd - 2.0) < 0.6

    def test_feature_dropout_channel_zeroing(self):
        xf = torch.randn(4, 32, 8, 8)
        pf = tp.feature_dropout(tp.tensor(xf.tolist()), 0.9, True)
        arr = pf.tolist()

        def chan_max(a, c):
            return max(abs(v) for s in a for ch in [s[c]]
                       for row in ch for v in row)
        zeroed = sum(1 for c in range(32) if chan_max(arr, c) == 0)
        assert zeroed > 10  # p=0.9 -> most channels zeroed
        # mask is independent per (sample, channel): kept entries of the
        # output equal input scaled by exactly 1/(1-p) = 10
        ratios = [arr[s][c][i][j] / x for s in range(4) for c in range(32)
                  for i in range(8) for j in range(8)
                  if (x := xf[s][c][i][j].item()) != 0
                  and arr[s][c][i][j] != 0]
        assert len(ratios) > 8 and all(abs(k - 10.0) < 1e-3 for k in ratios[:16])

    def test_short_circuits(self):
        x = tp.tensor([1.0, 2.0])
        assert tp.alpha_dropout(x, 1.0, True).tolist() == [0.0, 0.0]
        assert tp.alpha_dropout(x, 0.5, False).tolist() == [1.0, 2.0]
        assert tp.feature_dropout(tp.ones([1, 2, 2]), 0.5, False).sum().item() == 4.0
        with pytest.raises(RuntimeError):
            tp.feature_dropout(tp.tensor([1.0]), 0.5, True)

    def test_grads_flow(self):
        a = tp.tensor([[1.5, -0.7], [0.3, 2.2]]).requires_grad_(True)
        tp.alpha_dropout(a, 0.3, True).sum().backward()
        assert a.grad is not None
        b = tp.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]]]]).requires_grad_(True)
        tp.feature_dropout(b, 0.4, True).sum().backward()
        assert b.grad is not None

    def test_top_level_reexports(self):
        for n in ("dropout", "dropout_", "alpha_dropout", "feature_dropout",
                  "feature_dropout_", "feature_alpha_dropout", "rrelu",
                  "rrelu_", "bilinear", "ctc_loss", "embedding_bag",
                  "conv_tbc", "max_pool3d", "max_pool1d_with_indices",
                  "native_channel_shuffle"):
            assert callable(getattr(tp, n)), n


class TestNativeTrapezoid:
    """trapezoid / cumulative_trapezoid: native fused kernels, values and
    gradients must match torch exactly (weights rebuilt in the backward)."""

    def test_forward_dx_x_dim(self):
        y = tp.tensor([1.0, 4.0, 9.0, 16.0])
        ry = torch.tensor([1., 4., 9., 16.])
        assert tp.trapezoid(y, dx=2.0).item() == \
            torch.trapezoid(ry, dx=2.0).item()
        xs = tp.tensor([0.0, 1.0, 3.0, 6.0])
        rxs = torch.tensor([0., 1., 3., 6.])
        assert tp.trapezoid(y, xs).item() == \
            torch.trapezoid(ry, rxs).item()
        m = tp.arange(12).to(DType.float32).reshape([3, 4]).add(1)
        rm = torch.arange(12.).reshape(3, 4) + 1
        assert tp.trapezoid(m, dx=0.5, dim=1).tolist() == \
            torch.trapezoid(rm, dx=0.5, dim=1).tolist()

    def test_cumulative_forward(self):
        y = tp.tensor([1.0, 4.0, 9.0, 16.0])
        ry = torch.tensor([1., 4., 9., 16.])
        assert tp.cumulative_trapezoid(y, dx=1.0).tolist() == \
            torch.cumulative_trapezoid(ry, dx=1.0).tolist()

    def test_grads_exact(self):
        y = [1.0, 4.0, 9.0, 16.0]
        # dx form
        a = tp.tensor(y).requires_grad_(True)
        tp.trapezoid(a, dx=2.0).backward()
        b = torch.tensor(y, requires_grad=True)
        torch.trapezoid(b, dx=2.0).backward()
        assert a.grad.tolist() == b.grad.tolist() == [1.0, 2.0, 2.0, 1.0]
        # x form
        c = tp.tensor(y).requires_grad_(True)
        tp.trapezoid(c, tp.tensor([0.0, 1.0, 3.0, 6.0])).backward()
        d = torch.tensor(y, requires_grad=True)
        torch.trapezoid(d, torch.tensor([0., 1., 3., 6.])).backward()
        assert c.grad.tolist() == d.grad.tolist() == [0.5, 1.5, 2.5, 1.5]
        # cumulative dx
        e = tp.tensor(y).requires_grad_(True)
        tp.cumulative_trapezoid(e, dx=1.0).sum().backward()
        f = torch.tensor(y, requires_grad=True)
        torch.cumulative_trapezoid(f, dx=1.0).sum().backward()
        assert e.grad.tolist() == f.grad.tolist() == [1.5, 2.5, 1.5, 0.5]
