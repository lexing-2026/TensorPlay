"""Composable function transforms: values, structure handling and nesting."""
import pytest

import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.func import (
    chunk_vmap,
    debug_unwrap,
    functional_call,
    functionalize,
    grad,
    grad_and_value,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    linearize,
    rearrange,
    replace_all_batch_norm_modules_,
    stack_module_state,
    vjp,
    vmap,
)


def assert_close(got, want, tol=1e-4):
    assert (got - want).abs().max().item() < tol, f"\ngot  {got}\nwant {want}"


class TestGrad:
    def test_matches_analytic_derivative(self):
        x = tp.tensor(1.3)
        assert_close(grad(lambda t: t.sin())(x), x.cos())

    def test_result_is_detached_for_a_plain_input(self):
        g = grad(lambda t: t.sin())(tp.tensor(1.3))
        assert g.grad_fn is None

    def test_nests_to_higher_derivatives(self):
        x = tp.tensor(1.3)
        assert_close(grad(grad(lambda t: t.sin()))(x), -x.sin())
        assert_close(grad(grad(grad(lambda t: t.sin())))(x), -x.cos())

    def test_argnums_tuple(self):
        a, b = tp.ones([3]) * 2, tp.ones([3]) * 5
        ga, gb = grad(lambda p, q: (p * q).sum(), argnums=(0, 1))(a, b)
        assert_close(ga, b)
        assert_close(gb, a)

    def test_negative_argnums(self):
        g = grad(lambda p, q: (p * q).sum(), argnums=-1)(tp.ones([2]), tp.ones([2]) * 4)
        assert_close(g, tp.ones([2]))

    def test_has_aux(self):
        def f(p, q):
            return (p * q).sum(), p + q

        gs, aux = grad(f, argnums=(0, 1), has_aux=True)(tp.ones([3]) * 2, tp.ones([3]) * 5)
        assert_close(aux, tp.ones([3]) * 7)

    def test_grad_and_value_returns_both(self):
        g, value = grad_and_value(lambda t: (t * t).sum())(tp.ones([4]) * 3)
        assert_close(g, tp.ones([4]) * 6)
        assert_close(value, tp.tensor(36.0))

    def test_non_scalar_output_is_rejected(self):
        with pytest.raises(RuntimeError, match="scalar Tensor"):
            grad(lambda t: t * 2)(tp.ones([3]))


class TestVmap:
    def test_maps_over_the_leading_dimension(self):
        a, b = tp.randn([5, 3]), tp.randn([5, 3])
        assert_close(vmap(lambda p, q: (p * q).sum())(a, b), (a * b).sum(dim=1))

    def test_in_dims_none_passes_an_argument_through(self):
        a, b = tp.randn([5, 3]), tp.randn([3])
        got = vmap(lambda p, q: (p * q).sum(), in_dims=(0, None))(a, b)
        assert_close(got, (a * b).sum(dim=1))

    def test_out_dims_places_the_mapped_axis(self):
        a = tp.randn([5, 3])
        assert_close(vmap(lambda t: t * 2, out_dims=1)(a), (a * 2).movedim(0, 1))

    def test_maps_over_a_dict(self):
        a, b = tp.randn([5, 3]), tp.randn([5, 3])
        assert_close(vmap(lambda d: d["p"] * d["q"])({"p": a, "q": b}), a * b)

    def test_chunking_does_not_change_the_result(self):
        a, b = tp.randn([5, 3]), tp.randn([5, 3])
        want = (a * b).sum(dim=1)
        f = lambda p, q: (p * q).sum()
        assert_close(vmap(f, chunk_size=2)(a, b), want)
        assert_close(chunk_vmap(f, chunks=3)(a, b), want)

    def test_composes_with_grad_for_per_sample_gradients(self):
        a = tp.randn([5, 3])
        assert_close(vmap(grad(lambda t: (t ** 3).sum()))(a), 3 * a ** 2)

    def test_mismatched_batch_sizes_are_rejected(self):
        with pytest.raises(ValueError, match="same size"):
            vmap(lambda p, q: p + q)(tp.randn([3, 2]), tp.randn([4, 2]))

    def test_randomness_error_is_the_default(self):
        with pytest.raises(RuntimeError, match="randomness"):
            vmap(lambda t: t + tp.randn([3]))(tp.zeros([4, 3]))

    def test_randomness_different_varies_per_sample(self):
        out = vmap(lambda t: t + tp.randn([3]), randomness="different")(tp.zeros([4, 3]))
        assert not bool((out[0] == out[1]).all().item())

    def test_randomness_same_repeats(self):
        out = vmap(lambda t: t + tp.randn([3]), randomness="same")(tp.zeros([4, 3]))
        assert bool((out[0] == out[1]).all().item())


class TestVjp:
    def test_returns_output_and_a_cotangent_map(self):
        x = tp.randn([4])
        out, vjp_fn = vjp(lambda t: t.sin(), x)
        assert_close(out, x.sin())
        (g,) = vjp_fn(tp.ones([4]))
        assert_close(g, x.cos())

    def test_handles_a_tuple_output(self):
        x = tp.randn([4])
        _, vjp_fn = vjp(lambda t: (t.sin(), t.cos()), x)
        (g,) = vjp_fn((tp.ones([4]), tp.zeros([4])))
        assert_close(g, x.cos())

    def test_cotangent_structure_must_match_the_output(self):
        _, vjp_fn = vjp(lambda t: (t.sin(), t.cos()), tp.randn([4]))
        with pytest.raises(RuntimeError, match="structure"):
            vjp_fn(tp.ones([4]))


class TestJvp:
    def test_directional_derivative(self):
        x, v = tp.randn([4]), tp.randn([4])
        out, tangent = jvp(lambda t: t.sin(), (x,), (v,))
        assert_close(out, x.sin())
        assert_close(tangent, x.cos() * v)

    def test_two_inputs(self):
        x, y = tp.randn([4]), tp.randn([4])
        _, tangent = jvp(lambda a, b: a * b, (x, y), (tp.ones([4]), tp.ones([4]) * 2))
        assert_close(tangent, y + 2 * x)

    def test_carries_aux_through(self):
        _, _, aux = jvp(lambda t: (t.sin(), "meta"), (tp.randn([4]),), (tp.ones([4]),),
                        has_aux=True)
        assert aux == "meta"

    def test_primals_must_be_a_tuple(self):
        with pytest.raises(RuntimeError, match="tuple"):
            jvp(lambda t: t.sin(), tp.randn([4]), (tp.ones([4]),))


class TestJacobians:
    def test_both_modes_agree_on_a_square_jacobian(self):
        x = tp.randn([4])
        want = tp.diag(x.cos())
        assert_close(jacrev(tp.sin)(x), want)
        assert_close(jacfwd(tp.sin)(x), want)

    def test_both_modes_agree_on_a_rectangular_jacobian(self):
        x, m = tp.randn([4]), tp.randn([3, 4])
        f = lambda t: (m @ t).sin()
        rev, fwd = jacrev(f)(x), jacfwd(f)(x)
        assert list(rev.shape) == [3, 4]
        assert_close(rev, fwd)

    def test_linear_map_jacobian_is_the_matrix(self):
        x, m = tp.randn([4]), tp.randn([3, 4])
        assert_close(jacrev(lambda t: m @ t)(x), m)
        assert_close(jacfwd(lambda t: m @ t)(x), m)

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, None])
    def test_chunking_does_not_change_the_jacobian(self, chunk_size):
        x = tp.randn([4])
        assert_close(jacrev(tp.sin, chunk_size=chunk_size)(x), tp.diag(x.cos()))

    def test_argnums_tuple_gives_one_jacobian_per_input(self):
        x, y = tp.randn([4]), tp.randn([4])
        for jac in (jacrev, jacfwd):
            first, second = jac(lambda a, b: a * b, argnums=(0, 1))(x, y)
            assert_close(first, tp.diag(y))
            assert_close(second, tp.diag(x))

    def test_tuple_output_gives_one_jacobian_per_output(self):
        x = tp.randn([4])
        first, second = jacrev(lambda t: (t.sin(), t * 2))(x)
        assert_close(first, tp.diag(x.cos()))
        assert_close(second, tp.diag(tp.ones([4]) * 2))


class TestHessian:
    def test_diagonal_hessian(self):
        x = tp.randn([4])
        assert_close(hessian(lambda t: (t ** 3).sum())(x), tp.diag(6 * x))

    def test_quadratic_form(self):
        x = tp.randn([4])
        q = tp.randn([4, 4])
        q = q + q.t()
        assert_close(hessian(lambda t: t @ q @ t)(x), 2 * q, tol=1e-3)


class TestLinearize:
    def test_reuses_the_linearization_for_many_tangents(self):
        x, v = tp.randn([4]), tp.randn([4])
        out, jvp_fn = linearize(tp.sin, x)
        assert_close(out, x.sin())
        assert_close(jvp_fn(v), x.cos() * v)
        assert_close(jvp_fn(v * 3), x.cos() * v * 3)

    def test_tangent_shape_is_checked(self):
        _, jvp_fn = linearize(tp.sin, tp.randn([4]))
        with pytest.raises(RuntimeError, match="shape"):
            jvp_fn(tp.ones([3]))


class TestFunctionalize:
    def test_leaves_the_caller_tensor_untouched(self):
        def mutating(t):
            t.add_(1)
            return t * 2

        x = tp.zeros([3])
        out = functionalize(mutating)(x)
        assert_close(x, tp.zeros([3]))
        assert_close(out, tp.ones([3]) * 2)

    def test_removing_views_as_well(self):
        def mutating(t):
            t.add_(1)
            return t * 2

        x = tp.zeros([3])
        functionalize(mutating, remove="mutations_and_views")(x)
        assert_close(x, tp.zeros([3]))

    def test_unknown_remove_mode_is_rejected(self):
        with pytest.raises(RuntimeError, match="remove"):
            functionalize(lambda t: t, remove="everything")

    def test_debug_unwrap_is_the_identity(self):
        x = tp.zeros([3])
        assert debug_unwrap(x) is x


class TestFunctionalCall:
    def test_matches_the_module_called_normally(self):
        linear = nn.Linear(3, 2)
        params = {k: v.detach() for k, v in linear.named_parameters()}
        x = tp.randn([5, 3])
        assert_close(functional_call(linear, params, (x,)), linear(x))

    def test_accepts_several_state_dicts(self):
        linear = nn.Linear(3, 2)
        params = {k: v.detach() for k, v in linear.named_parameters()}
        x = tp.randn([5, 3])
        got = functional_call(
            linear, [{"weight": params["weight"]}, {"bias": params["bias"]}], (x,)
        )
        assert_close(got, linear(x))

    def test_overlapping_state_dicts_are_rejected(self):
        linear = nn.Linear(3, 2)
        weight = {"weight": linear.weight.detach()}
        with pytest.raises(ValueError, match="ambiguous"):
            functional_call(linear, [weight, weight], (tp.randn([5, 3]),))

    def test_module_state_is_restored_afterwards(self):
        linear = nn.Linear(3, 2)
        before = linear.weight.clone()
        functional_call(linear, {"weight": tp.zeros([2, 3])}, (tp.randn([5, 3]),))
        assert_close(linear.weight, before)


class TestStackModuleState:
    def test_evaluates_an_ensemble_under_vmap(self):
        models = [nn.Linear(3, 2) for _ in range(4)]
        params, buffers = stack_module_state(models)
        assert list(params["weight"].shape) == [4, 2, 3]

        base = nn.Linear(3, 2)
        x = tp.randn([4, 5, 3])
        got = vmap(lambda p, b, inp: functional_call(base, (p, b), (inp,)))(
            params, buffers, x
        )
        assert_close(got, tp.stack([models[i](x[i]) for i in range(4)]))

    def test_mixed_classes_are_rejected(self):
        with pytest.raises(RuntimeError, match="same class"):
            stack_module_state([nn.Linear(3, 2), nn.Identity()])

    def test_mixed_training_modes_are_rejected(self):
        first, second = nn.Linear(3, 2), nn.Linear(3, 2)
        second.eval()
        with pytest.raises(RuntimeError, match="training/eval"):
            stack_module_state([first, second])


class TestBatchNormReplacement:
    def test_drops_running_statistics_throughout(self):
        net = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        replace_all_batch_norm_modules_(net)
        assert net[1].running_mean is None
        assert net[1].running_var is None
        assert net[1].track_running_stats is False


class TestRearrange:
    def test_permutes_named_axes(self):
        x = tp.randn([2, 3, 4])
        assert_close(rearrange(x, "b h w -> b w h"), x.permute((0, 2, 1)))

    def test_merges_axes(self):
        x = tp.randn([2, 3, 4])
        assert list(rearrange(x, "b h w -> b (h w)").shape) == [2, 12]

    def test_splits_an_axis_with_a_given_length(self):
        x = tp.randn([2, 3, 4])
        assert list(rearrange(x, "(b1 b2) h w -> b1 b2 h w", b1=1).shape) == [1, 2, 3, 4]

    def test_ellipsis_passes_axes_through(self):
        x = tp.randn([2, 3, 4])
        assert list(rearrange(x, "b ... -> ... b").shape) == [3, 4, 2]

    def test_adds_a_singleton_axis(self):
        x = tp.randn([2, 3, 4])
        assert list(rearrange(x, "b h w -> b h w 1").shape) == [2, 3, 4, 1]

    def test_stacks_a_list_of_tensors(self):
        x = tp.randn([2, 3, 4])
        assert list(rearrange([x, x], "n b h w -> b n h w").shape) == [2, 2, 3, 4]

    def test_round_trips_through_a_merge(self):
        x = tp.randn([2, 3, 4])
        merged = rearrange(x, "b h w -> b (h w)")
        assert_close(rearrange(merged, "b (h w) -> b h w", h=3), x)

    def test_dropping_an_axis_is_rejected(self):
        with pytest.raises(ValueError):
            rearrange(tp.randn([2, 3]), "b h -> b")

    def test_missing_arrow_is_rejected(self):
        with pytest.raises(ValueError, match="->"):
            rearrange(tp.randn([2, 3]), "b h")


class TestSecondOrderThroughViewAndJoinOps:
    """The join and view backwards must stay in the graph, or every
    forward-over-reverse transform silently returns zeros."""

    @pytest.mark.parametrize(
        "name,f,diagonal",
        [
            ("cat", lambda x: tp.cat([x * x], dim=0).sum(), [2.0, 2.0]),
            ("stack", lambda x: tp.stack([x * x]).sum(), [2.0, 2.0]),
            ("slice", lambda x: (x * x)[0:2].sum(), [2.0, 2.0, 0.0]),
            ("select", lambda x: (x * x)[0], [2.0, 0.0, 0.0]),
            ("narrow", lambda x: (x * x).narrow(0, 0, 2).sum(), [2.0, 2.0, 0.0]),
            ("expand", lambda x: (x * x)[0:1].expand([2]).sum(), [4.0, 0.0, 0.0]),
        ],
    )
    def test_hessian_is_not_silently_zero(self, name, f, diagonal):
        x = tp.ones([len(diagonal)])
        assert_close(hessian(f)(x), tp.diag(tp.tensor(diagonal)))

    def test_expand_backward_is_recorded(self):
        x = tp.ones([2]).requires_grad_(True)
        y = x[0:1].expand([2])
        cotangent = tp.ones([2]).requires_grad_(True)
        (g,) = tp.autograd.grad(y, x, cotangent, create_graph=True)
        assert g.grad_fn is not None
