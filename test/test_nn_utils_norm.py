"""Coverage for the nn.utils helpers: gradient clipping, parameter vectors,
weight normalization and spectral normalization."""

import math

import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.utils as U


def _flat(t):
    return tp.reshape(t, [-1]).tolist()


def _grads(module):
    return [p.grad for p in module.parameters() if p.grad is not None]


# ------------------------------------------------------------- clipping


def test_get_total_norm_matches_the_concatenated_norm():
    a = tp.tensor([3.0, 4.0])
    b = tp.tensor([[12.0]])
    total = U.get_total_norm([a, b]).item()
    assert abs(total - math.sqrt(9 + 16 + 144)) < 1e-4


def test_get_total_norm_of_nothing_is_zero():
    assert U.get_total_norm([]).item() == 0.0


def test_get_total_norm_can_reject_nonfinite():
    bad = tp.tensor([float("inf")])
    assert math.isinf(U.get_total_norm([bad]).item())
    with pytest.raises(RuntimeError):
        U.get_total_norm([bad], error_if_nonfinite=True)


def test_clip_grad_norm_scales_down_and_reports_the_original_norm():
    p = tp.tensor([3.0, 4.0], requires_grad=True)
    (p * p).sum().backward()          # grad = 2p = [6, 8], norm 10
    reported = U.clip_grad_norm_([p], 1.0)

    assert abs(reported.item() - 10.0) < 1e-4
    assert abs(U.get_total_norm([p.grad]).item() - 1.0) < 1e-3
    # direction is preserved
    assert abs(p.grad.tolist()[0] / p.grad.tolist()[1] - 6.0 / 8.0) < 1e-5


def test_clip_grad_norm_leaves_small_gradients_alone():
    p = tp.tensor([0.3, 0.4], requires_grad=True)
    (p * p).sum().backward()          # grad = [0.6, 0.8], norm 1.0
    before = p.grad.tolist()
    U.clip_grad_norm_([p], 100.0)
    after = p.grad.tolist()
    for x, y in zip(before, after):
        assert abs(x - y) < 1e-6


def test_clip_grad_norm_accepts_a_single_tensor():
    p = tp.tensor([3.0, 4.0], requires_grad=True)
    (p * p).sum().backward()
    assert abs(U.clip_grad_norm_(p, 1.0).item() - 10.0) < 1e-4


def test_clip_grad_norm_over_a_module():
    lin = nn.Linear(4, 3)
    lin(tp.ones(2, 4)).sum().backward()
    U.clip_grad_norm_(lin.parameters(), 0.5)
    assert U.get_total_norm(_grads(lin)).item() <= 0.5 + 1e-4


def test_clip_grad_value_clamps_every_element():
    p = tp.tensor([3.0, -4.0], requires_grad=True)
    (p * p).sum().backward()          # grad = [6, -8]
    U.clip_grad_value_([p], 1.5)
    assert p.grad.tolist() == [1.5, -1.5]


def test_clip_grad_norm_foreach_path_matches_the_loop_path():
    grads = [tp.tensor([3.0, 4.0]), tp.tensor([[12.0]])]
    ps_a = []
    for g in grads:
        p = tp.nn.Parameter(g.detach().clone())
        p.grad = g.clone()
        ps_a.append(p)
    ps_b = []
    for g in grads:
        p = tp.nn.Parameter(g.detach().clone())
        p.grad = g.clone()
        ps_b.append(p)

    reported_a = U.clip_grad_norm_(ps_a, 1.0)
    reported_b = U.clip_grad_norm_(ps_b, 1.0, foreach=True)
    assert abs(reported_a.item() - reported_b.item()) < 1e-6
    for pa, pb in zip(ps_a, ps_b):
        for a, b in zip(_flat(pa.grad), _flat(pb.grad)):
            assert abs(a - b) < 1e-6


def test_clip_grad_norm_foreach_true_rejects_unsupported_devices():
    p = tp.nn.Parameter(tp.tensor([1.0]))
    p.grad = tp.tensor([1.0])
    real = U.clip_grad._device_has_foreach_support
    U.clip_grad._device_has_foreach_support = lambda device: False
    try:
        with pytest.raises(RuntimeError):
            U.clip_grad_norm_([p], 1.0, foreach=True)
    finally:
        U.clip_grad._device_has_foreach_support = real


def test_clip_grad_norm_deprecated_alias_warns():
    p = tp.tensor([3.0, 4.0], requires_grad=True)
    (p * p).sum().backward()
    with pytest.warns(FutureWarning):
        U.clip_grad_norm([p], 1.0)
    assert U.get_total_norm([p.grad]).item() <= 1.0 + 1e-3


def test_clip_grad_norm_can_reject_nonfinite_before_scaling():
    p = tp.nn.Parameter(tp.tensor([1.0]))
    p.grad = tp.tensor([float("nan")])
    with pytest.raises(RuntimeError):
        U.clip_grad_norm_([p], 1.0, error_if_nonfinite=True)
    # the gradients stay untouched when the error fires
    import math
    assert math.isnan(p.grad.tolist()[0])


def test_clip_grad_value_foreach_path_matches_the_loop_path():
    ps_a = []
    for g in ([3.0, -4.0], [[0.5, 2.0]]):
        p = tp.nn.Parameter(tp.tensor(g))
        p.grad = tp.tensor(g)
        ps_a.append(p)
    ps_b = []
    for g in ([3.0, -4.0], [[0.5, 2.0]]):
        p = tp.nn.Parameter(tp.tensor(g))
        p.grad = tp.tensor(g)
        ps_b.append(p)

    U.clip_grad_value_(ps_a, 1.5)
    U.clip_grad_value_(ps_b, 1.5, foreach=True)
    for pa, pb in zip(ps_a, ps_b):
        assert _flat(pa.grad) == _flat(pb.grad)


# -------------------------------------------------- parameter <-> vector


def test_parameters_to_vector_round_trips():
    lin = nn.Linear(3, 2)
    vec = U.parameters_to_vector(lin.parameters())
    assert vec.shape == (3 * 2 + 2,)

    U.vector_to_parameters(tp.ones_like(vec), lin.parameters())
    for p in lin.parameters():
        assert all(v == 1.0 for v in _flat(p))


def test_vector_to_parameters_rejects_a_non_tensor():
    lin = nn.Linear(2, 2)
    with pytest.raises(TypeError):
        U.vector_to_parameters([1.0], lin.parameters())


# ------------------------------------------------------------ weight norm


def test_weight_norm_splits_the_parameter_and_preserves_the_output():
    lin = nn.Linear(4, 3)
    reference = lin(tp.ones(1, 4)).tolist()

    U.weight_norm(lin)
    assert sorted(lin._parameters.keys()) == ["bias", "weight_g", "weight_v"]

    got = lin(tp.ones(1, 4)).tolist()
    for a, b in zip(_flat(tp.tensor(got)), _flat(tp.tensor(reference))):
        assert abs(a - b) < 1e-4


def test_weight_norm_magnitude_is_the_slice_norm():
    lin = nn.Linear(4, 3)
    weight = lin.weight.clone()
    U.weight_norm(lin)
    expected = tp._C.norm_except_dim(weight, 2, 0)
    for a, b in zip(_flat(lin.weight_g), _flat(expected)):
        assert abs(a - b) < 1e-5


def test_weight_norm_is_trainable_through_both_halves():
    lin = nn.Linear(4, 3)
    U.weight_norm(lin)
    lin(tp.ones(1, 4)).sum().backward()
    assert lin.weight_g.grad is not None
    assert lin.weight_v.grad is not None


def test_remove_weight_norm_restores_the_plain_parameter():
    lin = nn.Linear(4, 3)
    U.weight_norm(lin)
    before = lin(tp.ones(1, 4)).tolist()
    U.remove_weight_norm(lin)

    assert sorted(lin._parameters.keys()) == ["bias", "weight"]
    after = lin(tp.ones(1, 4)).tolist()
    for a, b in zip(_flat(tp.tensor(after)), _flat(tp.tensor(before))):
        assert abs(a - b) < 1e-4


def test_weight_norm_rejects_a_second_hook():
    lin = nn.Linear(4, 3)
    U.weight_norm(lin)
    with pytest.raises(RuntimeError):
        U.weight_norm(lin)


def test_weight_norm_rejects_uninitialized_parameters():
    lin = nn.LazyLinear(3)
    with pytest.raises(ValueError):
        U.weight_norm(lin)


# ---------------------------------------------------------- spectral norm


def test_spectral_norm_registers_the_original_and_the_iterates():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    assert sorted(lin._parameters.keys()) == ["bias", "weight_orig"]
    assert lin.weight_u.shape == (3,)
    assert lin.weight_v.shape == (4,)
    assert lin(tp.ones(2, 4)).shape == (2, 3)


def test_spectral_norm_drives_the_top_singular_value_to_one():
    lin = nn.Linear(6, 5)
    U.spectral_norm(lin)
    lin.train()
    for _ in range(80):
        lin(tp.ones(1, 6))
    top = tp.linalg.svdvals(tp.reshape(lin.weight, [5, -1])).tolist()[0]
    assert abs(top - 1.0) < 5e-2


def test_remove_spectral_norm_restores_the_plain_parameter():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    lin.eval()
    before = lin(tp.ones(1, 4)).tolist()
    U.remove_spectral_norm(lin)
    assert sorted(lin._parameters.keys()) == ["bias", "weight"]
    after = lin(tp.ones(1, 4)).tolist()
    for a, b in zip(_flat(tp.tensor(after)), _flat(tp.tensor(before))):
        assert abs(a - b) < 1e-4


def test_spectral_norm_rejects_a_second_hook():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    with pytest.raises(RuntimeError):
        U.spectral_norm(lin)


def test_spectral_norm_freezes_the_iterates_in_eval_mode():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    lin.train()
    lin(tp.ones(2, 4))
    u_train = lin.weight_u.clone()

    lin.eval()
    u_snap = lin.weight_u.clone()
    lin(tp.ones(2, 4))
    assert lin.weight_u.tolist() == u_snap.tolist()
    # eval still rescales with the stored estimate; switching back resumes
    lin.train()
    lin(tp.ones(2, 4))
    assert lin.weight_u.tolist() != u_train.tolist()


def test_spectral_norm_records_the_layout_version_in_metadata():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    state = lin.state_dict()
    metadata = list(state._metadata.values())[0]
    assert metadata.get("spectral_norm", {}).get("weight.version") == 1


def test_spectral_norm_state_dict_round_trips():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    for _ in range(10):
        lin(tp.ones(2, 4))
    state = lin.state_dict()

    other = nn.Linear(4, 3)
    U.spectral_norm(other)
    other.load_state_dict(state)
    # the iterates and the parameter carry over exactly
    assert other.weight_u.tolist() == lin.weight_u.tolist()
    assert other.weight_v.tolist() == lin.weight_v.tolist()
    assert other.weight_orig.tolist() == lin.weight_orig.tolist()


def test_spectral_norm_rebuilds_v_from_legacy_checkpoints():
    # Pre-versioning checkpoints stored the normalized weight alongside
    # weight_orig and u, but no v; the loader reconstructs v from them.
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    for _ in range(20):
        lin(tp.ones(2, 4))

    state = lin.state_dict()
    state._metadata.clear()
    legacy = {
        "bias": state["bias"],
        "weight": lin.weight.clone(),       # the normalized attribute
        "weight_orig": state["weight_orig"],
        "weight_u": state["weight_u"],
    }

    other = nn.Linear(4, 3)
    U.spectral_norm(other)
    other.load_state_dict(legacy)
    # the rebuilt v reproduces the stored normalized weight under eval
    other.eval()
    expected = lin.eval()(tp.ones(1, 4)).tolist()
    got = other(tp.ones(1, 4)).tolist()
    for a, b in zip(_flat(tp.tensor(got)), _flat(tp.tensor(expected))):
        assert abs(a - b) < 1e-4


def test_remove_spectral_norm_unregisters_every_hook():
    lin = nn.Linear(4, 3)
    U.spectral_norm(lin)
    U.remove_spectral_norm(lin)
    assert not lin._forward_pre_hooks
    assert not lin._state_dict_hooks
    assert not lin._load_state_dict_pre_hooks
    # metadata is no longer written after the removal
    metadata = list(lin.state_dict()._metadata.values())[0]
    assert "spectral_norm" not in metadata


def test_spectral_norm_rejects_uninitialized_parameters():
    lin = nn.LazyLinear(3)
    with pytest.raises(ValueError):
        U.spectral_norm(lin)


def test_spectral_norm_defaults_to_dim_one_for_transposed_convs():
    conv = nn.ConvTranspose2d(4, 3, 3)
    U.spectral_norm(conv)
    conv.train()
    conv(tp.ones(1, 4, 8, 8)).sum().backward()
    # dim=1 folds the output-channel axis, so u tracks the 3 output channels
    assert list(conv.weight_u.shape) == [3]
