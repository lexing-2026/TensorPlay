"""Behavior and numerics tests for the learning-rate scheduler package."""

import math
import warnings

import pytest

import tensorplay as tp
import tensorplay.optim as optim
from tensorplay.optim import lr_scheduler as L


def _param(lr=0.1, device="cpu"):
    return tp.Tensor([1.0], device=device, requires_grad=True)


def _optimizer(lr=0.1, momentum=None):
    params = [_param()]
    if momentum is None:
        return optim.SGD(params, lr=lr)
    return optim.SGD(params, lr=lr, momentum=momentum)


def _step_optimizer(opt):
    """Simulate a training step so scheduler/optimizer ordering checks pass."""
    opt.param_groups[0]["params"][0].grad = tp.Tensor([0.1])
    opt.step()


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _lr(opt):
    return float(opt.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# Base class behavior
# ---------------------------------------------------------------------------


def test_initial_state_sets_initial_lr_and_base_lrs():
    opt = _optimizer(lr=0.25)
    sched = L.StepLR(opt, step_size=1, gamma=0.5)
    assert opt.param_groups[0]["initial_lr"] == 0.25
    assert sched.base_lrs == [0.25]
    assert sched.last_epoch == 0
    assert _lr(opt) == 0.25


def test_resume_with_last_epoch_restores_schedule():
    opt = _optimizer(lr=0.25)
    opt.param_groups[0]["initial_lr"] = 0.25
    # Simulate a checkpoint restore: the optimizer already holds the learning
    # rate recorded at epoch 6; the scheduler continues counting from there.
    sched = L.StepLR(opt, step_size=2, gamma=0.5, last_epoch=6)
    assert sched.last_epoch == 7  # the initial step advances the counter
    assert _close(_lr(opt), 0.25)
    _step_optimizer(opt)
    sched.step()  # epoch 8: a decay boundary
    assert _close(_lr(opt), 0.25 * 0.5)


def test_step_before_optimizer_step_warns_once_per_first_step():
    opt = _optimizer()
    sched = L.StepLR(opt, step_size=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sched.step()
    assert any("optimizer.step()" in str(w.message) for w in caught)


def test_step_after_optimizer_step_does_not_warn():
    opt = _optimizer()
    sched = L.StepLR(opt, step_size=1)
    _step_optimizer(opt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sched.step()
        sched.step()
    assert not [w for w in caught if "optimizer.step()" in str(w.message)]


def test_overridden_optimizer_step_warns():
    opt = _optimizer()
    sched = L.StepLR(opt, step_size=1)

    # Rebind after scheduler init to simulate a user override of opt.step;
    # the plain function carries no scheduler tracking marker.
    inner = opt.step
    opt.step = lambda *args, **kwargs: inner(*args, **kwargs)
    with pytest.warns(UserWarning, match="overridden"):
        _step_optimizer(opt)
        sched.step()


def test_get_last_lr_tensor_values_do_not_alias_groups():
    lr0 = tp.Tensor([0.1])
    params = [_param()]
    opt = optim.SGD(params, lr=lr0)
    sched = L.StepLR(opt, step_size=2, gamma=0.5)
    _step_optimizer(opt)
    sched.step()
    _step_optimizer(opt)
    sched.step()
    last = sched.get_last_lr()
    last[0].fill_(999.0)
    assert _close(float(opt.param_groups[0]["lr"]), 0.05)


def test_get_lr_outside_step_warns():
    opt = _optimizer()
    sched = L.StepLR(opt, step_size=1, gamma=0.5)
    with pytest.warns(UserWarning, match="get_last_lr"):
        sched.get_lr()


def test_epoch_argument_uses_closed_form():
    opt = _optimizer()
    sched = L.StepLR(opt, step_size=5, gamma=0.5)
    _step_optimizer(opt)
    with pytest.warns(UserWarning, match="deprecated"):
        sched.step(20)
    assert _close(_lr(opt), 0.1 * 0.5**4)
    assert sched.last_epoch == 20


def test_scheduler_state_dict_roundtrip():
    opt = _optimizer()
    sched = L.StepLR(opt, step_size=3, gamma=0.5)
    _step_optimizer(opt)
    sched.step()
    sched.step()

    opt2 = _optimizer()
    sched2 = L.StepLR(opt2, step_size=3, gamma=0.5)
    sched2.load_state_dict(sched.state_dict())
    assert sched2.last_epoch == sched.last_epoch
    assert sched2._step_count == sched._step_count
    _step_optimizer(opt2)
    sched2.step()
    _step_optimizer(opt)
    sched.step()
    assert _close(_lr(opt), _lr(opt2))


def test_optimizer_without_lr_raises():
    class Fake:
        pass

    with pytest.raises(TypeError):
        L.StepLR(Fake(), step_size=1)


# ---------------------------------------------------------------------------
# Pointwise schedules
# ---------------------------------------------------------------------------


def test_lambda_lr():
    opt = _optimizer(lr=0.1)
    sched = L.LambdaLR(opt, lr_lambda=lambda e: 0.5 ** (e // 5))
    assert _close(_lr(opt), 0.1)  # epoch 0 factor is 1
    for epoch in range(1, 11):
        _step_optimizer(opt)
        sched.step()
        assert _close(_lr(opt), 0.1 * 0.5 ** (epoch // 5))


def test_lambda_lr_multiple_groups():
    params = [{"params": [_param()], "lr": 0.1}, {"params": [_param()], "lr": 0.2}]
    opt = optim.SGD(params, lr=0.1)
    sched = L.LambdaLR(opt, lr_lambda=[lambda e: e + 1, lambda e: 0.1])
    assert _close(float(opt.param_groups[0]["lr"]), 0.1)
    assert _close(float(opt.param_groups[1]["lr"]), 0.02)
    for p in opt.param_groups[0]["params"] + opt.param_groups[1]["params"]:
        p.grad = tp.Tensor([0.1])
    opt.step()
    sched.step()
    assert _close(float(opt.param_groups[0]["lr"]), 0.2)
    assert _close(float(opt.param_groups[1]["lr"]), 0.02)


def test_lambda_lr_group_count_mismatch():
    opt = _optimizer()
    with pytest.raises(ValueError, match="lr_lambdas"):
        L.LambdaLR(opt, lr_lambda=[lambda e: e, lambda e: e])


def test_multiplicative_lr():
    opt = _optimizer(lr=0.1)
    sched = L.MultiplicativeLR(opt, lr_lambda=lambda e: 0.9)
    assert _close(_lr(opt), 0.1)
    expected = 0.1
    for _ in range(5):
        _step_optimizer(opt)
        sched.step()
        expected *= 0.9
        assert _close(_lr(opt), expected)


def test_step_lr():
    opt = _optimizer(lr=0.1)
    sched = L.StepLR(opt, step_size=3, gamma=0.1)
    assert _close(_lr(opt), 0.1)
    for epoch in range(1, 10):
        _step_optimizer(opt)
        sched.step()
        assert _close(_lr(opt), 0.1 * 0.1 ** (epoch // 3))


def test_multi_step_lr():
    opt = _optimizer(lr=1.0)
    sched = L.MultiStepLR(opt, milestones=[2, 4, 7], gamma=0.5)
    expected = 1.0
    for epoch in range(1, 9):
        _step_optimizer(opt)
        sched.step()
        if epoch in (2, 4, 7):
            expected *= 0.5
        assert _close(_lr(opt), expected)


def test_multi_step_lr_milestone_multiplicity():
    opt = _optimizer(lr=1.0)
    sched = L.MultiStepLR(opt, milestones=[2, 2, 5], gamma=0.5)
    for _ in range(2):
        _step_optimizer(opt)
        sched.step()
    assert _close(_lr(opt), 0.25)  # two milestones hit on the same epoch


def test_constant_lr():
    opt = _optimizer(lr=0.1)
    sched = L.ConstantLR(opt, factor=0.5, total_iters=3)
    assert _close(_lr(opt), 0.05)
    for epoch in range(1, 7):
        _step_optimizer(opt)
        sched.step()
        if epoch < 3:
            assert _close(_lr(opt), 0.05)
        else:
            assert _close(_lr(opt), 0.1)


def test_constant_lr_factor_bounds():
    opt = _optimizer()
    with pytest.raises(ValueError):
        L.ConstantLR(opt, factor=1.5)


def test_linear_lr_matches_closed_form():
    opt = _optimizer(lr=0.1)
    sched = L.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=4)
    assert _close(_lr(opt), 0.01)
    for epoch in range(1, 8):
        _step_optimizer(opt)
        sched.step()
        factor = 0.1 + (1.0 - 0.1) * min(4, epoch) / 4
        assert _close(_lr(opt), 0.1 * factor), epoch


def test_linear_lr_factor_bounds():
    opt = _optimizer()
    with pytest.raises(ValueError):
        L.LinearLR(opt, start_factor=0.0)
    with pytest.raises(ValueError):
        L.LinearLR(opt, end_factor=1.5)


def test_exponential_lr():
    opt = _optimizer(lr=0.1)
    sched = L.ExponentialLR(opt, gamma=0.9)
    assert _close(_lr(opt), 0.1)
    for k in range(1, 6):
        _step_optimizer(opt)
        sched.step()
        assert _close(_lr(opt), 0.1 * 0.9**k)


def test_polynomial_lr():
    opt = _optimizer(lr=0.1)
    sched = L.PolynomialLR(opt, total_iters=4, power=2.0)
    assert _close(_lr(opt), 0.1)
    for epoch in range(1, 8):
        _step_optimizer(opt)
        sched.step()
        expected = 0.1 * (1.0 - min(4, epoch) / 4.0) ** 2.0
        assert _close(_lr(opt), expected, tol=1e-7), epoch


# ---------------------------------------------------------------------------
# Cosine family
# ---------------------------------------------------------------------------


def _closed_cosine(base, epoch, t_max, eta_min=0.0):
    return eta_min + (base - eta_min) * (1 + math.cos(math.pi * epoch / t_max)) / 2


def test_cosine_annealing_lr():
    opt = _optimizer(lr=0.1)
    sched = L.CosineAnnealingLR(opt, T_max=10, eta_min=0.01)
    assert _close(_lr(opt), 0.1)
    for epoch in range(1, 11):
        _step_optimizer(opt)
        sched.step()
        assert _close(_lr(opt), _closed_cosine(0.1, epoch, 10, 0.01), tol=1e-6), epoch


def test_cosine_annealing_lr_epoch_argument():
    opt = _optimizer(lr=0.1)
    sched = L.CosineAnnealingLR(opt, T_max=10, eta_min=0.01)
    _step_optimizer(opt)
    with pytest.warns(UserWarning, match="deprecated"):
        sched.step(7)
    assert _close(_lr(opt), _closed_cosine(0.1, 7, 10, 0.01), tol=1e-6)


def test_cosine_annealing_warm_restarts():
    opt = _optimizer(lr=0.1)
    sched = L.CosineAnnealingWarmRestarts(opt, T_0=4, T_mult=1, eta_min=0.02)
    assert _close(_lr(opt), 0.1)
    for epoch in range(1, 10):
        _step_optimizer(opt)
        sched.step()
        t_cur = epoch % 4
        expected = 0.02 + (0.1 - 0.02) * (1 + math.cos(math.pi * t_cur / 4)) / 2
        assert _close(_lr(opt), expected, tol=1e-9), epoch


def test_cosine_annealing_warm_restarts_t_mult_increases_period():
    opt = _optimizer(lr=0.1)
    sched = L.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)
    lrs = []
    for _ in range(9):
        _step_optimizer(opt)
        sched.step()
        lrs.append(_lr(opt))
    # Restart boundaries return to the maximum.
    assert _close(lrs[1], 0.1, tol=1e-9)  # end of first period (T_i == 2)
    assert _close(lrs[5], 0.1, tol=1e-9)  # end of second period (T_i == 4)
    assert sched.T_i == 8


def test_cosine_annealing_warm_restarts_validation():
    opt = _optimizer()
    with pytest.raises(ValueError):
        L.CosineAnnealingWarmRestarts(opt, T_0=0)
    with pytest.raises(ValueError):
        L.CosineAnnealingWarmRestarts(opt, T_0=4, T_mult=0)


# ---------------------------------------------------------------------------
# Composite schedules
# ---------------------------------------------------------------------------


def test_sequential_lr():
    opt = _optimizer(lr=0.1)
    child_a = L.ConstantLR(opt, factor=0.1, total_iters=3)
    child_b = L.ExponentialLR(opt, gamma=0.5)
    sched = L.SequentialLR(opt, [child_a, child_b], milestones=[3])
    assert _close(_lr(opt), 0.01)  # child A warmup at epoch 0
    # Expected curve (child B restarts at the milestone with the base lr):
    # e1/e2: 0.01, e3: 0.1, e4: 0.05, e5: 0.025, e6: 0.0125
    expected = {1: 0.01, 2: 0.01, 3: 0.1, 4: 0.05, 5: 0.025, 6: 0.0125}
    for epoch in range(1, 7):
        _step_optimizer(opt)
        sched.step()
        assert _close(_lr(opt), expected[epoch], tol=1e-7), epoch


def test_sequential_lr_requires_matching_optimizer():
    opt = _optimizer()
    other = _optimizer()
    with pytest.raises(ValueError, match="same optimizer"):
        L.SequentialLR(
            opt,
            [L.StepLR(opt, step_size=1), L.StepLR(other, step_size=1)],
            milestones=[1],
        )


def test_sequential_lr_rejects_plateau():
    opt = _optimizer()
    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        L.SequentialLR(
            opt,
            [L.StepLR(opt, step_size=1), L.ReduceLROnPlateau(opt)],
            milestones=[1],
        )


def test_chained_scheduler():
    opt = _optimizer(lr=0.1)
    sched = L.ChainedScheduler(
        [L.StepLR(opt, step_size=2, gamma=0.5), L.ExponentialLR(opt, gamma=0.9)]
    )
    assert _close(_lr(opt), 0.1)
    expected = 0.1
    for epoch in range(1, 7):
        _step_optimizer(opt)
        sched.step()
        if epoch % 2 == 0:
            expected *= 0.5
        expected *= 0.9
        assert _close(_lr(opt), expected, tol=1e-9), epoch


def test_chained_scheduler_validation():
    opt = _optimizer()
    other = _optimizer()
    with pytest.raises(ValueError, match="same optimizer"):
        L.ChainedScheduler(
            [L.StepLR(opt, step_size=1), L.StepLR(other, step_size=1)]
        )
    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        L.ChainedScheduler([L.StepLR(opt, step_size=1), L.ReduceLROnPlateau(opt)])
    with pytest.raises(ValueError, match="at least one"):
        L.ChainedScheduler([])


# ---------------------------------------------------------------------------
# ReduceLROnPlateau
# ---------------------------------------------------------------------------


def test_plateau_reduces_after_patience():
    opt = _optimizer(lr=0.1)
    sched = L.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    sched.step(1.0)  # baseline
    sched.step(0.9)  # improvement, bad counter reset
    sched.step(0.95)  # bad (1)
    sched.step(0.99)  # bad (2) -> reduce
    assert _close(_lr(opt), 0.05)
    # Cooldown is zero here: another immediate bad pair reduces again.
    sched.step(0.99)
    sched.step(0.99)
    assert _close(_lr(opt), 0.025)


def test_plateau_max_mode_and_abs_threshold():
    opt = _optimizer(lr=0.1)
    sched = L.ReduceLROnPlateau(
        opt, mode="max", factor=0.1, patience=0, threshold=0.1, threshold_mode="abs"
    )
    sched.step(0.5)
    sched.step(0.51)  # not better than best + 0.1 -> immediate reduce
    assert _close(_lr(opt), 0.01)
    assert sched.best == 0.5


def test_plateau_rel_threshold():
    opt = _optimizer(lr=0.1)
    sched = L.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=0, threshold=0.1, threshold_mode="rel"
    )
    sched.step(1.0)
    sched.step(0.95)  # 0.95 > 1.0 * (1 - 0.1) -> not an improvement
    assert _close(_lr(opt), 0.05)


def test_plateau_min_lr_floor():
    opt = _optimizer(lr=0.1)
    sched = L.ReduceLROnPlateau(opt, factor=0.1, patience=0, min_lr=0.05)
    sched.step(1.0)
    for _ in range(3):
        sched.step(1.1)
    assert _close(_lr(opt), 0.05)


def test_plateau_per_group_min_lr_and_bad_group_count():
    params = [{"params": [_param()], "lr": 0.1}, {"params": [_param()], "lr": 0.1}]
    opt = optim.SGD(params, lr=0.1)
    sched = L.ReduceLROnPlateau(opt, patience=0, min_lr=[0.08, 0.0])
    sched.step(1.0)
    sched.step(1.1)
    assert _close(float(opt.param_groups[1]["lr"]), 0.01)
    assert _close(float(opt.param_groups[0]["lr"]), 0.08)

    # Growing the optimizer's groups later raises a clear error when reducing.
    opt.add_param_group({"params": [_param()], "lr": 0.1})
    with pytest.raises(RuntimeError, match="param groups"):
        sched.step(1.2)


def test_plateau_cooldown():
    opt = _optimizer(lr=0.1)
    sched = L.ReduceLROnPlateau(opt, factor=0.5, patience=0, cooldown=2)
    sched.step(1.0)
    sched.step(1.1)  # reduce, then enter cooldown
    assert _close(_lr(opt), 0.05)
    assert sched.in_cooldown
    sched.step(1.1)  # cooldown tick: bad epochs ignored
    sched.step(1.1)  # cooldown ends
    assert not sched.in_cooldown
    assert _close(_lr(opt), 0.05)


def test_plateau_validates_mode_and_factor():
    opt = _optimizer()
    with pytest.raises(ValueError, match="mode"):
        L.ReduceLROnPlateau(opt, mode="sideways")
    with pytest.raises(ValueError, match="threshold mode"):
        L.ReduceLROnPlateau(opt, threshold_mode="sqrt")
    with pytest.raises(ValueError, match="Factor"):
        L.ReduceLROnPlateau(opt, factor=1.0)


def test_plateau_accepts_tensor_metric():
    opt = _optimizer(lr=0.1)
    sched = L.ReduceLROnPlateau(opt, factor=0.5, patience=0)
    sched.step(tp.Tensor([1.0]))
    sched.step(tp.Tensor([2.0]))
    assert _close(_lr(opt), 0.05)


def test_plateau_state_dict_roundtrip():
    opt = _optimizer()
    sched = L.ReduceLROnPlateau(opt, factor=0.5, patience=1, cooldown=1)
    sched.step(1.0)
    state = sched.state_dict()

    opt2 = _optimizer()
    sched2 = L.ReduceLROnPlateau(opt2, factor=0.5, patience=1, cooldown=1)
    sched2.load_state_dict(state)
    assert sched2.best == sched.best
    assert sched2.cooldown_counter == sched.cooldown_counter


# ---------------------------------------------------------------------------
# CyclicLR
# ---------------------------------------------------------------------------


def test_cyclic_lr_triangular_cycle():
    opt = _optimizer(lr=0.1)
    sched = L.CyclicLR(
        opt, base_lr=0.01, max_lr=0.1, step_size_up=4, step_size_down=4
    )
    assert _close(_lr(opt), 0.01)
    total = 8
    for i in range(1, total * 2 + 1):
        _step_optimizer(opt)
        sched.step()
        cycle_pos = i % total
        if cycle_pos <= 4:
            scale = cycle_pos / 4
        else:
            scale = 2 - cycle_pos / 4
        assert _close(_lr(opt), 0.01 + (0.1 - 0.01) * scale, tol=1e-9), i


def test_cyclic_lr_triangular2_scales_amplitude():
    opt = _optimizer(lr=0.1)
    sched = L.CyclicLR(opt, base_lr=0.0, max_lr=0.1, step_size_up=2, mode="triangular2")
    first_peak = None
    for i in range(1, 13):
        _step_optimizer(opt)
        sched.step()
        if i == 2:
            first_peak = _lr(opt)
        if i == 6:  # second cycle peak: amplitude halved once
            assert _close(_lr(opt), 0.0 + 0.05, tol=1e-9)


def test_cyclic_lr_exp_range():
    opt = _optimizer(lr=0.1)
    sched = L.CyclicLR(
        opt, base_lr=0.0, max_lr=0.1, step_size_up=4, mode="exp_range", gamma=0.99
    )
    total = 8
    for i in range(1, 9):
        _step_optimizer(opt)
        sched.step()
        x = i / total - (i // total)
        scale = x / 0.5 if x <= 0.5 else (x - 1) / (0.5 - 1)
        expected = (0.1 * scale) * (0.99**i)
        assert _close(_lr(opt), expected, tol=1e-9), i


def test_cyclic_lr_custom_scale_fn():
    opt = _optimizer(lr=0.1)
    sched = L.CyclicLR(
        opt,
        base_lr=0.0,
        max_lr=0.1,
        step_size_up=4,
        scale_fn=lambda x: 0.5,
        scale_mode="cycle",
    )
    _step_optimizer(opt)
    sched.step()
    # step 1 of 8: triangle position 0.125/0.5 -> 0.25 of the amplitude, times scale 0.5
    assert _close(_lr(opt), 0.0125)


def test_cyclic_lr_cycles_momentum():
    opt = _optimizer(momentum=0.9)
    opt.param_groups[0]["lr"] = 0.01
    sched = L.CyclicLR(
        opt,
        base_lr=0.01,
        max_lr=0.1,
        step_size_up=2,
        cycle_momentum=True,
        base_momentum=0.8,
        max_momentum=0.9,
    )
    assert opt.param_groups[0]["momentum"] == 0.9  # start of cycle: max momentum
    for _ in range(2):
        _step_optimizer(opt)
        sched.step()
    assert _close(opt.param_groups[0]["momentum"], 0.8, tol=1e-9)  # peak lr: min momentum


def test_cyclic_lr_momentum_requires_support():
    params = [_param()]
    opt = optim.Adagrad(params, lr=0.1)  # no momentum/betas in defaults
    with pytest.raises(ValueError, match="momentum"):
        L.CyclicLR(opt, base_lr=0.01, max_lr=0.1, step_size_up=2)


def test_cyclic_lr_invalid_mode():
    opt = _optimizer()
    with pytest.raises(ValueError, match="mode is invalid"):
        L.CyclicLR(opt, base_lr=0.01, max_lr=0.1, mode="zigzag")


# ---------------------------------------------------------------------------
# OneCycleLR
# ---------------------------------------------------------------------------


def test_one_cycle_lr_cosine_two_phase():
    opt = _optimizer(lr=0.1)
    total = 10
    sched = L.OneCycleLR(opt, max_lr=0.1, total_steps=total, pct_start=0.5)
    assert _close(_lr(opt), 0.1 / 25.0)  # initial lr = max/div_factor
    lrs = []
    for _ in range(total - 1):  # the cycle completes at last_epoch = total - 1
        _step_optimizer(opt)
        sched.step()
        lrs.append(_lr(opt))
    # lrs[k] holds the lr of last_epoch k+1: rise to the peak at the phase
    # boundary (last_epoch 4), then fall to the minimum (last_epoch 9).
    assert _close(max(lrs), 0.1, tol=1e-9)
    assert lrs.index(max(lrs)) == 3
    end = 0.1 / 25.0 / 1e4  # min lr = initial/final_div_factor
    assert _close(lrs[-1], end, tol=1e-9)
    assert all(a < b for a, b in zip(lrs[:3], lrs[1:4]))
    assert all(a > b for a, b in zip(lrs[4:], lrs[5:]))
    # One extra step is still permitted (cosine overshoots past the minimum),
    # and stepping beyond the schedule raises.
    _step_optimizer(opt)
    sched.step()
    assert _lr(opt) > end
    with pytest.raises(ValueError, match="total steps"):
        sched.step()


def test_one_cycle_lr_linear_annealing():
    opt = _optimizer(lr=0.1)
    total = 10
    sched = L.OneCycleLR(
        opt, max_lr=0.1, total_steps=total, pct_start=0.5, anneal_strategy="linear"
    )
    initial = 0.1 / 25.0
    assert _close(_lr(opt), initial)
    # First phase climbs linearly: last_epoch 2 -> halfway to the max.
    for _ in range(2):
        _step_optimizer(opt)
        sched.step()
    assert _close(_lr(opt), (0.1 - initial) * 0.5 + initial, tol=1e-9)
    for _ in range(total - 3):
        _step_optimizer(opt)
        sched.step()
    assert _close(_lr(opt), initial / 1e4, tol=1e-9)


def test_one_cycle_lr_three_phase():
    opt = _optimizer(lr=0.1)
    total = 12
    sched = L.OneCycleLR(
        opt, max_lr=0.12, total_steps=total, pct_start=0.5, three_phase=True
    )
    initial = 0.12 / 25.0
    # Phases (in last_epoch units): up to 5 (max), down to 10 (initial),
    # annihilate to 11 (min).
    targets = {5: 0.12, 10: initial, 11: initial / 1e4}
    for epoch in range(1, total):
        _step_optimizer(opt)
        sched.step()
        if epoch in targets:
            assert _close(_lr(opt), targets[epoch], tol=1e-9), epoch


def test_one_cycle_lr_infers_total_steps():
    opt = _optimizer()
    sched = L.OneCycleLR(opt, max_lr=0.1, epochs=2, steps_per_epoch=5)
    assert sched.total_steps == 10
    for _ in range(10):
        _step_optimizer(opt)
        sched.step()
    with pytest.raises(ValueError, match="total steps"):
        sched.step()


def test_one_cycle_lr_requires_step_spec():
    opt = _optimizer()
    with pytest.raises(ValueError, match="total_steps"):
        L.OneCycleLR(opt, max_lr=0.1)
    with pytest.raises(ValueError):
        L.OneCycleLR(opt, max_lr=0.1, total_steps=0)
    with pytest.raises(ValueError):
        L.OneCycleLR(opt, max_lr=0.1, total_steps=10, pct_start=2.0)
    with pytest.raises(ValueError):
        L.OneCycleLR(opt, max_lr=0.1, total_steps=10, anneal_strategy="step")


def test_one_cycle_lr_cycles_momentum():
    opt = _optimizer(momentum=0.9)
    sched = L.OneCycleLR(
        opt, max_lr=0.1, total_steps=4, pct_start=0.5, cycle_momentum=True
    )
    assert opt.param_groups[0]["momentum"] == 0.95  # max_momentum at start
    for epoch in range(1, 4):  # cycle completes at last_epoch = 3
        _step_optimizer(opt)
        sched.step()
        if epoch == 1:  # phase boundary: momentum at its minimum
            assert _close(opt.param_groups[0]["momentum"], 0.85, tol=1e-9)
    assert _close(opt.param_groups[0]["momentum"], 0.95, tol=1e-9)


# ---------------------------------------------------------------------------
# Tensor-valued learning rates
# ---------------------------------------------------------------------------


def test_tensor_valued_lr_is_preserved():
    lr0 = tp.Tensor([0.1])
    params = [_param()]
    opt = optim.SGD(params, lr=lr0)
    sched = L.StepLR(opt, step_size=2, gamma=0.5)
    assert isinstance(sched.base_lrs[0], tp.Tensor)
    for _ in range(2):
        _step_optimizer(opt)
        sched.step()
    lr = opt.param_groups[0]["lr"]
    assert isinstance(lr, tp.Tensor)
    assert _close(float(lr), 0.05)
    assert _close(float(sched.get_last_lr()[0]), 0.05)


# ---------------------------------------------------------------------------
# Integration: optimizer + scheduler stepping order
# ---------------------------------------------------------------------------


def test_scheduler_step_after_optimizer_step_sequence():
    opt = _optimizer(lr=0.1)
    sched = L.ExponentialLR(opt, gamma=0.5)
    for k in range(4):
        _step_optimizer(opt)
        sched.step()
        assert _close(_lr(opt), 0.1 * 0.5 ** (k + 1))
        assert _close(float(sched.get_last_lr()[0]), 0.1 * 0.5 ** (k + 1))
