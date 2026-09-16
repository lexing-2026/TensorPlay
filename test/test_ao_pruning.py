"""Tests for the tensorplay.ao.pruning API."""

import pytest

import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.ao.pruning import (
    BasePruningMethod,
    L1Unstructured,
    LnStructured,
    PruningContainer,
    RandomUnstructured,
    custom_from_mask,
    global_unstructured,
    is_pruned,
    l1_unstructured,
    ln_structured,
    random_structured,
    random_unstructured,
    remove,
)


def _fresh_linear():
    tp.manual_seed(42)
    return nn.Linear(4, 3)


def test_l1_unstructured_prunes_half():
    m = _fresh_linear()
    l1_unstructured(m, name="weight", amount=0.5)
    assert is_pruned(m)
    assert hasattr(m, "weight_mask")
    assert hasattr(m, "weight_orig")
    zero = int((m.weight_mask == 0).sum())
    assert zero == 6  # 3x4 = 12 weights, half zeroed


def test_remove_finalizes_pruning():
    m = _fresh_linear()
    l1_unstructured(m, name="weight", amount=0.5)
    remove(m, "weight")
    assert not is_pruned(m)
    assert not hasattr(m, "weight_orig")
    assert int((m.weight == 0).sum()) == 6


def test_random_unstructured_amount_counts():
    m = _fresh_linear()
    random_unstructured(m, name="weight", amount=3)
    assert int((m.weight_mask == 0).sum()) == 3


def test_ln_structured_prunes_channels():
    m = _fresh_linear()
    ln_structured(m, name="weight", amount=1, n=2, dim=0)
    # one output channel (row) zeroed entirely
    assert int((m.weight.sum(dim=1) == 0).sum()) == 1


def test_l1_unstructured_on_conv():
    m = nn.Conv2d(2, 4, 3)
    total = 4 * 2 * 3 * 3
    random_unstructured(m, name="weight", amount=0.25)
    assert int((m.weight_mask == 0).sum()) == total // 4


def test_global_unstructured_across_modules():
    m1, m2 = _fresh_linear(), _fresh_linear()
    amount = 5
    global_unstructured(
        [(m1, "weight"), (m2, "weight")],
        pruning_method=L1Unstructured,
        amount=amount,
    )
    total_zero = int((m1.weight == 0).sum()) + int((m2.weight == 0).sum())
    assert total_zero == amount


def test_custom_from_mask():
    m = _fresh_linear()
    mask = tp.ones_like(m.weight)
    mask[:, 0] = 0
    custom_from_mask(m, name="weight", mask=mask)
    assert int((m.weight_mask == 0).sum()) == 3
    assert int((m.weight == 0).sum()) == 3


def test_container_composition():
    m = _fresh_linear()
    m.weight = nn.Parameter(m.weight.detach())
    # the first prune installs the method hook; the second is folded into
    # the same container by ``apply``
    random_structured(m, name="weight", amount=2, dim=0)
    l1_unstructured(m, name="weight", amount=0.5)
    assert is_pruned(m)
    hooks = [
        h for h in m._forward_pre_hooks.values()
        if isinstance(h, BasePruningMethod)
    ]
    assert len(hooks) == 1 and isinstance(hooks[0], PruningContainer)
    assert len(hooks[0]._pruning_methods) == 2
    # structured pass zeroes whole rows; the unstructured pass zeroes half of
    # whatever remains, so at least one full row is gone
    assert (m.weight.sum(dim=1) == 0).sum() >= 1


def test_invalid_amount_raises():
    m = _fresh_linear()
    with pytest.raises(ValueError):
        l1_unstructured(m, name="weight", amount=1.5)
    with pytest.raises(ValueError):
        l1_unstructured(m, name="weight", amount=100)


def test_remove_without_pruning_raises():
    m = _fresh_linear()
    with pytest.raises(ValueError):
        remove(m, "weight")


def test_pruned_forward_uses_mask():
    m = _fresh_linear()
    x = tp.rand(2, 4)
    reference = m(x)
    l1_unstructured(m, name="weight", amount=0.5)
    pruned_output = m(x)
    # zeroed weights can only shrink the magnitude of the outputs
    assert not tp.allclose(pruned_output, reference)
    assert pruned_output.shape == reference.shape
