"""A training batch norm is generated, not handed to the framework.

The batch's mean and variance come from one pass, the normalized value is
written by a generated kernel, and the running statistics are moved toward the
batch's in place.  Each case checks the result, the gradients and the running
statistics against eager, and that the call was not left to the framework.
"""

import copy

import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.compiler.backends.stax.op_lowerings as lowerings

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(8, 6, 5, 7), (4, 16, 9)])
def test_training_batch_norm_matches_eager(device, shape, monkeypatch):
    handed_over = []
    fallback = lowerings._fallback_batch_norm

    def counting(*args, **kwargs):
        handed_over.append(args)
        return fallback(*args, **kwargs)

    monkeypatch.setattr(lowerings, "_fallback_batch_norm", counting)
    tp.manual_seed(0)
    norm = (nn.BatchNorm2d if len(shape) == 4 else nn.BatchNorm1d)(shape[1]).to(device)
    nn.init.uniform_(norm.weight, 0.5, 1.5)
    nn.init.uniform_(norm.bias, -1.0, 1.0)
    ref = copy.deepcopy(norm)
    x = (tp.randn(*shape, device=device) * 2 + 0.5).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad = tp.randn(*shape, device=device)

    out = tp.compile(norm, strict_native=True)(x)
    want = ref(x_ref)
    got = tp.autograd.grad(out, (x, norm.weight, norm.bias), grad)
    expected = tp.autograd.grad(want, (x_ref, ref.weight, ref.bias), grad)

    assert not handed_over
    assert tp.allclose(out, want, atol=1e-5, rtol=1e-5)
    for a, b in zip(got, expected):
        assert tp.allclose(a, b, atol=1e-4, rtol=1e-4)
    assert tp.allclose(norm.running_mean, ref.running_mean, atol=1e-6)
    assert tp.allclose(norm.running_var, ref.running_var, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("momentum", [0.1, None])
def test_training_batch_norm_counts_its_batches(device, momentum):
    # The count of batches seen moves with every compiled training step --
    # with no momentum the running statistics are averaged by it, and one
    # compiled region serves every count -- and a region calling a model it
    # closes over updates that model once per call.
    tp.manual_seed(0)
    norm = nn.BatchNorm1d(4, momentum=momentum).to(device)
    ref = copy.deepcopy(norm)
    compiled = [tp.compile(norm, strict_native=True)]
    if momentum is not None:
        closing = nn.BatchNorm1d(4).to(device)
        closing_ref = copy.deepcopy(closing)
        compiled.append(tp.compile(lambda x: closing(x), strict_native=True))
    for _ in range(3):
        x = tp.randn(8, 4, device=device, requires_grad=True)
        compiled[0](x).sum().backward()
        ref(x).sum().backward()
        if momentum is not None:
            compiled[1](x).sum().backward()
            closing_ref(x).sum().backward()
    assert len(compiled[0]._tensorplay_cache) == 1
    pairs = [(norm, ref)] + ([(closing, closing_ref)] if momentum is not None else [])
    for mine, theirs in pairs:
        assert mine.num_batches_tracked.item() == theirs.num_batches_tracked.item() == 3
        for name in ("running_mean", "running_var"):
            err = (getattr(mine, name) - getattr(theirs, name)).abs().max().item()
            assert err <= 1e-5
