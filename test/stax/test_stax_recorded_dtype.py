"""A compiled value has the element type the program computed it in.

A conversion is laid down in the type that was asked for, two operands of
different types are computed in the type both fit in, and an operation that
changes the type on its own under mixed precision answers in the type the run
recorded.  Each is checked on the result the caller reads and on the values a
compiled forward keeps for its backward.
"""
import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
import tensorplay.compiler._core.aot_autograd as aot_autograd

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def _compiled(fn):
    compiled = tp.compile(fn, backend="stax")

    def run(*args):
        out = compiled(*args)
        # The answer only says something about a region that was built.
        lowering = next(iter(compiled._tensorplay_cache.values()))
        assert lowering._tensorplay_codegen is not None
        return out

    return run


@pytest.mark.parametrize("device", DEVICES)
def test_conversion_answers_in_the_type_asked_for(device):
    x = tp.tensor([[0.5, -1.25, 3.0], [2.0, 0.125, -4.0]], device=device)

    def half(v):
        return v.to(tp.float16)

    def double(v):
        return (v * 2).to(tp.float64) + 1

    got = _compiled(half)(x)
    assert got.dtype == tp.float16
    assert got.float().tolist() == [[0.5, -1.25, 3.0], [2.0, 0.125, -4.0]]
    got = _compiled(double)(x)
    assert got.dtype == tp.float64
    assert got.tolist() == [[2.0, -1.5, 7.0], [5.0, 1.25, -7.0]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("half_first", [True, False])
def test_mixed_operands_answer_in_the_wider_type(device, half_first):
    # Values a half holds exactly, times float values a half cannot hold: the
    # products are only right when the float operand is not narrowed.
    x = tp.tensor([[1.0, 2.0, -0.5], [4.0, 0.25, 8.0]], device=device)
    w = tp.tensor([[1.0001, 3.0003, 5.0005], [7.0007, 9.0009, 11.0011]], device=device)
    expected = [[1.0001, 6.0006, -2.50025], [28.0028, 2.250225, 88.0088]]

    def mul(a, b):
        h = a.to(tp.float16)
        return h * b if half_first else b * h

    def sub(a, b):
        h = a.to(tp.float16)
        return (h - b) if half_first else (b - h)

    def div(a, b):
        h = a.to(tp.float16)
        return (h / b) if half_first else (b / h)

    got = _compiled(mul)(x, w)
    assert got.dtype == tp.float32
    for row, want in zip(got.tolist(), expected):
        assert row == pytest.approx(want, rel=1e-6)
    for fn in (sub, div):
        got = _compiled(fn)(x, w)
        want = fn(x, w)
        assert got.dtype == want.dtype == tp.float32
        for row, ref in zip(got.tolist(), want.tolist()):
            assert row == pytest.approx(ref, rel=1e-6)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.norm = nn.GroupNorm(2, 8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        h = self.c1(x)
        # A peak taken in single precision: its gradient needs the peak itself,
        # so the forward keeps a single-precision value as well as the
        # half-precision ones around it.
        peak = h.float().abs().amax(dim=(2, 3), keepdim=True)
        return self.c2(F.silu(self.norm(h))) / (peak + 1.0)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="mixed precision needs cuda")
def test_saved_values_keep_their_recorded_type(monkeypatch):
    seen = {}
    # The partitioner a training region is split by, so the forward it keeps
    # values for is the one compared below.
    partition = aot_autograd.min_cut_rematerialization_partition
    call = aot_autograd._call

    def remember_forward(joint, inputs, **kwargs):
        result = partition(joint, inputs, **kwargs)
        seen["forward"] = result[0]
        return result

    def compare(compiled, args):
        out = call(compiled, args)
        forward = seen.get("forward")
        if forward is not None and "pairs" not in seen:
            output = forward.graph.find_nodes(op="output")[0]
            nodes = output.args[0] if isinstance(output.args[0], (tuple, list)) else output.args
            if len(nodes) == len(out):
                seen["pairs"] = [
                    (node.name, node.meta["val"].dtype, value.dtype)
                    for node, value in zip(nodes, out)
                    if isinstance(value, tp.Tensor)
                ]
        return out

    monkeypatch.setattr(aot_autograd, "min_cut_rematerialization_partition", remember_forward)
    monkeypatch.setattr(aot_autograd, "_call", compare)

    tp.manual_seed(0)
    net = _Block().cuda()
    x = tp.randn(4, 4, 8, 8, device="cuda")
    compiled = tp.compile(net, backend="stax")
    with tp.autocast("cuda"):
        loss = compiled(x).float().pow(2).mean()
    loss.backward()

    # The comparison only says something about a region that was built.
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen is not None
    assert lowering._tensorplay_backward_codegen is not None

    pairs = seen["pairs"]
    assert len(pairs) > 4
    recorded = {dtype for _, dtype, _ in pairs}
    # The run keeps values of both precisions, so both directions are checked.
    assert tp.float16 in recorded and tp.float32 in recorded
    wrong = [(name, want, got) for name, want, got in pairs if want != got]
    assert wrong == []
