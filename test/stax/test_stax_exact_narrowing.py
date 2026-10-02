"""A value widened and then only moved is moved in the type it came from.

Picking nearest neighbours and laying tensors end to end move values without
changing them, so a compiled training region does both in the narrow type the
operand was widened from and widens the result: the same numbers, half the
memory.  Each case checks the numbers and the gradient against sums worked out
by hand, and checks what the recorded forward keeps in which type.
"""
import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

# Every value is one a half-precision number holds exactly, so narrowing loses
# nothing and the answers are exact.
SOURCE = [[0.5, -1.25], [2.0, 0.125]]
SKIP = [
    [0.25, 0.5, -0.75, 1.0],
    [1.5, -2.0, 0.0, 3.25],
    [-0.5, 4.0, 2.75, -1.5],
    [0.75, -3.0, 1.25, 2.5],
]


def _repeated(rows):
    return [[rows[i // 2][j // 2] for j in range(4)] for i in range(4)]


def _compiled(fn, *args):
    compiled = tp.compile(fn, backend="stax")
    out = compiled(*args)
    # The answers only say something about a region that was built.
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen is not None
    return out, lowering


def _forward_types(lowering, prefix):
    forward = lowering._tensorplay_aot_graphs[1]
    return [
        node.meta["val"].dtype
        for node in forward.graph.nodes
        if node.op == "call_function" and node.name.startswith(prefix)
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_moving_is_done_in_the_narrow_type(device):
    x = tp.tensor([[SOURCE]], device=device, requires_grad=True)
    skip = tp.tensor([[SKIP]], device=device).half()

    def region(v, s):
        repeated = F.interpolate(v.half().float(), scale_factor=2, mode="nearest")
        joined = tp.cat([repeated, s], dim=1)
        return joined * 1.5, joined.half() * 2

    (wide, narrow), lowering = _compiled(region, x, skip)
    assert wide.dtype == tp.float32 and narrow.dtype == tp.float16
    expected = [_repeated(SOURCE), SKIP]
    assert wide.detach().cpu().tolist() == [
        [[[1.5 * v for v in row] for row in plane] for plane in expected]
    ]
    assert narrow.detach().cpu().tolist() == [
        [[[2.0 * v for v in row] for row in plane] for plane in expected]
    ]

    (wide.sum() + narrow.float().sum()).backward()
    # Each element is read four times, once at 1.5 and once at 2.
    assert x.grad.cpu().tolist() == [[[[14.0, 14.0], [14.0, 14.0]]]]
    assert lowering._tensorplay_backward_codegen is not None

    # What is laid end to end is the narrow values; the wide tensor the program
    # asked for is a conversion of that.
    assert _forward_types(lowering, "cat") == [tp.float16]
    assert _forward_types(lowering, "upsample_nearest2d") == []


@pytest.mark.parametrize("device", DEVICES)
def test_a_source_written_to_since_is_not_read_again(device):
    x = tp.tensor([[SOURCE]], device=device, requires_grad=True)

    def region(v):
        narrow = v.half()
        wide = narrow.float()
        narrow.mul_(2)
        return F.interpolate(wide, scale_factor=2, mode="nearest") * 1.5, narrow * 1.0

    (repeated, doubled), lowering = _compiled(region, x)
    assert repeated.detach().cpu().tolist() == [
        [[[1.5 * v for v in row] for row in _repeated(SOURCE)]]
    ]
    assert doubled.detach().cpu().tolist() == [
        [[[2.0 * v for v in row] for row in SOURCE]]
    ]
    # The narrow tensor no longer holds what was widened, so the wide one is
    # what gets moved.
    assert _forward_types(lowering, "upsample_nearest2d") == [tp.float32]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="mixed precision on the device")
def test_mixed_precision_keeps_the_joined_value_narrow():
    tp.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3, padding=1)
            self.norm = nn.GroupNorm(4, 12)
            self.out = nn.Conv2d(12, 4, 3, padding=1)
            self.side = nn.Conv2d(12, 4, 1)

        def forward(self, x, skip):
            h = F.interpolate(self.conv(x), scale_factor=2, mode="nearest")
            h = tp.cat([h, skip], dim=1)
            return self.out(F.silu(self.norm(h))) + self.side(h)

    net = Net().cuda()
    x = tp.randn(2, 4, 6, 6, device="cuda")
    skip = tp.randn(2, 4, 12, 12, device="cuda").half()
    with tp.autocast("cuda"):
        reference = net(x, skip)
        out, lowering = _compiled(net, x, skip)
    assert out.dtype == reference.dtype
    assert float((out.float() - reference.float()).abs().max()) < 2e-2

    # The policy widens before the neighbours are picked; the region joins the
    # narrow values and keeps that, not the wide copy, for its backward.
    assert _forward_types(lowering, "cat") == [tp.float16]
    forward = lowering._tensorplay_aot_graphs[1]
    kept = forward.graph.find_nodes(op="output")[0].args[0]
    wide_kept = [
        node.name for node in kept
        if hasattr(node, "meta")
        and getattr(node.meta.get("val"), "dtype", None) == tp.float32
        and tuple(node.meta["val"].shape) == (2, 12, 12, 12)
    ]
    assert wide_kept == []
