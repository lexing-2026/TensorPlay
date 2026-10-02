"""A cpu training region with convolutions is lowered, and lowered correctly.

Convolutions on the cpu ask for channels-last layouts.  The region must
still come out of the lowering as built code for both passes -- a region
that silently falls back to the graph executor would also pass a value
check -- and its gradients must equal the eager ones.
"""
import numpy as np

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.n2 = nn.GroupNorm(4, 16)
        self.proj_a = nn.Linear(8, 16)
        self.proj_b = nn.Linear(8, 16)
        self.down = nn.Conv2d(16, 16, 3, padding=1)
        self.n3 = nn.GroupNorm(4, 32)
        self.merge = nn.Conv2d(32, 16, 3, padding=1)
        self.skip = nn.Conv2d(32, 16, 1)
        self.c3 = nn.Conv2d(16, 3, 1)

    def forward(self, x, emb):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.proj_a(emb)[:, :, None, None]
        h = h + F.silu(self.n2(self.c2(h)))
        h = h + self.proj_b(emb)[:, :, None, None]
        # A down/up path whose result is joined with the skip: the join is
        # read by a norm, a convolution and a 1x1 convolution, and is kept
        # for the backward pass.
        low = self.down(F.avg_pool2d(h, 2))
        up = F.interpolate(low, scale_factor=2.0, mode="nearest")
        joined = tp.cat([up, h], dim=1)
        h = self.merge(F.silu(self.n3(joined))) + self.skip(joined)
        return self.c3(h)


def _grads(net, fn, x, emb, target):
    net.zero_grad()
    loss = F.mse_loss(fn(x, emb), target)
    loss.backward()
    return float(loss), {
        name: p.grad.clone().numpy().astype(np.float64)
        for name, p in net.named_parameters()
    }


def test_cpu_conv_region_is_built_and_matches_eager():
    tp.manual_seed(0)
    net = _Net()
    x = tp.randn(8, 3, 12, 12)
    emb = tp.randn(8, 8)
    target = tp.randn(8, 3, 12, 12)

    want_loss, want = _grads(net, net, x, emb, target)
    compiled = tp.compile(net, backend="stax")
    got_loss, got = _grads(net, compiled, x, emb, target)

    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen is not None
    assert lowering._tensorplay_backward_codegen is not None

    assert abs(got_loss - want_loss) < 1e-6
    for name in want:
        scale = max(np.abs(want[name]).max(), 1e-12)
        assert np.abs(got[name] - want[name]).max() / scale < 1e-4, name
