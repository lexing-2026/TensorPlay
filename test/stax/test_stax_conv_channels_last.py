"""Reduced-precision convolutions in a compiled region run on channels-last operands.

The library computes a half-precision ungrouped convolution channels last
whatever it is handed, and repacks row-major operands itself.  A compiled
region asks for the order instead, so the kernel that writes an operand writes
it that way, and every gradient call of the backward reads its operands in the
same order.  The region reads each result with the strides it planned, so the
library must not pick another order on its own while the region runs: the
lowered-graph scope the region enters has to reach the kernels, which live in
another shared library than the binding that enters it.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(8, 32, 3, padding=1)
        self.norm = nn.GroupNorm(4, 32)
        self.c2 = nn.Conv2d(32, 16, 3, padding=1)

    def forward(self, x):
        return self.c2(F.silu(self.norm(self.c1(x))))


def test_compiled_training_matches_eager_on_channels_last_operands(monkeypatch):
    from tensorplay.compiler.backends.stax.templates.select_algorithm import extern_kernels

    seen = []
    convolution = extern_kernels.convolution

    def recording(x, weight, *args, **kwargs):
        seen.append((x.is_contiguous(memory_format=tp.channels_last),
                     weight.is_contiguous(memory_format=tp.channels_last),
                     tp._C._in_lowered_graph()))
        return convolution(x, weight, *args, **kwargs)

    monkeypatch.setattr(extern_kernels, "convolution", recording)

    tp.manual_seed(0)
    net = _Block().cuda()
    ref = _Block().cuda()
    ref.load_state_dict(net.state_dict())
    x = tp.randn(4, 8, 12, 12, device="cuda")
    compiled = tp.compile(net, strict_native=True)

    outs = []
    for model in (compiled, ref):
        with tp.autocast("cuda", dtype=tp.float16):
            out = model(x)
        out.float().pow(2).mean().backward()
        outs.append(out.float())

    assert seen and all(entry == (True, True, True) for entry in seen), seen
    assert tp.allclose(outs[0], outs[1], atol=2e-2, rtol=2e-2)
    for (name, p), q in zip(net.named_parameters(), ref.parameters()):
        err = ((p.grad - q.grad).abs().max() / (q.grad.abs().max() + 1e-6)).item()
        assert err < 1e-2, (name, err)


def test_lowered_graph_scope_reaches_the_convolution_kernels():
    # With repacking forced, an eager call hands back the channel-major
    # result; inside the scope the same call takes its operands as they lie.
    script = textwrap.dedent(
        """
        import tensorplay as tp
        import tensorplay.nn.functional as F
        x = tp.randn(4, 16, 8, 8, device="cuda").half()
        w = tp.randn(32, 16, 3, 3, device="cuda").half()
        eager = F.conv2d(x, w, padding=1)
        tp._C._enter_lowered_graph_scope()
        try:
            scoped = F.conv2d(x, w, padding=1)
        finally:
            tp._C._exit_lowered_graph_scope()
        print(eager.is_contiguous(memory_format=tp.channels_last), scoped.is_contiguous())
        print(float((eager.float() - scoped.float()).abs().max() / eager.float().abs().max()))
        """
    )
    env = dict(os.environ, TP_CONV_CHANNEL_MAJOR="1")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    layouts, diff = result.stdout.split("\n")[:2]
    assert layouts == "True True"
    assert float(diff) < 1e-2
