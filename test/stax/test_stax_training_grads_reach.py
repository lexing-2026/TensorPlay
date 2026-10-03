"""Every parameter of a compiled training region receives its gradient.

The backward is traced by running autograd under a dispatch mode, so a call
whose result carried no history there would cut every parameter before it
off from the loss: a convolution followed by a flatten, the common way into
a classifier head or a patch embedding, lost both of its gradients that way.
Each case compares every parameter's gradient with the eager one.
"""

import copy

import pytest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


class _Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = F.relu(self.norm(self.conv(x)))
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))


class _PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 16, 4, 4)
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        return self.norm(self.patch(x).flatten(2).transpose(1, 2))


@pytest.fixture
def exact_float32_convolutions():
    allowed = tp.backends.cudnn.allow_tf32
    tp.backends.cudnn.allow_tf32 = False
    yield
    tp.backends.cudnn.allow_tf32 = allowed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_type", [_Classifier, _PatchEmbedding])
def test_every_parameter_gets_the_eager_gradient(device, model_type, exact_float32_convolutions):
    tp.manual_seed(0)
    model = model_type().to(device)
    ref = copy.deepcopy(model)
    x = tp.randn(2, 3, 8, 8, device=device)
    tp.compile(model)(x).pow(2).mean().backward()
    ref(x).pow(2).mean().backward()
    for (name, p), q in zip(model.named_parameters(), ref.parameters()):
        assert p.grad is not None, name
        # A bias ahead of a batch norm has a zero gradient, so the scale has
        # a floor: what is compared there is rounding noise.
        err = ((p.grad - q.grad).abs().max() / (q.grad.abs().max() + 1e-3)).item()
        assert err < 1e-4, (name, err)
