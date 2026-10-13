"""Tests for the lumen attribution package.

The numerical checks use linear models, where every attribution method has a
closed-form answer, plus end-to-end smoke tests on a small MLP.
"""

import pytest
import tensorplay as tensorplay
from tensorplay import nn

from tensorplay.lumen import (
    FeatureAblation,
    InputXGradient,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from tensorplay.lumen._utils.graph import graph_gradient_func, node_names, trace


class LinearModel(nn.Module):
    """y = W x with a known weight matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        with tensorplay.no_grad():
            self.linear.weight.copy_(
                tensorplay.tensor(
                    [
                        [1.0, 0.0, 2.0, -1.0],
                        [0.5, 1.0, 0.0, 2.0],
                        [-1.0, 3.0, 1.0, 0.5],
                    ]
                )
            )

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        x = tensorplay.relu(self.fc1(x))
        return self.fc2(x)


def _target_gradient_rows(model: LinearModel, target: int) -> tensorplay.Tensor:
    with tensorplay.no_grad():
        return model.linear.weight[target].clone()


def test_input_x_gradient_linear() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 3.0, 1.0]])
    attr = InputXGradient(model).attribute(x, target=1)
    w = _target_gradient_rows(model, 1)  # [0.5, 1.0, 0.0, 2.0]
    expected = x * w
    assert attr.shape == x.shape
    assert tensorplay.allclose(attr, expected, atol=1e-5)


def test_saliency_linear() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5]])
    attr = Saliency(model).attribute(x, target=0)
    w = _target_gradient_rows(model, 0)  # [1.0, 0.0, 2.0, -1.0]
    assert tensorplay.allclose(attr, tensorplay.abs(w), atol=1e-5)

    signed = Saliency(model).attribute(x, target=0, abs=False)
    assert tensorplay.allclose(signed, w, atol=1e-5)


def test_integrated_gradients_linear() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5]])
    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(
        x, target=2, baselines=0.0, n_steps=100, return_convergence_delta=True
    )
    w = _target_gradient_rows(model, 2)  # [-1.0, 3.0, 1.0, 0.5]
    expected = x * w
    assert tensorplay.allclose(attr, expected, atol=1e-4)
    assert tensorplay.allclose(delta, tensorplay.zeros_like(delta), atol=1e-4)

    # completeness: sum(attr) == f(x) - f(baseline) for the target class
    with tensorplay.no_grad():
        out = model(x)
    assert tensorplay.allclose(attr.sum(dim=1), out[:, 2], atol=1e-4)


def test_integrated_gradients_multiple_inputs() -> None:
    class TwoInputLinearModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 3, bias=False)
            with tensorplay.no_grad():
                self.linear.weight.copy_(
                    tensorplay.arange(24, dtype=tensorplay.float32).reshape(3, 8) / 10
                )

        def forward(
            self, x1: tensorplay.Tensor, x2: tensorplay.Tensor
        ) -> tensorplay.Tensor:
            return self.linear(tensorplay.cat([x1, x2], dim=1))

    model = TwoInputLinearModel()
    x1 = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5]])
    x2 = tensorplay.tensor([[0.5, 1.5, 2.0, -2.0]])
    ig = IntegratedGradients(model)
    attrs = ig.attribute((x1, x2), target=0, n_steps=50)
    assert isinstance(attrs, tuple)
    assert len(attrs) == 2
    for a, x in zip(attrs, (x1, x2)):
        assert a.shape == x.shape
    # completeness holds per example across both inputs
    with tensorplay.no_grad():
        out = model(x1, x2)
    assert tensorplay.allclose(
        attrs[0].sum() + attrs[1].sum(), out[0, 0], atol=1e-3
    )


def test_feature_ablation_linear() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5]])
    attr = FeatureAblation(model).attribute(x, target=1, baselines=0.0)
    w = _target_gradient_rows(model, 1)
    expected = x * w
    assert tensorplay.allclose(attr, expected, atol=1e-5)


def test_feature_ablation_feature_mask() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5]])
    # group features (0, 1) and (2, 3)
    mask = tensorplay.tensor([[0, 0, 1, 1]])
    attr = FeatureAblation(model).attribute(x, target=1, feature_mask=mask)
    w = _target_gradient_rows(model, 1)
    # ablating a group removes all its features at once, so the group's
    # attribution is the summed feature contribution and is shared by every
    # scalar in the group.
    expected_group0 = x[0, 0] * w[0] + x[0, 1] * w[1]
    expected_group1 = x[0, 2] * w[2] + x[0, 3] * w[3]
    assert tensorplay.allclose(attr[:, 0], attr[:, 1], atol=1e-5)
    assert tensorplay.allclose(attr[:, 2], attr[:, 3], atol=1e-5)
    assert tensorplay.allclose(attr[:, 0], expected_group0, atol=1e-5)
    assert tensorplay.allclose(attr[:, 2], expected_group1, atol=1e-5)


def test_occlusion_linear() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5]])
    attr = Occlusion(model).attribute(
        x, target=1, sliding_window_shapes=(2,), strides=1, baselines=0.0
    )
    w = _target_gradient_rows(model, 1)
    # windows of width 2 with stride 1: [0,1], [1,2], [2,3].  A feature lying
    # in multiple windows gets the average of the window output differences.
    win01 = x[0, 0] * w[0] + x[0, 1] * w[1]
    win12 = x[0, 1] * w[1] + x[0, 2] * w[2]
    win23 = x[0, 2] * w[2] + x[0, 3] * w[3]
    expected = tensorplay.stack(
        [win01, (win01 + win12) / 2, (win12 + win23) / 2, win23]
    ).unsqueeze(0)
    assert tensorplay.allclose(attr, expected, atol=1e-5)


def test_noise_tunnel_mlp() -> None:
    model = MLP()
    x = tensorplay.randn(3, 4)
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    for nt_type in ("smoothgrad", "smoothgrad_sq", "vargrad"):
        attr = nt.attribute(x, target=1, nt_samples=3, nt_type=nt_type)
        assert attr.shape == x.shape


def test_noise_tunnel_convergence_delta() -> None:
    model = LinearModel()
    x = tensorplay.tensor([[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 1.0]])
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr, delta = nt.attribute(
        x,
        target=1,
        nt_samples=2,
        nt_type="smoothgrad",
        baselines=0.0,
        return_convergence_delta=True,
    )
    assert attr.shape == x.shape
    assert delta.shape[0] == x.shape[0] * 2


def test_graph_trace_and_gradient_func() -> None:
    model = MLP()
    traced = trace(model)
    assert traced is not None
    assert callable(traced)

    names = node_names(model)
    assert "fc1" in names
    assert "fc2" in names

    x = tensorplay.randn(2, 4, requires_grad=True)
    grads = graph_gradient_func(model, (x,), target_ind=1)
    assert len(grads) == 1
    assert grads[0].shape == x.shape

    # gradient computed through the traced graph matches eager gradient
    eager = Saliency(model).attribute(x, target=1, abs=False)
    assert tensorplay.allclose(grads[0], eager, atol=1e-5)


def test_convergence_delta_reported() -> None:
    ig = IntegratedGradients(LinearModel())
    assert ig.has_convergence_delta()
    sal = Saliency(LinearModel())
    assert not sal.has_convergence_delta()
    assert not FeatureAblation(LinearModel()).has_convergence_delta()