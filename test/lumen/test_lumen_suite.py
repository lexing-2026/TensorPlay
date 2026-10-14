"""Tests for the extended lumen attribution suite.

These are lightweight correctness/smoke tests for the algorithm families added
beyond the core six: GradientShap, DeepLift, LRP, FeaturePermutation, the
layer/neuron series, statistics, summarizers, visualization, and the
graph-rewrite / compiled-execution differentiators.
"""

import pytest
import tensorplay as tensorplay
from tensorplay import nn

from tensorplay.lumen import (
    ClassSummarizer,
    DataLoaderAttribution,
    DeepLift,
    DeepLiftShap,
    Deconvolution,
    FeatureAblation,
    FeaturePermutation,
    GradientShap,
    GraphDeconvolution,
    GraphGuidedBackprop,
    GuidedBackprop,
    InternalInfluence,
    LayerActivation,
    LayerConductance,
    LayerDeepLift,
    LayerDeepLiftShap,
    LayerFeatureAblation,
    LayerFeaturePermutation,
    LayerGradientXActivation,
    LayerGradientShap,
    LayerIntegratedGradients,
    LayerLRP,
    LRP,
    Mean,
    Min,
    Max,
    MSE,
    NeuronConductance,
    NeuronDeepLift,
    NeuronDeepLiftShap,
    NeuronFeatureAblation,
    NeuronGradient,
    NeuronGradientShap,
    NeuronGuidedBackprop,
    NeuronIntegratedGradients,
    StdDev,
    Sum,
    Summarizer,
    Var,
    Count,
    visualization,
)


class Linear(nn.Module):
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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _x() -> tensorplay.Tensor:
    return tensorplay.tensor([[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 1.0]])


def test_gradient_shap() -> None:
    model = Linear()
    x = _x()
    attr, delta = GradientShap(model).attribute(
        x,
        target=1,
        baselines=tensorplay.zeros_like(x),
        n_samples=5,
        return_convergence_delta=True,
    )
    assert attr.shape == x.shape
    assert delta.shape == (x.shape[0] * 5,)


def test_deep_lift_family() -> None:
    x = _x()
    for cls in (DeepLift, DeepLiftShap):
        attr = cls(MLP()).attribute(x, target=1, baselines=tensorplay.zeros_like(x))
        assert attr.shape == x.shape


def test_feature_permutation() -> None:
    model = Linear()
    x = _x()
    attr = FeaturePermutation(model).attribute(x, target=1)
    assert attr.shape == x.shape
    # permutation attribution of a linear model matches the per-feature
    # output sensitivity (x * W) up to permutation statistics.
    assert tensorplay.isfinite(attr).all()


def test_lrp() -> None:
    x = _x()
    assert LRP(MLP()).attribute(x, target=1).shape == x.shape


def test_guided_backprop_graph_and_hook() -> None:
    x = _x()
    net = MLP()
    target = 1
    gb = GraphGuidedBackprop(net).attribute(x, target=target)
    gb_hook = GuidedBackprop(net).attribute(x, target=target)
    assert tensorplay.allclose(gb, gb_hook, atol=1e-5)

    dc = GraphDeconvolution(net).attribute(x, target=target)
    dc_hook = Deconvolution(net).attribute(x, target=target)
    assert tensorplay.allclose(dc, dc_hook, atol=1e-5)


def test_guided_backprop_semantics() -> None:
    # GuidedBackprop gradient w.r.t. the input must equal the chain rule
    # through a ReLU whose backward is relu(grad) * (activation > 0).
    class TwoLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 3)

        def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
            return self.fc2(tensorplay.relu(self.fc1(x)))

    net = TwoLayer()
    x = _x()
    target = 1
    attr = GraphGuidedBackprop(net).attribute(x, target=target)

    with tensorplay.no_grad():
        h = tensorplay.relu(net.fc1(x))  # (2, 8) activation of the first ReLU
        w2 = net.fc2.weight[target]  # (8,) gradient at the hidden layer
        guided_grad_hidden = tensorplay.relu(w2) * (h > 0).to(h.dtype)
        expected = tensorplay.matmul(guided_grad_hidden, net.fc1.weight)
    assert tensorplay.allclose(attr, expected, atol=1e-5)


def test_feature_ablation_compiled_graph() -> None:
    net = MLP()
    x = _x()
    attr_eager = FeatureAblation(net).attribute(x, target=1)
    attr_compiled = FeatureAblation(net).attribute(
        x, target=1, use_compiled_graph=True
    )
    assert tensorplay.allclose(attr_eager, attr_compiled, atol=1e-5)


def test_layer_attributions() -> None:
    model = MLP()
    x = _x()
    layer = model.fc2
    assert LayerActivation(model, layer).attribute(x).shape == (2, 3)
    assert LayerGradientXActivation(model, layer).attribute(x, target=1).shape == (2, 3)
    assert (
        LayerIntegratedGradients(model, layer)
        .attribute(x, target=1, baselines=0.0, n_steps=10)
        .shape
        == (2, 3)
    )
    assert (
        LayerGradientShap(model, layer)
        .attribute(x, target=1, baselines=tensorplay.zeros_like(x), n_samples=3)
        .shape
        == (2, 3)
    )
    assert LayerConductance(model, layer).attribute(x, target=1).shape == (2, 3)
    assert LayerDeepLift(model, layer).attribute(x, target=1, baselines=0.0).shape == (2, 3)
    assert (
        LayerDeepLiftShap(model, layer)
        .attribute(x, target=1, baselines=tensorplay.zeros(2, 4))
        .shape
        == (2, 3)
    )
    assert LayerLRP(model, layer).attribute(x, target=1).shape == (2, 3)
    assert LayerFeatureAblation(model, layer).attribute(x, target=1).shape == (2, 3)
    assert LayerFeaturePermutation(model, layer).attribute(x, target=1).shape == (2, 3)
    assert InternalInfluence(model, layer).attribute(x, target=1).shape == (2, 3)


def test_neuron_attributions() -> None:
    model = MLP()
    x = _x()
    layer = model.fc2
    selector = (0,)
    assert NeuronGradient(model, layer).attribute(x, neuron_selector=selector).shape == (2, 4)
    assert (
        NeuronIntegratedGradients(model, layer)
        .attribute(x, neuron_selector=selector, baselines=0.0, n_steps=10)
        .shape
        == (2, 4)
    )
    assert (
        NeuronGradientShap(model, layer)
        .attribute(x, neuron_selector=selector, baselines=tensorplay.zeros_like(x), n_samples=3)
        .shape
        == (2, 4)
    )
    assert (
        NeuronConductance(model, layer).attribute(x, neuron_selector=selector, target=1).shape
        == (2, 4)
    )
    assert (
        NeuronDeepLift(model, layer)
        .attribute(x, neuron_selector=selector, baselines=0.0)
        .shape
        == (2, 4)
    )
    assert (
        NeuronDeepLiftShap(model, layer)
        .attribute(x, neuron_selector=selector, baselines=tensorplay.zeros(2, 4))
        .shape
        == (2, 4)
    )
    assert (
        NeuronFeatureAblation(model, layer).attribute(x, neuron_selector=selector).shape
        == (2, 4)
    )
    assert (
        NeuronGuidedBackprop(model, layer).attribute(x, neuron_selector=selector).shape
        == (2, 4)
    )


def test_dataloader_attribution() -> None:
    from tensorplay.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(tensorplay.randn(4, 4), tensorplay.zeros(4, dtype=tensorplay.int64))
    dl = DataLoader(ds, batch_size=2)
    attr = DataLoaderAttribution(FeatureAblation(MLP())).attribute(dl, input_roles=(0, 2))
    assert isinstance(attr, tuple)
    assert attr[0].shape[1] == 4


def test_summarizer_and_class_summarizer() -> None:
    stats = [Mean(), StdDev(), MSE(), Var(), Min(), Max(), Sum(), Count()]
    summ = Summarizer(stats)
    for _ in range(5):
        summ.update(tensorplay.randn(3))
    summary = summ.summary
    assert summary is not None
    assert "mean" in summary
    assert "count" in summary

    cs = ClassSummarizer([Mean(), Min(), Max(), Sum(), Count()])
    for i in range(4):
        cs.update(tensorplay.randn(3) + i, [i % 2])
    assert cs.summary is not None


def test_visualization_helpers() -> None:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend: never pop up a window
    import matplotlib.pyplot as plt

    attr = tensorplay.rand(4, 4)
    fig, _ = visualization.visualize_image_attr(
        attr, attr, method="heat_map", use_pyplot=False
    )
    plt.close(fig)
    fig, _ = visualization.visualize_image_attr_multiple(
        attr,
        attr,
        ["original_image", "heat_map"],
        ["all", "positive"],
        use_pyplot=False,
    )
    plt.close(fig)
    ts = tensorplay.randn(10, 2)
    fig, _ = visualization.visualize_timeseries_attr(ts, ts, use_pyplot=False)
    plt.close(fig)

    from tensorplay.lumen._utils.visualization import VisualizationDataRecord

    rec = VisualizationDataRecord(
        [0.5, -0.2],
        0.8,
        1,
        1,
        1,
        0.9,
        ["a", "b"],
        0.0,
    )
    out = visualization.visualize_text([rec])
    assert isinstance(out, str)  # fallback without IPython