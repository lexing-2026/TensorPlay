#!/usr/bin/env python3

"""lumen: model interpretability and attribution.

The package exposes attribution algorithms for explaining model predictions.
It offers the familiar surface of model interpretability libraries:

- gradient-based input attribution: :class:`InputXGradient`, :class:`Saliency`,
  :class:`IntegratedGradients`;
- perturbation-based input attribution: :class:`FeatureAblation`,
  :class:`Occlusion`;
- stochastic smoothing: :class:`NoiseTunnel`;
- shared base classes: :class:`Attribution`, :class:`GradientAttribution`,
  :class:`PerturbationAttribution`, :class:`LayerAttribution`,
  :class:`NeuronAttribution`.

In addition to the classic algorithms, the package can leverage the graph
capture stack: trace the forward function into a
:class:`~tensorplay.graph.GraphModule` once and reuse it for every forward and
backward evaluation (see :func:`lumen._utils.graph.trace` and
:func:`lumen._utils.graph.graph_gradient_func`).
"""

from tensorplay.lumen._core.feature_ablation import FeatureAblation
from tensorplay.lumen._core.input_x_gradient import InputXGradient
from tensorplay.lumen._core.integrated_gradients import IntegratedGradients
from tensorplay.lumen._core.noise_tunnel import NoiseTunnel
from tensorplay.lumen._core.occlusion import Occlusion
from tensorplay.lumen._core.saliency import Saliency
from tensorplay.lumen._utils.attribution import (
    Attribution,
    GradientAttribution,
    LayerAttribution,
    NeuronAttribution,
    PerturbationAttribution,
)
from tensorplay.lumen._utils.baselines import ProductBaselines
from tensorplay.lumen._utils.graph import graph_gradient_func, node_names, trace

__all__ = [
    "Attribution",
    "GradientAttribution",
    "PerturbationAttribution",
    "NeuronAttribution",
    "LayerAttribution",
    "IntegratedGradients",
    "InputXGradient",
    "Saliency",
    "FeatureAblation",
    "Occlusion",
    "NoiseTunnel",
    "ProductBaselines",
    "trace",
    "node_names",
    "graph_gradient_func",
]