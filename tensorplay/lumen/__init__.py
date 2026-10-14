#!/usr/bin/env python3

"""lumen: model interpretability and attribution.

The package exposes attribution algorithms for explaining model predictions,
covering the full model-interpretability surface:

- gradient-based input attribution: :class:`InputXGradient`, :class:`Saliency`,
  :class:`IntegratedGradients`, :class:`GradientShap`, :class:`DeepLift`,
  :class:`DeepLiftShap`, :class:`GuidedBackprop`, :class:`Deconvolution`,
  :class:`GuidedGradCam`;
- perturbation-based input attribution: :class:`FeatureAblation`,
  :class:`FeaturePermutation`, :class:`Occlusion`, :class:`ShapleyValues`,
  :class:`ShapleyValueSampling`;
- surrogate-model methods: :class:`Lime`, :class:`KernelShap`;
- propagation methods: :class:`LRP`;
- stochastic smoothing: :class:`NoiseTunnel`;
- layer and neuron attributions for every algorithm family;
- interpretable-input wrappers and embedding baselines for text and images;
- statistics, summarizers, and visualization helpers.

In addition to the classic algorithms, the package can leverage the graph
capture stack: trace the forward function into a
:class:`~tensorplay.graph.GraphModule` once and reuse it for every forward and
backward evaluation (see :func:`lumen._utils.graph.trace` and
:func:`lumen._utils.graph.graph_gradient_func`).
"""

from tensorplay.lumen._core.dataloader_attr import DataLoaderAttribution
from tensorplay.lumen._core.deep_lift import DeepLift, DeepLiftShap
from tensorplay.lumen._core.feature_ablation import FeatureAblation
from tensorplay.lumen._core.feature_permutation import FeaturePermutation
from tensorplay.lumen._core.gradient_shap import GradientShap
from tensorplay.lumen._core.graph_guided_backprop import (
    GraphDeconvolution,
    GraphGuidedBackprop,
)
from tensorplay.lumen._core.guided_backprop_deconvnet import Deconvolution, GuidedBackprop
from tensorplay.lumen._core.guided_grad_cam import GuidedGradCam
from tensorplay.lumen._core.input_x_gradient import InputXGradient
from tensorplay.lumen._core.integrated_gradients import IntegratedGradients
from tensorplay.lumen._core.kernel_shap import KernelShap
from tensorplay.lumen._core.layer.grad_cam import LayerGradCam
from tensorplay.lumen._core.layer.internal_influence import InternalInfluence
from tensorplay.lumen._core.layer.layer_activation import LayerActivation
from tensorplay.lumen._core.layer.layer_conductance import LayerConductance
from tensorplay.lumen._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap
from tensorplay.lumen._core.layer.layer_feature_ablation import LayerFeatureAblation
from tensorplay.lumen._core.layer.layer_feature_permutation import LayerFeaturePermutation
from tensorplay.lumen._core.layer.layer_gradient_shap import LayerGradientShap
from tensorplay.lumen._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from tensorplay.lumen._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from tensorplay.lumen._core.layer.layer_lrp import LayerLRP
from tensorplay.lumen._core.lime import Lime, LimeBase
from tensorplay.lumen._core.llm_attr import (
    LLMAttribution,
    LLMAttributionResult,
    LLMGradientAttribution,
    RemoteLLMAttribution,
)
from tensorplay.lumen._core.lrp import LRP
from tensorplay.lumen._core.neuron.neuron_conductance import NeuronConductance
from tensorplay.lumen._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from tensorplay.lumen._core.neuron.neuron_feature_ablation import NeuronFeatureAblation
from tensorplay.lumen._core.neuron.neuron_gradient import NeuronGradient
from tensorplay.lumen._core.neuron.neuron_gradient_shap import NeuronGradientShap
from tensorplay.lumen._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
from tensorplay.lumen._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)
from tensorplay.lumen._core.noise_tunnel import NoiseTunnel
from tensorplay.lumen._core.occlusion import Occlusion
from tensorplay.lumen._core.remote_provider import RemoteLLMProvider, VLLMProvider
from tensorplay.lumen._core.saliency import Saliency
from tensorplay.lumen._core.shapley_value import ShapleyValues, ShapleyValueSampling
from tensorplay.lumen._models.base import (
    configure_interpretable_embedding_layer,
    InterpretableEmbeddingBase,
    remove_interpretable_embedding_layer,
    TokenReferenceBase,
)
from tensorplay.lumen._utils import visualization
from tensorplay.lumen._utils.attribution import (
    Attribution,
    GradientAttribution,
    LayerAttribution,
    NeuronAttribution,
    PerturbationAttribution,
)
from tensorplay.lumen._utils.baselines import ProductBaselines
from tensorplay.lumen._utils.class_summarizer import ClassSummarizer
from tensorplay.lumen._utils.interpretable_input import (
    ImageMaskInput,
    InterpretableInput,
    TextTemplateInput,
    TextTokenInput,
)
from tensorplay.lumen._utils.stat import (
    CommonStats,
    Count,
    Max,
    Mean,
    Min,
    MSE,
    StatValue,
    StdDev,
    Sum,
    Var,
)
from tensorplay.lumen._utils.summarizer import Summarizer, SummarizerSingleTensor
from tensorplay.lumen._utils.graph import graph_gradient_func, node_names, trace

__all__ = [
    "Attribution",
    "GradientAttribution",
    "PerturbationAttribution",
    "NeuronAttribution",
    "LayerAttribution",
    "IntegratedGradients",
    "DataLoaderAttribution",
    "DeepLift",
    "DeepLiftShap",
    "InputXGradient",
    "Saliency",
    "GuidedBackprop",
    "Deconvolution",
    "GuidedGradCam",
    "FeatureAblation",
    "FeaturePermutation",
    "Occlusion",
    "ShapleyValueSampling",
    "ShapleyValues",
    "LimeBase",
    "Lime",
    "LRP",
    "KernelShap",
    "LayerConductance",
    "LayerGradientXActivation",
    "LayerActivation",
    "LayerFeaturePermutation",
    "LayerFeatureAblation",
    "LLMAttribution",
    "LLMAttributionResult",
    "LLMGradientAttribution",
    "RemoteLLMAttribution",
    "RemoteLLMProvider",
    "VLLMProvider",
    "InternalInfluence",
    "LayerGradCam",
    "LayerDeepLift",
    "LayerDeepLiftShap",
    "LayerGradientShap",
    "LayerIntegratedGradients",
    "LayerLRP",
    "NeuronConductance",
    "NeuronFeatureAblation",
    "NeuronGradient",
    "NeuronIntegratedGradients",
    "NeuronDeepLift",
    "NeuronDeepLiftShap",
    "NeuronGradientShap",
    "NeuronDeconvolution",
    "NeuronGuidedBackprop",
    "NoiseTunnel",
    "ProductBaselines",
    "GradientShap",
    "GraphGuidedBackprop",
    "GraphDeconvolution",
    "ImageMaskInput",
    "InterpretableEmbeddingBase",
    "TextTemplateInput",
    "TextTokenInput",
    "TokenReferenceBase",
    "visualization",
    "configure_interpretable_embedding_layer",
    "remove_interpretable_embedding_layer",
    "Summarizer",
    "CommonStats",
    "ClassSummarizer",
    "Mean",
    "StdDev",
    "MSE",
    "Var",
    "Min",
    "Max",
    "Sum",
    "Count",
    "StatValue",
    "SummarizerSingleTensor",
    "trace",
    "node_names",
    "graph_gradient_func",
]