"""Research-oriented helpers for transformer language models.

This subpackage is framework-native: every helper operates on models built with
tensorplay primitives, uses tensorplay tensors and hooks, and requires no
third-party runtime. Capabilities:

- :mod:`attention <tensorplay.lumen.llm.attention>`: attention-map extraction
  and head heatmap visualization.
- :mod:`patching <tensorplay.lumen.llm.patching>`: activation patching,
  ablation, and causal tracing.
- :mod:`sae <tensorplay.lumen.llm.sae>`: sparse autoencoder feature learning.
- :mod:`clt <tensorplay.lumen.llm.clt>`: cross-layer transcoders for
  residual-stream updates.
- :mod:`logit_lens <tensorplay.lumen.llm.logit_lens>`: per-layer token
  prediction decoding.
- :mod:`attribution_graph <tensorplay.lumen.llm.attribution_graph>`: feature
  attribution graphs over MLP features.
- :mod:`feature_circuit <tensorplay.lumen.llm.feature_circuit>`: causal
  feature-level circuit discovery (direct effects + joint-ablation edges).
"""

from tensorplay.lumen.llm.attention import (
    extract_attention,
    token_attention,
    visualize_attention_heads,
    visualize_attention_multi_layer,
    visualize_token_attention,
)
from tensorplay.lumen.llm.attribution_graph import (
    AttributionGraph,
    AttributionGraphData,
    GraphEdge,
    GraphNode,
)
from tensorplay.lumen.llm.clt import (
    CrossLayerTranscoder,
    collect_component_updates,
    train_clt,
)
from tensorplay.lumen.llm.feature_circuit import (
    CircuitEdge,
    CircuitNode,
    FeatureCircuit,
    FeatureCircuitData,
    top_attention_heads,
    top_mlp_features,
)
from tensorplay.lumen.llm.logit_lens import LogitLens
from tensorplay.lumen.llm.patching import ActivationPatching
from tensorplay.lumen.llm.sae import (
    SparseAutoencoder,
    collect_hidden_states,
    feature_frequency,
    reconstruction_error,
    top_activating_indices,
    train_sae,
)

__all__ = [
    "extract_attention",
    "visualize_attention_heads",
    "visualize_attention_multi_layer",
    "token_attention",
    "visualize_token_attention",
    "ActivationPatching",
    "SparseAutoencoder",
    "train_sae",
    "collect_hidden_states",
    "feature_frequency",
    "top_activating_indices",
    "reconstruction_error",
    "LogitLens",
    "AttributionGraph",
    "AttributionGraphData",
    "GraphNode",
    "GraphEdge",
    "CrossLayerTranscoder",
    "train_clt",
    "collect_component_updates",
    "FeatureCircuit",
    "FeatureCircuitData",
    "CircuitNode",
    "CircuitEdge",
    "top_mlp_features",
    "top_attention_heads",
]