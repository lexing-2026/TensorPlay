"""Research-oriented helpers for transformer language models.

This subpackage is framework-native: every helper operates on models built with
tensorplay primitives, uses tensorplay tensors and hooks, and requires no
third-party runtime. Capabilities:

- :mod:`attention <tensorplay.lumen.llm.attention>`: attention-map extraction
  and head heatmap visualization.
- :mod:`patching <tensorplay.lumen.llm.patching>`: activation patching,
  ablation, and causal tracing for locating where model components encode a
  behavior (the diagnostic step behind knowledge-editing research).
- :mod:`sae <tensorplay.lumen.llm.sae>`: sparse autoencoder feature learning
  from hidden states (dictionary learning).
- :mod:`logit_lens <tensorplay.lumen.llm.logit_lens>`: per-layer token
  prediction decoding.
- :mod:`attribution_graph <tensorplay.lumen.llm.attribution_graph>`: feature
  attribution graphs over MLP features.
- The ported :mod:`~tensorplay.lumen._core.llm_attr` attribution classes
  (``LLMAttribution`` / ``LLMGradientAttribution``) provide native gradient and
  perturbation attribution for text generation.
"""

from tensorplay.lumen.llm.attention import (
    extract_attention,
    visualize_attention_heads,
)
from tensorplay.lumen.llm.attribution_graph import (
    AttributionGraph,
    AttributionGraphData,
    GraphEdge,
    GraphNode,
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
]