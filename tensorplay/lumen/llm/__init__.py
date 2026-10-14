"""Research-oriented helpers for transformer language models.

This subpackage is framework-native: every helper operates on models built with
tensorplay primitives, uses tensorplay tensors and hooks, and requires no
third-party runtime. Three capabilities are provided:

- :mod:`attention <tensorplay.lumen.llm.attention>`: attention-map extraction
  (either from ``outputs.attentions`` or by capturing
  ``nn.MultiheadAttention`` weights) and head heatmap visualization.
- :mod:`patching <tensorplay.lumen.llm.patching>`: activation patching,
  ablation, and causal tracing for locating where model components encode a
  behavior (the diagnostic step behind knowledge-editing research).
- The ported :mod:`~tensorplay.lumen._core.llm_attr` attribution classes
  (``LLMAttribution`` / ``LLMGradientAttribution``) already provide native
  gradient and perturbation attribution for text generation on tensorplay
  models.
"""

from tensorplay.lumen.llm.attention import (
    extract_attention,
    visualize_attention_heads,
)
from tensorplay.lumen.llm.patching import ActivationPatching

__all__ = [
    "extract_attention",
    "visualize_attention_heads",
    "ActivationPatching",
]