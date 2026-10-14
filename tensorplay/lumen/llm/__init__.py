"""Research-oriented helpers for large language models.

This subpackage targets real transformer models (the Hugging Face ecosystem)
that live outside the tensorplay runtime. The two capabilities provided are:

- :mod:`attention <tensorplay.lumen.llm.attention>`: attention-map extraction
  and head heatmap visualization.
- :mod:`patching <tensorplay.lumen.llm.patching>`: activation patching,
  ablation, and causal tracing for locating where model components encode a
  behavior (the diagnostic step behind knowledge-editing research).

The framework used by the model itself is imported lazily inside the functions,
so importing this subpackage does not require it to be installed.
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