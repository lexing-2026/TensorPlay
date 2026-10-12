"""Optional dependency detection helpers.

Each predicate probes whether an external package can be imported without
actually importing it, which keeps the metrics package importable even when
optional extras are missing.
"""

import importlib.util
import shutil
import sys

_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _module_available(module_name: str) -> bool:
    """Return ``True`` when ``module_name`` is importable."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


_SCIPY_AVAILABLE = _module_available("scipy")
_SCIPY_GREATER_EQUAL_1_8 = _SCIPY_AVAILABLE
_MATPLOTLIB_AVAILABLE = _module_available("matplotlib")
_SCIENCEPLOT_AVAILABLE = _module_available("scienceplots")
_XLA_AVAILABLE = _module_available("torch_xla")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_TORCHAUDIO_AVAILABLE = _module_available("torchaudio")
_TRANSFORMERS_AVAILABLE = _module_available("transformers")
_NLTK_AVAILABLE = _module_available("nltk")
_ROUGE_SCORE_AVAILABLE = _module_available("rouge_score")
_BERTSCORE_AVAILABLE = _module_available("bert_score")
_LPIPS_AVAILABLE = _module_available("lpips")
_PYCOCOTOOLS_AVAILABLE = _module_available("pycocotools")
_PESQ_AVAILABLE = _module_available("pesq")
_PYSTOI_AVAILABLE = _module_available("pystoi")
_LIBROSA_AVAILABLE = _module_available("librosa")
_REQUESTS_AVAILABLE = _module_available("requests")
_TQDM_AVAILABLE = _module_available("tqdm")
_ONNXRUNTIME_AVAILABLE = _module_available("onnxruntime")
_TORCH_FIDELITY_AVAILABLE = _module_available("torch_fidelity")
_FAST_BSS_EVAL_AVAILABLE = _module_available("fast_bss_eval")
_EINOPS_AVAILABLE = _module_available("einops")
_LATEX_AVAILABLE: bool = shutil.which("latex") is not None