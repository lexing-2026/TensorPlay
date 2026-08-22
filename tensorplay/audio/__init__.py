"""tensorplay.audio — torchaudio-compatible API.

The Python surface is ported verbatim from torchaudio 2.11 (third_party/audio)
with import rewriting; the DSP primitives (fft family, windows, stft/istft)
are native p10 kernels mirroring ATen's SpectralOps semantics:

* CPU: pocketfft (p10/include/pocketfft_hdronly.h)
* CUDA: cuFFT   (p10/src/backend/cuda/SpectralKernels.cu)

I/O follows the classic torchaudio backend model (soundfile/scipy) with
torch-compatible load/save/info signatures.
"""
from .backend import (
    check_available,
    get_audio_backend,
    list_audio_backends,
    set_audio_backend,
    _SOUNDFILE_AVAILABLE,
    _SCIPY_AVAILABLE,
)
from .io import AudioMetaData, info, load, save
from . import compliance
from . import datasets
from . import functional
from . import models
from . import transforms
from . import utils

__all__ = [
    "check_available",
    "get_audio_backend",
    "list_audio_backends",
    "set_audio_backend",
    "AudioMetaData",
    "info",
    "load",
    "save",
    "compliance",
    "datasets",
    "functional",
    "models",
    "transforms",
    "utils",
]
