"""tensorplay.audio — tensorplay.audio-compatible API.

The Python surface provides the audio module family with local imports; the
DSP primitives (fft family, windows, stft/istft)

* CPU: pocketfft (p10/include/pocketfft_hdronly.h)
* CUDA: cuFFT   (p10/src/backend/cuda/SpectralKernels.cu)

I/O follows the classic tensorplay.audio backend model (soundfile/scipy) with
tensorplay-compatible load/save/info signatures.
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
