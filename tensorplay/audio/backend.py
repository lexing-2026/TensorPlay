import os
import sys

_SOUNDFILE_AVAILABLE = False
_SCIPY_AVAILABLE = False
_BACKEND = None

try:
    import soundfile
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    pass

try:
    import scipy.io.wavfile
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

def set_audio_backend(backend):
    """
    Specifies the package used to load audio files.
    Args:
        backend (str): Name of the backend. One of "soundfile", "scipy".
    """
    global _BACKEND
    if backend not in ["soundfile", "scipy"]:
        raise ValueError("Invalid backend. Supported backends: soundfile, scipy")
    
    if backend == "soundfile" and not _SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile not installed")
    if backend == "scipy" and not _SCIPY_AVAILABLE:
        raise ImportError("scipy not installed")
        
    _BACKEND = backend

def get_audio_backend():
    """
    Returns the name of the package used to load audio files.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
        
    if _SOUNDFILE_AVAILABLE:
        return "soundfile"
    if _SCIPY_AVAILABLE:
        return "scipy"
        
    return None

def check_available():
    return _SOUNDFILE_AVAILABLE or _SCIPY_AVAILABLE
