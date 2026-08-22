"""Audio I/O backend registry — torch-compatible backend model.

Mirrors the classic torchaudio backend API (set_audio_backend /
get_audio_backend / list_audio_backends).  Backends are provided by the
optional soundfile and scipy packages; soundfile is preferred when both are
present, matching torchaudio's historical default order.
"""
import os
import sys

_SOUNDFILE_AVAILABLE = False
_SCIPY_AVAILABLE = False
_BACKEND: str | None = None

try:
    import soundfile  # noqa: F401
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    pass

try:
    import scipy.io.wavfile  # noqa: F401
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

_ALLOWED = ("soundfile", "scipy")


def set_audio_backend(backend: str | None) -> None:
    """Specifies the package used to load audio files (torchaudio semantics).

    Args:
        backend: One of ``"soundfile"``, ``"scipy"`` or ``None``.
            ``None`` resets to automatic detection.
    """
    global _BACKEND
    if backend is None:
        _BACKEND = None
        return
    if backend not in _ALLOWED:
        raise ValueError(f'Invalid backend "{backend}". Supported backends: {_ALLOWED}')
    if backend == "soundfile" and not _SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile not installed")
    if backend == "scipy" and not _SCIPY_AVAILABLE:
        raise ImportError("scipy not installed")
    _BACKEND = backend


def get_audio_backend() -> str | None:
    """Returns the name of the package used to load audio files."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if _SOUNDFILE_AVAILABLE:
        return "soundfile"
    if _SCIPY_AVAILABLE:
        return "scipy"
    return None


def list_audio_backends() -> list[str]:
    """Returns available backends, best first (torchaudio semantics)."""
    out = []
    if _SOUNDFILE_AVAILABLE:
        out.append("soundfile")
    if _SCIPY_AVAILABLE:
        out.append("scipy")
    return out


def check_available() -> bool:
    return _SOUNDFILE_AVAILABLE or _SCIPY_AVAILABLE
