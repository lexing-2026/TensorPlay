"""Audio I/O — tensorplay-compatible load / save / info.

Signatures follow tensorplay.audio 2.x: ``load`` supports partial reads via
frame_offset/num_frames, ``save`` takes an explicit channels_first flag
instead of guessing, and ``info`` returns an ``AudioMetaData`` namedtuple.
Backends are soundfile (preferred) and scipy.
"""
from collections import namedtuple

import numpy as np
import tensorplay as tp
from .backend import get_audio_backend, _SCIPY_AVAILABLE, _SOUNDFILE_AVAILABLE

# torchaudio AudioMetaData (torchaudio/AudioMetaData.py)
AudioMetaData = namedtuple(
    "AudioMetaData",
    ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
)


def _sf_dtype(normalize, bits_per_sample):
    if normalize:
        return "float32"
    return {16: "int16", 32: "int32", 8: "uint8", 24: "int32"}.get(bits_per_sample, "int16")


def load(filepath, frame_offset=0, num_frames=-1, normalize=True,
         channels_first=True, format=None):
    """Loads an audio file into a Tensor (tensorplay.audio.load semantics).

    Args:
        filepath: Path to the audio file.
        frame_offset: Number of frames to skip before reading.
        num_frames: Maximum number of frames to read; -1 reads everything
            from frame_offset.
        normalize: If True, convert to float32 normalized to [-1, 1];
            otherwise keep the native integer encoding.
        channels_first: If True (default), return (Channels, Time).
        format: Ignored; the backend sniffs the container.

    Returns:
        (Tensor, int): waveform tensor and sample rate.
    """
    backend = get_audio_backend()
    if backend is None:
        raise ImportError(
            "No audio backend available. Please install soundfile or scipy.")

    audio_np = None
    sr = 0

    if backend == "soundfile":
        import soundfile as sf
        dtype = "float32" if normalize else None
        start = int(frame_offset)
        count = int(num_frames)
        if num_frames == -1:
            audio_np, sr = sf.read(filepath, dtype=dtype, start=start,
                                   always_2d=True)
        else:
            audio_np, sr = sf.read(filepath, dtype=dtype, start=start,
                                   frames=count, always_2d=True)

    elif backend == "scipy":
        from scipy.io import wavfile
        sr, raw = wavfile.read(filepath)
        if not isinstance(raw, np.ndarray):
            raw = np.array(raw)
        if raw.ndim == 1:
            raw = raw[:, None]
        start = max(0, int(frame_offset))
        stop = raw.shape[0] if num_frames == -1 else min(raw.shape[0], start + int(num_frames))
        audio_np = raw[start:stop]
        if normalize:
            if raw.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            elif raw.dtype == np.int32:
                audio_np = audio_np.astype(np.float32) / 2147483648.0
            elif raw.dtype == np.uint8:
                audio_np = (audio_np.astype(np.float32) - 128.0) / 128.0

    if audio_np is None:
        raise RuntimeError(f"Failed to load audio file: {filepath}")
    if not isinstance(audio_np, np.ndarray):
        audio_np = np.array(audio_np)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]

    # Fast path: C++ kernel converts (Time, Channels) -> (Channels, Time)
    # with int16/int32/uint8 normalization folded in.
    use_cpp = (
        hasattr(tp, "audio_to_tensor")
        and normalize
        and audio_np.dtype in (np.int16, np.int32, np.uint8, np.float32)
    )
    if use_cpp:
        tensor = tp.audio_to_tensor(audio_np.astype(np.float32))
    else:
        tensor = tp.tensor(audio_np.T.copy() if channels_first else audio_np.copy())
        return tensor, sr

    if not channels_first:
        tensor = tensor.t()
    return tensor, sr


def save(filepath, src, sample_rate, channels_first=True, format=None):
    """Saves a Tensor to an audio file (tensorplay.audio.save semantics).

    Args:
        filepath: Destination path (.wav etc. per backend support).
        src: Waveform tensor; interpreted as (Channels, Time) when
            channels_first=True (default), else (Time, Channels).
        sample_rate: Sampling rate in Hz.
        channels_first: Layout flag of ``src``.
        format: Ignored; inferred from the extension.
    """
    backend = get_audio_backend()
    if backend is None:
        raise ImportError("No audio backend available.")

    if isinstance(src, tp.Tensor):
        try:
            arr = src.numpy()
        except Exception:
            arr = np.asarray(src)
    else:
        arr = np.asarray(src)

    arr = np.asarray(arr)
    if channels_first:
        # torchaudio expects backend-facing (Time, Channels)
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.T
    elif arr.ndim == 1:
        arr = arr[:, None]

    if backend == "soundfile":
        import soundfile as sf
        sf.write(filepath, arr, int(sample_rate))
    elif backend == "scipy":
        from scipy.io import wavfile
        wavfile.write(filepath, int(sample_rate), arr)


def info(filepath, format=None, buffer_size=4096):
    """Returns signal information of an audio file (tensorplay.audio.info).

    Returns:
        AudioMetaData with fields
        (sample_rate, num_frames, num_channels, bits_per_sample, encoding).
    """
    backend = get_audio_backend()
    if backend == "soundfile":
        import soundfile as sf
        si = sf.info(filepath)
        subtype_bits = {
            "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
            "FLOAT": 32, "DOUBLE": 64, "ULAW": 8, "ALAW": 8,
        }
        return AudioMetaData(
            sample_rate=int(si.samplerate),
            num_frames=int(si.frames),
            num_channels=int(si.channels),
            bits_per_sample=subtype_bits.get(si.subtype, 16),
            encoding=si.subtype,
        )
    elif backend == "scipy":
        import wave
        with wave.open(filepath, 'rb') as f:
            encodings = {1: "PCM_S", 2: "ALAW", 3: "FLOAT", 6: "ALAW", 7: "ULAW"}
            return AudioMetaData(
                sample_rate=f.getframerate(),
                num_frames=f.getnframes(),
                num_channels=f.getnchannels(),
                bits_per_sample=f.getsampwidth() * 8,
                encoding=encodings.get(f.getcomptype(), "PCM_S"),
            )
    return None
