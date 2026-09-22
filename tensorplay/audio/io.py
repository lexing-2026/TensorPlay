"""Audio I/O — tensorplay-compatible load / save / info.

Signatures follow tensorplay.audio 2.x: ``load`` supports partial reads via
frame_offset/num_frames, ``save`` takes an explicit channels_first flag
instead of guessing, and ``info`` returns an ``AudioMetaData`` namedtuple.
Backends are soundfile (preferred) and scipy. Containers libsndfile cannot
open (mp4/m4a/aac/webm and similar) fall back to the FFmpeg decoder through
PyAV when it is installed, so any codec the installed FFmpeg knows is
loadable.
"""
from collections import namedtuple

import numpy as np
import tensorplay as tp
from .backend import get_audio_backend, _SCIPY_AVAILABLE, _SOUNDFILE_AVAILABLE

AudioMetaData = namedtuple(
    "AudioMetaData",
    ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
)


def _sf_dtype(normalize, bits_per_sample):
    if normalize:
        return "float32"
    return {16: "int16", 32: "int32", 8: "uint8", 24: "int32"}.get(bits_per_sample, "int16")


# Sample-format bits for the FFmpeg fallback's metadata; keys are PyAV
# AudioFormat names (planar variants carry the same width per sample).
_AV_FORMAT_BITS = {
    "u8": 8, "u8p": 8, "s16": 16, "s16p": 16, "s32": 32, "s32p": 32,
    "flt": 32, "fltp": 32, "dbl": 64, "dblp": 64, "s64": 64, "s64p": 64,
}


def _av_audio_stream(filepath):
    import av
    container = av.open(filepath)
    streams = [s for s in container.streams if s.type == "audio"]
    if not streams:
        container.close()
        raise RuntimeError(f"No audio stream found in file: {filepath}")
    return container, streams[0]


def _av_decode(filepath):
    """Decode an FFmpeg-readable container into (Time, Channels) samples.

    Decoded frames arrive either planar (Channels, Time) or packed
    (1, Channels*Time); both are normalized to (Time, Channels) in the
    stream's native sample dtype. Decoding runs to the end of the stream and
    the caller slices; this keeps offsets sample-exact regardless of packet
    boundaries and encoder priming.
    """
    container, stream = _av_audio_stream(filepath)
    try:
        channels = stream.codec_context.channels
        pieces = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim == 1:
                arr = arr[:, None]
            elif channels > 1 and arr.shape[0] == 1:
                arr = arr.reshape(channels, -1).T
            else:
                arr = arr.T
            pieces.append(np.ascontiguousarray(arr))
    finally:
        container.close()
    if not pieces:
        return np.zeros((0, channels), dtype=np.float32), int(stream.codec_context.sample_rate)
    samples = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
    return samples, int(stream.codec_context.sample_rate)


def _av_load(filepath, frame_offset, num_frames, normalize):
    samples, sr = _av_decode(filepath)
    start = max(0, frame_offset)
    stop = samples.shape[0] if num_frames < 0 else min(samples.shape[0], start + max(0, num_frames))
    samples = samples[start:stop]
    if normalize:
        kind = samples.dtype
        if kind == np.int8:
            samples = samples.astype(np.float32) * (1.0 / 128.0)
        elif kind == np.float64:
            samples = samples.astype(np.float32)
        # int16/int32/uint8/float32 pass through untouched: the fused
        # conversion kernel below performs their scaling itself.
    return np.ascontiguousarray(samples), sr


def _av_info(filepath):
    container, stream = _av_audio_stream(filepath)
    try:
        ctx = stream.codec_context
        rate = int(ctx.sample_rate)
        frames = 0
        if stream.duration is not None:
            frames = int(round(stream.duration * float(stream.time_base) * rate))
        elif container.duration is not None:
            frames = int(round(container.duration * rate / 1e6))
        fmt_name = ctx.format.name if ctx.format is not None else ""
        return AudioMetaData(
            sample_rate=rate,
            num_frames=frames,
            num_channels=int(ctx.channels),
            bits_per_sample=_AV_FORMAT_BITS.get(fmt_name, 32),
            encoding=ctx.name or "unknown",
        )
    finally:
        container.close()


def _native_io():
    """The ``tensorplay._C.io`` submodule when the extension provides it."""
    return getattr(getattr(tp, "_C", None), "io", None)


def _read_bytes(filepath):
    with open(filepath, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


def decode_wav(data, frame_offset=0, num_frames=-1):
    """Decodes WAV bytes into (waveform [channels, time] float32, sample rate).

    Uses the native RIFF/WAVE codec compiled into the extension; raises
    ImportError when the codec is not available in this build.
    """
    native = _native_io()
    decoder = getattr(native, "decode_wav", None) if native else None
    if decoder is None:
        raise ImportError("The native WAV decoder is not available in this build.")
    return decoder(data, int(frame_offset), int(num_frames))


def decode_wav_batch(data, frame_offset=0, num_frames=-1):
    """Decodes a sequence of WAV byte tensors in parallel (native codec).

    Returns a list of (waveform [channels, time] float32, sample rate) pairs
    in input order.
    """
    native = _native_io()
    decoder = getattr(native, "decode_wav_batch", None) if native else None
    if decoder is None:
        raise ImportError("The native WAV decoder is not available in this build.")
    return decoder(list(data), int(frame_offset), int(num_frames))


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

    # Native WAV path: playable clips decode without any backend package and
    # with byte-arithmetic seeking. On any failure (corrupt stream, unusual
    # layout) the backend path below takes over.
    if normalize and str(filepath).lower().endswith((".wav", ".wave")):
        native = _native_io()
        if native is not None and hasattr(native, "decode_wav"):
            try:
                waveform, sr = native.decode_wav(
                    tp.tensor(_read_bytes(filepath)), int(frame_offset), int(num_frames))
                if not channels_first:
                    waveform = waveform.t()
                return waveform, sr
            except Exception:
                pass

    if backend is None:
        raise ImportError(
            "No audio backend available. Please install soundfile or scipy.")

    audio_np = None
    sr = 0

    if backend == "soundfile":
        import soundfile as sf
        try:
            read_dtype = "float32" if normalize else None
            handle = sf.SoundFile(filepath)
            try:
                # Integer-PCM subtypes are read at their native width so the
                # conversion kernel below folds scaling and the (Time, Channels)
                # -> (Channels, Time) transpose into a single pass. For PCM_16/
                # PCM_24/PCM_32 the native-width read is bit-identical to the
                # float32 decode; encodings without an integer read (FLOAT,
                # ULAW, ...) stay on it.
                if normalize:
                    if handle.subtype == "PCM_16":
                        read_dtype = "int16"
                    elif handle.subtype in ("PCM_24", "PCM_32"):
                        read_dtype = "int32"
                sr = handle.samplerate
                # Slice-style bounds, as soundfile's own read() applies them:
                # a negative offset counts from the end and an offset past the
                # last frame clamps to EOF (yielding an empty read) instead of
                # failing the seek.
                start, stop, _ = slice(int(frame_offset), None).indices(handle.frames)
                count = int(num_frames)
                if count < 0:
                    count = stop - start
                handle.seek(start)
                audio_np = handle.read(count, dtype=read_dtype, always_2d=True)
            finally:
                handle.close()
        except Exception:
            # libsndfile does not know the container: hand it to the FFmpeg
            # decoder, which reads every codec it was built with. When PyAV
            # is missing, surface the original libsndfile failure instead.
            try:
                import av  # noqa: F401
            except ImportError:
                raise
            audio_np, sr = _av_load(filepath, int(frame_offset), int(num_frames), normalize)

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
        # For int16/int32/uint8 with normalize=True, normalization and the
        # (Time, Channels) -> (Channels, Time) transpose both fold into the
        # native conversion gated below; other encodings pass through
        # untouched, as before.

    if audio_np is None:
        raise RuntimeError(f"Failed to load audio file: {filepath}")
    if not isinstance(audio_np, np.ndarray):
        audio_np = np.array(audio_np)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]

    # Fast path: C++ kernel converts (Time, Channels) -> (Channels, Time)
    # with int16/int32/uint8 normalization folded in.
    native = _native_io()
    use_cpp = (
        native is not None
        and hasattr(native, "audio_to_tensor")
        and normalize
        and audio_np.dtype in (np.int16, np.int32, np.uint8, np.float32)
    )
    if use_cpp:
        # Pass the native encoding through: the kernel normalizes int16/int32/
        # uint8 and transposes in one pass. Casting to float32 here would
        # disable its normalization branches.
        tensor = native.audio_to_tensor(np.ascontiguousarray(audio_np))
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
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.T
    elif arr.ndim == 1:
        arr = arr[:, None]

    # Native WAV path: float32 clips encode through the extension codec; the
    # backend path remains for encodings it does not cover.
    native = _native_io()
    if (native is not None and hasattr(native, "encode_wav")
            and str(filepath).lower().endswith((".wav", ".wave"))):
        try:
            arr_f = arr.astype(np.float32)
            src = np.ascontiguousarray(arr_f.T) if arr_f.ndim == 2 else np.ascontiguousarray(arr_f)
            tensor = tp.tensor(src)
            if tensor.dtype == tp.float32:
                data = native.encode_wav(tensor, int(sample_rate), 16)
                with open(filepath, "wb") as f:
                    f.write(bytes(data.numpy().tobytes()))
                return
        except Exception:
            pass

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
    native = _native_io()
    if (native is not None and hasattr(native, "wav_info")
            and str(filepath).lower().endswith((".wav", ".wave"))):
        try:
            sample_rate, frames, channels, bits, encoding = native.wav_info(
                tp.tensor(_read_bytes(filepath)))
            name = {1: f"PCM_{bits}", 3: "FLOAT", 6: "ALAW", 7: "ULAW"}.get(
                int(encoding), "PCM")
            return AudioMetaData(int(sample_rate), int(frames), int(channels),
                                 int(bits), name)
        except Exception:
            pass
    backend = get_audio_backend()
    if backend == "soundfile":
        import soundfile as sf
        try:
            si = sf.info(filepath)
        except Exception as err:
            try:
                import av  # noqa: F401
            except ImportError:
                raise err
            return _av_info(filepath)
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


def encode_wav(data, sample_rate, bits=16):
    """Encodes a float32 (channels, time) tensor into WAV bytes (native codec).

    Returns a uint8 1-D tensor holding the RIFF/WAVE payload; ``bits`` may be
    8, 16, 24 or 32 (PCM).  Raises ImportError when the codec is not compiled
    into this build.
    """
    native = _native_io()
    encoder = getattr(native, "encode_wav", None) if native else None
    if encoder is None:
        raise ImportError("The native WAV encoder is not available in this build.")
    return encoder(data, int(sample_rate), int(bits))


def wav_info(data):
    """Returns metadata for WAV bytes: (sample_rate, frames, channels, bits, encoding).

    ``encoding`` is the numeric WAV format tag (1 = PCM, 3 = float,
    6 = A-law, 7 = u-law).
    """
    native = _native_io()
    info_fn = getattr(native, "wav_info", None) if native else None
    if info_fn is None:
        raise ImportError("The native WAV decoder is not available in this build.")
    return info_fn(data)
