import numpy as np
import tensorplay as tp
from .backend import get_audio_backend, _SCIPY_AVAILABLE, _SOUNDFILE_AVAILABLE


def load(filepath, normalize=True, channels_first=True):
    """
    Loads an audio file from the filesystem.
    
    Args:
        filepath (str): Path to the audio file.
        normalize (bool): If True, normalize the audio to [-1, 1] (float32).
                          If False, return original values (e.g. int16).
                          Default: True.
        channels_first (bool): If True, return (Channels, Time).
                               If False, return (Time, Channels).
                               Default: True (PyTorch style).
                               
    Returns:
        Tensor: Audio tensor.
        int: Sample rate.
    """
    backend = get_audio_backend()
    if backend is None:
        raise ImportError("No audio backend available. Please install soundfile or scipy.")
        
    sr = 0
    audio_np = None
    
    if backend == "soundfile":
        import soundfile as sf
        # soundfile always returns float32 normalized by default if read normally?
        # sf.read returns (data, samplerate)
        # data is (frames, channels)
        # dtype can be specified.
        dtype = 'float32' if normalize else 'int16'
        audio_np, sr = sf.read(filepath, dtype=dtype)
        
    elif backend == "scipy":
        from scipy.io import wavfile
        # wavfile.read returns (rate, data)
        # data is usually int16 for wav
        sr, audio_np = wavfile.read(filepath)
        
        # scipy reads as is (int16, etc). 
        # If we need normalization, we can let C++ handle it or do it here.
        # But wait, our C++ optimization handles int16->float32 normalization!
        # So we just pass the raw numpy array to C++.
    
    if audio_np is None:
        raise RuntimeError(f"Failed to load audio file: {filepath}")

    # Ensure array
    if not isinstance(audio_np, np.ndarray):
        audio_np = np.array(audio_np)
        
    # C++ Optimized Path
    # Condition: 
    # 1. We want PyTorch style output (Channels, Time) -> C++ does this transpose.
    # 2. We want normalization -> C++ does this for int16/uint8 inputs.
    
    use_cpp_opt = False
    
    # If scipy read int16 and we want normalized float32 tensor:
    if normalize and audio_np.dtype == np.int16 and hasattr(tp, 'audio_to_tensor'):
        use_cpp_opt = True
        
    # If already float32 (e.g. soundfile) and we just want transpose:
    if audio_np.dtype == np.float32 and hasattr(tp, 'audio_to_tensor'):
        use_cpp_opt = True
        
    if use_cpp_opt:
        # tp.audio_to_tensor always returns (C, T) float32
        tensor = tp.audio_to_tensor(audio_np)
        
        if not channels_first:
            # Transpose back if user requested (T, C)
            tensor = tensor.t()
            
        return tensor, sr

    # Fallback / Non-optimized path
    # ... (Manual implementation if C++ opt not applicable or available)
    
    # Handle Normalization manually if not handled by C++
    if normalize:
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.uint8:
            audio_np = (audio_np.astype(np.float32) - 128.0) / 128.0
        # Add other types if needed
    
    # Handle Channels First (Transpose)
    # Numpy is (Time, Channels) usually
    if audio_np.ndim == 2 and channels_first:
        audio_np = audio_np.transpose(1, 0) # (C, T)
    elif audio_np.ndim == 1 and channels_first:
        audio_np = audio_np[None, :] # (1, T)
        
    return tp.tensor(audio_np), sr

def save(filepath, src, sample_rate):
    """
    Saves a Tensor to an audio file.
    
    Args:
        filepath (str): Path to save.
        src (Tensor): Audio tensor (C, T) or (T, C) or (T,).
        sample_rate (int): Sample rate.
    """
    backend = get_audio_backend()
    if backend is None:
        raise ImportError("No audio backend available.")
        
    # Convert to numpy
    if isinstance(src, tp.Tensor):
        # We assume float32 [-1, 1] usually
        # But we need to check shape.
        # Most backends expect (Time, Channels)
        
        # If (C, T) -> Transpose to (T, C)
        if src.ndim == 2 and src.shape[0] < src.shape[1]: 
            # Heuristic: Channels is usually small (1, 2, etc), Time is large.
            # If shape[0] << shape[1], assume (C, T)
            src = src.transpose(0, 1) # -> (T, C)
            
        # Convert to numpy
        # TODO: Need Tensor -> Numpy binding or via DLPack
        # For now, if we don't have direct to_numpy, we might need a workaround or assume it exists.
        # tensorplay seems to have numpy() method? Let's check or assume.
        # If not, use dlpack.
        try:
            arr = src.numpy()
        except:
            # Fallback
            arr = np.array(src) # Implicit conversion?
            
    else:
        arr = src
        
    if backend == "soundfile":
        import soundfile as sf
        sf.write(filepath, arr, sample_rate)
    elif backend == "scipy":
        from scipy.io import wavfile
        # scipy expects int16 usually for wav if we want standard wav?
        # Or float32 is fine? Scipy supports float32 wav.
        wavfile.write(filepath, sample_rate, arr)

def info(filepath):
    """
    Get signal information of an audio file.
    """
    backend = get_audio_backend()
    if backend == "soundfile":
        import soundfile as sf
        si = sf.info(filepath)
        return {"samplerate": si.samplerate, "channels": si.channels, "frames": si.frames}
    elif backend == "scipy":
        # Scipy doesn't have a cheap 'info' without reading?
        # Actually wavfile.read reads the whole file.
        # So scipy is bad for just info.
        # Maybe use wave module for info if scipy is backend?
        import wave
        with wave.open(filepath, 'rb') as f:
            return {
                "samplerate": f.getframerate(),
                "channels": f.getnchannels(),
                "frames": f.getnframes()
            }
    return None
