# Backend availability
try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

# Global backend preference
# Options: "PIL", "OPENCV"
# Default to OPENCV if available, then PIL
if _OPENCV_AVAILABLE:
    _BACKEND = "OPENCV"
else:
    _BACKEND = "PIL"

def set_backend(backend):
    """
    Set the global image processing backend.
    Args:
        backend (str): "PIL" or "OPENCV"
    """
    global _BACKEND
    if backend == "OPENCV" and not _OPENCV_AVAILABLE:
        raise ImportError("OpenCV is not available. Please install opencv-python.")
    
    valid_backends = ["PIL", "OPENCV"]
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}. Available: {valid_backends}")
        
    _BACKEND = backend

def get_backend():
    """Get the current image processing backend."""
    return _BACKEND

def is_opencv_available():
    return _OPENCV_AVAILABLE
