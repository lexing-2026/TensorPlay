import sys
import os

# Try toimport torch first to avoid DLL conflicts with system cuDNN loaded by TensorPlay
try:
    import torch
except ImportError:
    pass

# Add project root to path so tests can find tensorplay
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
