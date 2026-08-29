import multiprocessing
import tensorplay
from multiprocessing import *
from multiprocessing.reduction import ForkingPickler

# This module wraps python's multiprocessing to provide support for
# shared memory passing of Tensor objects. 

def reduce_tensor(t):
    # Use the bytes-based (data copy) state: the SharedMemory zero-copy
    # branch mutates the source tensor via share_memory_() and cannot be
    # reconstructed reliably in forked children (empty tensor / segfault).
    if t.is_shared():
        t = t.clone()
    # Use the __newobj__ protocol (NEWOBJ + BUILD) which reconstructs via
    # __setstate__: the plain (cls, (), state) REDUCE form creates an empty
    # tensor and the setstate replacement is lost.
    import copyreg
    return copyreg.__newobj__, (t.__class__,), t.__getstate__()

# Register for TensorBase too since Tensor is an alias
try:
    from tensorplay._C import TensorBase
    ForkingPickler.register(TensorBase, reduce_tensor)
except ImportError:
    pass

ForkingPickler.register(tensorplay.Tensor, reduce_tensor)

__all__ = ['get_context', 'Queue', 'Event', 'Process', 'current_process', 'active_children']

def get_context(method=None):
    return multiprocessing.get_context(method)
