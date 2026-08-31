# mypy: allow-untyped-defs
"""Serialization reducers that move tensors between processes through shared memory.

Registering these reducers with the multiprocessing pickling machinery makes
every tensor sent through a queue, pipe, or connection hand over a shared
segment instead of a byte copy: the sender moves its data into memory visible
to other processes (either a named segment or a passed file descriptor), and
each recipient rebuilds a tensor aliasing those same pages.

Two sharing strategies are supported:

- ``file_system``: segments are identified by their name in the shared-memory
  filesystem, so any process that knows the name can attach to them.
- ``file_descriptor``: segments are anonymous ``memfd`` regions passed as
  duplicated descriptors over the connection, leaving no named entries behind.
"""

import mmap
import multiprocessing
import os
import threading
from multiprocessing import reduction
from multiprocessing.resource_tracker import unregister as _tracker_unregister
from multiprocessing.shared_memory import SharedMemory

import tensorplay
from tensorplay._C import DType

# Early load resource_sharer to prevent a partially initialized instance from
# being inherited in a forked child process. The descriptor-based reducer needs
# this module indirectly through DupFd(), and the built-in mp.Queue pickles
# arguments in a background thread which may overlap with the fork.
try:
    import multiprocessing.resource_sharer  # noqa: F401
except ImportError:
    pass

__all__ = ["init_reductions"]

_MEMFD_PREFIX = "tensorplay-mp-segment-"


class TensorWeakRef:
    r"""A cache entry for a tensor recently handed to another process.

    The reference is intentionally strong: it anchors the Python objects that
    keep a handed-over segment alive (the tensor, its memory view, and any
    descriptor it was created from) for as long as the cache holds the entry.
    ``releasers`` are no-argument callables run when the entry is dropped.
    """

    __slots__ = ["tensor", "_releasers"]

    def __init__(self, tensor, releasers=()):
        self.tensor = tensor
        self._releasers = list(releasers)

    def expired(self):
        return False

    def release(self):
        while self._releasers:
            try:
                self._releasers.pop()()
            except Exception:
                # Releasers may run during interpreter shutdown, when module
                # globals can already be torn down; never raise from here.
                pass

    def __del__(self):
        self.release()


class SharedCache(dict):
    """Dictionary from shared-segment handles to TensorWeakRef objects."""

    def __init__(self) -> None:
        # free_dead_references() is called if the len exceeds the current
        # limit. The limit scales with the number of remaining live objects.
        self.limit = 128
        # `fork` inherits lock state, so in case we fork while the lock is
        # held, we register a function to reset the lock to a new object in
        # the child to avoid possible deadlocks.
        self._after_fork()
        multiprocessing.util.register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):  # type: ignore[override]
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            if len(self) > self.limit:
                self.free_dead_references()

    def free_dead_references(self):
        live = 0
        for key, storage_ref in list(self.items()):
            if storage_ref.expired():
                del self[key]
            else:
                live += 1
        # Entries hold strong references, so they only leave the cache when
        # evicted. Dropping a named-segment entry is safe: the segment name
        # stays valid while the resource tracker keeps it registered for this
        # process. Dropping a descriptor entry is safe too: the recipient's
        # duplicated descriptor is held by the resource sharer until detach.
        if live > self.limit:
            excess = live - self.limit // 2
            for key in list(self.keys())[:excess]:
                self.pop(key).release()
            live -= excess
        self.limit = max(128, live * 2)


# mapping from segment handles to TensorWeakRef objects
shared_cache = SharedCache()


def fd_id(fd):
    # Returns a tuple which uniquely identifies a file descriptor. In Mac OS,
    # this doesn't work with shared memory handles, which is why we don't
    # support the "file_descriptor" sharing method on that platform.
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def rebuild_event(device, handle):
    return tensorplay.cuda.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def rebuild_empty_tensor(cls, shape, dtype, requires_grad):
    # Zero-element tensors cannot back a shared segment (there are no bytes to
    # map), so they are rebuilt from metadata alone. Strides carry no
    # information without elements.
    t = tensorplay.empty(shape, dtype=dtype)
    t.requires_grad = requires_grad
    return t


def rebuild_shared_tensor(
    cls, name, shape, strides, dtype, device_type, device_index, requires_grad
):
    # Attach to the named segment and rebuild a tensor aliasing its pages.
    shm = SharedMemory(name=name, create=False)
    # The attaching process must not unlink the segment when it exits: the
    # sender and any further receivers still need the name to stay valid.
    try:
        _tracker_unregister(shm._name, "shared_memory")
    except (ValueError, KeyError):
        pass
    tensor = cls.__new__(cls)
    tensor.__setstate__(
        ("shm", shm, shape, strides, dtype, device_type, device_index, requires_grad)
    )
    return tensor


def rebuild_shared_tensor_fd(cls, df, nbytes, shape, dtype, requires_grad):
    # Map the received descriptor and rebuild a tensor aliasing the same
    # pages the sender filled. The mapping keeps the segment alive after the
    # descriptor is closed.
    fd = df.detach()
    try:
        view = mmap.mmap(fd, nbytes)
    finally:
        os.close(fd)
    tensor = tensorplay.frombuffer(view, dtype=DType(dtype))
    tensor = tensor.reshape(shape)
    tensor.requires_grad = requires_grad
    return tensor


def _reduce_tensor_shared(tensor):
    if not tensor.is_shared():
        if tensor.is_contiguous() and tensor.storage_offset() == 0:
            # Move this tensor's storage into a named shared segment in
            # place; the sending process keeps reading and writing the very
            # same pages the recipients will attach to.
            tensor.share_memory_()
        else:
            # Only contiguous, unoffset storage can be moved in place; share
            # a compacted copy of the data instead.
            compact = tensor.contiguous()
            if compact.storage_offset() != 0:
                compact = compact.clone()
            compact.share_memory_()
            tensor = compact
    state = tensor.__getstate__()
    shm = state[1]
    shared_cache[shm.name] = TensorWeakRef(tensor)
    return (
        rebuild_shared_tensor,
        (type(tensor), shm.name) + tuple(state[2:]),
    )


def _reduce_tensor_fd(tensor):
    # Hand the segment over as a duplicated descriptor. The sender fills a
    # private memfd-backed region; the recipient maps the received descriptor
    # and rebuilds a tensor aliasing those pages. The sending tensor itself
    # keeps its original (private) storage.
    shape = tuple(tensor.shape)
    nbytes = tensor.numel() * tensor.itemsize()
    fd = os.memfd_create(_MEMFD_PREFIX + str(os.getpid()), flags=os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd, nbytes)
        view = mmap.mmap(fd, nbytes)
        host = tensorplay.frombuffer(view, dtype=tensor.dtype)
        host = host.reshape(shape)
        host.copy_(tensor)
    except BaseException:
        os.close(fd)
        raise
    releasers = [lambda: os.close(fd)]
    cache_key = fd_id(fd)
    shared_cache[cache_key] = TensorWeakRef(host, releasers)
    df = reduction.DupFd(fd)
    return (
        rebuild_shared_tensor_fd,
        (
            type(tensor),
            df,
            nbytes,
            shape,
            int(tensor.dtype),
            bool(tensor.requires_grad),
        ),
    )


def reduce_tensor(tensor):
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    if hasattr(tensor, "_backward_hooks"):
        from tensorplay.utils.hooks import warn_if_has_hooks

        warn_if_has_hooks(tensor)

    if tensor.device.type != "cpu":
        raise RuntimeError(
            f"Cannot pickle {tensor.device.type} tensor: only CPU tensors "
            "support cross-process shared memory"
        )

    if tensor.numel() == 0:
        return (
            rebuild_empty_tensor,
            (
                type(tensor),
                tuple(tensor.shape),
                tensor.dtype,
                bool(tensor.requires_grad),
            ),
        )

    from . import get_sharing_strategy

    if get_sharing_strategy() == "file_system" or tensor.is_shared():
        return _reduce_tensor_shared(tensor)
    return _reduce_tensor_fd(tensor)


def init_reductions():
    reduction.register(tensorplay.Tensor, reduce_tensor)

    try:
        from tensorplay._C import TensorBase

        if TensorBase is not tensorplay.Tensor:
            reduction.register(TensorBase, reduce_tensor)
    except ImportError:
        pass

    try:
        from tensorplay.nn.parameter import Parameter

        reduction.register(Parameter, reduce_tensor)
    except ImportError:
        pass

    try:
        cuda_module = getattr(tensorplay, "cuda", None)
        event_cls = getattr(cuda_module, "Event", None)
        if event_cls is not None and hasattr(event_cls, "ipc_handle"):
            reduction.register(event_cls, reduce_event)
    except ImportError:
        pass
