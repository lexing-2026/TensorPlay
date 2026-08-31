"""Process-group operations, rendezvous, and tensor marshaling.

Collectives dispatch on the process group backend: NCCL for CUDA groups,
gloo (and MPI) for CPU groups. Backends live in the C++ layer under
``tensorplay._C._distributed``.
"""

from __future__ import annotations

import datetime as _dt
import io
import os
import pickle
import threading
import warnings
from typing import List, Optional, Sequence

import tensorplay as tp
from tensorplay._C import _distributed as _C

__all__ = [
    "Backend",
    "GroupMember",
    "ProcessGroup",
    "ReduceOp",
    "Work",
    "group",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_into_tensor",
    "all_gather_object",
    "all_reduce",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "_allgather_base",
    "_reduce_scatter_base",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "broadcast_object_list",
    "destroy_process_group",
    "gather",
    "gather_object",
    "get_backend",
    "get_global_rank",
    "get_group_rank",
    "get_process_group_ranks",
    "get_rank",
    "get_world_size",
    "GradBucket",
    "init_process_group",
    "irecv",
    "is_available",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "monitored_barrier",
    "new_group",
    "new_subgroups",
    "new_subgroups_by_enumeration",
    "P2POp",
    "recv",
    "recv_object_list",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_coalesced",
    "reduce_scatter_tensor",
    "scatter",
    "scatter_object_list",
    "send",
    "send_object_list",
    "isend",
]

default_pg_timeout = _dt.timedelta(minutes=30)


class Backend:
    UNDEFINED = "undefined"
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"

    BACKENDS = [GLOO, MPI, NCCL]
    # ships the NCCL path so the remaining entries are absent.
    BACKEND_TO_MAP = {NCCL: None}


class ReduceOp:
    SUM = 0
    PRODUCT = PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4


class GroupMember:
    NON_GROUP_MEMBER = -100
    WORLD: Optional["ProcessGroup"] = None


#: default process group after ``init_process_group``.
group = GroupMember


class Work:
    """Represent asynchronous collective work and its completion future.

    ``get_future()`` returns a ``tensorplay.futures.Future``
    resolving to the list of output tensors once the collective completes.
    """

    def __init__(self, event, done=None, tensors=None) -> None:
        self._event = event
        self._done = done
        self._tensors = list(tensors) if tensors is not None else []

    def is_completed(self) -> bool:
        return bool(self._event.query())

    def wait(self, timeout: Optional[_dt.timedelta] = None) -> bool:
        self._event.synchronize()
        if self._done is not None:
            self._done()
        return True

    def get_future(self):
        from tensorplay import futures as _futures

        fut = _futures.Future()

        def _completer():
            try:
                self.wait()
                fut.set_result(self._result_tensors())
            except Exception as e:
                fut.set_exception(e)

        fut._completer = _completer
        return fut

    def _result_tensors(self):
        return list(self._tensors)


class ProcessGroup:
    def __init__(self, ranks: List[int], group_name: str,
                 backend: str = Backend.UNDEFINED) -> None:
        self.ranks = list(ranks)
        self.group_name = group_name
        self.backend = backend
        # NCCL communicator (cuda backends) / C++ process group (gloo, mpi).
        self.comm: Optional[int] = None
        self.gloo_pg = None
        self.mpi_pg = None
        self._timeout_s: float = default_pg_timeout.total_seconds()
        self._lock = threading.Lock()

    def size(self) -> int:
        return len(self.ranks)

    def rank(self) -> int:
        return self.group_rank(_global_rank())

    def group_size(self) -> int:
        return len(self.ranks)

    def name(self) -> str:
        return self.group_name

    def __repr__(self) -> str:
        if self.comm is not None:
            return f"{self.group_name}:{hex(id(self))}"
        else:
            return f"{self.group_name} (uninitialized)"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def global_rank(self, group_rank: int) -> int:
        return self.ranks[group_rank]

    def group_rank(self, global_rank: int) -> int:
        return self.ranks.index(global_rank)


Backend.BACKEND_TO_MAP[Backend.NCCL] = ProcessGroup
if hasattr(_C, "ProcessGroupGloo"):
    Backend.BACKEND_TO_MAP[Backend.GLOO] = _C.ProcessGroupGloo
if hasattr(_C, "ProcessGroupMPI"):
    Backend.BACKEND_TO_MAP[Backend.MPI] = _C.ProcessGroupMPI


_world_group: Optional[ProcessGroup] = None
_groups: dict[str, ProcessGroup] = {}
_group_count = 0
_backend = ""
_store_ref = [None]
_rank_state = [0]

_uninitialized_msg = (
    "Default process group has not been initialized, please make sure to call "
    "init_process_group."
)
_invalid_group_msg = "Invalid process group specified"


def is_available() -> bool:
    return _C.is_available()


def is_nccl_available() -> bool:
    return _C.is_available()


def is_gloo_available() -> bool:
    return hasattr(_C, "ProcessGroupGloo")


def is_mpi_available() -> bool:
    return hasattr(_C, "ProcessGroupMPI")


def is_initialized() -> bool:
    return _world_group is not None


def _check_default_pg() -> ProcessGroup:
    if _world_group is None:
        raise RuntimeError(_uninitialized_msg)
    return _world_group


def _resolve_group(group) -> ProcessGroup:
    if group is None:
        return _check_default_pg()
    if isinstance(group, ProcessGroup):
        return group
    if isinstance(group, str):
        # wrappers that stash ``pg.group_name`` in the context).
        if _world_group is not None and group == _world_group.group_name:
            return _world_group
        pg = _groups.get(group)
        if pg is not None:
            return pg
    raise ValueError(_invalid_group_msg)


def get_rank(group=None) -> int:
    pg = _resolve_group(group)
    return pg.group_rank(_global_rank())


def get_world_size(group=None) -> int:
    return _resolve_group(group).size()


def get_backend(group=None) -> str:
    return _resolve_group(group).backend


def _global_rank() -> int:
    return _rank_state[0]


def _current_store():
    if _store_ref[0] is None:
        raise RuntimeError("No store available; was init_process_group called?")
    return _store_ref[0]


#: Process-level default transport device for the gloo backend.
_gloo_device = [None]


def _default_gloo_device():
    if _gloo_device[0] is None:
        _gloo_device[0] = _C.ProcessGroupGloo.create_default_device()
    return _gloo_device[0]


def _ensure_gloo_comm(pg: ProcessGroup) -> object:
    with pg._lock:
        if pg.gloo_pg is not None:
            return pg.gloo_pg
        from ._store import PrefixStore

        opts = _C.GlooOptions()
        opts.threads = 2
        opts.group_name = pg.group_name
        opts.add_device(_default_gloo_device())
        # Prefixing keeps independent groups from reading each other's
        # rendezvous entries in a shared store.
        store = PrefixStore(f"tp_gloo/{pg.group_name}", _current_store())
        pg.gloo_pg = _C.ProcessGroupGloo(
            store=store,
            rank=pg.group_rank(_global_rank()),
            size=pg.size(),
            options=opts,
        )
        return pg.gloo_pg


def _ensure_mpi_comm(pg: ProcessGroup) -> object:
    with pg._lock:
        if pg.mpi_pg is not None:
            return pg.mpi_pg
        # The C++ side derives the communicator from the rank list (empty
        # ranks resolve to the world communicator).
        pg.mpi_pg = _C.ProcessGroupMPI.create(pg.ranks)
        return pg.mpi_pg


def _cpu_pg(pg: ProcessGroup):
    """Returns the C++ process group for CPU backends (gloo / MPI)."""
    if pg.backend == Backend.GLOO:
        return _ensure_gloo_comm(pg)
    if pg.backend == Backend.MPI:
        return _ensure_mpi_comm(pg)
    raise RuntimeError(f"Unknown backend: '{pg.backend}'")


def _cpu_timeout_ms(pg: ProcessGroup) -> int:
    return int(pg._timeout_s * 1000)


class _BackendWork(Work):
    """Work handle wrapping a C++ backend work object (gloo / MPI)."""

    def __init__(self, work, done=None, tensors=None) -> None:
        super().__init__(None, done=done, tensors=tensors)
        self._work = work

    def is_completed(self) -> bool:
        return bool(self._work.is_completed())

    def wait(self, timeout: Optional[_dt.timedelta] = None) -> bool:
        self._work.wait(-1)
        if self._done is not None:
            self._done()
        return True


class _ChainedWork(_BackendWork):
    """Completes when every underlying backend work has completed."""

    def __init__(self, works, done=None, tensors=None) -> None:
        super().__init__(None, done=done, tensors=tensors)
        self._works = list(works)

    def is_completed(self) -> bool:
        return all(w.is_completed() for w in self._works)

    def wait(self, timeout: Optional[_dt.timedelta] = None) -> bool:
        for w in self._works:
            w.wait(-1)
        if self._done is not None:
            self._done()
        return True


def _cpu_finish(work, tensors=None, restore=None, extra=None,
                async_op: bool = False):
    if async_op:
        def done():
            work.wait(-1)
            if restore is not None:
                restore()
            if extra is not None:
                extra()
        return _BackendWork(work, done=done, tensors=tensors)
    work.wait(-1)
    if restore is not None:
        restore()
    if extra is not None:
        extra()
    return None


def _ensure_comm(pg: ProcessGroup, timeout_s: float) -> int:
    if pg.backend != Backend.NCCL:
        _cpu_pg(pg)
        return None
    with pg._lock:
        if pg.comm is not None:
            return pg.comm
        _apply_nccl_autotune()
        _select_nccl_device(pg)
        store = _current_store()
        key = f"tensorplay_distributed/nccl_unique_id/{pg.group_name}"
        if _global_rank() == pg.ranks[0]:
            uid = _C.get_unique_id()
            store.set(key, uid.hex())
        else:
            uid = bytes.fromhex(store.get(key, timeout=timeout_s).decode())
        pg.comm = _C.comm_init_rank(pg.group_rank(_global_rank()), pg.size(), uid)
        return pg.comm


_NCCL_AUTOTUNED = False


def _apply_nccl_autotune() -> None:
    """Apply topology-aware NCCL defaults before the first communicator init.

    NCCL's auto-tuned channel grid under-utilizes intra-node GPUs that lack
    direct P2P (traffic is staged through host SHM); a wider channel grid
    raises large-message bandwidth by ~20% on such hosts. Only values the user
    has not explicitly set are filled in, so manual ``NCCL_*`` tuning always
    wins.
    """
    global _NCCL_AUTOTUNED
    if _NCCL_AUTOTUNED:
        return
    _NCCL_AUTOTUNED = True
    defaults = {
        "NCCL_MIN_NCHANNELS": "16",
        "NCCL_MAX_NCHANNELS": "16",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _select_nccl_device(pg) -> None:
    """Select a CUDA device that avoids rank collisions in one communicator.

    NCCL forbids two ranks of
    one communicator on the same device, and a fresh process defaults to
    cuda:0, so single-node multi-rank runs collide unless the rank selects
    A rank that already pinned a device (or world_size == 1) is untouched.
    """
    try:
        from tensorplay.cuda import current_device, device_count, set_device
        if device_count() <= 1 or current_device() != 0:
            return
        rank = _global_rank()
        if pg.size() <= 1 or rank >= device_count():
            return
        import os
        idx = int(os.environ.get("LOCAL_RANK", rank % device_count()))
        if idx != 0:
            set_device(idx)
    except Exception:
        pass


def _get_process_group_name(pg) -> str:
    return getattr(pg, "group_name", "")

def _get_process_group_store(pg):
    return _current_store()

def _parse_init_method(init_method: str, rank: int, world_size: int,
                       timeout_s: float):
    from ._store import FileStore, TCPStore

    if init_method.startswith("env://"):
        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")
        if not master_addr or not master_port:
            raise RuntimeError(
                "env:// init_method requires MASTER_ADDR and MASTER_PORT "
                "environment variables"
            )
        return TCPStore(master_addr, int(master_port), world_size,
                        is_master=(rank == 0), timeout=timeout_s)
    if init_method.startswith("file://"):
        return FileStore(init_method[len("file://"):], world_size)
    if init_method.startswith("tcp://"):
        host, _, port = init_method[len("tcp://"):].rpartition(":")
        return TCPStore(host, int(port), world_size, is_master=(rank == 0),
                        timeout=timeout_s)
    raise ValueError(f"Unsupported init_method: {init_method}")


def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    world_size: int = -1,
    rank: int = -1,
    store=None,
    group_name: str = "",
    timeout: _dt.timedelta = default_pg_timeout,
) -> None:
    global _world_group, _backend, _group_count

    if _world_group is not None:
        raise RuntimeError("trying to initialize the default process group twice!")
    if backend is None:
        backend = "nccl"
    backend = str(backend).lower()
    if backend == Backend.NCCL:
        if not is_available():
            raise RuntimeError(
                "Distributed package is not available (NCCL library could "
                "not be loaded)"
            )
    elif backend == Backend.GLOO:
        if not is_gloo_available():
            raise RuntimeError(
                "The gloo backend was not compiled into this build"
            )
    elif backend == Backend.MPI:
        if not is_mpi_available():
            raise RuntimeError(
                "The MPI backend was not compiled into this build; MPI is "
                "only included when the host has an MPI runtime"
            )
    else:
        raise ValueError(
            f"Invalid backend: '{backend}'. TensorPlay currently supports: "
            f"'{Backend.NCCL}', '{Backend.GLOO}', '{Backend.MPI}'"
        )

    timeout_s = timeout.total_seconds()
    if init_method is None:
        init_method = "env://"

    if store is None:
        # falls back to RANK/WORLD_SIZE environment variables.
        from .rendezvous import rendezvous

        store, rank, world_size = next(rendezvous(init_method, rank, world_size,
                                                  timeout=timeout))
    else:
        if rank < 0 or world_size < 0:
            raise RuntimeError(
                "rank and world_size must be provided when a store is given"
            )

    _rank_state[0] = rank
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    _store_ref[0] = store
    _backend = backend
    _group_count += 1
    _world_group = ProcessGroup(list(range(world_size)), group_name or "0",
                                backend=backend)
    _world_group._timeout_s = timeout_s
    GroupMember.WORLD = _world_group
    _ensure_comm(_world_group, timeout_s)


def destroy_process_group(group=None) -> None:
    global _world_group, _backend
    if group is None:
        groups = ([_world_group] if _world_group is not None else []) + list(
            _groups.values()
        )
        for pg in groups:
            if pg.comm is not None:
                _C.comm_destroy(pg.comm)
                pg.comm = None
            pg.gloo_pg = None
            pg.mpi_pg = None
        _world_group = None
        GroupMember.WORLD = None
        _groups.clear()
        _store_ref[0] = None
        _backend = ""
        return
    pg = _resolve_group(group)
    if pg.comm is not None:
        _C.comm_destroy(pg.comm)
        pg.comm = None
    pg.gloo_pg = None
    pg.mpi_pg = None
    if pg is _world_group:
        _world_group = None
        GroupMember.WORLD = None
        _groups.clear()
        _store_ref[0] = None
        _backend = ""


def new_group(ranks: Optional[List[int]] = None,
              timeout: _dt.timedelta = default_pg_timeout,
              backend: Optional[str] = None, pg_options=None):
    global _group_count
    _check_default_pg()
    if backend is None:
        backend = _backend
    else:
        backend = str(backend).lower()
        if backend not in (Backend.NCCL, Backend.GLOO, Backend.MPI):
            raise ValueError(
                f"Invalid backend: '{backend}'. TensorPlay currently supports: "
                f"'{Backend.NCCL}', '{Backend.GLOO}', '{Backend.MPI}'"
            )
        if backend == Backend.NCCL and not is_available():
            raise RuntimeError("NCCL library could not be loaded")
        if backend == Backend.GLOO and not is_gloo_available():
            raise RuntimeError("The gloo backend was not compiled into this build")
        if backend == Backend.MPI and not is_mpi_available():
            raise RuntimeError("The MPI backend was not compiled into this build")
    if ranks is None:
        ranks = list(range(get_world_size()))
    ranks = sorted(set(ranks))
    if _global_rank() not in ranks:
        return GroupMember.NON_GROUP_MEMBER
    _group_count += 1
    pg = ProcessGroup(ranks, group_name=str(_group_count), backend=backend)
    pg._timeout_s = timeout.total_seconds()
    _groups[pg.group_name] = pg
    _ensure_comm(pg, pg._timeout_s)
    return pg


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------
def _device_index_of(t: tp.Tensor) -> int:
    idx = t.device.index
    if idx < 0:
        idx = tp.cuda.current_device()
    return idx


def _contiguous_view(t: tp.Tensor):
    """Return (buffer, restore) aliasing ``t`` when contiguous.

    Works on ``TensorBase`` (factory outputs), which lacks ``.contiguous()``:
    a strided tensor is copied through a fresh dense buffer instead.
    """
    if t.is_contiguous():
        return t, None
    buf = tp.zeros(t.shape, dtype=t.dtype, device=t.device)
    buf.copy_(t)
    return buf, (lambda: t.copy_(buf))


def _finish(event, restore=None, extra=None, async_op: bool = False):
    if async_op:
        def done():
            event.synchronize()
            if restore is not None:
                restore()
            if extra is not None:
                extra()
        return Work(event, done=done)
    event.synchronize()
    if restore is not None:
        restore()
    if extra is not None:
        extra()
    return None


def _finish_with_tensors(event, tensors, restore=None, extra=None,
                         async_op: bool = False):
    """Like :func:`_finish` but attaches output tensors for ``get_future``."""
    if async_op:
        def done():
            event.synchronize()
            if restore is not None:
                restore()
            if extra is not None:
                extra()
        return Work(event, done=done, tensors=tensors)
    event.synchronize()
    if restore is not None:
        restore()
    if extra is not None:
        extra()
    return None


def broadcast(tensor: tp.Tensor, src: int, group=None, async_op: bool = False):
    pg = _resolve_group(group)
    if pg.backend != Backend.NCCL:
        buf, restore = _contiguous_view(tensor)
        work = _cpu_pg(pg).broadcast(
            [buf], pg.group_rank(src), 0, _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=[tensor], restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, restore = _contiguous_view(tensor)
    # NCCL roots are communicator-relative (= group rank).
    root = pg.group_rank(src)
    _C.broadcast(buf, root, comm)
    return _finish_with_tensors(_record_event(_device_index_of(tensor)),
                                [tensor], restore=restore, async_op=async_op)


def all_reduce(tensor: tp.Tensor, op: int = ReduceOp.SUM, group=None,
               async_op: bool = False):
    pg = _resolve_group(group)
    if pg.backend != Backend.NCCL:
        buf, restore = _contiguous_view(tensor)
        work = _cpu_pg(pg).allreduce([buf], int(op), _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=[tensor], restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, restore = _contiguous_view(tensor)
    _C.all_reduce(buf, int(op), comm)
    return _finish_with_tensors(_record_event(_device_index_of(tensor)),
                                [tensor], restore=restore, async_op=async_op)


def reduce(tensor: tp.Tensor, dst: int, op: int = ReduceOp.SUM, group=None,
           async_op: bool = False):
    pg = _resolve_group(group)
    if pg.backend != Backend.NCCL:
        buf, restore = _contiguous_view(tensor)
        work = _cpu_pg(pg).reduce(
            [buf], pg.group_rank(dst), 0, int(op), _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=[tensor], restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, restore = _contiguous_view(tensor)
    _C.reduce(buf, int(op), pg.group_rank(dst), comm)
    return _finish_with_tensors(_record_event(_device_index_of(tensor)),
                                [tensor], restore=restore, async_op=async_op)


def all_gather(tensor_list: List[tp.Tensor], tensor: tp.Tensor, group=None,
               async_op: bool = False):
    pg = _resolve_group(group)
    if len(tensor_list) != pg.size():
        raise RuntimeError(
            f"Number of tensors in tensor_list ({len(tensor_list)}) does not match "
            f"the group world size ({pg.size()})"
        )
    if pg.backend != Backend.NCCL:
        send_t, restore = _contiguous_view(tensor)
        work = _cpu_pg(pg).allgather(
            [list(tensor_list)], [send_t], _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=tensor_list, restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    n = tensor.numel()
    out = tp.zeros(pg.size() * n, dtype=tensor.dtype,
                   device=tensor.device)
    send_t, restore = _contiguous_view(tensor)
    _C.all_gather(out, send_t, comm)

    def _split():
        for i, t in enumerate(tensor_list):
            t.copy_(out[i * n : (i + 1) * n].view(t.shape))

    return _finish_with_tensors(_record_event(_device_index_of(tensor)),
                                tensor_list, restore=restore,
                                extra=_split, async_op=async_op)


def gather(tensor: tp.Tensor, gather_list: Optional[List[tp.Tensor]] = None,
           dst: int = 0, group=None, async_op: bool = False):
    pg = _resolve_group(group)
    my_group_rank = pg.group_rank(_global_rank())
    if pg.backend != Backend.NCCL:
        if my_group_rank == dst and gather_list is None:
            raise RuntimeError("gather_list must be specified on the destination rank")
        send_t, restore = _contiguous_view(tensor)
        outputs = [list(gather_list)] if my_group_rank == dst else []
        work = _cpu_pg(pg).gather(outputs, [send_t], dst, _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=gather_list or [], restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    recv_obj = None
    if my_group_rank == dst:
        if gather_list is None:
            raise RuntimeError("gather_list must be specified on the destination rank")
        n = tensor.numel()
        recv_obj = tp.zeros(pg.size() * n, dtype=tensor.dtype, device=tensor.device)
    send_t, restore = _contiguous_view(tensor)
    _C.gather(recv_obj, send_t, dst, comm)
    n = tensor.numel()

    def _split():
        if my_group_rank != dst:
            return
        for i, t in enumerate(gather_list):
            t.copy_(recv_obj[i * n : (i + 1) * n].view(t.shape))

    return _finish_with_tensors(_record_event(_device_index_of(tensor)),
                                gather_list or [], restore=restore,
                                extra=_split, async_op=async_op)


def scatter(tensor: tp.Tensor, scatter_list: Optional[List[tp.Tensor]] = None,
            src: int = 0, group=None, async_op: bool = False):
    pg = _resolve_group(group)
    my_group_rank = pg.group_rank(_global_rank())
    if pg.backend != Backend.NCCL:
        if my_group_rank == src and scatter_list is None:
            raise RuntimeError("scatter_list must be specified on the source rank")
        inputs = []
        if my_group_rank == src:
            inputs = [list(scatter_list)]
        recv_t, restore = _contiguous_view(tensor)
        work = _cpu_pg(pg).scatter([recv_t], inputs, src, _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=[tensor], restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    send_obj = None
    if my_group_rank == src:
        if scatter_list is None:
            raise RuntimeError("scatter_list must be specified on the source rank")
        chunks = []
        for t in scatter_list:
            c, r = _contiguous_view(t)
            chunks.append(c.reshape(1, -1))
        send_obj = tp.cat(chunks, 0)
    recv_t, restore = _contiguous_view(tensor)
    _C.scatter(recv_t, send_obj, src, comm)

    return _finish_with_tensors(_record_event(_device_index_of(tensor)),
                                [tensor], restore=restore, async_op=async_op)


def reduce_scatter(output: tp.Tensor, input_list: List[tp.Tensor],
                   op: int = ReduceOp.SUM, group=None, async_op: bool = False):
    pg = _resolve_group(group)
    if len(input_list) != pg.size():
        raise RuntimeError(
            f"Number of tensors in input_list ({len(input_list)}) does not match "
            f"the group world size ({pg.size()})"
        )
    if pg.backend != Backend.NCCL:
        recv_t, restore = _contiguous_view(output)
        work = _cpu_pg(pg).reduce_scatter(
            [recv_t], [list(input_list)], int(op), _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=[output], restore=restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    chunks = []
    for t in input_list:
        c, _ = _contiguous_view(t)
        chunks.append(c.reshape(1, -1))
    send_t = tp.cat(chunks, 0)
    recv_t, restore = _contiguous_view(output)
    _C.reduce_scatter(recv_t, send_t, int(op), comm)

    return _finish_with_tensors(_record_event(_device_index_of(output)),
                                [output], restore=restore, async_op=async_op)


def isend(tensor: tp.Tensor, dst: int, group=None, tag: int = 0):
    """

    Returns a Work handle, or None if not part of the group.
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("isend")
        return None
    pg = _resolve_group(group)
    buf, _ = _contiguous_view(tensor)
    if pg.backend != Backend.NCCL:
        return _BackendWork(_cpu_pg(pg).send(
            [buf], pg.group_rank(dst), tag))
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    # NCCL p2p peers are communicator-relative (= group rank).
    _C.send(buf, pg.group_rank(dst), comm)
    return Work(_record_event(_device_index_of(tensor)))


def irecv(tensor: tp.Tensor, src: Optional[int] = None, group=None,
          tag: int = 0):
    """

    Returns a Work handle whose ``wait()`` completes the copy, or None if
    not part of the group.
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("irecv")
        return None
    pg = _resolve_group(group)
    if src is None and pg.backend != Backend.NCCL:
        buf, restore = _contiguous_view(tensor)
        work = _cpu_pg(pg).recv_anysource([buf], tag)
        return _BackendWork(work, done=restore)
    if src is None:
        raise RuntimeError(
            "TensorPlay does not support recv from any source yet; "
            "specify src"
        )
    buf, restore = _contiguous_view(tensor)
    if pg.backend != Backend.NCCL:
        return _BackendWork(_cpu_pg(pg).recv(
            [buf], pg.group_rank(src), tag), done=restore)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    _C.recv(buf, pg.group_rank(src), comm)
    event = _record_event(_device_index_of(tensor))
    return Work(event, done=restore)


def send(tensor: tp.Tensor, dst: int, group=None, tag: int = 0):
    work = isend(tensor, dst, group=group, tag=tag)
    if work is not None:
        work.wait()


def recv(tensor: tp.Tensor, src: Optional[int] = None, group=None, tag: int = 0):
    """Receives a tensor synchronously; returns the sender rank."""
    work = irecv(tensor, src=src, group=group, tag=tag)
    if work is None:
        return -1
    work.wait()
    return src


def barrier(group=None, async_op: bool = False, device_ids: Optional[List[int]] = None):
    pg = _resolve_group(group)
    if pg.backend != Backend.NCCL:
        work = _cpu_pg(pg).barrier(_cpu_timeout_ms(pg))
        return _cpu_finish(work, async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    device_index = device_ids[0] if device_ids else tp.cuda.current_device()
    flag = tp.zeros(1, dtype=tp.float32, device=f"cuda:{device_index}")
    _C.all_reduce(flag, ReduceOp.SUM, comm)
    return _finish(_record_event(device_index), None, async_op=async_op)


def _record_event(device_index: int):
    return tp.cuda.current_stream(device_index).record_event()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _get_default_group() -> ProcessGroup:
    return _check_default_pg()


def _rank_not_in_group(group) -> bool:
    if group is None:
        return not is_initialized()
    if group is GroupMember.NON_GROUP_MEMBER:
        return True
    if isinstance(group, ProcessGroup):
        if group is _world_group:
            return False
        return group.group_name not in _groups
    return True


def _warn_not_in_group(op_name: str) -> None:
    warnings.warn(
        f"{op_name} is not supported when the given group is not a sub-group of "
        "the default process group."
    )


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    if group is GroupMember.WORLD or (isinstance(group, ProcessGroup) and group is _world_group):
        return global_rank
    if not isinstance(group, ProcessGroup) or group.group_name not in _groups:
        raise ValueError(
            f"Group {group} is not registered, please create group with "
            "tensorplay.distributed.new_group API"
        )
    if global_rank not in group.ranks:
        raise ValueError(f"Global rank {global_rank} is not part of group {group}")
    return group.ranks.index(global_rank)


def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    if group is GroupMember.WORLD or (isinstance(group, ProcessGroup) and group is _world_group):
        return group_rank
    if not isinstance(group, ProcessGroup) or group.group_name not in _groups:
        raise ValueError(
            f"Group {group} is not registered, please create group with "
            "tensorplay.distributed.new_group API"
        )
    if group_rank < 0 or group_rank >= len(group.ranks):
        raise ValueError(f"Group rank {group_rank} is not part of group {group}")
    return group.ranks[group_rank]


def get_process_group_ranks(group) -> List[int]:
    return list((group or _get_default_group()).ranks)


def _validate_output_list_for_rank(my_rank: int, dst: int, gather_list) -> None:
    if dst < 0:
        raise ValueError("Invalid dst rank (-1)")
    if my_rank == dst and (gather_list is None or len(gather_list) == 0):
        raise ValueError("Argument gather_list must be specified on the dst rank")
    if my_rank != dst and gather_list is not None:
        raise ValueError(
            "Argument gather_list must NOT be specified on non-dst ranks"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def all_gather_into_tensor(output_tensor: tp.Tensor, input_tensor: tp.Tensor,
                           group=None, async_op: bool = False):
    """Gather tensors from all ranks into one flat output tensor.

    ``output_tensor`` must have ``world_size * input_tensor.numel()`` elements,
    """
    pg = _resolve_group(group)
    if pg.backend != Backend.NCCL:
        out_t, out_restore = _contiguous_view(output_tensor)
        in_t, in_restore = _contiguous_view(input_tensor)
        work = _cpu_pg(pg).all_gather_into_tensor(
            out_t, in_t, _cpu_timeout_ms(pg))

        def _restore():
            if out_restore is not None:
                out_restore()
            if in_restore is not None:
                in_restore()

        return _cpu_finish(work, tensors=[output_tensor], restore=_restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    expected = pg.size() * input_tensor.numel()
    if output_tensor.numel() != expected:
        raise RuntimeError(
            f"output_tensor has {output_tensor.numel()} elements but expected "
            f"{expected} (world_size {pg.size()} x input numel {input_tensor.numel()})"
        )
    out, out_restore = _contiguous_view(output_tensor)
    send_t, send_restore = _contiguous_view(input_tensor)
    _C.all_gather(out, send_t, comm)

    def _restore():
        if out_restore is not None:
            out_restore()
        if send_restore is not None:
            send_restore()

    return _finish(_record_event(_device_index_of(input_tensor)),
                   restore=_restore if (out_restore or send_restore) else None,
                   async_op=async_op)


_allgather_base = all_gather_into_tensor


def reduce_scatter_tensor(output: tp.Tensor, input: tp.Tensor,
                          op: int = ReduceOp.SUM, group=None,
                          async_op: bool = False):
    """

    ``input`` must have ``world_size * output.numel()`` elements.
    """
    pg = _resolve_group(group)
    expected = pg.size() * output.numel()
    if input.numel() != expected:
        raise RuntimeError(
            f"input has {input.numel()} elements but expected {expected} "
            f"(world_size {pg.size()} x output numel {output.numel()})"
        )
    if pg.backend != Backend.NCCL:
        out_t, out_restore = _contiguous_view(output)
        in_t, in_restore = _contiguous_view(input)
        work = _cpu_pg(pg).reduce_scatter_tensor(
            out_t, in_t, int(op), _cpu_timeout_ms(pg))

        def _restore():
            if out_restore is not None:
                out_restore()
            if in_restore is not None:
                in_restore()

        return _cpu_finish(work, tensors=[output], restore=_restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    recv_t, recv_restore = _contiguous_view(output)
    send_t, send_restore = _contiguous_view(input)
    _C.reduce_scatter(recv_t, send_t, int(op), comm)

    def _restore():
        if recv_restore is not None:
            recv_restore()
        if send_restore is not None:
            send_restore()

    return _finish(_record_event(_device_index_of(output)),
                   restore=_restore if (recv_restore or send_restore) else None,
                   async_op=async_op)


_reduce_scatter_base = reduce_scatter_tensor


# ---------------------------------------------------------------------------
# native ncclGroupStart/ncclGroupEnd primitive (_C.group_start/_C.group_end),
# exactly how ProcessGroupNCCL implements its *_coalesced entry points.
# ---------------------------------------------------------------------------
def all_reduce_coalesced(tensors: List[tp.Tensor], op: int = ReduceOp.SUM,
                         group=None, async_op: bool = False):
    """All-reduce a list of tensors in one coalesced (grouped) launch."""
    pg = _resolve_group(group)
    if pg.backend != Backend.NCCL:
        # CPU backends run one collective per tensor; the list form is a
        # convenience wrapper, not a fused exchange.
        works = []
        restores = []
        for t in tensors:
            buf, restore = _contiguous_view(t)
            restores.append(restore)
            works.append(_cpu_pg(pg).allreduce(
                [buf], int(op), _cpu_timeout_ms(pg)))

        def _restore():
            for r in restores:
                if r is not None:
                    r()

        return _cpu_finish(_ChainedWork(works), tensors=list(tensors),
                           restore=_restore, async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    restores = []
    _C.group_start()
    try:
        for t in tensors:
            buf, restore = _contiguous_view(t)
            restores.append(restore)
            _C.all_reduce(buf, int(op), comm)
    except BaseException:
        _C.group_end()
        raise
    _C.group_end()

    def _restore():
        for r in restores:
            if r is not None:
                r()

    dev = _device_index_of(tensors[0]) if tensors else tp.cuda.current_device()
    return _finish_with_tensors(_record_event(dev), list(tensors),
                                restore=_restore if any(restores) else None,
                                async_op=async_op)


def all_gather_coalesced(output_tensor_lists: List[List[tp.Tensor]],
                         input_tensor_list: List[tp.Tensor], group=None,
                         async_op: bool = False):
    """All-gather each input tensor into its own output list, in one coalesced
"""
    pg = _resolve_group(group)
    if len(output_tensor_lists) != len(input_tensor_list):
        raise ValueError(
            "output_tensor_lists and input_tensor_list must have the same length"
        )
    if pg.backend != Backend.NCCL:
        works = []
        for out_list, tensor in zip(output_tensor_lists, input_tensor_list):
            if len(out_list) != pg.size():
                raise RuntimeError(
                    f"output list length ({len(out_list)}) does not match the "
                    f"group world size ({pg.size()})"
                )
            send_t, _ = _contiguous_view(tensor)
            works.append(_cpu_pg(pg).allgather(
                [list(out_list)], [send_t], _cpu_timeout_ms(pg)))
        return _cpu_finish(_ChainedWork(works),
                           tensors=[t for ol in output_tensor_lists for t in ol],
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    flats = []
    restores = []
    _C.group_start()
    try:
        for out_list, tensor in zip(output_tensor_lists, input_tensor_list):
            if len(out_list) != pg.size():
                raise RuntimeError(
                    f"output list length ({len(out_list)}) does not match the "
                    f"group world size ({pg.size()})"
                )
            n = tensor.numel()
            out = tp.zeros(pg.size() * n, dtype=tensor.dtype,
                           device=tensor.device)
            send_t, restore = _contiguous_view(tensor)
            restores.append(restore)
            _C.all_gather(out, send_t, comm)
            flats.append((out, out_list, n))
    except BaseException:
        _C.group_end()
        raise
    _C.group_end()

    def _split():
        for out, out_list, n in flats:
            for i, t in enumerate(out_list):
                t.copy_(out[i * n:(i + 1) * n].view(t.shape))
        for r in restores:
            if r is not None:
                r()

    dev = (_device_index_of(input_tensor_list[0]) if input_tensor_list
           else tp.cuda.current_device())
    outs = [t for ol in output_tensor_lists for t in ol]
    return _finish_with_tensors(_record_event(dev), outs, extra=_split,
                                async_op=async_op)


def reduce_scatter_coalesced(output_tensor_list: List[tp.Tensor],
                             input_tensor_lists: List[List[tp.Tensor]],
                             op: int = ReduceOp.SUM, group=None,
                             async_op: bool = False):
    """Reduce-scatter each input tensor list into its output tensor, in one
"""
    pg = _resolve_group(group)
    if len(output_tensor_list) != len(input_tensor_lists):
        raise ValueError(
            "output_tensor_list and input_tensor_lists must have the same length"
        )
    if pg.backend != Backend.NCCL:
        works = []
        for output, inputs in zip(output_tensor_list, input_tensor_lists):
            if len(inputs) != pg.size():
                raise RuntimeError(
                    "input list length must equal the group world size"
                )
            out_t, _ = _contiguous_view(output)
            works.append(_cpu_pg(pg).reduce_scatter(
                [out_t], [list(inputs)], int(op), _cpu_timeout_ms(pg)))
        return _cpu_finish(_ChainedWork(works),
                           tensors=list(output_tensor_list),
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    restores = []
    _C.group_start()
    try:
        for output, in_list in zip(output_tensor_list, input_tensor_lists):
            if len(in_list) != pg.size():
                raise RuntimeError(
                    f"input list length ({len(in_list)}) does not match the "
                    f"group world size ({pg.size()})"
                )
            flat_in = tp.cat([t.reshape(-1) for t in in_list])
            send_t, s_restore = _contiguous_view(flat_in)
            recv_t, r_restore = _contiguous_view(output)
            restores.extend([s_restore, r_restore])
            _C.reduce_scatter(recv_t, send_t, int(op), comm)
    except BaseException:
        _C.group_end()
        raise
    _C.group_end()

    def _restore():
        for r in restores:
            if r is not None:
                r()

    dev = (_device_index_of(output_tensor_list[0]) if output_tensor_list
           else tp.cuda.current_device())
    return _finish_with_tensors(_record_event(dev), list(output_tensor_list),
                                restore=_restore if any(restores) else None,
                                async_op=async_op)


_mon_barrier_seq = [0]


def monitored_barrier(group=None, timeout=None, wait_all_ranks: bool = False):
    """

    Uses the rendezvous store to detect membership, then a real NCCL barrier
    to synchronize. Rank 0 monitors by default; ``wait_all_ranks=True`` makes
    every rank wait for every other rank.
    """
    pg = _resolve_group(group)
    if pg.backend == Backend.MPI:
        raise RuntimeError(
            "monitored_barrier is only supported by the gloo and nccl backends"
        )
    if pg.backend == Backend.GLOO:
        if timeout is None:
            timeout_s = default_pg_timeout.total_seconds()
        elif isinstance(timeout, _dt.timedelta):
            timeout_s = timeout.total_seconds()
        else:
            timeout_s = float(timeout)
        _cpu_pg(pg).monitored_barrier(int(timeout_s * 1000), wait_all_ranks)
        return
    if timeout is None:
        timeout_s = default_pg_timeout.total_seconds()
    elif isinstance(timeout, _dt.timedelta):
        timeout_s = timeout.total_seconds()
    else:
        timeout_s = float(timeout)
    store = _get_process_group_store(pg)
    _mon_barrier_seq[0] += 1
    prefix = f"_tp_monbar_{pg.group_name}_{_mon_barrier_seq[0]}"
    my_rank = pg.rank()
    store.set(f"{prefix}_rank_{my_rank}", "1")
    keys = [f"{prefix}_rank_{r}" for r in range(pg.size())]
    if my_rank == 0 or wait_all_ranks:
        if not store.wait(keys, timeout=timeout_s):
            missing = [r for r in range(pg.size())
                       if not store.wait([keys[r]], timeout=0.05)]
            raise RuntimeError(
                f"Timed out after {timeout_s}s in monitored_barrier waiting for "
                f"rank(s) {missing} to join the barrier."
            )
    barrier(group=group)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _get_object_coll_device(group=None) -> str:
    """Device for object collectives: CPU backends marshal on the CPU."""
    if group is not None or (
            _backend != "" and _backend != Backend.NCCL):
        return "cpu"
    try:
        return f"cuda:{tp.cuda.current_device()}"
    except Exception:
        return "cpu"


def _object_to_tensor(obj, device, group=None):
    f = io.BytesIO()
    pickle.Pickler(f).dump(obj)
    import numpy as np

    # np.frombuffer yields a read-only array (pickle bytes); tp tensors
    # require writable storage, hence the copy.
    byte_tensor = tp.as_tensor(
        np.frombuffer(f.getvalue(), dtype=np.uint8).copy(), dtype=tp.uint8
    ).to(device)
    local_size = tp.tensor([byte_tensor.numel()], dtype=tp.int64, device=device)
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size, group=None):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()
    size = tensor_size if isinstance(tensor_size, int) else int(tensor_size.item())
    return pickle.Unpickler(io.BytesIO(buf[:size])).load()


def all_gather_object(object_list, obj, group=None) -> None:
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_object")
        return

    current_device = _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    group_size = get_world_size(group=group)
    object_sizes_tensor = tp.zeros(group_size, dtype=tp.int64,
                                   device=current_device)
    object_size_list = [
        object_sizes_tensor[i : i + 1] for i in range(group_size)
    ]
    all_gather(object_size_list, local_size, group=group)
    max_object_size = max(int(s.item()) for s in object_size_list)
    # Pad input to the max size across all ranks (tp has no resize_).
    padded_input = tp.empty(max_object_size, dtype=tp.uint8,
                            device=current_device)
    padded_input[: input_tensor.numel()].copy_(input_tensor)
    coalesced_output_tensor = tp.empty(
        max_object_size * group_size, dtype=tp.uint8, device=current_device
    )
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    all_gather(output_tensors, padded_input, group=group)
    for i, tensor in enumerate(output_tensors):
        object_list[i] = _tensor_to_object(
            tensor, int(object_size_list[i].item()), group
        )


def gather_object(obj, object_gather_list=None, dst: Optional[int] = None,
                  group=None) -> None:
    """Gathers picklable objects from the whole group in a single process.

    """
    if _rank_not_in_group(group):
        _warn_not_in_group("gather_object")
        return
    pg = _resolve_group(group)
    group_dst = pg.group_rank(dst) if dst is not None else 0

    my_group_rank = get_rank(group)
    _validate_output_list_for_rank(my_group_rank, group_dst, object_gather_list)
    current_device = _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    group_size = pg.size()
    object_sizes_tensor = tp.zeros(group_size, dtype=tp.int64,
                                   device=current_device)
    object_size_list = [
        object_sizes_tensor[i : i + 1] for i in range(group_size)
    ]
    # All-gather sizes even though this is a gather: each rank needs to send
    # a tensor of the same (maximal) size.
    all_gather(object_size_list, local_size, group=group)
    max_object_size = max(int(s.item()) for s in object_size_list)
    padded_input = tp.empty(max_object_size, dtype=tp.uint8,
                            device=current_device)
    padded_input[: input_tensor.numel()].copy_(input_tensor)
    output_tensors = None
    if my_group_rank == group_dst:
        coalesced_output_tensor = tp.empty(
            max_object_size * group_size, dtype=tp.uint8, device=current_device
        )
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    gather(
        padded_input,
        gather_list=output_tensors,
        dst=group_dst,
        group=pg,
    )
    if my_group_rank != group_dst:
        return

    if object_gather_list is None:
        raise RuntimeError("Must provide object_gather_list on dst rank")
    for i, tensor in enumerate(output_tensors):
        object_gather_list[i] = _tensor_to_object(
            tensor, int(object_size_list[i].item()), group
        )


def send_object_list(object_list: Sequence[object], dst: int, group=None,
                     device=None) -> None:
    if _rank_not_in_group(group):
        _warn_not_in_group("send_object_list")
        return

    current_device = device or _get_object_coll_device(group)
    tensor_list, size_list = zip(
        *[_object_to_tensor(obj, current_device, group) for obj in object_list]
    )
    object_sizes_tensor = tp.cat(list(size_list))
    send(object_sizes_tensor, dst, group=group)
    object_tensor = (
        tensor_list[0] if len(tensor_list) == 1 else tp.cat(list(tensor_list))
    )
    send(object_tensor, dst, group=group)


def recv_object_list(object_list: list, src: Optional[int] = None, group=None,
                     device=None) -> int:
    """

    Returns the sender's global rank.
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("recv_object_list")
        return -1

    pg = _resolve_group(group)
    current_device = device or _get_object_coll_device(group)
    object_sizes_tensor = tp.empty(len(object_list), dtype=tp.int64,
                                   device=current_device)
    rank_sizes = recv(object_sizes_tensor, src=src, group=group)
    total = sum(int(s.item()) for s in object_sizes_tensor)
    object_tensor = tp.empty(total, dtype=tp.uint8, device=current_device)
    rank_objects = recv(object_tensor, src=src, group=group)
    if rank_sizes != rank_objects:
        raise RuntimeError(
            "Mismatch in return ranks for object sizes and objects."
        )
    offset = 0
    for i in range(len(object_list)):
        obj_size = int(object_sizes_tensor[i].item())
        obj_view = object_tensor[offset : offset + obj_size]
        offset += obj_size
        object_list[i] = _tensor_to_object(obj_view, obj_size, group)
    return rank_objects


def broadcast_object_list(object_list, src: int = 0, group=None,
                          device=None):
    """Broadcasts picklable objects in ``object_list`` to the whole group.

    Non-source ranks may pass ``None``; the populated list is returned on
    every rank.
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("broadcast_object_list")
        return object_list

    pg = _resolve_group(group)
    group_src = pg.group_rank(src)
    current_device = device or _get_object_coll_device(group)
    my_group_rank = pg.rank()

    # The object count goes out first so non-source ranks can size their
    # buffers without knowing ``len(object_list)`` up front.
    if my_group_rank == group_src:
        num_objects = tp.tensor([len(object_list)], dtype=tp.int64,
                                device=current_device)
    else:
        num_objects = tp.zeros(1, dtype=tp.int64, device=current_device)
    broadcast(num_objects, src, group=pg)
    count = int(num_objects.item())

    if object_list is None:
        object_list = [None] * count
    elif len(object_list) != count:
        raise ValueError(
            f"object_list has {len(object_list)} entries but rank {src} "
            f"broadcast {count}"
        )

    if my_group_rank == group_src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj, current_device, group) for obj in object_list]
        )
        object_sizes_tensor = tp.cat(list(size_list))
    else:
        object_sizes_tensor = tp.zeros(count, dtype=tp.int64,
                                       device=current_device)

    broadcast(object_sizes_tensor, src, group=pg)

    if my_group_rank == group_src:
        object_tensor = (
            tensor_list[0] if len(tensor_list) == 1 else tp.cat(list(tensor_list))
        )
    else:
        total = sum(int(s.item()) for s in object_sizes_tensor)
        object_tensor = tp.empty(total, dtype=tp.uint8, device=current_device)

    broadcast(object_tensor, src, group=pg)
    offset = 0
    if my_group_rank != group_src:
        for i in range(count):
            obj_size = int(object_sizes_tensor[i].item())
            obj_view = object_tensor[offset : offset + obj_size]
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)
    return object_list


def scatter_object_list(scatter_object_output_list: list,
                        scatter_object_input_list: Optional[Sequence[object]] = None,
                        src: int = 0, group=None) -> None:
    """

    ``src`` is a global rank. On each rank the scattered object is stored as
    the first element of ``scatter_object_output_list``.
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("scatter_object_list")
        return

    if not isinstance(scatter_object_output_list, list) or \
            len(scatter_object_output_list) < 1:
        raise ValueError(
            "Expected argument scatter_object_output_list to be a list of "
            "size at least 1."
        )

    pg = _resolve_group(group)
    group_src = pg.group_rank(src)
    my_group_rank = pg.rank()
    pg_device = _get_object_coll_device(group)
    if my_group_rank == group_src:
        if scatter_object_input_list is None:
            raise ValueError(
                "source rank must provide non-None scatter_object_input_list"
            )
        tensor_list, tensor_sizes = zip(
            *[
                _object_to_tensor(obj, pg_device, group)
                for obj in scatter_object_input_list
            ]
        )
        tensor_list, tensor_sizes = list(tensor_list), [t for t in tensor_sizes]
        # The src rank broadcasts the maximum tensor size because all ranks
        # are expected to call scatter with equal-sized tensors.
        max_tensor_size = max(int(s.item()) for s in tensor_sizes)
        padded_list = []
        for tensor in tensor_list:
            padded = tp.empty(max_tensor_size, dtype=tp.uint8, device=pg_device)
            padded[: tensor.numel()].copy_(tensor)
            padded_list.append(padded)
        tensor_list = padded_list
    else:
        max_tensor_size = 0
    max_size_tensor = tp.tensor([max_tensor_size], dtype=tp.int64,
                                device=pg_device)
    broadcast(max_size_tensor, src, group=pg)
    max_tensor_size = int(max_size_tensor.item())

    output_tensor = tp.empty(max_tensor_size, dtype=tp.uint8, device=pg_device)
    scatter(
        output_tensor,
        scatter_list=None if my_group_rank != group_src else tensor_list,
        src=group_src,
        group=pg,
    )

    obj_tensor_size = tp.tensor([0], dtype=tp.int64, device=pg_device)
    scatter(
        obj_tensor_size,
        scatter_list=None if my_group_rank != group_src else tensor_sizes,
        src=group_src,
        group=pg,
    )

    scatter_object_output_list[0] = _tensor_to_object(
        output_tensor, obj_tensor_size, group
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def all_to_all_single(output: tp.Tensor, input: tp.Tensor,
                      output_split_sizes: Optional[List[int]] = None,
                      input_split_sizes: Optional[List[int]] = None,
                      group=None, async_op: bool = False):
    """Splits ``input`` evenly (or by split sizes) and scatters the chunks.

    """
    pg = _resolve_group(group)
    if input.dtype != output.dtype:
        raise ValueError("output tensor must have the same type as input tensor")
    if (output_split_sizes is None) != (input_split_sizes is None):
        raise ValueError(
            "output_split_sizes and input_split_sizes must both be specified "
            "or both be None"
        )
    if pg.backend != Backend.NCCL:
        out_counts = list(output_split_sizes or [])
        in_counts = list(input_split_sizes or [])
        out_t, out_restore = _contiguous_view(output)
        in_t, in_restore = _contiguous_view(input)
        work = _cpu_pg(pg).all_to_all_single(
            out_t, in_t, out_counts, in_counts, _cpu_timeout_ms(pg))

        def _restore():
            if out_restore is not None:
                out_restore()
            if in_restore is not None:
                in_restore()

        return _cpu_finish(work, tensors=[output], restore=_restore,
                           async_op=async_op)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    if output_split_sizes is None:
        if output.numel() != input.numel():
            raise ValueError(
                "output tensor must have the same number of elements as "
                "input tensor for equal splits"
            )
        _C.all_to_all_single_equal_split(output, input, comm)
    else:
        if len(input_split_sizes) != pg.size() or \
                len(output_split_sizes) != pg.size():
            raise RuntimeError(
                "split sizes length must equal the group world size"
            )
        if sum(input_split_sizes) != input.numel():
            raise ValueError("input_split_sizes sum must equal input numel")
        if sum(output_split_sizes) != output.numel():
            raise ValueError("output_split_sizes sum must equal output numel")
        _C.all_to_all_single_unequal_split(
            output, input, list(output_split_sizes), list(input_split_sizes),
            comm)

    return _finish(_record_event(_device_index_of(input)), None,
                   async_op=async_op)


def all_to_all(output_tensor_list: List[tp.Tensor],
               input_tensor_list: List[tp.Tensor], group=None,
               async_op: bool = False):
    """Scatters a list of tensors to ranks and collects one from each.

    Per-rank splits are the tensor numels, executed as one grouped send/recv
    exchange.
    """
    pg = _resolve_group(group)
    if len(output_tensor_list) != pg.size() or \
            len(input_tensor_list) != pg.size():
        raise RuntimeError(
            "all_to_all expects input/output tensor lists of length "
            f"world_size ({pg.size()})"
        )
    if pg.backend != Backend.NCCL:
        # The backend copies in place into the output tensors; reshaped
        # views share storage with the caller's tensors.
        outs = [t.reshape(-1) for t in output_tensor_list]
        ins = [t.reshape(-1) for t in input_tensor_list]
        work = _cpu_pg(pg).alltoall(outs, ins, _cpu_timeout_ms(pg))
        return _cpu_finish(work, tensors=list(output_tensor_list),
                           async_op=async_op)
    dtype = input_tensor_list[0].dtype
    for t in list(input_tensor_list) + list(output_tensor_list):
        if t.dtype != dtype:
            raise ValueError(
                "all_to_all tensors must have identical dtypes across lists"
            )
    input_splits = [t.numel() for t in input_tensor_list]
    output_splits = [t.numel() for t in output_tensor_list]
    flat_in = tp.cat([t.reshape(-1) for t in input_tensor_list])
    flat_out = tp.empty(sum(output_splits), dtype=dtype,
                        device=input_tensor_list[0].device)
    work = all_to_all_single(flat_out, flat_in, output_splits, input_splits,
                             group=pg, async_op=True)

    def done():
        if hasattr(work, "wait"):
            work.wait()
        offset = 0
        for t, n in zip(output_tensor_list, output_splits):
            t.copy_(flat_out[offset : offset + n].reshape(t.shape))
            offset += n

    if async_op:
        return Work(work._event, done=done)
    done()
    return None


def _check_op(op) -> None:
    if op not in [isend, irecv]:
        raise ValueError(
            "Invalid ``op``. Expected ``op`` "
            "to be of type ``tensorplay.distributed.isend`` or "
            "``tensorplay.distributed.irecv``."
        )


def _check_p2p_op_list(p2p_op_list) -> None:
    if not isinstance(p2p_op_list, list) or not all(
        isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list
    ):
        raise ValueError(
            "Invalid ``p2p_op_list``. Each op is expected to "
            "be of type ``tensorplay.distributed.P2POp``."
        )
    group = p2p_op_list[0].group
    if not all(group == p2p_op.group for p2p_op in p2p_op_list):
        raise ValueError("All ops need to use the same group.")


class P2POp:
    """A class to build point-to-point operations for ``batch_isend_irecv``.

    is a global rank (or ``group_peer`` a group rank).
    """

    def __init__(self, op, tensor, peer: Optional[int] = None, group=None,
                 tag: int = 0, group_peer: Optional[int] = None):
        self.op = op
        self.tensor = tensor
        if group_peer is not None:
            if peer is not None:
                raise ValueError("Can't specify both peer and group_peer")
            self.peer = get_global_rank(_resolve_group(group), group_peer)
            self.group_peer = group_peer
        else:
            if peer is None:
                raise ValueError("Must specify peer or group_peer")
            self.peer = peer
            self.group_peer = get_group_rank(_resolve_group(group), peer)
        self.group = group
        self.tag = tag

    def __repr__(self) -> str:
        op_name = getattr(self.op, "__name__", repr(self.op))
        return (
            f"P2POp({op_name}, peer={self.peer}, group={self.group}, "
            f"tag={self.tag})"
        )


def batch_isend_irecv(p2p_op_list: List[P2POp]) -> List[Work]:
    """

    All operations are treated as a single NCCL group so ordering of sends
    vs receives cannot deadlock. Every rank in ``group`` must participate.
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    pg = _resolve_group(group)

    works: List[Work] = []
    # NCCL-style coalescing: enqueue every p2p op inside one ncclGroup.
    _C.group_start()
    try:
        for p2p_op in p2p_op_list:
            work = p2p_op.op(p2p_op.tensor, p2p_op.peer,
                             group=p2p_op.group, tag=p2p_op.tag)
            if work is not None:
                works.append(work)
    finally:
        _C.group_end()
    del pg
    return works


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _compute_bucket_assignment_by_size(tensors, bucket_size_limits,
                                       expect_sparse_gradient=None,
                                       tensor_indices=None):
    """Compute communication buckets grouped by dtype and device.

    Returns ``(bucket_indices, per_bucket_size_limits)``. Tensors are binned
    per (dtype, device) key; sparse-gradient tensors get their own bucket.
    """
    if expect_sparse_gradient is not None and \
            len(expect_sparse_gradient) not in (0, len(tensors)):
        raise AssertionError("expect_sparse_gradient has invalid length")
    if not tensors:
        raise ValueError("tensors must not be empty")

    result = []  # [(indices, size_limit)] like the C++ vector<tuple<..>>
    k_no_size_limit = 0
    buckets: dict = {}   # key -> {"indices": [...], "size": int, "limit": int}
    iterators: dict = {}  # key -> index into bucket_size_limits

    for i, tensor in enumerate(tensors):
        if tensor.is_sparse:
            raise RuntimeError("No support for sparse tensors.")
        tensor_index = i
        if tensor_indices:
            tensor_index = tensor_indices[i]
        if expect_sparse_gradient and expect_sparse_gradient[tensor_index]:
            result.append(([tensor_index], k_no_size_limit))
            continue

        key = (tensor.dtype, str(tensor.device))
        bucket = buckets.setdefault(key, {"indices": [], "size": 0,
                                          "limit": k_no_size_limit})
        bucket["indices"].append(tensor_index)
        bucket["size"] += tensor.numel() * tensor.element_size()

        if key not in iterators:
            iterators[key] = 0
        limit_idx = iterators[key]
        bucket_size_limit = bucket_size_limits[limit_idx]
        bucket["limit"] = bucket_size_limit
        if bucket["size"] >= bucket_size_limit:
            result.append((bucket["indices"], bucket_size_limit))
            buckets[key] = {"indices": [], "size": 0,
                            "limit": k_no_size_limit}
            next_idx = limit_idx + 1
            if next_idx < len(bucket_size_limits):
                iterators[key] = next_idx

    for bucket in buckets.values():
        if bucket["indices"]:
            result.append((bucket["indices"], bucket["limit"]))

    result.sort(key=lambda item: min(item[0]))
    bucket_indices = [list(indices) for indices, _ in result]
    per_bucket_size_limits = [limit for _, limit in result]
    return bucket_indices, per_bucket_size_limits


def _flatten_dense_tensors(tensors):
    return tp.cat([t.detach().reshape(-1) for t in tensors])


def _unflatten_dense_tensors(flat, tensors):
    outputs = []
    offset = 0
    for t in tensors:
        numel = t.numel()
        outputs.append(flat[offset : offset + numel].view(t.shape))
        offset += numel
    return outputs


def _broadcast_coalesced(process_group, tensors, buffer_size,
                         authoritative_rank=0):
    """Broadcast many tensors, coalesced into flat buckets.

    Buckets are formed per (dtype, device); each bucket is flattened into a
    single buffer, broadcast with one collective, then copied back. At most
    two buckets stay in flight to bound peak memory.
    """
    from collections import deque

    if not tensors:
        return
    src = get_global_rank(process_group, authoritative_rank)

    # Bucket per (dtype, device) key, splitting each stream at buffer_size,
    # exactly as compute_bucket_assignment_by_size does for broadcast_coalesced.
    streams: dict = {}
    order = []
    for idx, tensor in enumerate(tensors):
        key = (tensor.dtype, str(tensor.device))
        if key not in streams:
            streams[key] = {"idx": [], "size": 0}
            order.append(key)
        g = streams[key]
        g["idx"].append(idx)
        g["size"] += tensor.numel() * tensor.element_size()

    buckets = []
    for key in order:
        g = streams.pop(key)
        acc_idx, acc_size = [], 0
        for i in g["idx"]:
            t = tensors[i]
            tbytes = t.numel() * t.element_size()
            if acc_idx and acc_size + tbytes > buffer_size:
                buckets.append(acc_idx)
                acc_idx, acc_size = [], 0
            acc_idx.append(i)
            acc_size += tbytes
        if acc_idx:
            buckets.append(acc_idx)

    in_flight = deque()
    max_in_flight = 2

    class _BroadcastWork:
        def __init__(self, bucket_tensors, root_rank):
            self.bucket_tensors = bucket_tensors
            self.flat = _flatten_dense_tensors(bucket_tensors)
            broadcast(self.flat, root_rank, group=process_group)

        def finish(self):
            outs = _unflatten_dense_tensors(self.flat, self.bucket_tensors)
            for t, o in zip(self.bucket_tensors, outs):
                if t.numel() != 0:
                    t.copy_(o)

    for bucket in buckets:
        if len(in_flight) >= max_in_flight:
            in_flight.popleft().finish()
        in_flight.append(_BroadcastWork([tensors[i] for i in bucket], src))
    while in_flight:
        in_flight.popleft().finish()


def _verify_params_across_processes(process_group, tensors, logger=None) -> bool:
    """Broadcast every tensor from rank 0 and verify values match locally."""
    if get_world_size(process_group) == 1 or not tensors:
        return True
    src = get_global_rank(process_group, 0)
    for tensor in tensors:
        ref = tensor.detach().clone()
        broadcast(tensor, src, group=process_group)
        diff = (tensor - ref).abs().max().item()
        del ref
        if diff != 0:
            return False
    return True


class GradBucket:

    def __init__(self, index: int, buffer: tp.Tensor, offsets: List[int],
                 lengths: List[int], sizes: List[List[int]],
                 parameters: List[tp.Tensor], num_total_buckets: int = 0):
        self._index = index
        self._buffer = buffer
        self._offsets = offsets
        self._lengths = lengths
        self._sizes = sizes
        self._parameters = parameters
        self._num_total_buckets = num_total_buckets

    def index(self) -> int:
        return self._index

    def buffer(self) -> tp.Tensor:
        return self._buffer

    def set_buffer(self, buffer: tp.Tensor) -> None:
        self._buffer = buffer

    def gradients(self) -> List[tp.Tensor]:
        return [
            self._buffer[o : o + l].view(shape)
            for o, l, shape in zip(self._offsets, self._lengths, self._sizes)
        ]

    def parameters(self) -> List[tp.Tensor]:
        return list(self._parameters)

    def is_last(self) -> bool:
        return self._index == self._num_total_buckets - 1


def new_subgroups_by_enumeration(
    ranks_per_subgroup_list,
    timeout=None,
    backend=None,
    pg_options=None,
    group_desc=None,
):
    """

    The division is specified by a nested list of ranks. The subgroups
    cannot have overlap, and some ranks may not have to be in any subgroup.
    """
    import logging

    logger = logging.getLogger(__name__)
    if ranks_per_subgroup_list is None or len(ranks_per_subgroup_list) == 0:
        raise ValueError("The arg 'ranks_per_subgroup_list' cannot be empty")

    subgroups = []
    cur_subgroup = None
    # Create a mapping from rank to subgroup to check if there is any subgroup overlap.
    rank_to_ranks_dict: dict[int, list[int]] = {}
    for ranks in ranks_per_subgroup_list:
        subgroup = new_group(
            ranks=ranks,
            timeout=timeout if timeout is not None else default_pg_timeout,
            backend=backend,
            pg_options=pg_options,
        )
        subgroups.append(subgroup)
        my_rank = get_rank()
        for rank in ranks:
            if rank in rank_to_ranks_dict:
                raise ValueError(
                    f"Rank {rank} has appeared in both subgroup "
                    f"{rank_to_ranks_dict[rank]} and {ranks}"
                )
            rank_to_ranks_dict[rank] = ranks
            if my_rank == rank:
                cur_subgroup = subgroup
                logger.info("Rank %s is assigned to subgroup %s", rank, ranks)

    return cur_subgroup, subgroups


def new_subgroups(
    group_size: int | None = None,
    group=None,
    timeout=None,
    backend=None,
    pg_options=None,
    group_desc=None,
):
    """

    By default, it creates intra-machine subgroups, where each of which
    contains all the ranks of a machine, based on the assumption that each
    machine has the same number of devices.

    Returns:
        The subgroup containing the current rank, and all the subgroups used
        for cleanup.
    """
    if group_size is None:
        if not tp.cuda.is_available():
            raise ValueError(
                "Default group size only takes effect when CUDA/XPU is available."
                "If your subgroup using a backend that does not depend on CUDA/XPU,"
                "please pass in 'group_size' correctly."
            )
        group_size = tp.cuda.device_count()
    if group_size <= 0:
        raise ValueError(f"The arg 'group_size' ({group_size}) must be positive")

    world_size = get_world_size(group=group)
    if world_size < group_size:
        raise ValueError(
            f"The arg 'group_size' ({group_size}) must not exceed the world size ({world_size})"
        )
    if world_size % group_size != 0:
        raise ValueError(
            f"The world size ({world_size}) must be divisible by '{group_size=}'"
        )

    ranks = get_process_group_ranks(group=group)
    ranks_per_subgroup_list = [
        ranks[i : i + group_size] for i in range(0, len(ranks), group_size)
    ]
    subgroup, subgroups = new_subgroups_by_enumeration(
        ranks_per_subgroup_list,
        timeout=timeout,
        backend=backend,
        pg_options=pg_options,
        group_desc=group_desc,
    )
    if not isinstance(subgroup, ProcessGroup):
        raise AssertionError("Current rank was not assigned to a subgroup")
    return subgroup, subgroups
