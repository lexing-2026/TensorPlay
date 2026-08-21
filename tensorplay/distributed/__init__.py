"""tensorplay.distributed — torch.distributed-compatible API on NCCL.

Layering mirrors torch: ``p10`` NCCLContext (C++, against nccl.h) plays the
role of c10d's ProcessGroupNCCL core; this package plays the role of
``distributed_c10d.py`` (groups, rendezvous, tensor marshaling). The store
layer (FileStore/TCPStore) is pure Python by design.
"""

from __future__ import annotations

import datetime as _dt
import os
import threading
from typing import List, Optional

import tensorplay as tp
from tensorplay._C import _distributed as _C

__all__ = [
    "GroupMember",
    "ProcessGroup",
    "ReduceOp",
    "Work",
    "all_gather",
    "all_reduce",
    "barrier",
    "broadcast",
    "destroy_process_group",
    "gather",
    "get_backend",
    "get_rank",
    "get_world_size",
    "init_process_group",
    "is_available",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "new_group",
    "recv",
    "reduce",
    "reduce_scatter",
    "scatter",
    "send",
]

default_pg_timeout = _dt.timedelta(minutes=30)


class ReduceOp:
    SUM = 0
    PRODUCT = PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4


class GroupMember:
    NON_GROUP_MEMBER = -100


class Work:
    """Handle for an async collective (torch.distributed.Work subset)."""

    def __init__(self, event, done=None) -> None:
        self._event = event
        self._done = done

    def is_completed(self) -> bool:
        return bool(self._event.query())

    def wait(self, timeout: Optional[_dt.timedelta] = None) -> bool:
        self._event.synchronize()
        if self._done is not None:
            self._done()
        return True


class ProcessGroup:
    def __init__(self, ranks: List[int], group_name: str) -> None:
        self.ranks = list(ranks)
        self.group_name = group_name
        self.comm: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self.ranks)

    def global_rank(self, group_rank: int) -> int:
        return self.ranks[group_rank]

    def group_rank(self, global_rank: int) -> int:
        return self.ranks.index(global_rank)


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
    return False


def is_mpi_available() -> bool:
    return False


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
    raise ValueError(_invalid_group_msg)


def get_rank(group=None) -> int:
    pg = _resolve_group(group)
    return pg.group_rank(_global_rank())


def get_world_size(group=None) -> int:
    return _resolve_group(group).size


def get_backend(group=None) -> str:
    _resolve_group(group)
    return _backend


def _global_rank() -> int:
    return _rank_state[0]


def _current_store():
    if _store_ref[0] is None:
        raise RuntimeError("No store available; was init_process_group called?")
    return _store_ref[0]


def _ensure_comm(pg: ProcessGroup, timeout_s: float) -> int:
    with pg._lock:
        if pg.comm is not None:
            return pg.comm
        store = _current_store()
        key = f"tensorplay_distributed/nccl_unique_id/{pg.group_name}"
        if _global_rank() == pg.ranks[0]:
            uid = _C.get_unique_id()
            store.set(key, uid.hex())
        else:
            uid = bytes.fromhex(store.get(key, timeout=timeout_s).decode())
        pg.comm = _C.comm_init_rank(pg.group_rank(_global_rank()), pg.size, uid)
        return pg.comm


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
    if backend != "nccl":
        raise ValueError(
            f"Invalid backend: '{backend}'. TensorPlay currently supports: 'nccl'"
        )
    if not is_available():
        raise RuntimeError(
            "Distributed package is not available (NCCL library could not be loaded)"
        )

    timeout_s = timeout.total_seconds()
    if init_method is None:
        init_method = "env://"

    if store is None:
        if rank < 0:
            rank = int(os.environ["RANK"])
        if world_size < 0:
            world_size = int(os.environ["WORLD_SIZE"])
        store = _parse_init_method(init_method, rank, world_size, timeout_s)
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
    _world_group = ProcessGroup(list(range(world_size)), group_name or "0")
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
        _world_group = None
        _groups.clear()
        _store_ref[0] = None
        _backend = ""
        return
    pg = _resolve_group(group)
    if pg.comm is not None:
        _C.comm_destroy(pg.comm)
        pg.comm = None
    if pg is _world_group:
        _world_group = None
        _groups.clear()
        _store_ref[0] = None
        _backend = ""


def new_group(ranks: Optional[List[int]] = None,
              timeout: _dt.timedelta = default_pg_timeout,
              backend: Optional[str] = None, pg_options=None):
    global _group_count
    _check_default_pg()
    if backend not in (None, "nccl"):
        raise ValueError(f"Invalid backend: '{backend}'")
    if ranks is None:
        ranks = list(range(get_world_size()))
    ranks = sorted(set(ranks))
    if _global_rank() not in ranks:
        return GroupMember.NON_GROUP_MEMBER
    _group_count += 1
    pg = ProcessGroup(ranks, group_name=str(_group_count))
    _groups[pg.group_name] = pg
    _ensure_comm(pg, timeout.total_seconds())
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


def broadcast(tensor: tp.Tensor, src: int, group=None, async_op: bool = False):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, restore = _contiguous_view(tensor)
    _C.broadcast(buf, src, comm)
    return _finish(_record_event(_device_index_of(tensor)), restore=restore, async_op=async_op)


def all_reduce(tensor: tp.Tensor, op: int = ReduceOp.SUM, group=None,
               async_op: bool = False):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, restore = _contiguous_view(tensor)
    _C.all_reduce(buf, int(op), comm)
    return _finish(_record_event(_device_index_of(tensor)), restore=restore, async_op=async_op)


def reduce(tensor: tp.Tensor, dst: int, op: int = ReduceOp.SUM, group=None,
           async_op: bool = False):
    # ``dst`` is a global rank (torch semantics).
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, restore = _contiguous_view(tensor)
    _C.reduce(buf, int(op), dst, comm)
    return _finish(_record_event(_device_index_of(tensor)), restore=restore, async_op=async_op)


def all_gather(tensor_list: List[tp.Tensor], tensor: tp.Tensor, group=None,
               async_op: bool = False):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    if len(tensor_list) != pg.size:
        raise RuntimeError(
            f"Number of tensors in tensor_list ({len(tensor_list)}) does not match "
            f"the group world size ({pg.size})"
        )
    n = tensor.numel()
    out = tp.zeros(pg.size * n, dtype=tensor.dtype,
                   device=tensor.device)
    send_t, restore = _contiguous_view(tensor)
    _C.all_gather(out, send_t, comm)

    def _split():
        flat = out.cpu()
        for i, t in enumerate(tensor_list):
            t.copy_(flat[i * n : (i + 1) * n].reshape(t.shape))

    return _finish(_record_event(_device_index_of(tensor)),
                   restore=restore, extra=_split, async_op=async_op)


def gather(tensor: tp.Tensor, gather_list: Optional[List[tp.Tensor]] = None,
           dst: int = 0, group=None, async_op: bool = False):
    # ``dst`` is a group rank (torch semantics).
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    my_group_rank = pg.group_rank(_global_rank())
    recv_obj = None
    if my_group_rank == dst:
        if gather_list is None:
            raise RuntimeError("gather_list must be specified on the destination rank")
        n = tensor.numel()
        recv_obj = tp.zeros(pg.size * n, dtype=tensor.dtype, device=tensor.device)
    send_t, restore = _contiguous_view(tensor)
    _C.gather(recv_obj, send_t, pg.global_rank(dst), comm)
    n = tensor.numel()

    def _split():
        if my_group_rank != dst:
            return
        flat = recv_obj.cpu()
        for i, t in enumerate(gather_list):
            t.copy_(flat[i * n : (i + 1) * n].reshape(t.shape))

    return _finish(_record_event(_device_index_of(tensor)),
                   restore=restore, extra=_split, async_op=async_op)


def scatter(tensor: tp.Tensor, scatter_list: Optional[List[tp.Tensor]] = None,
            src: int = 0, group=None, async_op: bool = False):
    # ``src`` is a group rank (torch semantics).
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    my_group_rank = pg.group_rank(_global_rank())
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
    _C.scatter(recv_t, send_obj, pg.global_rank(src), comm)

    return _finish(_record_event(_device_index_of(tensor)),
                   restore=restore, async_op=async_op)


def reduce_scatter(output: tp.Tensor, input_list: List[tp.Tensor],
                   op: int = ReduceOp.SUM, group=None, async_op: bool = False):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    if len(input_list) != pg.size:
        raise RuntimeError(
            f"Number of tensors in input_list ({len(input_list)}) does not match "
            f"the group world size ({pg.size})"
        )
    chunks = []
    for t in input_list:
        c, _ = _contiguous_view(t)
        chunks.append(c.reshape(1, -1))
    send_t = tp.cat(chunks, 0)
    recv_t, restore = _contiguous_view(output)
    _C.reduce_scatter(recv_t, send_t, int(op), comm)

    return _finish(_record_event(_device_index_of(output)),
                   restore=restore, async_op=async_op)


def send(tensor: tp.Tensor, dst: int, group=None, tag: int = 0):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    buf, _ = _contiguous_view(tensor)
    _C.send(buf, dst, comm)
    _record_event(_device_index_of(tensor)).synchronize()


def recv(tensor: tp.Tensor, src: Optional[int] = None, group=None, tag: int = 0):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    if src is None:
        src = pg.global_rank((pg.group_rank(_global_rank()) - 1) % pg.size)
    buf, restore = _contiguous_view(tensor)
    _C.recv(buf, src, comm)
    event = _record_event(_device_index_of(tensor))
    return _finish(event, restore=restore, async_op=False) or src


def barrier(group=None, async_op: bool = False, device_ids: Optional[List[int]] = None):
    pg = _resolve_group(group)
    comm = _ensure_comm(pg, default_pg_timeout.total_seconds())
    device_index = device_ids[0] if device_ids else tp.cuda.current_device()
    flag = tp.zeros(1, dtype=tp.float32, device=f"cuda:{device_index}")
    _C.all_reduce(flag, ReduceOp.SUM, comm)
    return _finish(_record_event(device_index), None, async_op=async_op)


def _record_event(device_index: int):
    return tp.cuda.current_stream(device_index).record_event()
