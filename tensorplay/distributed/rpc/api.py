from __future__ import annotations

import contextlib
import collections
import functools
import threading
import time
import uuid
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from . import constants as rpc_constants
from .backend_registry import BackendType, construct_rpc_backend_options, init_backend
from .internal import (
    RPCExecMode,
    _InternalRPCPickler,
    _build_rpc_profiling_key,
    _handle_exception,
    _internal_rpc_pickler,
    PythonUDF,
)

T = TypeVar("T")
_MISSING = object()
_state_lock = threading.RLock()
_thread_local = threading.local()
_agent: Any = None
_current_worker: "WorkerInfo | None" = None
_workers: dict[str, "WorkerInfo"] = {}
_executor: Any = None
_native_runtime: Any = None
_pending: set["_Future[Any]"] = set()
_rrefs: dict[Any, "RRef[Any]"] = {}
_default_pickler: _InternalRPCPickler = _internal_rpc_pickler

__all__ = [
    "shutdown",
    "WorkerInfo",
    "get_worker_info",
    "remote",
    "rpc_sync",
    "rpc_async",
    "RRef",
    "AllGatherStates",
    "method_factory",
    "new_method",
]


@dataclass(frozen=True)
class WorkerInfo:
    name: str
    id: int

    def __str__(self) -> str:
        return f"WorkerInfo(name='{self.name}', id={self.id})"


def get_worker_info(worker_name: Any = None) -> WorkerInfo:
    if _agent is None:
        raise RuntimeError("RPC has not been initialized")
    if worker_name is None:
        return _agent.get_worker_info()
    return _agent.get_worker_info(worker_name)


class _Future(Generic[T]):
    def __init__(self, future: Any = None) -> None:
        self._future = future or ConcurrentFuture()

    @classmethod
    def completed(cls, value: T) -> "_Future[T]":
        result = cls()
        result.set_result(value)
        return result

    def wait(self, timeout: float | None = None) -> T:
        if hasattr(self._future, "result"):
            if timeout in (None, 0, rpc_constants.UNSET_RPC_TIMEOUT):
                return self._future.result()
            return self._future.result(timeout=float(timeout))
        if timeout in (None, 0, rpc_constants.UNSET_RPC_TIMEOUT):
            return self._future.wait()
        return self._future.wait(float(timeout))

    def value(self) -> T:
        return self.wait()

    def done(self) -> bool:
        return self._future.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if hasattr(self._future, "exception"):
            if timeout in (None, 0, rpc_constants.UNSET_RPC_TIMEOUT):
                return self._future.exception()
            return self._future.exception(timeout=float(timeout))
        try:
            self.wait(timeout)
        except BaseException as exc:
            return exc
        return None

    def set_result(self, value: T) -> None:
        if not self._future.done():
            self._future.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(exc)

    def then(self, callback: Any) -> "_Future[Any]":
        result: _Future[Any] = _Future()

        def source_value(source: Any) -> Any:
            return source.result() if hasattr(source, "result") else source.wait()

        def adopt(value: Any) -> None:
            try:
                result.set_result(value.wait() if isinstance(value, _Future) else source_value(value))
            except BaseException as exc:
                result.set_exception(exc)

        def complete(source: ConcurrentFuture[T]) -> None:
            try:
                source_value(source)
                value = callback(self)
                if isinstance(value, _Future):
                    value._future.add_done_callback(adopt)
                elif isinstance(value, ConcurrentFuture):
                    value.add_done_callback(adopt)
                else:
                    result.set_result(value)
            except BaseException as exc:
                result.set_exception(exc)

        self._future.add_done_callback(complete)
        return result


Future = _Future


class AllGatherStates:
    def __init__(self) -> None:
        self.gathered_objects: dict[str, Any] = {}
        self.proceed_signal = threading.Event()


_ALL_WORKER_NAMES: set[str] = set()
_all_gather_dict_lock = threading.RLock()
_all_gather_sequence_id: dict[str, int] = {}
_all_gather_sequence_id_to_states: collections.defaultdict[str, AllGatherStates] = (
    collections.defaultdict(AllGatherStates)
)


class _NativeRpcAgent:
    def __init__(self, name: str, rank: int, world_size: int, options: Any, native: Any) -> None:
        self.name = str(name)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.options = options
        self.native = native
        self.store: Any = None
        self._workers = [
            WorkerInfo(str(info.name), int(info.id))
            for info in native.workers()
        ]

    def get_worker_info(self, worker_name: str | int | None = None) -> WorkerInfo:
        if worker_name is None:
            info = self.native.current_worker()
            return WorkerInfo(str(info.name), int(info.id))
        if isinstance(worker_name, int):
            for info in self._workers:
                if info.id == worker_name:
                    return info
        for info in self._workers:
            if info.name == str(worker_name):
                return info
        raise ValueError(f"worker {worker_name!r} is not registered")

    def get_worker_infos(self) -> list[WorkerInfo]:
        return list(self._workers)

    def get_backend_options(self) -> Any:
        return self.options

    def _get_backend_options(self) -> Any:
        return self.options

    def shutdown(self) -> None:
        self.native.shutdown()

    def join(self, shutdown: bool = True, timeout: float = 0) -> None:
        self.native.join(bool(shutdown), float(timeout))

    def _update_group_membership(self, worker_info: WorkerInfo, devices: list[Any], reverse_device_map: dict[Any, Any], is_join: bool) -> None:
        membership = getattr(self, "group_membership", None)
        if membership is None:
            membership = self.group_membership = {}
        if is_join:
            membership[worker_info.name] = {
                "devices": list(devices),
                "reverse_device_map": dict(reverse_device_map),
            }
        else:
            membership.pop(worker_info.name, None)


def _construct_native_agent(store: Any, name: str, rank: int, world_size: int, rpc_backend_options: Any) -> Any:
    native_module = _load_native_runtime()
    transports = getattr(rpc_backend_options, "_transports", None)
    channels = getattr(rpc_backend_options, "_channels", None)
    native_options = native_module.TensorPipeRpcBackendOptions(
        int(getattr(rpc_backend_options, "num_worker_threads", 16)),
        transports,
        channels,
        float(getattr(rpc_backend_options, "rpc_timeout", rpc_constants.DEFAULT_RPC_TIMEOUT_SEC)),
        str(getattr(rpc_backend_options, "init_method", rpc_constants.DEFAULT_INIT_METHOD)),
    )
    native_options.devices = [
        str(device) for device in getattr(rpc_backend_options, "devices", [])
    ]
    for worker, mapping in getattr(rpc_backend_options, "device_maps", {}).items():
        native_options.set_device_map(
            str(worker),
            {str(source): str(target) for source, target in mapping.items()},
        )
    return native_module.TensorPipeAgent(
        store,
        str(name),
        int(rank),
        int(world_size),
        native_options,
    )


def _create_native_agent(store: Any, name: str, rank: int, world_size: int, rpc_backend_options: Any, native: Any = None) -> _NativeRpcAgent:
    if native is None:
        native = _construct_native_agent(store, name, rank, world_size, rpc_backend_options)
    agent = _NativeRpcAgent(
        name,
        rank,
        world_size,
        rpc_backend_options,
        native,
    )
    agent.store = store
    return agent


def _is_current_rpc_agent_set() -> bool:
    return _agent is not None


def is_available() -> bool:
    try:
        return _load_native_runtime() is not None
    except RuntimeError:
        return False


def _load_native_runtime() -> Any:
    try:
        import tensorplay

        runtime = getattr(tensorplay._C, "_distributed_rpc")
    except (AttributeError, ImportError) as exc:
        raise RuntimeError("the native RPC runtime is not built") from exc
    return runtime


def _init_rpc_states(agent: Any) -> None:
    global _agent, _current_worker, _workers, _executor, _native_runtime
    global _ALL_WORKER_NAMES
    with _state_lock:
        if _agent is not None and _agent is not agent:
            raise RuntimeError("RPC is already initialized")
        _agent = agent
        native = getattr(agent, "native", None)
        if native is not None and hasattr(native, "start"):
            native.start()
        _current_worker = agent.get_worker_info()
        _workers = {info.name: info for info in agent.get_worker_infos()}
        _workers.setdefault(_current_worker.name, _current_worker)
        _ALL_WORKER_NAMES = set(_workers)
        _executor = None


def _gather_to_leader(
    sequence_id: str,
    worker_name: str,
    obj: Any,
    worker_names: set[str] | None = None,
) -> None:
    with _all_gather_dict_lock:
        expected = set(worker_names) if worker_names else set(_ALL_WORKER_NAMES)
        if worker_name not in expected:
            raise AssertionError(f"{worker_name} is not expected by leader")
        states = _all_gather_sequence_id_to_states[sequence_id]
        if worker_name in states.gathered_objects:
            raise AssertionError(
                f"{worker_name} reported sequence id {sequence_id} twice"
            )
        states.gathered_objects[worker_name] = obj
        if expected == set(states.gathered_objects):
            states.proceed_signal.set()


def _broadcast_to_followers(sequence_id: str, objects_map: dict[str, Any]) -> None:
    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]
    if states.proceed_signal.is_set():
        raise AssertionError(
            f"termination signal sequence id {sequence_id} was set twice"
        )
    states.gathered_objects = dict(objects_map)
    states.proceed_signal.set()


def init_rpc(
    name: str,
    backend: Any = BackendType.TENSORPIPE,
    rank: int = -1,
    world_size: int = -1,
    rpc_backend_options: Any = None,
) -> None:
    global _native_runtime
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    with _state_lock:
        if _agent is not None:
            raise RuntimeError("RPC is already initialized")
    if rank == -1:
        rank = 0
    if world_size == -1:
        world_size = 1
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank and world_size must describe a valid worker")
    if rpc_backend_options is None:
        rpc_backend_options = construct_rpc_backend_options(backend)
    timeout = float(getattr(rpc_backend_options, "rpc_timeout", rpc_constants.DEFAULT_RPC_TIMEOUT_SEC))
    native_agent = None
    try:
        agent = init_backend(
            backend,
            None,
            name,
            rank,
            world_size,
            rpc_backend_options,
        )
        native_agent = agent.native
        _native_runtime = native_agent
        if _agent is None:
            _init_rpc_states(agent)
        if world_size > 1:
            _all_gather(None, timeout=timeout)
            native_agent.barrier(list(_workers), timeout)
    except BaseException:
        if native_agent is not None:
            native_agent.shutdown()
        _reset_current_rpc_agent()
        raise


def _reset_current_rpc_agent() -> None:
    global _agent, _current_worker, _workers, _executor, _native_runtime
    global _ALL_WORKER_NAMES
    with _state_lock:
        _agent = None
        _current_worker = None
        _workers = {}
        _executor = None
        _native_runtime = None
        _ALL_WORKER_NAMES = set()
        with _all_gather_dict_lock:
            _all_gather_sequence_id.clear()
            _all_gather_sequence_id_to_states.clear()
        _pending.clear()
        _rrefs.clear()


def _get_current_rpc_agent() -> Any:
    if _agent is None:
        raise RuntimeError("RPC has not been initialized")
    return _agent


def get_rpc_timeout() -> float:
    if _agent is None:
        return rpc_constants.DEFAULT_RPC_TIMEOUT_SEC
    native = getattr(_agent, "native", None)
    if native is not None and hasattr(native, "get_rpc_timeout"):
        return float(native.get_rpc_timeout())
    return float(getattr(_agent.get_backend_options(), "rpc_timeout", rpc_constants.DEFAULT_RPC_TIMEOUT_SEC))


def _set_rpc_timeout(timeout: float) -> None:
    timeout = float(timeout)
    if timeout < 0.0:
        raise ValueError("RPC timeout must be non-negative")
    if _agent is None:
        raise RuntimeError("RPC has not been initialized")
    native = getattr(_agent, "native", None)
    if native is not None and hasattr(native, "set_rpc_timeout"):
        native.set_rpc_timeout(timeout)
    options = _agent.get_backend_options()
    if hasattr(options, "rpc_timeout"):
        options.rpc_timeout = timeout


def _require_initialized(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _agent is None:
            raise RuntimeError("RPC has not been initialized. Call init_rpc first.")
        return fn(*args, **kwargs)

    return wrapper


def _to_worker_info(to: Any) -> WorkerInfo:
    if isinstance(to, WorkerInfo):
        return to
    if isinstance(to, int):
        return get_worker_info(to)
    if isinstance(to, str):
        if to.startswith("rank:") and to[5:].isdigit():
            return _to_worker_info(int(to[5:]))
        if to in _workers:
            return _workers[to]
        if _agent is not None:
            return _agent.get_worker_info(to)
    raise ValueError(f"cannot resolve worker {to!r}")


def _resolve_timeout(timeout: float | None) -> float | None:
    if timeout in (rpc_constants.UNSET_RPC_TIMEOUT, None):
        timeout = get_rpc_timeout()
    if timeout == 0:
        return None
    return float(timeout)


def _validate_rpc_call(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if not callable(func):
        raise TypeError("function should be callable")
    _default_pickler.serialize(PythonUDF(func, args, kwargs))


def _resolve_future(value: Any) -> Any:
    if isinstance(value, _Future):
        return value.wait()
    if isinstance(value, ConcurrentFuture):
        return value.result()
    if hasattr(value, "wait") and callable(value.wait) and hasattr(value, "then"):
        return value.wait()
    return value


def _execute(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], target: WorkerInfo) -> Any:
    started = time.perf_counter()
    previous = getattr(_thread_local, "in_rpc", False)
    _thread_local.in_rpc = True
    try:
        result = func(*args, **kwargs)
        if hasattr(func, "_wrapped_async_rpc_function"):
            result = _resolve_future(result)
        if result.__class__.__name__ == "RemoteException":
            _handle_exception(result)
        return result
    finally:
        _thread_local.in_rpc = previous
        try:
            from .server_process_global_profiler import _record_server_event

            current_name = _current_worker.name if _current_worker is not None else "local"
            _record_server_event(
                getattr(func, "__qualname__", repr(func)),
                started,
                time.perf_counter(),
                current_name,
                target.name,
            )
        except Exception:
            pass


def _track_future(future: _Future[Any]) -> _Future[Any]:
    with _state_lock:
        _pending.add(future)

    def remove(_: Any) -> None:
        with _state_lock:
            _pending.discard(future)

    future._future.add_done_callback(remove)
    return future


def _submit(
    target: WorkerInfo,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float = rpc_constants.UNSET_RPC_TIMEOUT,
) -> _Future[Any]:
    if _native_runtime is not None:
        _validate_rpc_call(func, args, kwargs)
        native_future = _native_runtime.submit(
            target.name,
            func,
            tuple(args),
            dict(kwargs),
            float(timeout),
        )
        return _track_future(_Future(native_future))
    raise RuntimeError("native RPC runtime is not running")


@contextlib.contextmanager
def _use_rpc_pickler(rpc_pickler: _InternalRPCPickler):
    global _default_pickler
    old = _default_pickler
    _default_pickler = rpc_pickler
    try:
        yield
    finally:
        _default_pickler = old


@contextlib.contextmanager
def _wait_all():
    futures: list[_Future[Any]] = []
    old = getattr(_thread_local, "future_list", None)
    _thread_local.future_list = futures
    try:
        yield
    finally:
        try:
            for future in futures:
                future.wait()
        finally:
            if old is None:
                del _thread_local.future_list
            else:
                _thread_local.future_list = old


def _all_gather(
    obj: Any,
    worker_names: set[str] | None = None,
    timeout: float = rpc_constants.UNSET_RPC_TIMEOUT,
) -> dict[str, Any]:
    if _native_runtime is not None:
        names = list(worker_names or _workers)
        return dict(_native_runtime.all_gather(obj, names, float(timeout)))
    if not _ALL_WORKER_NAMES:
        raise RuntimeError("RPC has not been initialized")
    expected = set(worker_names or _ALL_WORKER_NAMES)
    leader_name = min(expected)
    self_name = get_worker_info().name
    with _all_gather_dict_lock:
        concat_names = "".join(sorted(expected))
        sequence_num = _all_gather_sequence_id.get(concat_names, 0)
        _all_gather_sequence_id[concat_names] = sequence_num + 1
        sequence_id = concat_names + str(sequence_num)

    is_leader = leader_name == self_name
    if timeout == rpc_constants.UNSET_RPC_TIMEOUT:
        rpc_timeout = get_rpc_timeout()
        signal_timeout = None
    elif timeout == rpc_constants.DEFAULT_SHUTDOWN_TIMEOUT:
        rpc_timeout = timeout
        signal_timeout = None
    else:
        rpc_timeout = timeout
        signal_timeout = timeout

    if is_leader:
        _gather_to_leader(sequence_id, self_name, obj, expected)
    else:
        rpc_sync(
            leader_name,
            _gather_to_leader,
            args=(sequence_id, self_name, obj, expected),
            timeout=rpc_timeout,
        )

    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]
    states.proceed_signal.wait(timeout=signal_timeout)

    if is_leader:
        futures: dict[str, _Future[Any]] = {}
        for follower_name in expected - {leader_name}:
            futures[follower_name] = rpc_async(
                follower_name,
                _broadcast_to_followers,
                args=(sequence_id, states.gathered_objects),
                timeout=rpc_timeout,
            )
        for future in futures.values():
            future.wait()

    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states.pop(sequence_id)
    return dict(states.gathered_objects)


def _barrier(
    worker_names: list[str] | set[str] | None = None,
    timeout: float = rpc_constants.UNSET_RPC_TIMEOUT,
) -> None:
    _all_gather(None, set(worker_names or _workers), timeout=timeout)


def _wait_all_workers(timeout: float = rpc_constants.DEFAULT_SHUTDOWN_TIMEOUT) -> None:
    _all_gather(None, timeout=timeout)


def _finalize_shutdown() -> None:
    agent = _agent
    if agent is None:
        return
    try:
        agent.shutdown()
    finally:
        _reset_current_rpc_agent()


@_require_initialized
def shutdown(graceful: bool = True, timeout: float = rpc_constants.DEFAULT_SHUTDOWN_TIMEOUT) -> None:
    try:
        if graceful:
            _wait_all_workers(timeout)
            _get_current_rpc_agent().join(shutdown=True, timeout=timeout)
    finally:
        _finalize_shutdown()


def _rref_typeof_on_owner(rref: "RRef[Any]", blocking: bool = True) -> Any:
    result = type(rref.local_value())
    return result if blocking else _Future.completed(result)


def _rref_typeof_on_user(rref: "RRef[Any]", timeout: float = rpc_constants.UNSET_RPC_TIMEOUT, blocking: bool = True) -> Any:
    future = rpc_async(rref.owner(), _rref_typeof_on_owner, args=(rref,), timeout=timeout)
    return future.wait() if blocking else future


class RRef(Generic[T]):
    def __init__(
        self,
        value: T | object = _MISSING,
        owner: Any = None,
        _future: _Future[T] | None = None,
        _native: Any = None,
    ) -> None:
        self._native = _native
        if _native is not None:
            self._future = None
            self._owner = str(_native.owner())
            self._id = tuple(_native.rref_id())
            self._confirmed = bool(_native.confirmed_by_owner())
            _rrefs[self._id] = self
            return
        if _future is None:
            if value is _MISSING:
                raise TypeError("RRef requires a value or future")
            _future = _Future.completed(value)  # type: ignore[arg-type]
        self._future = _future
        self._owner = _to_worker_info(owner).name if owner is not None else (_current_worker.name if _current_worker is not None else "local")
        self._id = uuid.uuid4().hex
        self._confirmed = True
        _rrefs[self._id] = self

    def owner(self) -> str:
        if self._native is not None:
            return str(self._native.owner())
        return self._owner

    def to_here(self, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT) -> T:
        if self._native is not None:
            return self._native.to_here(float(_resolve_timeout(timeout) or -1.0))
        return self._future.wait(_resolve_timeout(timeout))

    def local_value(self) -> T:
        if self._native is not None:
            return self._native.local_value()
        return self.to_here()

    def backward(
        self,
        dist_autograd_ctx_id: int = -1,
        retain_graph: bool = False,
    ) -> None:
        if self._native is None:
            raise RuntimeError("RRef is not backed by the native runtime")
        self._native.backward(int(dist_autograd_ctx_id), bool(retain_graph))

    def confirmed_by_owner(self) -> bool:
        if self._native is not None:
            return bool(self._native.confirmed_by_owner())
        return self._confirmed and self._future.done()

    def is_owner(self) -> bool:
        if self._native is not None:
            return bool(self._native.is_owner())
        return _current_worker is not None and self._owner == _current_worker.name

    def fork(self) -> "RRef[T]":
        if self._native is not None:
            return RRef(_native=self._native.fork())
        forked = object.__new__(RRef)
        forked._native = None
        forked._future = self._future
        forked._owner = self._owner
        forked._id = uuid.uuid4().hex
        forked._confirmed = self._confirmed
        _rrefs[forked._id] = forked
        return forked

    def _get_type(self, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT, blocking: bool = True) -> Any:
        if self._native is not None and not self.is_owner():
            return _rref_typeof_on_user(self, timeout, blocking)
        result = type(self.local_value())
        return result if blocking else _Future.completed(result)

    def _serialize(self) -> dict[str, Any]:
        owner_id = _workers.get(self.owner())
        if self._native is not None:
            return {
                "owner": self.owner(),
                "owner_id": int(owner_id.id if owner_id is not None else self._native.owner_id()),
                "id": tuple(self._native.rref_id()),
                "fork_id": tuple(self._native.fork_id()),
            }
        return {"owner": self._owner, "id": self._id, "value": self.to_here()}

    @classmethod
    def _deserialize(cls, data: dict[str, Any]) -> "RRef[Any]":
        if "fork_id" in data and _native_runtime is not None:
            return cls(
                _native=_native_runtime.restore_rref(
                    str(data["owner"]),
                    int(data["owner_id"]),
                    tuple(data["id"]),
                    tuple(data["fork_id"]),
                )
            )
        return cls(data["value"], owner=data.get("owner"))

    def __reduce__(self):
        return (type(self)._deserialize, (self._serialize(),))

    def rpc_sync(self, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT):
        from .rref_proxy import RRefProxy

        return RRefProxy(self, rpc_sync, timeout)

    def rpc_async(self, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT):
        from .rref_proxy import RRefProxy

        return RRefProxy(self, rpc_async, timeout)

    def remote(self, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT):
        from .rref_proxy import RRefProxy

        return RRefProxy(self, remote, timeout)

    def __repr__(self) -> str:
        return f"RRef(owner={self._owner!r}, id={self._id!r})"


def method_factory(method_name: str, docstring: str | None = None):
    def method(self: RRef[Any], *args: Any, **kwargs: Any) -> Any:
        return getattr(super(RRef, self), method_name)(*args, **kwargs)

    method.__doc__ = docstring
    return method


new_method = method_factory


@_require_initialized
def remote(to: Any, func: Any, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT) -> RRef[Any]:
    target = _to_worker_info(to)
    if _native_runtime is None:
        raise RuntimeError("native RPC runtime is not running")
    call_args = tuple(args or ())
    call_kwargs = dict(kwargs or {})
    _validate_rpc_call(func, call_args, call_kwargs)
    native_rref = _native_runtime.remote(
        target.name,
        func,
        call_args,
        call_kwargs,
        float(timeout),
    )
    return RRef(_native=native_rref)


def _invoke_rpc(to: Any, func: Any, rpc_type: RPCExecMode, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT) -> Any:
    call_args = tuple(args or ())
    call_kwargs = dict(kwargs or {})
    if not callable(func):
        raise TypeError("function should be callable")
    if rpc_type is RPCExecMode.SYNC:
        return rpc_sync(to, func, call_args, call_kwargs, timeout)
    if rpc_type is RPCExecMode.REMOTE:
        return remote(to, func, call_args, call_kwargs, timeout)
    return rpc_async(to, func, call_args, call_kwargs, timeout)


@_require_initialized
def rpc_sync(to: Any, func: Any, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT) -> Any:
    target = _to_worker_info(to)
    call_args = tuple(args or ())
    call_kwargs = dict(kwargs or {})
    if getattr(_thread_local, "in_rpc", False) and _current_worker is not None and target.name == _current_worker.name:
        return _execute(func, call_args, call_kwargs, target)
    return _submit(target, func, call_args, call_kwargs, timeout).wait(_resolve_timeout(timeout))


@_require_initialized
def rpc_async(to: Any, func: Any, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, timeout: float = rpc_constants.UNSET_RPC_TIMEOUT) -> _Future[Any]:
    target = _to_worker_info(to)
    future = _submit(target, func, tuple(args or ()), dict(kwargs or {}), timeout)
    future.rpc_timeout = timeout  # type: ignore[attr-defined]
    future_list = getattr(_thread_local, "future_list", None)
    if future_list is not None:
        future_list.append(future)
    return future


def _get_should_profile() -> bool:
    return False


def _enable_rpc_profiler(should_profile: bool, qualified_name: str | None, func: Any, rpc_type: RPCExecMode, dst_worker_info: WorkerInfo):
    if not should_profile:
        return contextlib.nullcontext()
    name = qualified_name or getattr(func, "__qualname__", repr(func))
    key = _build_rpc_profiling_key(rpc_type, name, get_worker_info().name, dst_worker_info.name)
    return _record_context(key)


@contextlib.contextmanager
def _record_context(key: str):
    started = time.perf_counter()
    try:
        yield key
    finally:
        try:
            from .server_process_global_profiler import _record_server_event

            current = get_worker_info().name
            _record_server_event(key, started, time.perf_counter(), current, current)
        except Exception:
            pass
