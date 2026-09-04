from __future__ import annotations

import collections
import contextlib
import copyreg
import io
import pickle
import threading
import time
import traceback
from enum import Enum
from typing import Any

__all__ = ["RPCExecMode", "serialize", "deserialize", "PythonUDF", "RemoteException"]


class RPCExecMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    ASYNC_JIT = "async_jit"
    REMOTE = "remote"


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg", "exception_type"])
_thread_local_tensor_tables = threading.local()


class _InternalRPCPickler:
    def __init__(self) -> None:
        self._dispatch_table = copyreg.dispatch_table.copy()
        self._class_reducer_dict: dict[type, Any] = {}
        from tensorplay import Tensor

        self._dispatch_table[Tensor] = self._tensor_reducer

    def _register_reducer(self, obj_class: type, reducer: Any) -> None:
        self._class_reducer_dict.setdefault(obj_class, reducer)

    @classmethod
    def _tensor_receiver(cls, tensor_index: int) -> Any:
        return _thread_local_tensor_tables.recv_tables[tensor_index]

    def _tensor_reducer(self, tensor: Any) -> tuple[Any, tuple[int]]:
        send_tables = getattr(_thread_local_tensor_tables, "send_tables", None)
        if send_tables is None:
            raise RuntimeError("tensor reducer is only active during serialization")
        send_tables.append(tensor)
        return self._tensor_receiver, (len(send_tables) - 1,)

    @classmethod
    def _py_rref_receiver(cls, data: Any) -> Any:
        return data

    def _py_rref_reducer(self, rref: Any) -> tuple[Any, tuple[Any]]:
        return self._py_rref_receiver, (rref,)

    def _rref_reducer(self, rref: Any) -> tuple[Any, tuple[Any]]:
        return self._py_rref_reducer(rref)

    def serialize(self, obj: Any) -> tuple[bytes, list[Any]]:
        stream = io.BytesIO()
        old_tables = getattr(_thread_local_tensor_tables, "send_tables", None)
        _thread_local_tensor_tables.send_tables = []
        try:
            pickler = pickle.Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickler.dispatch_table = dict(self._dispatch_table)
            pickler.dispatch_table.update(self._class_reducer_dict)
            pickler.dump(obj)
            tensors = list(_thread_local_tensor_tables.send_tables)
        finally:
            if old_tables is None:
                del _thread_local_tensor_tables.send_tables
            else:
                _thread_local_tensor_tables.send_tables = old_tables
        return stream.getvalue(), tensors

    def deserialize(self, binary_data: bytes, tensor_table: list[Any]) -> Any:
        old_tables = getattr(_thread_local_tensor_tables, "recv_tables", None)
        _thread_local_tensor_tables.recv_tables = tensor_table
        try:
            try:
                return pickle.Unpickler(io.BytesIO(binary_data)).load()
            except (AttributeError, ModuleNotFoundError) as exc:
                error = AttributeError(
                    f"unable to resolve a serialized callable: {exc}"
                )
                error.__cause__ = exc
                return error
        finally:
            if old_tables is None:
                del _thread_local_tensor_tables.recv_tables
            else:
                _thread_local_tensor_tables.recv_tables = old_tables


_internal_rpc_pickler = _InternalRPCPickler()


def serialize(obj: Any) -> tuple[bytes, list[Any]]:
    return _internal_rpc_pickler.serialize(obj)


def deserialize(binary_data: bytes, tensor_table: list[Any]) -> Any:
    return _internal_rpc_pickler.deserialize(binary_data, tensor_table)


def _run_function(python_udf: PythonUDF | Any) -> Any:
    try:
        if isinstance(python_udf, AttributeError):
            raise python_udf
        return python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as exc:
        return RemoteException(f"{exc!r}\n{traceback.format_exc()}", type(exc))


def _handle_exception(result: Any) -> None:
    if not isinstance(result, RemoteException):
        return
    try:
        raise result.exception_type(result.msg)
    except Exception as exc:
        if type(exc) is result.exception_type:
            raise
        raise RuntimeError(f"unable to recreate remote exception: {result.msg}") from exc


def _build_rpc_profiling_key(exec_type: RPCExecMode, func_name: str, current_worker_name: str, dst_worker_name: str) -> str:
    mode = exec_type.value if isinstance(exec_type, RPCExecMode) else str(exec_type)
    return f"rpc_{mode}#{func_name}({current_worker_name} -> {dst_worker_name})"


class _RecordFunction:
    def __init__(self, key: str) -> None:
        self.key = key
        self.start = time.perf_counter()
        self.end: float | None = None

    def end_record(self) -> None:
        if self.end is None:
            self.end = time.perf_counter()

    def __enter__(self) -> "_RecordFunction":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> bool:
        self.end_record()
        return False


def _start_record_function(exec_type: RPCExecMode, func_name: str, current_worker_name: str, dest_worker_name: str) -> _RecordFunction:
    return _RecordFunction(
        _build_rpc_profiling_key(exec_type, func_name, current_worker_name, dest_worker_name)
    )
