"""Rendezvous backend using an atomic shared key/value store."""

from __future__ import annotations

import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from typing import Any

from tensorplay.distributed import FileStore, Store, TCPStore
from tensorplay.distributed.elastic.events import (
    NodeState,
    construct_and_record_rdzv_event,
)

from .api import (
    RendezvousConnectionError,
    RendezvousError,
    RendezvousParameters,
    RendezvousStateError,
)
from .core_rendezvous_backend import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint

__all__ = ["C10dRendezvousBackend", "create_backend"]


logger = logging.getLogger(__name__)
DEFAULT_PORT = 29400


class C10dRendezvousBackend(RendezvousBackend):
    """Store rendezvous state behind compare-and-set operations."""

    _NULL_SENTINEL = "Y2FuaW1hZGFt"

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")
        self._store = store
        self._key = "tp.rendezvous." + run_id
        self._call_store("compare_set", self._key, "", self._NULL_SENTINEL)

    @property
    def name(self) -> str:
        return "c10d"

    def get_state(self) -> tuple[bytes, Token] | None:
        return self._decode_state(self._call_store("get", self._key))

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        encoded = b64encode(state).decode()
        if token:
            if not isinstance(token, bytes):
                current = self.get_state()
                return (*current, False) if current is not None else None
            expected = token.decode()
        else:
            expected = self._NULL_SENTINEL

        observed = self._call_store(
            "compare_set", self._key, expected, encoded
        )
        state_token = self._decode_state(observed)
        if state_token is None:
            return None
        current_state, current_token = state_token
        return current_state, current_token, current_state == state

    def _call_store(self, store_op: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to the rendezvous store has failed. "
                "See the inner exception for details."
            ) from exc

    def _decode_state(self, encoded: bytes) -> tuple[bytes, Token] | None:
        if encoded == self._NULL_SENTINEL.encode():
            return None
        try:
            state = b64decode(encoded)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise RendezvousStateError(
                "The rendezvous state object is corrupt. "
                "See the inner exception for details."
            ) from exc
        return state, encoded


def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)
    configured_host = params.get_as_bool("is_host")
    is_host = (
        configured_host
        if configured_host is not None
        else _matches_machine_hostname(host)
    )
    read_timeout = params.get_as_int("read_timeout", 60)
    if read_timeout is None or read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    for is_server in (is_host, False):
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                timeout=float(read_timeout),
                wait_for_workers=False,
            )
            if is_server:
                message = f"Process {os.getpid()} hosts the rendezvous TCP store."
                construct_and_record_rdzv_event(
                    run_id=params.run_id,
                    message=message,
                    node_state=NodeState.INITIALIZED,
                )
                logger.info(message)
            return store
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            if not is_server or configured_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the rendezvous store has failed. "
                    "See the inner exception for details."
                ) from exc
    raise RendezvousConnectionError(
        "The connection to the rendezvous store has failed."
    )


def _create_file_store(params: RendezvousParameters) -> FileStore:
    path = params.endpoint
    if not path:
        try:
            _, path = tempfile.mkstemp()
        except OSError as exc:
            raise RendezvousError(
                "The file creation for the rendezvous store has failed. "
                "See the inner exception for details."
            ) from exc
    try:
        return FileStore(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RendezvousConnectionError(
            "The connection to the rendezvous store has failed. "
            "See the inner exception for details."
        ) from exc


def create_backend(
    params: RendezvousParameters,
) -> tuple[C10dRendezvousBackend, Store]:
    store_type = str(params.get("store_type", "tcp")).strip().lower()
    try:
        if store_type == "file":
            store = _create_file_store(params)
        elif store_type == "tcp":
            store = _create_tcp_store(params)
        else:
            raise ValueError(
                "Invalid store type. Supported values are 'file' and 'tcp'."
            )
        backend = C10dRendezvousBackend(store, params.run_id)
    except Exception as exc:
        construct_and_record_rdzv_event(
            message=f"{type(exc).__name__}: {exc}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
    return backend, store
