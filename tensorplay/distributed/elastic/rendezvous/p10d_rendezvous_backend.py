from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
from datetime import timedelta
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

__all__ = ["P10dRendezvousBackend", "create_backend"]


logger = logging.getLogger(__name__)
DEFAULT_PORT = 29400


class P10dRendezvousBackend(RendezvousBackend):
    """Rendezvous state backend backed by a TP key/value Store."""

    _NULL_SENTINEL = b"tp-rendezvous-null"

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._store = store
        self._key = "tp.rendezvous." + run_id

        try:
            self._store.compare_and_swap(self._key, b"", self._NULL_SENTINEL)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to the rendezvous store has failed. "
                "See the inner exception for details."
            ) from exc

    @property
    def name(self) -> str:
        return "p10d"

    def get_state(self) -> tuple[bytes, Token] | None:
        raw = self._call_store("get", self._key)
        return self._decode_state(raw)

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token] | None:
        if token is None:
            token = 0
        if not isinstance(token, int):
            current = self.get_state()
            return current

        current_raw = self._call_store("get", self._key)
        current = self._decode_state(current_raw)
        if current is None:
            if token != 0:
                return None
            expected = self._NULL_SENTINEL
        else:
            _, current_token = current
            if current_token != token:
                return current
            expected = current_raw

        new_token = token + 1
        encoded = base64.b64encode(state).decode("ascii")
        new_raw = f"{new_token}:{encoded}".encode("ascii")
        swapped, observed = self._call_store(
            "compare_and_swap", self._key, expected, new_raw
        )
        if swapped:
            return state, new_token
        return self._decode_state(observed)

    def _call_store(self, store_op: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to the rendezvous store has failed. "
                "See the inner exception for details."
            ) from exc

    def _decode_state(self, raw: bytes) -> tuple[bytes, Token] | None:
        if raw == self._NULL_SENTINEL:
            return None

        try:
            token_text, separator, payload = raw.decode("ascii").partition(":")
            if not separator:
                raise ValueError("missing state token")
            token = int(token_text)
            state = base64.b64decode(payload, validate=True)
        except (binascii.Error, UnicodeDecodeError, ValueError, TypeError) as exc:
            raise RendezvousStateError(
                "The rendezvous state object is corrupt. "
                "See the inner exception for details."
            ) from exc

        return state, token


def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)

    configured_host = params.get_as_bool("is_host")
    if configured_host is not None:
        is_host = configured_host
    else:
        is_host = _matches_machine_hostname(host)

    read_timeout = params.get_as_int("read_timeout", 60)
    if read_timeout is None or read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    store: TCPStore | None = None
    for is_server in (is_host, False):
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                timeout=timedelta(seconds=read_timeout).total_seconds(),
                wait_for_workers=False,
            )
            if is_server:
                message = (
                    f"Process {os.getpid()} hosts the rendezvous TCP store."
                )
                construct_and_record_rdzv_event(
                    run_id=params.run_id,
                    message=message,
                    node_state=NodeState.INITIALIZED,
                )
                logger.info(message)
            break
        except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
            if not is_server or configured_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the rendezvous store has failed. "
                    "See the inner exception for details."
                ) from exc

    if store is None:
        raise RendezvousConnectionError(
            "The connection to the rendezvous store has failed."
        )
    return store


def _create_file_store(params: RendezvousParameters) -> FileStore:
    path = params.endpoint
    if not path:
        try:
            _, path = tempfile.mkstemp(prefix="tp_rendezvous_")
        except OSError as exc:
            raise RendezvousError(
                "The rendezvous store file could not be created. "
                "See the inner exception for details."
            ) from exc

    try:
        return FileStore(path)
    except (ValueError, RuntimeError, OSError) as exc:
        raise RendezvousConnectionError(
            "The connection to the rendezvous store has failed. "
            "See the inner exception for details."
        ) from exc


def create_backend(
    params: RendezvousParameters,
) -> tuple[P10dRendezvousBackend, Store]:
    """Create a rendezvous backend and its shared Store."""
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
        backend = P10dRendezvousBackend(store, params.run_id)
    except Exception as exc:
        construct_and_record_rdzv_event(
            message=f"{type(exc).__name__}: {exc}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise

    return backend, store
