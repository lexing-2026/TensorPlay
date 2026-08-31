"""Store-backed rendezvous state persistence.

The rendezvous state is a small JSON document guarded by a version token:
readers fetch ``(state, token)``, writers attempt an atomic
compare-and-swap and retry on conflict. The document is base64 encoded so
it survives both the line-oriented file store and the hex-encoded TCP
wire protocol.
"""
import base64
import json
import os
import tempfile

from tensorplay.distributed import FileStore, Store, StoreTimeoutError, TCPStore

from .api import RendezvousConnectionError, RendezvousParameters
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint

Token = int


class RendezvousBackend:
    """Persistence interface of the rendezvous state."""

    @property
    def name(self) -> str:
        """Backend name."""
        raise NotImplementedError

    def get_state(self) -> tuple[bytes, Token] | None:
        """Return ``(state, token)`` or None when no state exists yet."""
        raise NotImplementedError

    def set_state(self, value: bytes, token: Token) -> tuple[bytes, Token] | None:
        """CAS-write ``value`` under ``token``; returns the freshest state on conflict."""
        raise NotImplementedError


class CoreRendezvousBackend(RendezvousBackend):
    """Rendezvous state stored in a shared key/value store.

    The store is typically a TCPStore hosted next to the rendezvous endpoint
    or a FileStore on a shared filesystem. Every store mutation goes through
    an atomic compare-and-swap, so multiple agents race on the same key
    without corrupting the state document.
    """

    def __init__(self, store: Store, run_id: str) -> None:
        self._store = store
        self._run_id = run_id
        self._key = f"tp_elastic/rdzv/{run_id}/state"
        self._unavailable = False

    @property
    def name(self) -> str:
        return "core"

    @property
    def store(self) -> Store:
        return self._store

    @property
    def unavailable(self) -> bool:
        """True once the store proved unreachable (fast-fails all calls)."""
        return self._unavailable

    def get_state(self) -> tuple[bytes, Token] | None:
        self._check_reachable()
        if not self._exists(self._key):
            return None
        base64_state = self._call_store("get", self._key, timeout=1)
        return self._decode_state(base64_state)

    def set_state(self, value: bytes, token: Token) -> tuple[bytes, Token] | None:
        self._check_reachable()
        current_raw = self._read_raw()
        current = self._decode_state(current_raw)
        if current is not None:
            current_state, current_token = current
            if current_token != token:
                return current
        else:
            current_raw = b""
        new_raw = self._encode_state(value, token + 1).encode()
        try:
            swapped, current_raw = self._store.compare_and_swap(
                self._key, current_raw, new_raw
            )
        except Exception as e:
            raise self._store_error("compare-and-swap", e) from e
        if swapped:
            return value, token + 1
        decoded = self._decode_state(current_raw)
        if decoded is None:
            return b"", 0
        return decoded

    def _check_reachable(self) -> None:
        if self._unavailable:
            raise RendezvousConnectionError(
                "The rendezvous store was marked unreachable"
            )

    def _store_error(self, op: str, error: Exception) -> RendezvousConnectionError:
        """Classify a store failure; connection-level errors poison the backend."""
        if isinstance(error, (ConnectionError, OSError)) or "could not connect" in str(
            error
        ):
            self._mark_unreachable(error)
        return RendezvousConnectionError(
            f"Rendezvous store operation '{op}' failed: {error}"
        )

    def _mark_unreachable(self, error: Exception) -> None:
        self._unavailable = True

    def _read_raw(self) -> bytes:
        if not self._exists(self._key):
            return b""
        try:
            return self._call_store("get", self._key, timeout=1)
        except RendezvousConnectionError:
            return b""

    def _exists(self, key: str) -> bool:
        probe = getattr(self._store, "has", None)
        if probe is not None:
            try:
                return bool(probe(key))
            except Exception as e:
                raise self._store_error("probe", e) from e
        try:
            self._store.get(key, timeout=0.3)
            return True
        except Exception as e:
            if isinstance(e, StoreTimeoutError) and "could not connect" not in str(e):
                return False
            raise self._store_error("probe", e) from e

    def _call_store(self, store_op: str, *args, **kwargs):
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except RendezvousConnectionError:
            raise
        except Exception as e:
            raise self._store_error(store_op, e) from e

    def _encode_state(self, state: bytes, token: Token) -> str:
        payload = base64.b64encode(state).decode()
        return f"{token}:{payload}"

    def _decode_state(self, raw: bytes) -> tuple[bytes, Token] | None:
        if not raw:
            return None
        try:
            text = raw.decode() if isinstance(raw, bytes) else raw
            token_str, _, payload = text.partition(":")
            state = base64.b64decode(payload)
            return state, int(token_str)
        except (ValueError, TypeError):
            return None


def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    """Create the rendezvous TCPStore for ``params``.

    The node matching the endpoint host starts the server (unless
    ``start_daemon`` is disabled); when the port is already served by a
    co-located agent it attaches as a client instead of failing.
    """
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=29400)
    if not params.endpoint:
        port = 0
    is_host = _matches_machine_hostname(host)
    start_daemon = params.get_as_bool("start_daemon", True)
    read_timeout = params.get_as_int("read_timeout", 20)
    if is_host and start_daemon:
        try:
            return TCPStore(
                host,
                port,
                world_size=params.max_nodes * params.local_world_size,
                is_master=True,
                timeout=float(read_timeout),
                wait_for_workers=False,
            )
        except OSError:
            pass
    try:
        return TCPStore(
            host,
            port,
            world_size=params.max_nodes * params.local_world_size,
            is_master=False,
            timeout=float(read_timeout),
            wait_for_workers=False,
        )
    except Exception as e:
        raise RendezvousConnectionError(
            f"Failed to connect to the rendezvous endpoint {host}:{port}: {e}"
        ) from e


def _create_file_store(params: RendezvousParameters) -> FileStore:
    """Create a FileStore shared by all rendezvous participants."""
    path = params.get("store_path")
    if not path:
        path = os.path.join(
            tempfile.gettempdir(), f"tp_rdzv_{params.run_id}", "state"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return FileStore(path, world_size=-1)


def create_backend(
    params: RendezvousParameters,
    store_type: str = "tcp",
) -> tuple[CoreRendezvousBackend, Store]:
    """Assemble the backend plus its store for ``params``.

    ``store_type`` selects ``tcp`` (default) or ``file``.
    """
    if store_type == "file":
        store = _create_file_store(params)
    else:
        store = _create_tcp_store(params)
    backend = CoreRendezvousBackend(store, params.run_id)
    return backend, store
