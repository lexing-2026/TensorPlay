from __future__ import annotations

import binascii
from base64 import b64decode, b64encode

from . import _etcd_stub as etcd
from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .core_rendezvous_backend import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint

__all__ = ["EtcdRendezvousBackend", "create_backend"]


class EtcdRendezvousBackend(RendezvousBackend):
    _DEFAULT_TTL = 7200

    def __init__(self, client, run_id: str, key_prefix: str | None = None, ttl: int | None = None):
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self._client = client
        self._key = f"{key_prefix.rstrip('/')}/{run_id}" if key_prefix else run_id
        self._ttl = int(ttl) if ttl and ttl > 0 else self._DEFAULT_TTL

    @property
    def name(self) -> str:
        return "etcd-v2"

    def get_state(self) -> tuple[bytes, Token] | None:
        try:
            result = self._client.read(self._key)
        except etcd.EtcdKeyNotFound:
            return None
        except Exception as exc:
            raise RendezvousConnectionError(f"etcd state read failed: {exc}") from exc
        try:
            return b64decode((result.value or "").encode()), int(result.modifiedIndex)
        except (ValueError, TypeError, binascii.Error) as exc:
            raise RendezvousStateError("invalid rendezvous state") from exc

    def set_state(self, state: bytes, token: Token | None = None):
        encoded = b64encode(state).decode()
        try:
            if token:
                current = self._client.read(self._key)
                if int(current.modifiedIndex) != int(token):
                    return self.get_state()
                result = self._client.write(self._key, encoded, ttl=self._ttl, prevIndex=int(token))
            else:
                result = self._client.write(self._key, encoded, ttl=self._ttl, prevExist=False)
            return b64decode(result.value.encode()), int(result.modifiedIndex)
        except (etcd.EtcdAlreadyExist, etcd.EtcdCompareFailed):
            return self.get_state()
        except Exception as exc:
            raise RendezvousConnectionError(f"etcd state update failed: {exc}") from exc


def _create_etcd_client(params: RendezvousParameters):
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)
    timeout = params.get_as_int("read_timeout", 60)
    if timeout is not None and timeout <= 0:
        raise ValueError("read_timeout must be positive")
    try:
        return etcd.Client(host, port, read_timeout=timeout or 60)
    except Exception as exc:
        raise RendezvousConnectionError(f"unable to create etcd client: {exc}") from exc


def create_backend(params: RendezvousParameters):
    client = _create_etcd_client(params)
    prefix = params.get("etcd_prefix", "/tp/elastic/rendezvous")
    backend = EtcdRendezvousBackend(client, params.run_id, key_prefix=prefix, ttl=params.get_as_int("ttl", 7200))
    store = EtcdStore(client, params.get("store_prefix", "/tp/elastic/store"))
    return backend, store
