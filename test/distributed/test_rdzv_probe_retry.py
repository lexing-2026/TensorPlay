from tensorplay.distributed.elastic.rendezvous.api import RendezvousConnectionError
from tensorplay.distributed.elastic.rendezvous.core_rendezvous_backend import (
    CoreRendezvousBackend,
)


class _FlakyStore:
    """Store whose first ``failures`` probes raise a transport error."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def has(self, key: str) -> bool:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("TCPStore check: request failed")
        return True

    def get(self, key: str, timeout=None) -> bytes:
        return b"1:e30="


def test_probe_retries_transient_failures():
    store = _FlakyStore(failures=2)
    backend = CoreRendezvousBackend(store, "run")

    state, token = backend.get_state()

    assert (state, token) == (b"{}", 1)
    assert store.calls == 3


def test_probe_gives_up_after_configured_attempts():
    store = _FlakyStore(failures=99)
    backend = CoreRendezvousBackend(store, "run")

    try:
        backend.get_state()
    except RendezvousConnectionError as error:
        assert "probe" in str(error)
    else:
        raise AssertionError("expected RendezvousConnectionError")
    assert store.calls == backend._PROBE_ATTEMPTS
