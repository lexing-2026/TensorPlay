"""Helper utilities shared by rendezvous backends."""
import fnmatch
import random
import socket
import threading
import time
from datetime import timedelta


def _parse_rendezvous_config(config_str: str) -> dict[str, str]:
    """Parse ``key=value,key2=value2`` into a dict; empty string yields ``{}``."""
    out: dict[str, str] = {}
    for token in filter(None, config_str.split(",")):
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(
                f"Malformed rendezvous config entry '{token}'; expected key=value"
            )
        out[key.strip()] = value.strip()
    return out


def _try_parse_port(port_str: str) -> int | None:
    """Return ``port_str`` as int in [0, 65536] or None when not numeric."""
    try:
        port = int(port_str)
    except ValueError:
        return None
    if port < 0 or port > 65536:
        return None
    return port


def parse_rendezvous_endpoint(endpoint: str, default_port: int) -> tuple[str, int]:
    """Split ``[IPv4][IPv6][hostname]:port`` into (host, port).

    ``port`` defaults to ``default_port`` when the endpoint omits it. IPv6
    addresses must be bracketed to disambiguate the port separator.
    """
    endpoint = endpoint.strip()
    if endpoint.startswith("["):
        host, sep, rest = endpoint.partition("]")
        if not sep:
            raise ValueError(f"Invalid IPv6 endpoint '{endpoint}': missing ']'")
        host = host[1:]
        port_str = rest[1:] if rest.startswith(":") else ""
    else:
        host, sep, port_str = endpoint.partition(":")
        if ":" in port_str:
            # Bare IPv6 literal without a port.
            host, port_str = endpoint, ""
    port = _try_parse_port(port_str) if port_str else None
    if port is None:
        port = default_port
    if not host:
        host = "localhost"
    return host, port


def _matches_machine_hostname(host: str) -> bool:
    """Whether ``host`` refers to this machine (name, FQDN, or local address)."""
    if not host:
        return True
    if host == "*":
        return True
    host = host.lower().strip()
    if fnmatch.fnmatch(socket.gethostname().lower(), host):
        return True
    try:
        fqdn = socket.getfqdn().lower()
        if fnmatch.fnmatch(fqdn, host):
            return True
    except OSError:
        pass
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        addr = socket.gethostbyname(host)
        if addr.startswith("127.") or addr == _primary_addr():
            return True
    except OSError:
        pass
    return False


def _primary_addr() -> str:
    """Address this host would use to reach the internet (no traffic sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return ""
    finally:
        sock.close()


def _delay(seconds: float | tuple[float, float]) -> None:
    """Sleep for a fixed or random-in-range duration (jittered retries)."""
    if isinstance(seconds, tuple):
        seconds = random.uniform(*seconds)
    time.sleep(seconds)


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


class _PeriodicTimer:
    """Background thread invoking an action every ``interval`` seconds."""

    def __init__(
        self,
        interval: timedelta | float,
        action,
        name: str | None = None,
        run_at_start: bool = False,
        start_daemon: bool = True,
    ) -> None:
        if isinstance(interval, timedelta):
            interval = interval.total_seconds()
        self._interval = interval
        self._action = action
        self._name = name
        self._run_at_start = run_at_start
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._daemon = start_daemon

    @property
    def name(self) -> str | None:
        return self._name

    def set_name(self, name: str) -> None:
        self._name = name

    def start(self) -> None:
        """Launch the timer thread; repeated calls are no-ops."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested = False
        self._thread = threading.Thread(target=self._run, daemon=self._daemon)
        if self._name:
            self._thread.name = self._name
        self._thread.start()

    def _run(self) -> None:
        if self._run_at_start:
            self._invoke()
        while not self._stop_requested:
            deadline = time.monotonic() + self._interval
            while not self._stop_requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.1))
            if self._stop_requested:
                break
            self._invoke()

    def _invoke(self) -> None:
        try:
            self._action()
        except Exception:
            pass

    def cancel(self) -> None:
        """Stop the timer; the in-flight action is allowed to finish."""
        self._stop_requested = True
