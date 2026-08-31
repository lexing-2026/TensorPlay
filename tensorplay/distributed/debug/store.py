"""Handler inspecting a remote store's keys and values."""
from tensorplay.distributed import Store, StoreTimeoutError, TCPStore

from .handlers import DebugHandler

__all__ = ["StoreDumpHandler"]


class StoreDumpHandler(DebugHandler):
    """Read keys from a TCPStore endpoint (set up via the request args).

    Request args: ``host``, ``port``, ``keys`` (list) or ``prefix``. Only
    explicitly requested keys are read; there is no key-listing primitive,
    so ``prefix`` expects clients to know candidate key names.
    """

    def __init__(self) -> None:
        super().__init__(name="store")

    def handle_request(self, request: dict) -> dict:
        host = request.get("host")
        port = request.get("port")
        if not host or not port:
            return {"error": "store handler requires 'host' and 'port'"}
        try:
            store = TCPStore(host, int(port), world_size=-1, is_master=False,
                             timeout=5.0, wait_for_workers=False)
        except Exception as e:
            return {"error": f"cannot reach store at {host}:{port}: {e!r}"}
        keys = request.get("keys") or []
        prefix = request.get("prefix")
        if prefix and not keys:
            return {
                "error": "prefix scanning is not supported by the store; pass 'keys'"
            }
        out: dict[str, str] = {}
        for key in keys:
            try:
                out[str(key)] = store.get(str(key), timeout=2).decode(errors="replace")
            except (StoreTimeoutError, Exception) as e:
                out[str(key)] = f"<unreadable: {e!r}>"
        return {"values": out}
