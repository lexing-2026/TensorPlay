"""Handler registry of the debug server."""
import abc

__all__ = ["DebugHandler", "register_handler", "get_handler", "list_handlers"]

_handlers: dict[str, "DebugHandler"] = {}


class DebugHandler(abc.ABC):
    """One named report served over the debug endpoint."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def handle_request(self, request: dict) -> dict:
        """Produce a JSON-serializable response for ``request``."""
        ...


def register_handler(handler: DebugHandler) -> None:
    """Register (or replace) ``handler`` under its name."""
    _handlers[handler.name] = handler


def get_handler(name: str) -> DebugHandler | None:
    return _handlers.get(name)


def list_handlers() -> list[str]:
    return sorted(_handlers)
