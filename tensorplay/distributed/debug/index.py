"""Handler listing the available debug handlers."""
from .handlers import DebugHandler, list_handlers

__all__ = ["IndexHandler"]


class IndexHandler(DebugHandler):
    """Report the handlers available on this server."""

    def __init__(self) -> None:
        super().__init__(name="index")

    def handle_request(self, request: dict) -> dict:
        return {"handlers": list_handlers()}
