"""Handler dumping live thread stacks of this process."""
import sys
import traceback

from .handlers import DebugHandler

__all__ = ["StacksHandler"]


class StacksHandler(DebugHandler):
    """Report the stack of every live thread (optionally one thread)."""

    def __init__(self) -> None:
        super().__init__(name="stacks")

    def handle_request(self, request: dict) -> dict:
        thread_id = request.get("thread_id")
        frames = sys._current_frames()
        threads = {t.ident: t.name for t in __import__("threading").enumerate()}
        out = {}
        for ident, frame in frames.items():
            if thread_id is not None and str(ident) != str(thread_id):
                continue
            stack = "".join(traceback.format_stack(frame)).strip().splitlines()
            out[str(ident)] = {
                "name": threads.get(ident, "unknown"),
                "stack": stack[-40:],
            }
        return {"stacks": out}
