from __future__ import annotations

__all__ = ["read_proc_state", "is_uninterruptible_state"]


def read_proc_state(pid: int) -> str | None:
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as stream:
            data = stream.read()
    except (OSError, ValueError):
        return None
    closing = data.rfind(")")
    if closing < 0:
        return None
    fields = data[closing + 1 :].split()
    return fields[0] if fields else None


def is_uninterruptible_state(state: str | None) -> bool:
    return state is not None and state.startswith("D")
