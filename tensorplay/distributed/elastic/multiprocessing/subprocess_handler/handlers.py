from __future__ import annotations

from .subprocess_handler import SubprocessHandler

__all__ = ["get_subprocess_handler"]


def get_subprocess_handler(
    entrypoint: str,
    args: tuple,
    env: dict[str, str],
    stdout: str | None,
    stderr: str | None,
    local_rank_id: int,
    numa_options=None,
) -> SubprocessHandler:
    del numa_options
    command = (entrypoint, *tuple(str(value) for value in args))
    return SubprocessHandler(
        args=command,
        env=env,
        stdout=stdout,
        stderr=stderr,
        local_rank_id=local_rank_id,
    )
