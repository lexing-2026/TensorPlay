from __future__ import annotations

import pickle
from collections.abc import Sequence

from .components.builder import build_db, transform_ft
from .components.config_manager import JobConfig
from .components.loader import read_dir
from .components.types import types

__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    config = JobConfig()
    parsed = config.parse_args(args)
    if not parsed.trace_dir:
        raise AssertionError("trace_dir is required")
    details, version = read_dir(parsed)
    if parsed.transform_ft:
        if not parsed.group_world_size:
            raise AssertionError("group world size is required")
        details = transform_ft(details, parsed.group_world_size)
    database = build_db(details, parsed, version)
    if parsed.output:
        with open(parsed.output, "wb") as stream:
            pickle.dump((types, database), stream)


if __name__ == "__main__":
    main()
