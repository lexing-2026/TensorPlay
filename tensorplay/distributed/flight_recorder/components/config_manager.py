from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .fr_logger import FlightRecorderLogger

__all__ = ["JobConfig"]


class JobConfig:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Flight recorder trace analyzer")
        self.parser.add_argument("trace_dir", nargs="?")
        self.parser.add_argument("--selected-ranks", default=None, nargs="+", type=int)
        self.parser.add_argument("--allow-incomplete-ranks", action="store_true")
        self.parser.add_argument("--pg-filters", default=None, nargs="+", type=str)
        self.parser.add_argument("-o", "--output", default=None)
        self.parser.add_argument("-p", "--prefix", default=None)
        self.parser.add_argument("-j", "--just_print_entries", action="store_true")
        self.parser.add_argument("-v", "--verbose", action="store_true")
        self.parser.add_argument("--print_stack_trace", action="store_true")
        self.parser.add_argument("--mismatch_cap", type=int, default=10)
        self.parser.add_argument("--transform-ft", action="store_true")
        self.parser.add_argument("--group-world-size", type=int, default=None)

    def parse_args(self, args: Sequence[str] | None) -> argparse.Namespace:
        parsed = self.parser.parse_args(args)
        if (parsed.selected_ranks is not None or parsed.pg_filters is not None) and not parsed.just_print_entries:
            raise AssertionError("rank and process-group filters require raw entry output")
        if parsed.verbose:
            FlightRecorderLogger().set_log_level(logging.DEBUG)
        return parsed
