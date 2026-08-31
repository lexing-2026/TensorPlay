from __future__ import annotations

import enum
import os

__all__ = [
    "AggressiveGuardFreeMode",
    "backed_size_oblivious",
    "check_shape_env_recorded_events",
    "extended_debug_create_symbol",
    "extended_debug_current_loc",
    "extended_debug_guard_added",
    "no_data_dependent_graph_break",
    "soft_pending_unbacked_not_found_error",
    "symbol_guard_limit_before_specialize",
    "translation_validation",
    "translation_validation_no_bisect",
    "translation_validation_timeout",
    "use_duck_shape",
]

no_data_dependent_graph_break = os.environ.get(
    "TENSORPLAY_NO_DATA_DEPENDENT_GRAPH_BREAK", "0"
) == "1"
translation_validation = os.environ.get(
    "TENSORPLAY_TRANSLATION_VALIDATION", "0"
) == "1"
translation_validation_timeout = int(
    os.environ.get("TENSORPLAY_TRANSLATION_VALIDATION_TIMEOUT", "600000")
)
translation_validation_no_bisect = os.environ.get(
    "TENSORPLAY_TRANSLATION_NO_BISECT", "0"
) == "1"
check_shape_env_recorded_events = False
extended_debug_guard_added = os.environ.get(
    "TENSORPLAY_EXTENDED_DEBUG_GUARD_ADDED"
)
extended_debug_create_symbol = os.environ.get(
    "TENSORPLAY_EXTENDED_DEBUG_CREATE_SYMBOL"
)
extended_debug_current_loc = os.environ.get(
    "TENSORPLAY_EXTENDED_DEBUG_CURRENT_LOC", "0"
) == "1"
soft_pending_unbacked_not_found_error = False
symbol_guard_limit_before_specialize: int | None = None
use_duck_shape = True
backed_size_oblivious = False


class AggressiveGuardFreeMode(enum.IntEnum):
    DISABLED = 0
    VALUE_RANGE_ANALYSIS = 1
    SKIP_RANGE_ANALYSIS = 2
