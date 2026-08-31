from __future__ import annotations

import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)
__all__ = ["log_debug_info_for_expired_timers"]


def log_debug_info_for_expired_timers(run_id: str, expired_timers: dict[int, list[str]]):
    if expired_timers:
        logger.info("Timers expired for run %s: %s", run_id, expired_timers)
