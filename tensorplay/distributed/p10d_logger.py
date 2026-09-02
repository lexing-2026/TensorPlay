"""Logging decorators for process-group operations."""
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import tensorplay.distributed as dist
from tensorplay.distributed.logging_handlers import _log_handlers

__all__: list[str] = []

_DEFAULT_DESTINATION = "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"p10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = f"{type(log_handler).__name__}-{destination}"
    return (log_handler, log_handler_name)


global _p10d_logger
_p10d_logger = _get_or_create_logger()


def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]:
    if dist.is_initialized():
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict = {
            "func_name": f"{func_name}",
            "pg_name": f"{getattr(kwargs.get('pg'), 'group_name', '')}",
            "backend": f"{dist.get_backend(group)}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(group)}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(group)}",
        }
        if msg_dict["backend"] == "nccl":
            from tensorplay._C import _distributed as _C

            msg_dict["nccl_version"] = _C.version()
    else:
        msg_dict = {
            "func_name": f"{func_name}",
        }
    return msg_dict


_T = TypeVar("_T")


def _exception_logger(func: Callable[..., _T]) -> Callable[..., _T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
            msg_dict["error"] = f"{error}"
            _p10d_logger.debug(msg_dict)
            raise

    return wrapper


def _time_logger(func: Callable[..., _T]) -> Callable[..., _T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        start = time.monotonic()
        try:
            return func(*args, **kwargs)
        finally:
            _p10d_logger.debug(
                {"func_name": func.__name__,
                 "elapsed_ms": (time.monotonic() - start) * 1000.0}
            )

    return wrapper
