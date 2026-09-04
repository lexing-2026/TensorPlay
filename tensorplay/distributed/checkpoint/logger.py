from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable
from typing import ParamSpec, TypeVar
from uuid import uuid4

from .logging_handlers import DCP_LOGGER_NAME

__all__: list[str] = []

logger = logging.getLogger()
_dcp_logger = logging.getLogger(DCP_LOGGER_NAME)
if not _dcp_logger.handlers:
    _dcp_logger.addHandler(logging.NullHandler())

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _msg_dict_from_dcp_method_args(*args: Any, **kwargs: Any) -> dict[str, Any]:
    del args
    storage_writer = kwargs.get("storage_writer")
    storage_reader = kwargs.get("storage_reader")
    planner = kwargs.get("planner")
    checkpoint_id = kwargs.get("checkpoint_id")
    if checkpoint_id is None:
        serializer = storage_writer or storage_reader
        checkpoint_id = getattr(serializer, "checkpoint_id", None)
    result: dict[str, Any] = {
        "checkpoint_id": str(checkpoint_id) if checkpoint_id is not None else None,
        "uuid": str(uuid4().int),
    }
    if storage_writer is not None:
        result["storage_writer"] = type(storage_writer).__name__
    if storage_reader is not None:
        result["storage_reader"] = type(storage_reader).__name__
    if planner is not None:
        result["planner"] = type(planner).__name__
    return result


def _get_msg_dict(func_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"method": func_name, **_msg_dict_from_dcp_method_args(*args, **kwargs)}


def _dcp_method_logger(
    log_exceptions: bool = False,
    **wrapper_kwargs: Any,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg_dict = _get_msg_dict(
                target.__name__, *args, **{**wrapper_kwargs, **kwargs}
            )
            msg_dict.update({"event": "start", "time": time.time_ns()})
            msg_dict["log_exceptions"] = log_exceptions
            _dcp_logger.debug(msg_dict)
            started = msg_dict["time"]
            succeeded = False
            try:
                result = target(*args, **kwargs)
                succeeded = True
                return result
            except BaseException as error:
                if log_exceptions:
                    msg_dict.update(
                        {
                            "event": "exception",
                            "error": str(error),
                            "time": time.time_ns(),
                        }
                    )
                    _dcp_logger.error(msg_dict)
                raise
            finally:
                if succeeded:
                    finished = time.time_ns()
                    msg_dict.update(
                        {
                            "event": "end",
                            "time": finished,
                            "times_spent": finished - started,
                        }
                    )
                    _dcp_logger.debug(msg_dict)

        return wrapper

    return decorator


def _init_logger(rank: int):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            f"[{rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)
