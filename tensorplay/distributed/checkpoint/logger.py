from __future__ import annotations

import functools
from typing import Any, Callable


def _msg_dict_from_dcp_method_args(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"args": args, "kwargs": kwargs}


def _get_msg_dict(func_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"method": func_name, **_msg_dict_from_dcp_method_args(*args, **kwargs)}


def _dcp_method_logger(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper


def _init_logger(rank: int):
    import logging
    logger = logging.getLogger(f"tensorplay.checkpoint.rank{rank}")
    return logger
