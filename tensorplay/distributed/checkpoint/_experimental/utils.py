from __future__ import annotations

from concurrent.futures import Future
from typing import Any


def wrap_future(original_result: Any) -> Future[None]:
    result: Future[None] = Future()
    if isinstance(original_result, Future):
        def done(future: Future[Any]) -> None:
            try:
                future.result()
                result.set_result(None)
            except BaseException as error:
                result.set_exception(error)
        original_result.add_done_callback(done)
    else:
        result.set_result(None)
    return result
