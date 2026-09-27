"""Caching for things that take a while and whose answer does not change.

Two shapes, for two situations.  One is for a value that is the same for the
whole process -- the answer depends on the source code, which does not change
while the process runs -- but that is expensive enough to be worth starting
before it is needed, and worth being able to fill in by hand.

The other is for a value that depends on an object, where the usual decorator
is the wrong tool: it would keep that object alive for as long as the process
runs, for an answer that object may not even still be good for.
"""

from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Concatenate, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future


_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")

# Sentinel used to indicate that cache lookup failed.
_cache_sentinel = object()

_prefetch_executor = ThreadPoolExecutor(max_workers=2)


def prefetchable_cache(func: Callable[[], _T]) -> Callable[[], _T]:
    """A cache for one value, which can be started before it is asked for.

    The value is the same for the whole process -- what it is computed from
    does not change while the process runs -- but computing it costs enough to
    be worth doing while something else is happening, and worth being able to
    supply directly rather than compute at all.

    So the answer is either already there, being computed, or computed on the
    spot; and a caller that is about to be the one to ask can start it early,
    or hand it over.
    """

    _cache: _T | object = _cache_sentinel
    _lock = threading.Lock()
    _future: Future[_T] | None = None

    def wrapper() -> _T:
        nonlocal _cache, _future
        with _lock:
            if _cache is not _cache_sentinel:
                return _cache  # type: ignore[return-value]
            if _future is not None:
                _cache = _future.result()
                _future = None
                return _cache  # type: ignore[return-value]
            _cache = func()
            return _cache  # type: ignore[return-value]

    def set_val(val: _T) -> None:
        nonlocal _cache
        with _lock:
            if _cache is not _cache_sentinel:
                raise RuntimeError("prefetchable_cache value already set")
            _cache = val

    def clear() -> None:
        nonlocal _cache, _future
        with _lock:
            _cache = _cache_sentinel
            _future = None

    def prefetch() -> None:
        nonlocal _future
        with _lock:
            if _cache is not _cache_sentinel or _future is not None:
                return
            _future = _prefetch_executor.submit(func)

    wrapper.set = set_val  # type: ignore[attr-defined]
    wrapper.clear = clear  # type: ignore[attr-defined]
    wrapper.prefetch = prefetch  # type: ignore[attr-defined]
    return wrapper


def cache_method(
    f: Callable[Concatenate[_C, _P], _T],
) -> Callable[Concatenate[_C, _P], _T]:
    """Like a plain cache, but for a method.

    A plain cache on a method keeps the object alive for as long as the process
    runs, which for an object that is supposed to be finished with is a leak.
    This one keys on the arguments alone and so does not.

    Note that this ignores everything about the object except its identity, so
    it is only right where the object does not change in a way that matters --
    the wrapped call being pure, for instance.
    """

    cache_name = "_cache_method_" + f.__name__

    @functools.wraps(f)
    def wrap(self: _C, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if kwargs:
            raise AssertionError("cache_method does not accept keyword arguments")
        if not (cache := getattr(self, cache_name, None)):
            cache = {}
            setattr(self, cache_name, cache)
        cached_value = cache.get(args, _cache_sentinel)
        if cached_value is not _cache_sentinel:
            return cached_value
        value = f(self, *args, **kwargs)
        cache[args] = value
        return value

    return wrap
