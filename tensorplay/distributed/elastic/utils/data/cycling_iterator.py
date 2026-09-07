"""Iterator that restarts a finite iterable forever, tracking epoch counts."""
from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = ["CyclingIterator"]

_T = TypeVar("_T")


class CyclingIterator(Iterator[_T]):
    """Wrap a finite iterable into an endless one.

    Each cycle increments ``epoch``; the underlying iterable is re-materialized
    through its factory (or by calling ``iter`` on it again when it supports
    that) so per-epoch shuffling keeps working.
    """

    def __init__(
        self,
        *args,
        start_epoch: int = 0,
        n: int | None = None,
        generator_fn=None,
    ) -> None:
        if len(args) >= 2 and isinstance(args[0], int):
            n = int(args[0])
            generator_fn = args[1]
            if len(args) > 2:
                start_epoch = int(args[2])
        elif args:
            generator_fn = args[0]
        if generator_fn is None:
            raise TypeError("an iterable factory is required")
        self._n = n
        self._factory = generator_fn
        self._epoch = start_epoch
        self._current: Iterator[_T] = self._materialize()

    def __iter__(self) -> "CyclingIterator[_T]":
        return self

    def _materialize(self) -> Iterator[_T]:
        produced = self._factory(self._epoch)
        return iter(produced)

    def __next__(self) -> _T:
        try:
            return next(self._current)
        except StopIteration:
            if self._n is not None and self._epoch >= self._n - 1:
                raise
            self._epoch += 1
            self._current = self._materialize()
            return next(self._current)

    @property
    def epoch(self) -> int:
        """Epoch number of the data currently being produced."""
        return self._epoch
