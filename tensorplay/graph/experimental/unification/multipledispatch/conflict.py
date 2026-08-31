from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Sequence

from .utils import _toposort, groupby
from .variadic import isvariadic

__all__ = [
    "AmbiguityWarning",
    "ambiguous",
    "ambiguities",
    "consistent",
    "edge",
    "ordering",
    "super_signature",
    "supercedes",
]


class AmbiguityWarning(Warning):
    pass


def supercedes(first: tuple[type, ...], second: tuple[type, ...]) -> bool:
    if len(first) < len(second):
        return not first and len(second) == 1 and isvariadic(second[-1])
    if len(first) == len(second):
        return all(issubclass(left, right) for left, right in zip(first, second))
    left_index = right_index = 0
    while left_index < len(first) and right_index < len(second):
        left, right = first[left_index], second[right_index]
        if not isvariadic(left) and not isvariadic(right):
            if not issubclass(left, right):
                return False
            left_index += 1
            right_index += 1
        elif isvariadic(left):
            return right_index == len(second) - 1 and issubclass(left, right)
        else:
            if not issubclass(left, right):
                return False
            left_index += 1
    return left_index == len(first) and right_index == len(second)


def consistent(first: tuple[type, ...], second: tuple[type, ...]) -> bool:
    if not first:
        return not second or isvariadic(second[0])
    if not second:
        return not first or isvariadic(first[0])
    if len(first) == len(second):
        return all(issubclass(a, b) or issubclass(b, a) for a, b in zip(first, second))
    left_index = right_index = 0
    while left_index < len(first) and right_index < len(second):
        left, right = first[left_index], second[right_index]
        if not (issubclass(left, right) or issubclass(right, left)):
            return False
        if isvariadic(left):
            right_index += 1
        elif isvariadic(right):
            left_index += 1
        else:
            left_index += 1
            right_index += 1
    return (
        right_index == len(second) and left_index == len(first)
    ) or (
        left_index == len(first) - 1
        and left_index < len(first)
        and isvariadic(first[-1])
    ) or (
        right_index == len(second) - 1
        and right_index < len(second)
        and isvariadic(second[-1])
    )


def ambiguous(first: tuple[type, ...], second: tuple[type, ...]) -> bool:
    return consistent(first, second) and not (supercedes(first, second) or supercedes(second, first))


def ambiguities(signatures: Iterable[tuple[type, ...]]) -> set[tuple[tuple[type, ...], tuple[type, ...]]]:
    values = list(map(tuple, signatures))
    return {
        (left, right)
        for left in values
        for right in values
        if hash(left) < hash(right)
        and ambiguous(left, right)
        and not any(supercedes(other, left) and supercedes(other, right) for other in values)
    }


def super_signature(signatures: Sequence[tuple[type, ...]]) -> list[type]:
    if not signatures or not all(len(item) == len(signatures[0]) for item in signatures):
        raise AssertionError("signatures must have equal arity")
    return [max((type.mro(item[index]) for item in signatures), key=len)[0] for index in range(len(signatures[0]))]


def edge(first: tuple[type, ...], second: tuple[type, ...], tie_breaker: Callable[[tuple[type, ...]], int] = hash) -> bool:
    return supercedes(first, second) and (
        not supercedes(second, first) or tie_breaker(first) > tie_breaker(second)
    )


def ordering(signatures: Iterable[tuple[type, ...]]) -> list[tuple[type, ...]]:
    values = list(map(tuple, signatures))
    edges = groupby(operator.itemgetter(0), ((a, b) for a in values for b in values if edge(a, b)))
    for value in values:
        edges.setdefault(value, [])
    return _toposort({key: [right for _, right in pairs] for key, pairs in edges.items()})
