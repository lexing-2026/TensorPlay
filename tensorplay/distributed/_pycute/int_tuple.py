from functools import reduce
from itertools import chain
from typing import TypeAlias

from .typing import Integer

IntTuple: TypeAlias = int | tuple["IntTuple", ...]


def is_int(value: object) -> bool:
    return isinstance(value, Integer)


def is_tuple(value: object) -> bool:
    return isinstance(value, tuple)


def as_tuple(value: IntTuple) -> tuple[IntTuple, ...]:
    return (value,) if is_int(value) else value


def match_structure(left: IntTuple, right: IntTuple) -> bool:
    if is_int(left) and is_int(right):
        return True
    if is_tuple(left) and is_tuple(right):
        return len(left) == len(right) and all(
            match_structure(a, b) for a, b in zip(left, right)
        )
    return False


def flatten(value: IntTuple) -> tuple[int, ...]:
    if is_tuple(value):
        return tuple(item for child in value for item in flatten(child))
    return (value,)


def signum(value: int) -> int:
    return int(value > 0) - int(value < 0)


def product(value: IntTuple) -> int:
    if is_tuple(value):
        return reduce(lambda result, item: result * product(item), value, 1)
    return value


def inner_product(left: IntTuple, right: IntTuple) -> int:
    if is_tuple(left) and is_tuple(right):
        if len(left) != len(right):
            raise AssertionError
        return sum(inner_product(a, b) for a, b in zip(left, right))
    if is_tuple(left) or is_tuple(right):
        raise AssertionError
    return left * right


def tuple_max(value: IntTuple) -> int:
    return max(tuple_max(item) for item in value) if is_tuple(value) else value


def elem_scale(left: IntTuple, right: IntTuple) -> IntTuple:
    if is_tuple(left):
        if not is_tuple(right) or len(left) != len(right):
            raise AssertionError
        return tuple(elem_scale(a, b) for a, b in zip(left, right))
    if is_tuple(right):
        return elem_scale(left, product(right))
    return left * right


def shape_div(left: IntTuple, right: IntTuple) -> IntTuple:
    if is_tuple(left):
        if is_tuple(right):
            if len(left) != len(right):
                raise AssertionError
            return tuple(shape_div(a, b) for a, b in zip(left, right))
        result = []
        divisor = right
        for item in left:
            result.append(shape_div(item, divisor))
            divisor = shape_div(divisor, product(item))
        return tuple(result)
    if is_tuple(right):
        return shape_div(left, product(right))
    if not (left % right == 0 or right % left == 0):
        raise AssertionError
    return (left + right - 1) // right


def suffix_product(value: IntTuple, init: IntTuple = 1) -> IntTuple:
    if is_tuple(value):
        if is_tuple(init):
            if len(value) != len(init):
                raise AssertionError
            return tuple(suffix_product(a, b) for a, b in zip(value, init))
        result = []
        running = init
        for item in reversed(value):
            result.append(suffix_product(item, running))
            running *= product(item)
        return tuple(reversed(result))
    if is_tuple(init):
        raise AssertionError
    return init


def idx2crd(idx: IntTuple, shape: IntTuple, stride: IntTuple | None = None) -> IntTuple:
    if stride is None:
        stride = suffix_product(shape)
    if is_tuple(idx):
        if not (is_tuple(shape) and is_tuple(stride)) or len(idx) != len(shape):
            raise AssertionError
        return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
    if is_tuple(shape) and is_tuple(stride):
        if len(shape) != len(stride):
            raise AssertionError
        return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
    if is_tuple(shape) or is_tuple(stride):
        raise AssertionError
    return (idx // stride) % shape


def crd2idx(crd: IntTuple | None, shape: IntTuple, stride: IntTuple | None = None) -> int:
    if stride is None:
        stride = suffix_product(shape)
    if is_tuple(crd):
        if not (is_tuple(shape) and is_tuple(stride)) or len(crd) != len(shape):
            raise AssertionError
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    if crd is None:
        crd = 0
    if is_tuple(shape) and is_tuple(stride):
        if len(shape) != len(stride):
            raise AssertionError
        result = 0
        remaining = crd
        for index in range(len(shape) - 1, 0, -1):
            extent = product(shape[index])
            result += crd2idx(remaining % extent, shape[index], stride[index])
            remaining //= extent
        if shape:
            result += crd2idx(remaining, shape[0], stride[0])
        return result
    if is_tuple(shape) or is_tuple(stride):
        raise AssertionError
    return crd * stride


def crd2crd(crd: IntTuple, dst_shape: IntTuple, src_shape: IntTuple | None = None) -> IntTuple:
    if is_tuple(crd):
        if is_tuple(dst_shape):
            if len(crd) != len(dst_shape):
                raise AssertionError
            return tuple(crd2crd(a, b) for a, b in zip(crd, dst_shape))
        if src_shape is None:
            raise AssertionError
        return crd2idx(crd, src_shape)
    if is_tuple(dst_shape):
        return idx2crd(crd, dst_shape)
    if crd >= dst_shape:
        raise AssertionError
    return crd


def slice_(crd: tuple | int | None, target: tuple | int) -> tuple | int:
    if is_tuple(crd):
        if not is_tuple(target) or len(crd) != len(target):
            raise AssertionError
        return tuple(
            item
            for coord, value in zip(crd, target)
            for item in (slice_(coord, value),)
            if item != ()
            for item in (item if is_tuple(item) else (item,))
        )
    if crd is None:
        return (target,)
    return ()


def has_none(value: tuple | int | None) -> bool:
    return any(has_none(item) for item in value) if is_tuple(value) else value is None
