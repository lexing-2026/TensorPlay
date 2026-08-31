from itertools import chain
from typing import TypeAlias

from .int_tuple import (
    crd2idx,
    flatten,
    has_none,
    IntTuple,
    is_int,
    is_tuple,
    product,
    slice_,
    suffix_product,
)

CoordinateType: TypeAlias = int | IntTuple | tuple[object, ...] | None


class LayoutBase:
    pass


def is_layout(value: object) -> bool:
    return isinstance(value, LayoutBase)


class Layout(LayoutBase):
    def __init__(self, shape: IntTuple, stride: IntTuple | None = None) -> None:
        self.shape = shape
        self.stride = suffix_product(shape) if stride is None else stride

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Layout) and (self.shape, self.stride) == (
            other.shape,
            other.stride,
        )

    def __len__(self) -> int:
        return len(self.shape) if is_tuple(self.shape) else 1

    def __call__(self, *args: CoordinateType) -> "Layout | int":
        if has_none(args):
            return Layout(
                slice_(args[0], self.shape) if len(args) == 1 else slice_(args, self.shape),
                slice_(args[0], self.stride) if len(args) == 1 else slice_(args, self.stride),
            )
        if len(args) == 1:
            return crd2idx(args[0], self.shape, self.stride)
        return crd2idx(args, self.shape, self.stride)

    def __getitem__(self, index: int) -> "Layout":
        if is_tuple(self.shape):
            return Layout(self.shape[index], self.stride[index])
        if index != 0:
            raise AssertionError
        return Layout(self.shape, self.stride)

    def size(self) -> int:
        return product(self.shape)

    def cosize(self) -> int:
        return self(self.size() - 1) + 1

    def __str__(self) -> str:
        return f"{self.shape}:{self.stride}"

    def __repr__(self) -> str:
        return f"Layout({self.shape},{self.stride})"


LayoutOrIntTuple: TypeAlias = Layout | IntTuple
LayoutProfile: TypeAlias = tuple[object, ...] | Layout | None
LayoutInput: TypeAlias = Layout | IntTuple | tuple[object, ...] | None


def make_layout(*layouts: Layout | tuple[Layout, ...]) -> Layout:
    if len(layouts) == 1 and not is_layout(layouts[0]):
        layouts = layouts[0]
    shape, stride = zip(*((item.shape, item.stride) for item in layouts))
    return Layout(shape, stride)


def size(layout: LayoutOrIntTuple) -> int:
    return layout.size() if is_layout(layout) else product(layout)


def cosize(layout: Layout) -> int:
    return layout.cosize()


def coalesce(layout: Layout, profile: LayoutProfile = None) -> Layout:
    if is_tuple(profile):
        if len(layout) < len(profile):
            raise AssertionError
        return make_layout(
            chain(
                (coalesce(layout[index], profile[index]) for index in range(len(profile))),
                (layout[index] for index in range(len(profile), len(layout))),
            )
        )
    result_shape = [1]
    result_stride = [0]
    for shape, stride in zip(reversed(flatten(layout.shape)), reversed(flatten(layout.stride))):
        if shape == 1:
            continue
        if result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] *= shape
        else:
            result_shape.append(shape)
            result_stride.append(stride)
    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    result_shape.reverse()
    result_stride.reverse()
    return Layout(tuple(result_shape), tuple(result_stride))


def filter(layout: Layout, profile: LayoutProfile = None) -> Layout:
    if is_tuple(profile):
        if len(layout) < len(profile):
            raise AssertionError
        return make_layout(
            chain(
                (filter(layout[index], profile[index]) for index in range(len(profile))),
                (layout[index] for index in range(len(profile), len(layout))),
            )
        )
    shapes = []
    strides = []
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        if shape != 1 and stride != 0:
            shapes.append(shape)
            strides.append(stride)
    if not shapes:
        return Layout(1, 0)
    return coalesce(Layout(tuple(shapes), tuple(strides)))


def composition(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    if layout_b is None:
        return layout_a
    if is_int(layout_b):
        return composition(layout_a, Layout(layout_b))
    if is_tuple(layout_b):
        if len(layout_a) < len(layout_b):
            raise AssertionError
        return make_layout(
            chain(
                (composition(layout_a[index], layout_b[index]) for index in range(len(layout_b))),
                (layout_a[index] for index in range(len(layout_b), len(layout_a))),
            )
        )
    if is_tuple(layout_b.shape):
        return make_layout(composition(layout_a, item) for item in layout_b)
    if layout_b.stride == 0:
        return Layout(layout_b.shape, 0)
    result_shape = []
    result_stride = []
    rest_shape = layout_b.shape
    rest_stride = layout_b.stride
    flat_a = coalesce(layout_a)
    flat_shapes = flatten(flat_a.shape)
    flat_strides = flatten(flat_a.stride)
    for current_shape, current_stride in zip(reversed(flat_shapes[1:]), reversed(flat_strides[1:])):
        if not (current_shape % rest_stride == 0 or rest_stride % current_shape == 0):
            raise AssertionError
        new_shape = min(max(1, current_shape // rest_stride), rest_shape)
        if new_shape != 1:
            result_shape.append(new_shape)
            result_stride.append(rest_stride * current_stride)
        rest_shape //= new_shape
        rest_stride = -( -rest_stride // current_shape )
    if rest_shape != 1 or not result_shape:
        result_shape.append(rest_shape)
        result_stride.append(rest_stride * flat_strides[0])
    result_shape.reverse()
    result_stride.reverse()
    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    return Layout(tuple(result_shape), tuple(result_stride))


def complement(layout: LayoutOrIntTuple, max_idx: int = 1) -> Layout:
    if is_int(layout):
        return complement(Layout(layout))
    shapes = []
    strides = []
    current = 1
    for stride, shape in sorted(zip(flatten(layout.stride), flatten(layout.shape))):
        if stride == 0 or shape == 1:
            continue
        if current > shape * stride:
            raise AssertionError
        shapes.append(stride // current)
        strides.append(current)
        current = shape * stride
    shapes.append((max_idx + current - 1) // current)
    strides.append(current)
    shapes.reverse()
    strides.reverse()
    return coalesce(Layout(tuple(shapes), tuple(strides)))


def right_inverse(layout: LayoutOrIntTuple | None) -> Layout | None:
    if layout is None:
        return None
    if is_int(layout):
        return Layout(layout)
    shapes = []
    strides = []
    current = 1
    flat_shape = flatten(layout.shape)
    flat_stride = flatten(layout.stride)
    for stride, shape, rstride in sorted(zip(flat_stride, flat_shape, suffix_product(flat_shape))):
        if shape == 1:
            continue
        if current != stride:
            break
        shapes.append(shape)
        strides.append(rstride)
        current = shape * stride
    shapes.reverse()
    strides.reverse()
    return coalesce(Layout(tuple(shapes), tuple(strides)))


def left_inverse(layout: LayoutOrIntTuple | None) -> Layout | None:
    if layout is None or is_int(layout):
        return None if layout is None else Layout(layout)
    return right_inverse(make_layout(complement(layout), layout))


def logical_divide(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    if layout_b is None:
        return layout_a
    if is_int(layout_b):
        return logical_divide(layout_a, Layout(layout_b))
    if is_tuple(layout_b):
        if len(layout_a) < len(layout_b):
            raise AssertionError
        return make_layout(
            chain(
                (logical_divide(layout_a[index], layout_b[index]) for index in range(len(layout_b))),
                (layout_a[index] for index in range(len(layout_b), len(layout_a))),
            )
        )
    return composition(layout_a, make_layout(layout_b, complement(layout_b, size(layout_a))))


def logical_product(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    if layout_b is None:
        return layout_a
    if is_int(layout_b):
        return logical_divide(layout_a, Layout(layout_b))
    if is_tuple(layout_b):
        if len(layout_a) < len(layout_b):
            raise AssertionError
        return make_layout(
            chain(
                (logical_product(layout_a[index], layout_b[index]) for index in range(len(layout_b))),
                (layout_a[index] for index in range(len(layout_b), len(layout_a))),
            )
        )
    return make_layout(
        layout_a,
        composition(complement(layout_a, size(layout_a) * cosize(layout_b)), layout_b),
    )


def hier_unzip(splitter: object, layout_a: Layout, layout_b: LayoutInput) -> Layout:
    if layout_b is None:
        return make_layout(Layout(1, 0), layout_a)
    if is_tuple(layout_b):
        if len(layout_a) < len(layout_b):
            raise AssertionError
        split = make_layout(
            hier_unzip(splitter, layout_a[index], layout_b[index])
            for index in range(len(layout_b))
        )
        return make_layout(
            make_layout(split[index][0] for index in range(len(layout_b))),
            make_layout(
                chain(
                    (split[index][1] for index in range(len(layout_b))),
                    (layout_a[index] for index in range(len(layout_b), len(layout_a))),
                )
            ),
        )
    return splitter(layout_a, layout_b)


def zipped_divide(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    return hier_unzip(logical_divide, layout_a, layout_b)


def tiled_divide(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    result = zipped_divide(layout_a, layout_b)
    return make_layout([result[0]] + [result[1][index] for index in range(len(result[1]))])


def zipped_product(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    return hier_unzip(logical_product, layout_a, layout_b)


def tiled_product(layout_a: Layout, layout_b: LayoutInput) -> Layout:
    result = zipped_product(layout_a, layout_b)
    return make_layout([result[0]] + [result[1][index] for index in range(len(result[1]))])


def slice_and_offset(coordinate: tuple[object, ...], layout: Layout) -> tuple[Layout, int]:
    return (
        Layout(slice_(coordinate, layout.shape), slice_(coordinate, layout.stride)),
        crd2idx(coordinate, layout.shape, layout.stride),
    )
