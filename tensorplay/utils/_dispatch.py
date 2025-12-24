from collections.abc import Sequence
from typing import Optional, overload, Protocol, Union
from typing_extensions import TypeIs

import tensorplay


_is_in_tensorplay_dispatch_mode = False
_is_in_non_infra_tensorplay_dispatch_mode = False
# If inside any mode that has ignore_compile_internals() = False
_is_in_any_mode_without_ignore_compile_internals = False


def is_in_tensorplay_dispatch_mode(include_infra_modes=True) -> bool:
    return (
        _is_in_tensorplay_dispatch_mode
        if include_infra_modes
        else _is_in_non_infra_tensorplay_dispatch_mode
    )


def is_in_any_mode_without_ignore_compile_internals() -> bool:
    return _is_in_any_mode_without_ignore_compile_internals


# Subtypes which have __tensor_flatten__ and __tensor_unflatten__.
class TensorWithFlatten(Protocol):
    def __tensor_flatten__(self) -> tuple[Sequence[str], object]: ...

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: int, flatten_spec: int, outer_size: int, outer_stride: int
    ) -> tensorplay.Tensor: ...

    # It would be really nice to be able to say that the return of
    # is_traceable_wrapper_subclass() is Intersection[tensorplay.Tensor,
    # TensorWithFlatten] - but that doesn't exist.

    shape: tensorplay._C.Size

    @overload
    def stride(self, dim: None = None) -> tuple[int, ...]: ...

    @overload
    def stride(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = None) -> tuple[int, ...]: ...

    @overload
    def size(self, dim: int) -> int: ...

    def storage_offset(self) -> int: ...

    def dim(self) -> int: ...

    @overload
    def to(
        self,
        dtype: tensorplay.types._dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional['tensorplay.memory_format'] = None,
    ) -> tensorplay.Tensor: ...

    @overload
    def to(
        self,
        device: Optional["tensorplay._prims_common.DeviceLikeType"] = None,
        dtype: Optional[tensorplay.types._dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional['tensorplay.memory_format'] = None,
    ) -> tensorplay.Tensor: ...

    @overload
    def to(
        self,
        other: tensorplay.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional['tensorplay.memory_format'] = None,
    ) -> tensorplay.Tensor: ...


def is_traceable_wrapper_subclass(t: object) -> TypeIs[TensorWithFlatten]:
    """
    Returns whether a tensor subclass that implements __tensorplay_dispatch__
    is 'traceable' with tensorplay.compile.
    In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.
    It is also expected to obey some restrictions around traceability and aliasing:
        * The subclass's __tensorplay_dispatch__() implementation should desugar into pytensorplay
            dispatcher operations that can be traced into a graph.
        * The subclass should use return_and_correct_aliasing(). This is needed today to make
            sure that tensorplay.compile does the right thing in a few cases around input mutation
            and output aliasing.

    Expected magic method signatures:
        attrs, ctx = t.__tensor_flatten__()
            attrs: list of attribute name strings for inner tensors
            ctx: dict containing any other subclass-specific metadata needed for unflattening

        t = MySubClass.__tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride)
            inner_tensors: dict mapping attribute name -> tensor for each inner tensor
            ctx: dict with subclass metadata in the form that __tensor_flatten__() produces
            outer_size: expected (possibly symbolic) size that the returned subclass
                instance should have. Note that this arg is useful for certain subclasses
                that require the shape info to be constructed. In most cases, this arg can be
                safely ignored.
            outer_stride: expected (possibly symbolic) stride that the returned subclass
                instance should have. Note that this arg is useful for certain subclasses
                that require the stride info to be constructed. In most cases, this arg can be
                safely ignored.
    """
    is_subclass = isinstance(t, tensorplay.Tensor) and type(t) is not tensorplay.Tensor
    return (
        is_subclass
        and hasattr(t, "__tensor_flatten__")
        and hasattr(t, "__tensor_unflatten__")
    )


def is_traceable_wrapper_subclass_type(t: type) -> TypeIs[type[TensorWithFlatten]]:
    """Same as above, but takes a type argument instead of an instance."""
    return (
        issubclass(t, tensorplay.Tensor)
        and t is not tensorplay.Tensor
        and hasattr(t, "__tensor_flatten__")
        and hasattr(t, "__tensor_unflatten__")
    )


def transform_subclass(t, callback, outer_size=None, outer_stride=None):
    """
    Given a traceable, wrapper tensor subclass ``t`` that implements
    ``__tensorplay_dispatch__`` and holds some inner tensors,
    and a callback of type ``Callable[[str, tensorplay.Tensor], tensorplay.Tensor]``,
    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.
    It will do so by grabbing each inner tensor attribute from the wrapper,
    passing them into ``callback`` to get a transformed tensor,
    and putting each transformed tensor into the fresh tensor subclass instance.

    Note: this function will not handle ensuring that the fresh subclass
    gets the same (autograd, and aliasing) metadata as the original tensor.
    This is generally handled in other subsystems like AOTAutograd.
    """
    outer_size = outer_size if outer_size is not None else t.size()
    outer_stride = outer_stride if outer_stride is not None else t.stride()

    attrs, ctx = t.__tensor_flatten__()
    transformed_tensors_dict = {}
    for attr in attrs:
        transformed_tensors_dict[attr] = callback(attr, getattr(t, attr))
    sub = type(t).__tensor_unflatten__(
        transformed_tensors_dict, ctx, outer_size, outer_stride
    )

    # NB: Purposefully guard here to simplify the inner / outer symbols.
    # Using sym_eq() for symbolic comparison can result in an expression that's too
    # difficult to guard on, so we use == here.
    assert sub.shape == outer_size, (
        f"Expected return value from {type(t)}__tensor_unflatten__() to have "
        f"shape equal to {outer_size}, but got: {sub.shape}"
    )
    assert sub.stride() == outer_stride, (
        f"Expected return value from {type(t)}__tensor_unflatten__() to have "
        f"stride equal to {outer_stride}, but got: {sub.stride()}"
    )

    return sub