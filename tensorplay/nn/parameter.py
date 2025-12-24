from typing import Any
from abc import ABCMeta
import tensorplay as tp
from tensorplay import Tensor

__all__ = [
    "Parameter",
    "UninitializedParameter",
    "is_lazy",
    "Buffer",
    "UninitializedBuffer",
    "UninitializedTensorMixin",
]

class Parameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~tensorplay.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the tensorplay.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will always have ``requires_grad=True``
            unless given explicitly. Default: `True`
    """
    def __init__(self, data=None, requires_grad=True):
        if data is None:
            data = tp.empty(0)
        super().__init__()
        # In TensorPlay, assigning to self.data updates the underlying implementation
        self.data = data
        self.requires_grad = requires_grad

    def __repr__(self):
        return 'Parameter containing:\n' + super().__repr__()

    def __reduce_ex__(self, proto):
        # Support for pickling
        return (
            Parameter,
            (self.data, self.requires_grad)
        )

class UninitializedTensorMixin(metaclass=ABCMeta):
    def materialize(self, shape, device=None, dtype=None):
        r"""Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`tensorplay.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`tensorplay.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        
        # Create new tensor data
        new_data = tp.empty(shape, device=device, dtype=dtype)
        
        # Update self to become the new class (Parameter or Tensor)
        self.data = new_data
        self.__class__ = self.cls_to_become

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access the shape of an uninitialized parameter or buffer. "
            "This error usually happens in `load_state_dict` when trying to load "
            "an uninitialized parameter into an initialized one. "
            "Call `forward` to initialize the parameters before accessing their attributes."
        )

    def share_memory_(self):
        raise RuntimeError(
            "Can't share memory on an uninitialized parameter or buffer. "
            "Call `forward` to initialize the parameters before calling "
            "`module.share_memory()`."
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __reduce_ex__(self, proto):
        return (self.__class__, (self.requires_grad,))


def is_lazy(param: Any) -> bool:
    """
    Returns whether ``param`` is an ``UninitializedParameter`` or ``UninitializedBuffer``.

    Args:
        param (Any): the input to check.
    """
    return isinstance(param, UninitializedTensorMixin)


class UninitializedParameter(Parameter):
    r"""A parameter that is not initialized.

    Uninitialized Parameters are a special case of :class:`tensorplay.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`tensorplay.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`tensorplay.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """
    cls_to_become = Parameter

    def __init__(self, requires_grad=True, device=None, dtype=None):
        # Create empty 0-element tensor to hold device/dtype info
        factory_kwargs = {"device": device, "dtype": dtype}
        # Filter out None values to avoid bad cast in bindings
        factory_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
        data = tp.empty(0, **factory_kwargs)
        super().__init__(data, requires_grad)

    # Mixin methods copied due to nanobind multiple inheritance limitation
    def materialize(self, shape, device=None, dtype=None):
        UninitializedTensorMixin.materialize(self, shape, device, dtype)

    @property
    def shape(self):
        return UninitializedTensorMixin.shape.fget(self)

    def share_memory_(self):
        UninitializedTensorMixin.share_memory_(self)

    def __repr__(self):
        return UninitializedTensorMixin.__repr__(self)

    def __reduce_ex__(self, proto):
        return UninitializedTensorMixin.__reduce_ex__(self, proto)

# Register as virtual subclass
UninitializedTensorMixin.register(UninitializedParameter)


class Buffer(Tensor):
    r"""A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

    Buffers are :class:`~tensorplay.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s -- when they're
    assigned as Module attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~tensorplay.nn.Module.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~tensorplay.nn.Module.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """
    def __init__(self, data=None, persistent=True):
        if data is None:
            data = tp.empty(0)
        super().__init__()
        self.data = data
        self.requires_grad = data.requires_grad
        self.persistent = persistent
        self._is_buffer = True

    def __reduce_ex__(self, proto):
        return (
            Buffer,
            (self.data, self.persistent)
        )


# Internal class to allow UninitializedBuffer to become a Tensor-like object
# with Python attributes (layout compatibility)
class _MaterializedTensor(Tensor):
    pass

class UninitializedBuffer(Tensor):
    r"""A buffer that is not initialized.

    Uninitialized Buffer is a a special case of :class:`tensorplay.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`tensorplay.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`tensorplay.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """
    cls_to_become = _MaterializedTensor

    def __init__(self, requires_grad=False, device=None, dtype=None, persistent=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        # Filter out None values to avoid bad cast in bindings
        factory_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
        data = tp.empty(0, **factory_kwargs)
        super().__init__()
        self.data = data
        self.requires_grad = requires_grad
        self.persistent = persistent
        self._is_buffer = True

    # Mixin methods copied due to nanobind multiple inheritance limitation
    def materialize(self, shape, device=None, dtype=None):
        UninitializedTensorMixin.materialize(self, shape, device, dtype)

    @property
    def shape(self):
        return UninitializedTensorMixin.shape.fget(self)

    def share_memory_(self):
        UninitializedTensorMixin.share_memory_(self)

    def __repr__(self):
        return UninitializedTensorMixin.__repr__(self)

    def __reduce_ex__(self, proto):
        return UninitializedTensorMixin.__reduce_ex__(self, proto)

# Register as virtual subclass
UninitializedTensorMixin.register(UninitializedBuffer)
