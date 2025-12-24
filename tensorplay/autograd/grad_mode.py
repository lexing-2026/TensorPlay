from typing import Any, Union

import tensorplay
import tensorplay._C._autograd as _autograd
from tensorplay.utils._contextlib import (
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
    F,
)


__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "is_grad_enabled",
]


class no_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
        If you want to disable forward AD for a computation, you can unpack
        your dual tensors.

    Example::
        >>> x = tensorplay.tensor([1.], requires_grad=True)
        >>> with tensorplay.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @tensorplay.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @tensorplay.no_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with tensorplay.no_grad():
        ...     a = tensorplay.nn.Parameter(tensorplay.rand(10))
        >>> a.requires_grad
        True
    """

    def __init__(self) -> None:
        super().__init__()
        self.prev = False

    def __enter__(self):
        self.prev = _autograd.is_grad_enabled()
        _autograd.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _autograd.set_grad_enabled(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        enable_grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = tensorplay.tensor([1.], requires_grad=True)
        >>> with tensorplay.no_grad():
        ...     with tensorplay.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        tensor([2.])
        >>> @tensorplay.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with tensorplay.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True
        >>> @tensorplay.enable_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> with tensorplay.no_grad():
        ...     z = tripler(x)
        >>> z.requires_grad
        True

    """

    def __enter__(self) -> None:
        self.prev = _autograd.is_grad_enabled()
        _autograd.set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _autograd.set_grad_enabled(self.prev)


def is_grad_enabled():
    return _autograd.is_grad_enabled()


class set_grad_enabled(_DecoratorContextManager):
    r"""Context-manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = tensorplay.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with tensorplay.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> _ = tensorplay.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> _ = tensorplay.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode: bool) -> None:
        self.mode = mode
        self.prev = False
        self.prev = is_grad_enabled()
        _autograd.set_grad_enabled(mode)

    def __call__(self, orig_func: F) -> F:
        _autograd.set_grad_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        _autograd.set_grad_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _autograd.set_grad_enabled(self.prev)

    def __str__(self) -> str:
        return f"{tensorplay.typename(self)}(mode={self.mode})"

    def __repr__(self) -> str:
        return str(self)

    def clone(self) -> "set_grad_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class inference_mode(_DecoratorContextManager):
    r"""Context manager that enables or disables inference mode.

    InferenceMode is analogous to :class:`~no_grad` and should be used
    when you are certain your operations will not interact with autograd
    (e.g., during data loading or model evaluation). Compared to
    :class:`~no_grad`, it removes additional overhead by disabling view
    tracking and version counter bumps. It is also more restrictive, in
    that tensors created in this mode cannot be used in computations
    recorded by autograd.

    This context manager is thread-local; it does not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        Inference mode is one of several mechanisms that can locally enable
        or disable gradients. See :ref:`locally-disable-grad-doc` for a
        comparison. If avoiding the use of tensors created in inference mode
        in autograd-tracked regions is difficult, consider benchmarking your
        code with and without inference mode to weigh the performance benefits
        against the trade-offs. You can always use :class:`~no_grad` instead.

    .. note::
       Unlike some other mechanisms that locally enable or disable grad,
       entering inference_mode also disables :ref:`forward-mode AD <forward-mode-ad>`.

    Args:
        mode (bool or function): Either a boolean flag to enable or disable
            inference mode, or a Python function to decorate with inference
            mode enabled.

    Example::
        >>> import tensorplay
        >>> x = tensorplay.ones(1, 2, 3, requires_grad=True)
        >>> with tensorplay.inference_mode():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @tensorplay.inference_mode()
        ... def func(x):
        ...     return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False
        >>> @tensorplay.inference_mode()
        ... def doubler(x):
        ...     return x * 2
        >>> out = doubler(x)
        >>> out.requires_grad
        False

    """

    def __init__(self, mode: bool = True) -> None:
        super().__init__()
        self.mode = mode

    def __new__(cls, mode=True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self._inference_mode_context = _autograd._InferenceMode(self.mode)
        self._inference_mode_context.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._inference_mode_context.__exit__(exc_type, exc_value, traceback)

    def clone(self) -> "inference_mode":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


def _enter_inference_mode(mode):
    mode_context = _autograd._InferenceMode(mode)
    mode_context.__enter__()
    return mode_context


def _exit_inference_mode(mode):
    mode.__exit__(None, None, None)