"""Copied semantics from torchaudio/_internal/module_utils.py.

Torchaudio marks optional-dependency functions with these decorators so that
importing the module never fails; calling an unavailable function raises a
helpful error. TensorPlay audio has no optional native deps, so ``no_op``
is a passthrough and ``fail_with_message`` always raises.
"""
import functools


def no_op(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def fail_with_message(message):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            raise RuntimeError(message)
        return wrapper
    return decorator
