# mypy: allow-untyped-defs
"""Backend property helpers.

Provides the flag plumbing (``ContextProp``/``PropModule``/freezing) and the
native precision and backend capability controls.
"""

import sys
import types
from contextlib import contextmanager

import tensorplay

__all__ = [
    "cpu",
    "cuda",
    "cudnn",
    "disable_global_flags",
    "flags",
    "flags_frozen",
    "fp32_precision",
    "mkl",
    "mkldnn",
    "nnpack",
    "openmp",
    "set_flags",
]


# The idea for this parameter is that we forbid bare assignment
# to tensorplay.backends.<cudnn|mkldnn>.enabled and friends when running our
# test suite, where it's very easy to forget to undo the change
# later.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen():
    return not __allow_nonbracketed_mutation_flag


@contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
            )


class PropModule(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


def set_flags(_fp32_precision=None):
    """Set the process-wide matrix multiplication precision."""
    original = tensorplay._C.get_float32_matmul_precision()
    if _fp32_precision is not None:
        tensorplay._C._set_float32_matmul_precision(_fp32_precision)
    return (original,)


@contextmanager
def flags(fp32_precision=None):
    """Temporarily set the process-wide matrix multiplication precision."""
    with __allow_nonbracketed_mutation():
        original = set_flags(fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*original)


def _get_fp32_precision():
    return tensorplay._C.get_float32_matmul_precision()


def _set_fp32_precision(value):
    tensorplay._C._set_float32_matmul_precision(value)


class GenericModule(PropModule):
    fp32_precision = ContextProp(_get_fp32_precision, _set_fp32_precision)


sys.modules[__name__] = GenericModule(sys.modules[__name__], __name__)

from . import cpu as cpu
from . import cuda as cuda
from . import cudnn as cudnn
from . import mkl as mkl
from . import mkldnn as mkldnn
from . import nnpack as nnpack
from . import openmp as openmp
