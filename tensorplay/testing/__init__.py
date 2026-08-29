from tensorplay._C import FileCheck as FileCheck

from . import _comparison
from ._comparison import (
    assert_allclose as assert_allclose,
    assert_close as assert_close,
    default_tolerances as default_tolerances,
    get_tolerances as get_tolerances,
)
from ._creation import make_tensor as make_tensor

__all__ = [
    "FileCheck",
    "assert_close",
    "assert_allclose",
    "default_tolerances",
    "get_tolerances",
    "make_tensor",
]
