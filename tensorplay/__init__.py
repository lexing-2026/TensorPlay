"""
The tensorplay package offers a simple deep-learning framework
designed for educational purposes and small-scale experiments.
It defines a data structure for multidimensional arrays called Tensor,
on which it encapsulates mathematical operations.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import builtins
import ctypes
import glob
import importlib
import inspect
import os
import sys
import threading
from typing import (
    Any as _Any,
    TYPE_CHECKING,
)
from typing_extensions import TypeIs as _TypeIs


__version__ = "1.0.0rc0"


# -------------------------------------------------------------------------
# DLL Loading (Windows)
# -------------------------------------------------------------------------
if sys.platform == 'win32':
    def _load_dll_libraries():
        # Adapted from TensorPlay
        import sysconfig
        
        # Helper to add DLL directory safely
        def _add_dll_directory(path):
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                except (OSError, AttributeError):
                    pass
        
        # 1. Package's own lib directory (p10.dll, tpx.dll, stax.dll, dnnl.dll, etc.)
        package_lib_path = os.path.join(os.path.dirname(__file__), 'lib')
        _add_dll_directory(package_lib_path)
        # Add to PATH immediately as fallback
        os.environ["PATH"] = package_lib_path + ";" + os.environ["PATH"]
        
        # 2. Package's root directory (sometimes DLLs are here)
        _add_dll_directory(os.path.dirname(__file__))

        # 3. Conda/Python Library/bin (for MKL, OneDNN, etc.)
        py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
        _add_dll_directory(py_dll_path)
        
        # 4. VirtualEnv support
        if sys.exec_prefix != sys.base_exec_prefix:
            base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
            _add_dll_directory(base_py_dll_path)
        
        # 5. User site-packages Library/bin
        userbase = sysconfig.get_config_var('userbase')
        if userbase:
            user_dll_path = os.path.join(userbase, 'Library', 'bin')
            _add_dll_directory(user_dll_path)
            
        # 6. Explicitly load DLLs to ensure dependencies are resolved
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        # Pre-load critical dependencies in order
        package_lib_path = os.path.join(os.path.dirname(__file__), 'lib')
        
        # Order matters! Dependencies first.
        # MKL -> CUDA/cuDNN -> p10 -> tpx -> stax
        
        # 1. Pre-load MKL (if present)
        mkl_dlls = glob.glob(os.path.join(package_lib_path, "mkl_*.dll"))
        # Load mkl_core and mkl_sequential first if they exist
        priority_mkl = ["mkl_core", "mkl_sequential", "mkl_intel_lp64", "mkl_def", "mkl_avx2"]
        sorted_mkl = []
        for name in priority_mkl:
            for dll in mkl_dlls:
                if name in os.path.basename(dll):
                    sorted_mkl.append(dll)
        # Add remaining MKL DLLs
        for dll in mkl_dlls:
            if dll not in sorted_mkl:
                sorted_mkl.append(dll)
                
        # 2. Pre-load CUDA/cuDNN (if present)
        cuda_dlls = glob.glob(os.path.join(package_lib_path, "cudart*.dll")) + \
                    glob.glob(os.path.join(package_lib_path, "cublas*.dll")) + \
                    glob.glob(os.path.join(package_lib_path, "cudnn*.dll")) + \
                    glob.glob(os.path.join(package_lib_path, "curand*.dll"))
                    
        # 3. Core Libraries
        core_dlls = [
            os.path.join(package_lib_path, "p10.dll"),
            os.path.join(package_lib_path, "tpx.dll"),
            os.path.join(package_lib_path, "stax.dll")
        ]
        
        all_dlls = sorted_mkl + cuda_dlls + core_dlls
        
        path_patched = False
        for dll in all_dlls:
            if not os.path.exists(dll):
                continue
                
            if "OpenCL" in dll:
                continue
            
            is_loaded = False
            if with_load_library_flags:
                # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                if res:
                    is_loaded = True
            
            if not is_loaded:
                # Fallback
                if not path_patched:
                    os.environ["PATH"] = package_lib_path + ";" + os.environ["PATH"]
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                
        kernel32.SetErrorMode(prev_error_mode)
            
    _load_dll_libraries()
    del _load_dll_libraries

# -------------------------------------------------------------------------
# Core Imports
# -------------------------------------------------------------------------
from ._tensor import Tensor
from ._C import (tensor, DType, Size, Scalar, Device, DeviceType,
                from_dlpack, to_dlpack, set_printoptions,
                default_generator, manual_seed, seed, initial_seed, Generator)
from .autograd import no_grad, enable_grad, set_grad_enabled, is_grad_enabled
from .serialization import save, load

# -------------------------------------------------------------------------
# DType Aliases
# -------------------------------------------------------------------------
device = Device
dtype = DType
uint8 = DType.uint8
int8 = DType.int8
int16 = DType.int16
uint16 = DType.uint16
int32 = DType.int32
int64 = DType.int64
float32 = DType.float32
float64 = DType.float64
bool = DType.bool
undefined = DType.undefined


__all__ = [
    "Tensor", "tensor", "from_dlpack", "Scalar", "DeviceType", "device", "dtype", "Size",
    "uint8", "int8", "int16", "uint16", "int32", "int64", "float32", "float64", "bool",
    "save", "load", "as_tensor",
    "no_grad", "enable_grad", "set_grad_enabled", "is_grad_enabled",
    "allclose",
    "__config__",
]

# Append functional API to __all__
__all__.extend([
    "abs", "acos", "acosh", "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_backward", "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward", "all", "angle", "any", "arange",
    "argmax", "argmin", "asin", "asinh", "atan", "atan2", "atanh",
    "avg_pool2d", "avg_pool2d_backward", "batch_norm",
    "batch_norm_backward", "bernoulli", "cat", "ceil", "chunk", "clamp",
    "clamp_backward", "constant_pad_nd", "constant_pad_nd_backward",
    "conv1d", "conv1d_grad_bias", "conv1d_grad_input",
    "conv1d_grad_weight", "conv2d", "conv2d_grad_bias",
    "conv2d_grad_input", "conv2d_grad_weight", "conv3d",
    "conv3d_grad_bias", "conv3d_grad_input", "conv3d_grad_weight",
    "conv_transpose2d", "conv_transpose2d_grad_bias",
    "conv_transpose2d_grad_input", "conv_transpose2d_grad_weight",
    "conv_transpose3d", "conv_transpose3d_grad_bias",
    "conv_transpose3d_grad_input", "conv_transpose3d_grad_weight", "cos",
    "cosh", "embedding", "embedding_dense_backward", "empty",
    "empty_like", "eq", "exp", "eye", "floor", "full", "full_like", "ge",
    "gelu", "group_norm", "group_norm_backward", "gt", "instance_norm",
    "instance_norm_backward", "layer_norm", "layer_norm_backward", "le",
    "lerp", "linspace", "log", "log_softmax", "logspace", "lt",
    "masked_select", "matmul", "max", "max_pool2d", "max_pool2d_backward",
    "mean", "median", "min", "mm", "mse_loss", "mse_loss_backward", "ne",
    "neg", "nll_loss", "nll_loss_backward", "norm", "normal", "ones",
    "ones_like", "permute", "permute_backward", "poisson", "pow", "prod",
    "rand", "rand_like", "randint", "randint_like", "randn", "randn_like",
    "randperm", "relu", "reshape", "round", "rsqrt", "sigmoid", "sign",
    "silu", "sin", "sinh", "softmax", "split", "sqrt", "square",
    "squeeze", "squeeze_backward", "stack", "std", "sum", "t", "tan",
    "tanh", "threshold_backward", "transpose", "unbind", "unsqueeze",
    "var", "zeros", "zeros_like",
])

# Please keep this list sorted
# assert __all__ == sorted(__all__)

# The tensorplay._C submodule is already loaded via `from tensorplay._C import *` above
# Make an explicit reference to the _C submodule to appease linters
from tensorplay import _C as _C, multiprocessing

import functools

from ._C import (
    abs, acos, acosh, adaptive_avg_pool2d, adaptive_avg_pool2d_backward, 
    adaptive_max_pool2d, adaptive_max_pool2d_backward, all, angle, any, 
    arange, argmax, argmin, asin, asinh, atan, atan2, atanh, avg_pool2d, 
    avg_pool2d_backward, batch_norm, batch_norm_backward, bernoulli, cat, 
    ceil, chunk, clamp, clamp_backward, constant_pad_nd, 
    constant_pad_nd_backward, conv1d, conv1d_grad_bias, conv1d_grad_input, 
    conv1d_grad_weight, conv2d, conv2d_grad_bias, conv2d_grad_input, 
    conv2d_grad_weight, conv3d, conv3d_grad_bias, conv3d_grad_input, 
    conv3d_grad_weight, conv_transpose2d, conv_transpose2d_grad_bias, 
    conv_transpose2d_grad_input, conv_transpose2d_grad_weight, 
    conv_transpose3d, conv_transpose3d_grad_bias, 
    conv_transpose3d_grad_input, conv_transpose3d_grad_weight, cos, cosh, 
    embedding, embedding_dense_backward, empty, empty_like, eq, exp, eye, 
    floor, full, full_like, ge, gelu, group_norm, group_norm_backward, gt, 
    instance_norm, instance_norm_backward, layer_norm, 
    layer_norm_backward, le, lerp, linspace, log, log_softmax, logspace, 
    lt, masked_select, matmul, max, max_pool2d, max_pool2d_backward, mean, 
    median, min, mm, mse_loss, mse_loss_backward, ne, neg, nll_loss, 
    nll_loss_backward, norm, normal, ones, ones_like, permute, 
    permute_backward, poisson, pow, prod, rand, rand_like, randint, 
    randint_like, randn, randn_like, randperm, relu, reshape, round, 
    rsqrt, sigmoid, sign, silu, sin, sinh, softmax, split, sqrt, square, 
    squeeze, squeeze_backward, stack, std, sum, t, tan, tanh, 
    threshold_backward, transpose, unbind, unsqueeze, var, zeros, 
    zeros_like, as_tensor,
)

__attr_name, __obj = "", None
for __attr_name in dir(_C):
    if __attr_name[0] != "_" and not __attr_name.endswith("Base"):
        __all__.append(__attr_name)
        __obj = getattr(_C, __attr_name)
        if callable(__obj) or inspect.isclass(__obj):
            if os.getenv("TENSORPLAY_BUILDING_STUBS") == "1":
                continue

            if __obj.__module__ != __name__:
                try:
                    __obj.__module__ = __name__
                except AttributeError:
                    # Fallback: wrap it if it's a function and module wasn't updated
                    # Check if it is a nanobind function (type name usually nb_func)
                    if "nb_func" in type(__obj).__name__:
                         def make_wrapper(f):
                             @functools.wraps(f)
                             def wrapper(*args, **kwargs):
                                 return f(*args, **kwargs)
                             wrapper.__module__ = __name__
                             return wrapper
                         
                         wrapper = make_wrapper(__obj)
                         # Overwrite in globals if it exists (imported from _C)
                         if __attr_name in globals():
                             globals()[__attr_name] = wrapper

    elif __attr_name == "TensorBase":
        if hasattr(sys.modules[__name__], __attr_name):
            delattr(sys.modules[__name__], __attr_name)

del __attr_name, __obj

if not TYPE_CHECKING:
    def _import_extension_to_sys_modules(module, memo=None):
        """
        Recursively import submodules of a C extension module into sys.modules.
        """
        if memo is None:
            memo = set()
        if module in memo:
            return
        memo.add(module)
        module_name = module.__name__
        for name in dir(module):
            member = getattr(module, name)
            member_name = getattr(member, "__name__", "")
            if inspect.ismodule(member) and member_name.startswith(module_name):
                sys.modules.setdefault(member_name, member)
                _import_extension_to_sys_modules(member, memo)

    _import_extension_to_sys_modules(_C)
    del _import_extension_to_sys_modules


from .functional import *
from .utils.comparison import allclose

# -------------------------------------------------------------------------
# Submodules
# -------------------------------------------------------------------------
from . import cuda
from . import stax
from . import backends
from . import optim
from . import nn
from . import autograd
from . import utils
from . import __config__

# -------------------------------------------------------------------------
# Lazy Loading for Heavy Submodules
# -------------------------------------------------------------------------
def __getattr__(name):
    if name == "vision":
        import tensorplay.vision as vision
        return vision
    elif name == "audio":
        import tensorplay.audio as audio
        return audio
    return None


def typename(obj: _Any, /) -> str:
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = tensorplay.tensor([1, 2, 3])
        >>> tensorplay.typename(x)
        'tensorplay.LongTensor'
        >>> tensorplay.typename(tensorplay.nn.Parameter)
        'tensorplay.nn.parameter.Parameter'
    """
    if isinstance(obj, tensorplay.Tensor):
        return obj.type()

    module = getattr(obj, "__module__", "") or ""
    qualname = ""

    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__

    if module in {"", "builtins"}:
        return qualname
    return f"{module}.{qualname}"


def is_tensor(obj: _Any, /) -> _TypeIs["tensorplay.Tensor"]:
    r"""Returns True if `obj` is a TensorPlay tensor.

    Note that this function is simply doing ``isinstance(obj, Tensor)``.
    Using that ``isinstance`` check is better for type checking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_tensor``.

    Args:
        obj (object): Object to test
    Example::

        >>> x = tensorplay.tensor([1, 2, 3])
        >>> tensorplay.is_tensor(x)
        True

    """
    return isinstance(obj, tensorplay.Tensor)


_GLOBAL_DEVICE_CONTEXT = threading.local()

newaxis: None = None

__all__.extend(["e", "pi", "nan", "inf", "newaxis"])

from tensorplay._tensor import Tensor

# The _tensor_classes set is initialized by the call to initialize_python_bindings.
_tensor_classes: set[type[Tensor]] = set()

import tensorplay

__all__.extend(
    name for name in dir(tensorplay) if isinstance(getattr(tensorplay, name), tensorplay.dtype)
)

# needs to be after the above c++ bindings so we can overwrite from Python side
from tensorplay import functional as functional
from tensorplay.functional import *

################################################################################
# Import most common subpackages
################################################################################

# Use the redundant form so that type checkers know that these are a part of
# the public API. The "regular" import lines are there solely for the runtime
# side effect of adding to the imported module's members for other users.

# needs to be before import tensorplay.nn as nn to avoid circular dependencies
from tensorplay.autograd import (
    enable_grad as enable_grad,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
    is_grad_enabled as is_grad_enabled,
)

from tensorplay import (
    __config__ as __config__,
    autograd as autograd,
    backends as backends,
    cpu as cpu,
    cuda as cuda,
    hub as hub,
    multiprocessing as multiprocessing,
    nn as nn,
    optim as optim,
    types as types,
    utils as utils,
    version as version,
)


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.
    from tensorplay import (
        onnx as onnx,
    )

else:
    _lazy_modules = {
        "onnx",
    }

    def __getattr__(name):
        # Lazy modules
        if name in _lazy_modules:
            return importlib.import_module(f".{name}", __name__)
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _import_device_backends():
    """
    Leverage the Python plugin mechanism to load out-of-the-tree device extensions.
    """
    from importlib.metadata import entry_points

    group_name = "tensorplay.backends"
    if sys.version_info < (3, 10):
        backend_extensions = entry_points().get(group_name, ())
    else:
        backend_extensions = entry_points(group=group_name)

    for backend_extension in backend_extensions:
        try:
            # Load the extension
            entrypoint = backend_extension.load()
            # Call the entrypoint
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the backend extension: {backend_extension.name}. "
                f"You can disable extension auto-loading with TENSORPLAY_DEVICE_BACKEND_AUTOLOAD=0."
            ) from err


def _is_device_backend_autoload_enabled() -> builtins.bool:
    """
    Whether autoloading out-of-the-tree device extensions is enabled.
    The switch depends on the value of the environment variable
    `TENSORPLAY_DEVICE_BACKEND_AUTOLOAD`.

    Returns:
        bool: Whether to enable autoloading the extensions. Enabled by default.

    Examples:
        >>> tensorplay._is_device_backend_autoload_enabled()
        True
    """
    # enabled by default
    return os.getenv("TENSORPLAY_DEVICE_BACKEND_AUTOLOAD", "1") == "1"


def _as_tensor_fullprec(t):
    """
    Like tensorplay.as_tensor, but when given Python data types it will keep
    them in full precision.  Used for calling convention for Dynamo.
    """
    ty = type(t)
    if ty is builtins.float:
        return tensorplay.as_tensor(t, dtype=tensorplay.float64)
    elif ty is builtins.int:
        return tensorplay.as_tensor(t, dtype=tensorplay.int64)
    else:
        return tensorplay.as_tensor(t)


# `_import_device_backends` should be kept at the end to ensure
# all the other functions in this module that may be accessed by
# an autoloaded backend are defined
if _is_device_backend_autoload_enabled():
    _import_device_backends()