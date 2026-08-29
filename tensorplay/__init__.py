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
from enum import IntEnum
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

# Generated at build time by tools/generate_tensorplay_version.py (CMake
# generated version module.
from tensorplay.version import __version__ as __version__


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
# -------------------------------------------------------------------------
# preloads component CUDA wheels when the extension reports a missing SONAME.
# Keep the same behavior here: a normal CPU-only import does not eagerly load
# CUDA, while a CUDA build remains importable from NVIDIA Python wheels.
elif sys.platform.startswith('linux'):
    def _get_cuda_dep_paths(path, lib_folder, lib_name):
        paths = []
        # Exact wheel layout only (nvidia/cuda_runtime, nvidia/cublas, ...).
        # A broad 'nvidia/cu*' glob would also swallow foreign major-version
        # wheels like nvidia/cu13 and dlopen the wrong libcudart SONAME,
        paths.extend(glob.glob(os.path.join(path, 'nvidia', lib_folder, 'lib', lib_name)))
        paths.extend(glob.glob(os.path.join(path, lib_folder, 'lib', lib_name)))
        if not paths and '.so.' in lib_name:
            stem = lib_name.split('.so.', 1)[0]
            paths.extend(glob.glob(os.path.join(path, 'nvidia', lib_folder, 'lib', stem + '.so')))
            paths.extend(glob.glob(os.path.join(path, lib_folder, 'lib', stem + '.so')))
        return paths

    def _preload_cuda_lib(lib_folder, lib_name, required=True):
        for path in sys.path:
            candidates = _get_cuda_dep_paths(path, lib_folder, lib_name)
            if candidates:
                ctypes.CDLL(candidates[0])
                return
        package_lib_path = os.path.join(os.path.dirname(__file__), 'lib')
        candidates = glob.glob(os.path.join(package_lib_path, lib_name))
        if not candidates and '.so.' in lib_name:
            stem = lib_name.split('.so.', 1)[0]
            candidates = glob.glob(os.path.join(package_lib_path, stem + '.so'))
        if candidates:
            ctypes.CDLL(candidates[0])
            return
        if required:
            raise ImportError(f'{lib_name} not found in TensorPlay CUDA dependency paths')

    def _preload_cuda_deps(err=None):
        if err is not None:
            message = str(err)
            cuda_error = any(
                token in message
                for token in ('libcud', 'libcublas', 'libcurand', 'libnvrtc')
            )
            if not cuda_error:
                raise err

        # a mixed CUDA installation through cublas' transitive RUNPATH.
        for lib_folder, lib_name, required in (
            ('cublas', 'libcublasLt.so.*[0-9]', True),
            ('cublas', 'libcublas.so.*[0-9]', True),
            ('cudnn', 'libcudnn.so.*[0-9]', True),
            ('cuda_nvrtc', 'libnvrtc.so.*[0-9]', False),
            ('cuda_runtime', 'libcudart.so.*[0-9]', True),
            ('curand', 'libcurand.so.*[0-9]', True),
        ):
            _preload_cuda_lib(lib_folder, lib_name, required=required)

# -------------------------------------------------------------------------
# Core Imports
# -------------------------------------------------------------------------
try:
    from . import _C as _C
except (ImportError, OSError) as _load_error:
    if sys.platform.startswith('linux') and '_preload_cuda_deps' in globals():
        _preload_cuda_deps(_load_error)
        from . import _C as _C
    else:
        raise

if not hasattr(_C, "_log_api_usage_once"):
    _C._log_api_usage_once = lambda *args, **kwargs: None

from ._tensor import Tensor
from ._C import (tensor, DType, Size, Scalar, Device, DeviceType,
                from_dlpack, to_dlpack, from_numpy, frombuffer, set_printoptions,
                default_generator, manual_seed, seed, initial_seed, Generator,
                get_rng_state, set_rng_state,
                set_num_threads, get_num_threads, get_thread_num,
                in_parallel_region, get_parallel_info)
from .autograd import (no_grad, enable_grad, set_grad_enabled, is_grad_enabled,
                       inference_mode)
from .serialization import save, load, inspect_checkpoint
from .random import fork_rng

# -------------------------------------------------------------------------
# DType Aliases
# -------------------------------------------------------------------------
device = Device
dtype = DType
uint8 = DType.uint8
int8 = DType.int8
int16 = DType.int16
uint16 = DType.uint16
uint32 = DType.uint32
uint64 = DType.uint64
int32 = DType.int32
int64 = DType.int64
float32 = DType.float32
float64 = DType.float64
float16 = DType.float16
bfloat16 = DType.bfloat16
complex32 = DType.complex32
complex64 = DType.complex64
complex128 = DType.complex128
bcomplex32 = DType.bcomplex32
bool = DType.bool
undefined = DType.undefined

half = DType.float16
float = DType.float32
double = DType.float64
short = DType.int16
int = DType.int32
long = DType.int64
cfloat = DType.complex64
cdouble = DType.complex128
chalf = DType.complex32


class MemoryFormat(IntEnum):
    CONTIGUOUS = 0
    PRESERVE = 1
    CHANNELS_LAST = 2
    CHANNELS_LAST_3D = 3
    contiguous_format = 0
    preserve_format = 1
    channels_last = 2
    channels_last_3d = 3



class Layout(IntEnum):
    """

    ``tensor.layout`` returns one of these values; dense tensors report
    STRIDED.
    """

    SPARSE_COO = 0
    SPARSE_CSR = 1
    STRIDED = 2
    sparse_coo = 0
    sparse_csr = 1
    strided = 2


sparse_coo = Layout.SPARSE_COO
sparse_csr = Layout.SPARSE_CSR
strided = Layout.STRIDED
contiguous_format = MemoryFormat.CONTIGUOUS
preserve_format = MemoryFormat.PRESERVE
channels_last = MemoryFormat.CHANNELS_LAST
channels_last_3d = MemoryFormat.CHANNELS_LAST_3D


__all__ = [
    "Tensor", "tensor", "from_dlpack", "Scalar", "DeviceType", "device", "dtype", "Size",
    "MemoryFormat", "contiguous_format", "preserve_format", "channels_last", "channels_last_3d",
    "uint8", "int8", "int16", "uint16", "uint32", "uint64", "int32", "int64",
    "float16", "bfloat16", "float32", "float64", "complex32", "complex64", "complex128", "bcomplex32", "bool",
    "half", "float", "double", "short", "int", "long", "cfloat", "cdouble", "chalf",
    "save", "load", "inspect_checkpoint", "as_tensor",
    "no_grad", "enable_grad", "set_grad_enabled", "is_grad_enabled",
    "allclose",
    "compile", "compiler", "graph", "library",
    "set_num_threads", "get_num_threads", "get_thread_num",
    "in_parallel_region", "get_parallel_info",
    "default_generator", "manual_seed", "seed", "initial_seed", "Generator",
    "get_rng_state", "set_rng_state", "fork_rng",
    "DeviceMismatchError",
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
    "where", "maximum", "minimum", "addcmul", "addcdiv",
    "view_as_real", "view_as_complex", "is_complex",
])

# Please keep this list sorted
# assert __all__ == sorted(__all__)

# The tensorplay._C submodule is already loaded above; import multiprocessing
from . import multiprocessing

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

# Catchable device-mismatch error raised by the C++ core (RuntimeError
# subclass); kept explicit because it is an exception type, not an op.
DeviceMismatchError = _C.DeviceMismatchError

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
                    # (pybind11 functions are builtin_function_or_method, which
                    # rejects __module__ writes)
                    if callable(__obj):
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


def unique(input, sorted=True, return_inverse=False, return_counts=False):
    """

    The native op always computes all three outputs; this wrapper mirrors
    ``return_inverse`` / ``return_counts``.
    """
    values, inverse, counts = _C.unique(input, sorted, True, True)
    if return_inverse and return_counts:
        return values, inverse, counts
    if return_counts:
        return values, counts
    if return_inverse:
        return values, inverse
    return values

from ._shape_funcs import *
from ._composite_funcs import *
from ._einsum import einsum
from .utils.comparison import allclose

from ._ops import ops as ops
from . import compiler
from .compiler import compile
from . import library
from . import profiler

# -------------------------------------------------------------------------
# Submodules
# -------------------------------------------------------------------------
from . import cuda
from . import stax
from . import backends
# nn before optim (upstream order): optim.swa_utils pulls in tensorplay.nn,
# which must be fully initialized by then to avoid a partial-import cycle.
from . import nn
from . import optim
from . import graph
from . import autograd
from . import distributed
from . import utils
from . import __config__

from . import amp

from .amp.autocast_mode import (  # noqa: E402
    autocast_decrement_nesting,
    autocast_increment_nesting,
    clear_autocast_cache,
    get_autocast_cpu_dtype,
    get_autocast_dtype,
    get_autocast_gpu_dtype,
    is_autocast_available,
    is_autocast_cache_enabled,
    is_autocast_enabled,
    set_autocast_cache_enabled,
    set_autocast_dtype,
    set_autocast_enabled,
)

from .amp import (  # noqa: E402
    GradScaler,
    autocast,
    custom_bwd,
    custom_fwd,
)

__all__.extend([
    "GradScaler",
    "amp",
    "cuda",
    "autocast",
    "autocast_decrement_nesting",
    "autocast_increment_nesting",
    "clear_autocast_cache",
    "custom_bwd",
    "custom_fwd",
    "get_autocast_cpu_dtype",
    "get_autocast_dtype",
    "get_autocast_gpu_dtype",
    "is_autocast_available",
    "is_autocast_cache_enabled",
    "is_autocast_enabled",
    "set_autocast_cache_enabled",
    "set_autocast_dtype",
    "set_autocast_enabled",
])

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


def as_tensor(data, dtype=None, device=None):
    r"""Convert ``data`` into a tensor, sharing storage when possible.

    If ``data`` is already a tensor with the requested dtype and device, it is
    returned as-is (no copy).  Otherwise it is converted, mirroring

    Args:
        data (tensor, list, or scalar): Initial data for the tensor.
        dtype (:class:`tensorplay.DType`, optional): the desired data type of
            the returned tensor.
        device (:class:`tensorplay.Device`, str, optional): the device of the
            constructed tensor.

    Example::

        >>> x = tensorplay.tensor([1.0, 2.0])
        >>> tensorplay.as_tensor(x) is x
        True
        >>> tensorplay.as_tensor([0, 1, 2], dtype=tensorplay.int64).dtype == tensorplay.int64
        True
    """
    if isinstance(data, tensorplay.Tensor):
        if dtype is not None and data.dtype != dtype:
            data = data.to(dtype)
        if device is not None and data.device != tensorplay.device(device):
            data = data.to(device)
        return data
    return tensor(data, dtype=dtype, device=device)


_GLOBAL_DEVICE_CONTEXT = threading.local()

# thread-local mode stack semantics); _GLOBAL_DEVICE_CONTEXT is kept for API


def get_default_device() -> "tensorplay.device":
    r"""Gets the default ``tensorplay.Tensor`` to be allocated on ``device``"""
    return _C.get_default_device()


def set_default_device(device: Device) -> None:
    """Sets the default ``tensorplay.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with tensorplay.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`tensorplay.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the tensorplay API (not just factory functions).

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`tensorplay.from_numpy` and :func:`tensorplay.frombuffer`

    Args:
        device (:class:`tensorplay.device`, str, int, or None): the device to set as
            default, or ``None`` to clear the override. An integer is
            interpreted as an index for the current accelerator.

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> tensorplay.get_default_device()
        device(type='cpu')
        >>> tensorplay.set_default_device('cuda')  # current device is 0
        >>> tensorplay.get_default_device()
        device(type='cuda', index=0)
        >>> tensorplay.set_default_device('cuda')
        >>> tensorplay.cuda.set_device('cuda:1')  # current device is 1
        >>> tensorplay.get_default_device()
        device(type='cuda', index=1)
        >>> tensorplay.set_default_device('cuda:1')
        >>> tensorplay.get_default_device()
        device(type='cuda', index=1)

    """
    if isinstance(device, str):
        device = Device(device)
    elif isinstance(device, builtins.int):
        # An integer is interpreted as an index for the current accelerator
        # (CUDA is TensorPlay's accelerator device type).
        device = Device(DeviceType.CUDA, device)
    _C._set_default_device(device)


_default_dtype: DType = float32


def get_default_dtype() -> DType:
    """Returns the current default floating point dtype (float32 initially,
    changed by :func:`set_default_dtype`)."""
    return _default_dtype


def set_default_dtype(d: DType, /) -> None:
    r"""

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause tensorplay to raise an exception.

    When TensorPlay is initialized its default floating point dtype is float32,
    and the intent of set_default_dtype(float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`tensorplay.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is float32
        >>> # Python floats are interpreted as float32
        >>> tensorplay.tensor([1.2, 3]).dtype
        tensorplay.float32
        >>> # initial default for floating point is complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> tensorplay.tensor([1.2, 3j]).dtype
        tensorplay.complex64

        >>> tensorplay.set_default_dtype(tensorplay.float64)
        >>> # Python floats are now interpreted as float64
        >>> tensorplay.tensor([1.2, 3]).dtype  # a new floating point tensor
        tensorplay.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> tensorplay.tensor([1.2, 3j]).dtype  # a new complex tensor
        tensorplay.complex128

        >>> tensorplay.set_default_dtype(tensorplay.float16)
        >>> # Python floats are now interpreted as float16
        >>> tensorplay.tensor([1.2, 3]).dtype  # a new floating point tensor
        tensorplay.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> tensorplay.tensor([1.2, 3j]).dtype  # a new complex tensor
        tensorplay.complex32

    """
    _C._set_default_dtype(d)
    global _default_dtype
    _default_dtype = d


def use_deterministic_algorithms(
    mode: builtins.bool,
    *,
    warn_only: builtins.bool = False,
) -> None:
    r"""Sets whether TensorPlay operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    .. note:: This setting alone is not always enough to make an application
        reproducible. Refer to :ref:`reproducibility` for more information.

    .. note:: :func:`tensorplay.set_deterministic_debug_mode` offers an alternative
        interface for this feature.

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.

    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Keyword args:
        warn_only (:class:`bool`, optional): If True, operations that do not
            have a deterministic implementation will throw a warning instead of
            an error. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> tensorplay.use_deterministic_algorithms(True)
    """
    # has no Inductor counterpart.
    _C._set_deterministic_algorithms(mode, warn_only=warn_only)


def are_deterministic_algorithms_enabled() -> builtins.bool:
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`tensorplay.use_deterministic_algorithms` documentation for more details.
    """
    return _C._get_deterministic_algorithms()


def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool:
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`tensorplay.use_deterministic_algorithms` documentation for more
    details.
    """
    return _C._get_deterministic_algorithms_warn_only()


def set_deterministic_debug_mode(debug_mode: builtins.int | str) -> None:
    r"""Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`tensorplay.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """

    # NOTE: builtins.int is used here because int in this scope resolves
    # to tensorplay.int
    if not isinstance(debug_mode, (builtins.int, str)):
        raise TypeError(f"debug_mode must be str or int, but got {type(debug_mode)}")

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, "
                f"`warn`, `error`, but got {debug_mode}"
            )

    if debug_mode == 0:
        _C._set_deterministic_algorithms(False)
    elif debug_mode == 1:
        _C._set_deterministic_algorithms(True, warn_only=True)
    elif debug_mode == 2:
        _C._set_deterministic_algorithms(True)
    else:
        raise RuntimeError(
            f"invalid value of debug_mode, expected 0, 1, or 2, but got {debug_mode}"
        )


def get_deterministic_debug_mode() -> builtins.int:
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`tensorplay.set_deterministic_debug_mode`
    documentation for more details.
    """

    if _C._get_deterministic_algorithms():
        if _C._get_deterministic_algorithms_warn_only():
            return 1
        else:
            return 2
    else:
        return 0


def get_float32_matmul_precision() -> str:
    r"""Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`tensorplay.set_float32_matmul_precision` documentation for more details.
    """
    return _C.get_float32_matmul_precision()


def set_float32_matmul_precision(precision: str) -> None:
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `tensorplay.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `tensorplay.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `tensorplay.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """
    if not isinstance(precision, str):
        raise TypeError("set_float32_matmul_precision expects a str, "
                        f"but got {type(precision)}")
    _C._set_float32_matmul_precision(precision)


__all__.extend([
    "inference_mode",
    "get_default_dtype",
    "set_default_dtype",
    "get_default_device",
    "set_default_device",
    "use_deterministic_algorithms",
    "are_deterministic_algorithms_enabled",
    "is_deterministic_algorithms_warn_only_enabled",
    "set_deterministic_debug_mode",
    "get_deterministic_debug_mode",
    "get_float32_matmul_precision",
    "set_float32_matmul_precision",
])

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

# Re-pin the einsum shim: functional.py's thin wrapper predates the sublist
# quantile/nanquantile/histogram need the same treatment: the generated
# functional.py wrappers forward a raw Python-number `q` (the _C binding
# requires a Tensor) and drop `histogram`'s `range` keyword, so the
# hand-written _composite_funcs versions (scalar-q -> input-dtype Tensor
from ._einsum import einsum as einsum
from ._composite_funcs import quantile as quantile
from ._composite_funcs import nanquantile as nanquantile
from ._composite_funcs import histogram as histogram
from ._finfo import finfo, iinfo
from . import jit

# Python's ``import *`` intentionally omits underscore-prefixed names, but
# only generated ``_foreach_*`` wrappers; their implementation is still the
# native dispatcher/backend and this block does not introduce a Python
# composite operator.  ``_amp_*`` dispatcher hooks follow the same rule
for _foreach_name in dir(functional):
    if (_foreach_name.startswith("_foreach_") or _foreach_name.startswith("_amp_")
            or _foreach_name.startswith("_fused_")):
        globals()[_foreach_name] = getattr(functional, _foreach_name)
        if _foreach_name not in __all__:
            __all__.append(_foreach_name)
del _foreach_name


# -------------------------------------------------------------------------
# ``from tensorplay.functional import *`` above, which shadows earlier defs).
#
# reduction returning a named tuple with ``values``/``indices``, and the
# reduction faces only, so the binary face and the named-tuple contract are
# restored here at the package boundary.
# -------------------------------------------------------------------------
import collections as _collections

_max_return_type = _collections.namedtuple("max_return_type", ["values", "indices"])
_min_return_type = _collections.namedtuple("min_return_type", ["values", "indices"])


def max(input, other=None, *, dim=None, keepdim=False):
    if other is not None:
        return _C.maximum(input, other)
    result = functional.max(input, dim=dim, keepdim=keepdim)
    if dim is not None and isinstance(result, tuple):
        return _max_return_type(*result)
    return result


def min(input, other=None, *, dim=None, keepdim=False):
    if other is not None:
        return _C.minimum(input, other)
    result = functional.min(input, dim=dim, keepdim=keepdim)
    if dim is not None and isinstance(result, tuple):
        return _min_return_type(*result)
    return result


def gradient(input, *, spacing=None, dim=None, edge_order=1):
    """

    The native binding only takes ``Tensor[]`` spacing / ``int[]`` dim;
    materialize python numbers into scalar tensors typed like ``input`` (a
    single scalar applies to every dimension).
    """
    dims = [dim] if isinstance(dim, builtins.int) else (
        list(dim) if dim is not None else list(builtins.range(input.dim())))
    if spacing is None:
        sp = []
    elif isinstance(spacing, (builtins.int, builtins.float)):
        sp = [full([], spacing, dtype=input.dtype) for _ in dims]
    elif isinstance(spacing, Tensor):
        # A single coordinate tensor (numpy-style) applies to every
        # requested dimension; the kernel broadcasts numel()==n vs ==1.
        sp = [spacing]
    else:
        sp = [v if isinstance(v, Tensor) else full([], v, dtype=input.dtype)
              for v in spacing]
    return _C.gradient(self=input, spacing=sp, dim=dims, edge_order=edge_order)

################################################################################
# Import most common subpackages
################################################################################

# Use the redundant form so that type checkers know that these are a part of
# the public API. The "regular" import lines are there solely for the runtime
# side effect of adding to the imported module's members for other users.

# needs to be before import tensorplay.nn as nn to avoid circular dependencies
from tensorplay.autograd import (
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
    is_grad_enabled as is_grad_enabled,
)

from tensorplay import (
    __config__ as __config__,
    autograd as autograd,
    backends as backends,
    cuda as cuda,
    futures as futures,
    hub as hub,
    multiprocessing as multiprocessing,
    nn as nn,
    optim as optim,
    types as types,
    utils as utils,
)


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion for tools like pylance, even when these modules are not explicitly
    # imported in user code.
    from tensorplay import (
        export as export,
    )

else:
    _lazy_modules = {
        "audio",
        "export",
        "fft",
        "linalg",
        "quantization",
        "sparse",
        "special",
        "vision",
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
