import builtins as _builtins

builtins_int = _builtins.int
builtins_complex = _builtins.complex

from collections import OrderedDict

from . import _C

# Alias Tensor to _C.TensorBase so that objects created by C++ (like via tp.tensor())
# are instances of this class.
Tensor = _C.TensorBase

def ndimension(self) -> int:
    """
    Alias for dim()
    """
    return self.dim()

def long(self):
    return self.to(_C.int64)

def float(self):
    return self.to(_C.float32)

def int(self):
    return self.to(_C.int32)

def double(self):
    return self.to(_C.float64)

def cuda(self, device=None, non_blocking=False):
    """
    Returns a copy of this object in CUDA memory.
    If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.
    """
    if device is None:
        device_idx = _C._cuda.current_device() if _C._cuda.is_available() else 0
    elif isinstance(device, int):
        device_idx = device
    elif isinstance(device, str):
        parsed = _C.Device(device)
        device_idx = parsed.index
        if device_idx < 0:
            device_idx = _C._cuda.current_device() if _C._cuda.is_available() else 0
    elif isinstance(device, _C.Device):
        if not device.is_cuda():
            raise ValueError(f"Expected a CUDA device, got {device}")
        device_idx = device.index
        if device_idx < 0:
            device_idx = _C._cuda.current_device() if _C._cuda.is_available() else 0
    else:
        raise TypeError(f"Invalid CUDA device: {device!r}")
    
    return self.to(_C.Device(_C.DeviceType.CUDA, device_idx), non_blocking=non_blocking)

def cpu(self):
    """
    Returns a copy of this object in CPU memory.
    If this object is already in CPU memory, then no copy is performed and the original object is returned.
    """
    return self.to(_C.Device(_C.DeviceType.CPU))

def t(self):
    """
    Returns the transpose of the tensor.
    Aliased to transpose(0, 1) to ensure correct autograd behavior (TransposeBackward).
    """
    ndim = self.dim()
    if ndim > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {ndim}D")
    if ndim < 2:
        return self
    return self.transpose(0, 1)

def type(self, dtype=None, non_blocking=False, **kwargs):
    """
    Returns the type if dtype is not provided, else casts this object to the specified type.
    """
    if dtype is None and not kwargs:
        device_str = ""
        if self.is_cuda:
            device_str = "cuda."
        
        dtype_map = {
            _C.float32: "FloatTensor",
            _C.float64: "DoubleTensor",
            _C.float16: "HalfTensor",
            _C.bfloat16: "BFloat16Tensor",
            _C.int32: "IntTensor",
            _C.int64: "LongTensor",
            _C.int16: "ShortTensor",
            _C.int8: "CharTensor",
            _C.uint8: "ByteTensor",
            _C.uint16: "UInt16Tensor",
            _C.uint32: "UInt32Tensor",
            _C.uint64: "UInt64Tensor",
            _C.complex32: "ComplexHalfTensor",
            _C.complex64: "ComplexFloatTensor",
            _C.complex128: "ComplexDoubleTensor",
            _C.bcomplex32: "BComplex32Tensor",
            _C.bool: "BoolTensor",
        }
        
        dt = self.dtype
        if dt in dtype_map:
            return f"tensorplay.{device_str}{dtype_map[dt]}"
        return f"tensorplay.{device_str}Tensor"
    
    return self.to(dtype, non_blocking=non_blocking, **kwargs)


Tensor.ndimension = ndimension
Tensor.long = long
Tensor.float = float
Tensor.int = int
Tensor.double = double
Tensor.cuda = cuda
Tensor.cpu = cpu
Tensor.t = t
Tensor.type = type




def unfold(self, dimension, size, step):
    """Returns a view of the original tensor which contains all slices of
    size :attr:`size` from :attr:`self` in the dimension :attr:`dimension`,

    view semantics, including 0-d inputs and error messages).
    """
    return _C.unfold(self, dimension, size, step)


Tensor.unfold = unfold


def register_hook(self, hook):
    """

    The hook is called every time a gradient with respect to this tensor is
    computed. It may modify the gradient by returning a replacement Tensor;
    returning ``None`` leaves the gradient unchanged. Hooks compose in
    registration order.

    Returns a :class:`~tensorplay.utils.hooks.RemovableHandle` whose
    ``remove()`` method (or context-manager form) unregisters the hook.
    """
    from tensorplay.utils.hooks import RemovableHandle

    if not self.requires_grad:
        raise RuntimeError(
            "cannot register a hook on a tensor that doesn't require gradient"
        )

    hooks_dict = OrderedDict()

    def _apply_hooks(grad):
        for h in list(hooks_dict.values()):
            result = h(grad)
            if result is not None:
                grad = result
        return grad

    if self.is_leaf or self.grad_fn is None:
        node = self._accumulate_grad_node
        if node is None:
            raise RuntimeError(
                "cannot register a hook on a tensor whose AccumulateGrad node "
                "has not been created yet; run a backward pass first"
            )
        pos = 0
    else:
        node = self.grad_fn
        pos = self._output_nr

    def pre_hook(grads):
        grad = grads[pos]
        new_grad = _apply_hooks(grad)
        if new_grad is grad:
            return grads
        out = list(grads)
        out[pos] = new_grad
        return out

    node.add_pre_hook(pre_hook)

    handle = RemovableHandle(hooks_dict)
    hooks_dict[handle.id] = hook
    return handle


Tensor.register_hook = register_hook


def register_post_accumulate_grad_hook(self, hook):
    """

    The hook runs after the gradient has been accumulated into ``self.grad``.
    It receives the tensor (the parameter) and its return value is ignored;
    unlike :meth:`register_hook` it cannot replace the gradient, but it may
    modify ``self.grad`` in place. Only leaf tensors that require grad and
    are used in the autograd graph support this hook.

    Returns a :class:`~tensorplay.utils.hooks.RemovableHandle`.
    """
    from tensorplay.utils.hooks import RemovableHandle

    if not self.requires_grad:
        raise RuntimeError(
            "cannot register a hook on a tensor that doesn't require gradient"
        )
    if not self.is_leaf:
        raise RuntimeError(
            "Registering a hook on a tensor that is not a leaf will "
            "cause an error"
        )

    hooks_dict = OrderedDict()

    def post_hook(_inputs, outputs):
        for h in list(hooks_dict.values()):
            h(self)
        return outputs

    node = self._accumulate_grad_node
    if node is None:
        raise RuntimeError(
            "cannot register a hook on a tensor whose AccumulateGrad node "
            "has not been created yet; run a backward pass first"
        )
    node.add_post_hook(post_hook)

    handle = RemovableHandle(hooks_dict)
    hooks_dict[handle.id] = hook
    return handle


Tensor.register_post_accumulate_grad_hook = register_post_accumulate_grad_hook


# ---------------------------------------------------------------------------
# The C++ binding takes a single sequence; normalize the variadic form here.
# ---------------------------------------------------------------------------
_orig_permute = Tensor.permute


def _permute(self, *dims):
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        dims = dims[0]
    return _orig_permute(self, list(dims))


Tensor.permute = _permute


# ---------------------------------------------------------------------------
# generated TensorMethods binding also names the parameter `size`, so
# t.expand(size=[2, 3]) is valid there; keep that surface here.
# ---------------------------------------------------------------------------
_orig_expand = Tensor.expand


def _expand(self, *args, size=None, implicit=False):
    if size is not None:
        if args:
            raise TypeError("expand() got multiple values for argument 'size'")
        size = size
    else:
        size = args
    if len(size) == 1:
        s0 = size[0]
        if isinstance(s0, (list, tuple)) or hasattr(s0, "__iter__"):
            size = tuple(s0)
    if implicit:
        return _orig_expand(self, list(size), implicit=True)
    return _orig_expand(self, list(size))


Tensor.expand = _expand


# ---------------------------------------------------------------------------
# __bool__ lives in the C extension (nb_bool slot on TensorBase): empty ->
# RuntimeError "no values is ambiguous", one element -> value != 0, more ->


def _norm_new_size(size):
    if len(size) == 1 and hasattr(size[0], "__iter__") and \
            not isinstance(size[0], _C.TensorBase):
        return [builtins_int(x) for x in size[0]]
    return [builtins_int(x) for x in size]


def _flag(out, requires_grad):
    return out.requires_grad_(True) if requires_grad else out


def _new_zeros(self, *size, dtype=None, device=None, requires_grad=False):
    shape = _norm_new_size(size)
    if shape:
        out = _C.full(shape, 0.0, dtype=dtype or self.dtype,
                      device=device or self.device)
    else:
        out = _C.zeros_like(self, dtype=dtype or self.dtype,
                            device=device or self.device)
    return _flag(out, requires_grad)


def _new_ones(self, *size, dtype=None, device=None, requires_grad=False):
    shape = _norm_new_size(size)
    if shape:
        out = _C.full(shape, 1.0, dtype=dtype or self.dtype,
                      device=device or self.device)
    else:
        out = _C.ones_like(self, dtype=dtype or self.dtype,
                           device=device or self.device)
    return _flag(out, requires_grad)


def _new_full(self, size, fill_value, *, dtype=None, device=None,
              requires_grad=False):
    out = _C.full(list(size), fill_value, dtype=dtype or self.dtype,
                  device=device or self.device)
    return out.requires_grad_(requires_grad) if requires_grad else out


def _new_empty(self, size, *, dtype=None, device=None, requires_grad=False):
    if isinstance(size, builtins_int):
        size = [size]
    shape = _norm_new_size(tuple(size))
    if shape:
        out = _C.empty(shape, dtype=dtype or self.dtype,
                       device=device or self.device)
    else:
        out = _C.empty_like(self, dtype=dtype or self.dtype,
                            device=device or self.device)
    return _flag(out, requires_grad)


def _new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
    out = _C.tensor(data, dtype=dtype or self.dtype,
                    device=device or self.device)
    return out.requires_grad_(requires_grad) if requires_grad else out


Tensor.new_zeros = _new_zeros
Tensor.new_ones = _new_ones
Tensor.new_full = _new_full
Tensor.new_empty = _new_empty
Tensor.new_tensor = _new_tensor


# ---------------------------------------------------------------------------
# as generated bindings; add the rest of the family.
# ---------------------------------------------------------------------------
_DTYPE_SHORTCUTS = {
    "bool": "bool",
    "byte": "uint8",
    "char": "int8",
    "short": "int16",
    "half": "float16",
    "bfloat16": "bfloat16",
}


def _make_dtype_shortcut(attr, dt_name):
    def op(self):
        return self.to(getattr(_C.DType, dt_name))
    op.__name__ = attr
    return op


for _attr, _dt in _DTYPE_SHORTCUTS.items():
    setattr(Tensor, _attr, _make_dtype_shortcut(_attr, _dt))
del _attr, _dt


# ---------------------------------------------------------------------------
# pointwise method forms routed through the top-level composites so the
# ---------------------------------------------------------------------------
def _cf(name):
    from . import _composite_funcs
    return getattr(_composite_funcs, name)


def _as_tensor(x):
    from . import as_tensor
    return as_tensor(x)


# floor_divide/remainder/fmod/true_divide are native methods; only the
# reflected forms need a wrapper, since the left operand arrives as a plain
# Python number.
def __rfloordiv__(self, other):
    return _as_tensor(other).floor_divide(self)


def __rmod__(self, other):
    import tensorplay
    return tensorplay.remainder(other, self)


Tensor.__rfloordiv__ = __rfloordiv__
Tensor.__rmod__ = __rmod__


def repeat_interleave(self, repeats, dim=None, *, output_size=None):
    return _cf("repeat_interleave")(self, repeats, dim=dim,
                                    output_size=output_size)


Tensor.repeat_interleave = repeat_interleave


# ---------------------------------------------------------------------------
# reduction faces returning plain tuples.  Restore (a) the elementwise binary
# form ``t.max(Tensor)`` and (b) the named-tuple contract with
# ---------------------------------------------------------------------------
_native_tensor_max = Tensor.max
_native_tensor_min = Tensor.min


def max(self, *args, **kwargs):
    if args and hasattr(args[0], "shape"):
        return _C.maximum(self, args[0])
    result = _native_tensor_max(self, *args, **kwargs)
    if ((args and args[0] is not None) or kwargs.get("dim") is not None) \
            and isinstance(result, tuple):
        from ._return_types import max_return_type
        return max_return_type(*result)
    return result


def min(self, *args, **kwargs):
    if args and hasattr(args[0], "shape"):
        return _C.minimum(self, args[0])
    result = _native_tensor_min(self, *args, **kwargs)
    if ((args and args[0] is not None) or kwargs.get("dim") is not None) \
            and isinstance(result, tuple):
        from ._return_types import min_return_type
        return min_return_type(*result)
    return result


Tensor.max = max
Tensor.min = min


def amax(self, dim=None, keepdim=False):
    from . import functional
    return functional.amax(self, dim=dim, keepdim=keepdim)


def amin(self, dim=None, keepdim=False):
    from . import functional
    return functional.amin(self, dim=dim, keepdim=keepdim)


Tensor.amax = amax
Tensor.amin = amin


def count_nonzero(self, dim=None):
    nz = self.ne(0).to(_C.DType.int64)
    return nz.sum() if dim is None else nz.sum(dim=dim)


Tensor.count_nonzero = count_nonzero


def nonzero(self):
    from . import functional
    return functional.nonzero(self)


Tensor.nonzero = nonzero


def unique(self, sorted=True, return_inverse=False, return_counts=False):
    # public contract of returning 1/2/3 tensors depending on the flags.
    values, inverse, counts = _C.unique(self, sorted, True, True)
    outs = [values]
    if return_inverse:
        outs.append(inverse)
    if return_counts:
        outs.append(counts)
    return outs[0] if len(outs) == 1 else tuple(outs)


Tensor.unique = unique


def unique_consecutive(self, return_inverse=False, return_counts=False,
                       dim=None):
    values, inverse, counts = _C.unique_consecutive(
        self, return_inverse, return_counts, dim)
    outs = [values]
    if return_inverse:
        outs.append(inverse)
    if return_counts:
        outs.append(counts)
    return outs[0] if len(outs) == 1 else tuple(outs)


Tensor.unique_consecutive = unique_consecutive


def topk(self, k, dim=None, largest=True, sorted=True):
    d = self.dim() - 1 if dim is None else builtins_int(dim)
    if d < 0:
        d += self.dim()
    desc = largest
    values, indices = _C.sort(self, dim=d, descending=desc)
    n = self.size(d)
    k = builtins_int(k)
    if k < 0 or k > n:
        raise RuntimeError(
            f"selected index k: {k} out of range for dimension {d} "
            f"of size {n}")
    sl = [slice(None)] * self.dim()
    sl[d] = slice(0, k)
    return values[tuple(sl)], indices[tuple(sl)]


Tensor.topk = topk


# ---------------------------------------------------------------------------
# operator dunders: floor-div/mod reflect + bitwise family. Bitwise ops
# dispatch to the module functions (integers and bool; shifts take the
# bit width modulo through the unsigned domain in the kernel).
# ---------------------------------------------------------------------------
def __ifloordiv__(self, other):
    res = self.floor_divide(other)
    with _C_no_grad():
        self.copy_(res)
    return self


def __imod__(self, other):
    res = self.remainder(other)
    with _C_no_grad():
        self.copy_(res)
    return self


def _C_no_grad():
    from .autograd import no_grad
    return no_grad()


_BITWISE_NAMES = {
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "bitwise_left_shift",
    "rshift": "bitwise_right_shift",
}


def _bitwise_fn(name, reflect=False):
    def op(self, other):
        import tensorplay
        fn = getattr(tensorplay, _BITWISE_NAMES[name])
        if reflect:
            return fn(_as_tensor(other), self)
        return fn(self, other)

    return op


def _bitwise_inplace(name):
    def op(self, other):
        res = _bitwise_fn(name)(self, other)
        with _C_no_grad():
            self.copy_(res)
        return self

    return op


def __invert__(self):
    import tensorplay
    return tensorplay.bitwise_not(self)


def __pos__(self):
    return self


def __abs__(self):
    return self.abs()


Tensor.__floordiv__ = lambda self, other: self.floor_divide(other)
Tensor.__mod__ = lambda self, other: self.remainder(other)
Tensor.__ifloordiv__ = __ifloordiv__
Tensor.__imod__ = __imod__
Tensor.__and__ = _bitwise_fn("and")
Tensor.__rand__ = _bitwise_fn("and", reflect=True)
Tensor.__iand__ = _bitwise_inplace("and")
Tensor.__or__ = _bitwise_fn("or")
Tensor.__ror__ = _bitwise_fn("or", reflect=True)
Tensor.__ior__ = _bitwise_inplace("or")
Tensor.__xor__ = _bitwise_fn("xor")
Tensor.__rxor__ = _bitwise_fn("xor", reflect=True)
Tensor.__ixor__ = _bitwise_inplace("xor")
Tensor.__invert__ = __invert__
Tensor.__lshift__ = _bitwise_fn("lshift")
Tensor.__rlshift__ = _bitwise_fn("lshift", reflect=True)
Tensor.__ilshift__ = _bitwise_inplace("lshift")
Tensor.__rshift__ = _bitwise_fn("rshift")
Tensor.__rrshift__ = _bitwise_fn("rshift", reflect=True)
Tensor.__irshift__ = _bitwise_inplace("rshift")
Tensor.__pos__ = __pos__
Tensor.__abs__ = __abs__


def __complex__(self):
    return builtins_complex(self.item())


def __index__(self):
    return builtins_int(self.item())


Tensor.__complex__ = __complex__
Tensor.__index__ = __index__


# ---------------------------------------------------------------------------
# (cpu/cuda/to/pin_memory/record_stream) is bound natively; these are the
# remaining queries and aliases.
# ---------------------------------------------------------------------------
def _is_cpu(self):
    return str(self.device.type) == "cpu"


def _is_cuda(self):
    return str(self.device.type) == "cuda"


def _is_meta(self):
    dt = getattr(_C.DeviceType, "META", None)
    return dt is not None and str(self.device.type) == str(dt)


Tensor.is_cpu = property(_is_cpu)
Tensor.is_cuda = property(_is_cuda)
if hasattr(_C.DeviceType, "META"):
    Tensor.is_meta = property(_is_meta)
else:
    Tensor.is_meta = property(lambda self: False)
del _is_cpu, _is_cuda


def xpu(self, device=None):
    if device is None:
        return self.to(_C.Device(_C.DeviceType.XPU))
    if builtins_int(device) == device and not isinstance(device, str):
        return self.to(_C.Device(_C.DeviceType.XPU, device))
    parsed = _C.Device(device)
    return self.to(parsed)


Tensor.xpu = xpu
