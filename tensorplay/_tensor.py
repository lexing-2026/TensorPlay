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

def flatten(self, start_dim=0, end_dim=-1):
    """
    Flattens a contiguous range of dims.
    """
    input_dim = self.dim()
    if start_dim < 0:
        start_dim += input_dim
    if end_dim < 0:
        end_dim += input_dim
        
    if start_dim < 0 or start_dim >= input_dim:
         raise IndexError(f"Dimension out of range (expected to be in range of [{0}, {input_dim-1}], but got {start_dim})")
    if end_dim < 0 or end_dim >= input_dim:
         raise IndexError(f"Dimension out of range (expected to be in range of [{0}, {input_dim-1}], but got {end_dim})")
    
    if start_dim > end_dim:
        return self

    new_shape = []
    for i in range(start_dim):
        new_shape.append(self.size(i))
        
    flattened_size = 1
    for i in range(start_dim, end_dim + 1):
        flattened_size *= self.size(i)
    new_shape.append(flattened_size)
    
    for i in range(end_dim + 1, input_dim):
        new_shape.append(self.size(i))
        
    return self.reshape(new_shape)

def unflatten(self, dim, sizes):
    """
    Expands a dimension of the input tensor over multiple dimensions.
    """
    input_dim = self.dim()
    if dim < 0:
        dim += input_dim
        
    if dim < 0 or dim >= input_dim:
         raise IndexError(f"Dimension out of range (expected to be in range of [{0}, {input_dim-1}], but got {dim})")
    
    current_size = self.size(dim)
    
    # Calculate product of explicit sizes and handle -1
    product = 1
    infer_idx = -1
    for i, s in enumerate(sizes):
        if s == -1:
            if infer_idx >= 0:
                raise RuntimeError("unflatten: only one dimension can be inferred (-1)")
            infer_idx = i
        else:
            product *= s
            
    if infer_idx >= 0:
        if current_size % product != 0:
             raise RuntimeError(f"unflatten: provided sizes {sizes} don't match the size of dimension {dim} ({current_size})")
        sizes = list(sizes)
        sizes[infer_idx] = current_size // product
    else:
        if product != current_size:
            raise RuntimeError(f"unflatten: provided sizes {sizes} don't match the size of dimension {dim} ({current_size})")
            
    new_shape = []
    for i in range(dim):
        new_shape.append(self.size(i))
        
    new_shape.extend(sizes)
    
    for i in range(dim + 1, input_dim):
        new_shape.append(self.size(i))
        
    return self.reshape(new_shape)

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
Tensor.flatten = flatten
Tensor.unflatten = unflatten
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
    stepping by :attr:`step` (torch's ``Tensor.unfold``).

    Port of ``aten/src/ATen/native/TensorShape.cpp``: the view appends a new
    trailing dimension of length ``size`` and re-strides ``dimension`` by
    ``step``.
    """
    sizes = list(self.shape)
    strides = list(self.strides)
    if dimension < 0:
        dimension += len(sizes)
    if dimension < 0 or dimension >= len(sizes):
        raise IndexError(f"Dimension out of range (expected to be in range of [0, {len(sizes) - 1}], but got {dimension})")
    if sizes[dimension] < size:
        raise ValueError(f"maximum size for tensor at dimension {dimension} is {sizes[dimension]} but size is {size}")
    sizes[dimension] = (sizes[dimension] - size) // step + 1
    # torch appends the ORIGINAL stride of `dimension`, then scales it by step
    strides.append(strides[dimension])
    sizes.append(size)
    strides[dimension] *= step
    return self.as_strided(sizes, strides)


Tensor.unfold = unfold


def register_hook(self, hook):
    """Registers a backward hook (torch's ``Tensor.register_hook``).

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
