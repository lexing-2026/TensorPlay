from . import _C

# Alias Tensor to _C.TensorBase so that objects created by C++ (like via tp.tensor())
# are instances of this class.
Tensor = _C.TensorBase

def is_float(self) -> bool:
    """
    Check if tensor is floating point.
    """
    return self.dtype == _C.DType.float32 or self.dtype == _C.DType.float64

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
        device_idx = 0
    elif isinstance(device, int):
        device_idx = device
    else:
        # Assuming device is a tensorplay.device object or similar if passed
        # But for now let's support int index
        device_idx = 0 # Fallback or error?
    
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
            _C.int32: "IntTensor",
            _C.int64: "LongTensor",
            _C.int16: "ShortTensor",
            _C.int8: "CharTensor",
            _C.uint8: "ByteTensor",
            _C.bool: "BoolTensor",
        }
        
        dt = self.dtype
        if dt in dtype_map:
            return f"tensorplay.{device_str}{dtype_map[dt]}"
        return f"tensorplay.{device_str}Tensor"
    
    return self.to(dtype, non_blocking=non_blocking, **kwargs)

Tensor.is_float = is_float
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

