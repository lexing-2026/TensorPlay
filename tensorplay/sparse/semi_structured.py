from collections import namedtuple
import warnings

import tensorplay
from tensorplay import Tensor
from tensorplay import _C

from ._semi_structured_ops import (
    semi_sparse_addmm,
    semi_sparse_clone,
    semi_sparse_detach,
    semi_sparse_detach_,
    semi_sparse_indices,
    semi_sparse_is_same_size,
    semi_sparse_linear,
    semi_sparse_mm,
    semi_sparse_scaled_mm,
    semi_sparse_t,
    semi_sparse_to,
    semi_sparse_to_copy,
    semi_sparse_to_dense,
    semi_sparse_transpose,
    semi_sparse_values,
    semi_sparse_view,
)


_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG",
    "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols",
)


def _native(name):
    try:
        return getattr(_C, name)
    except AttributeError as error:
        raise RuntimeError(
            "semi-structured native operators are unavailable; rebuild the extension"
        ) from error


def _scalar_value(value):
    if isinstance(value, tensorplay.Scalar):
        return _C.scalar_tensor(
            value, dtype=tensorplay.float32, device=tensorplay.device("cpu")
        ).item()
    return value


def _scale(value, scale):
    scale = _scalar_value(scale)
    return value if scale == 1 else value * scale


def _raw_representation(sparse):
    if sparse.packed is not None and sparse.meta is not None:
        return sparse.packed, sparse.meta, False
    if sparse.packed_t is not None and sparse.meta_t is not None:
        return sparse.packed_t, sparse.meta_t, True
    raise RuntimeError("the compressed tensor has no compressed representation")


def _native_left(packed, meta, dense):
    return _native("_sparse_semi_structured_mm")(packed, meta, dense)


def _native_right(dense, packed, meta):
    return _native("_sparse_semi_structured_mm_right")(dense, packed, meta)


class SparseSemiStructuredTensor(Tensor):
    _DEFAULT_ALG_ID = 0
    _DTYPE_SHAPE_CONSTRAINTS = {}
    _FORCE_CUTLASS = False
    _FUSE_TRANSPOSE = False
    BACKEND = ""
    SPARSE_DISPATCH = None

    def __init__(
        self,
        shape,
        packed=None,
        meta=None,
        packed_t=None,
        meta_t=None,
        compressed_swizzled_bitmask=None,
        fuse_transpose_cusparselt=False,
        alg_id_cusparselt=0,
        requires_grad=False,
        source=None,
        source_transposed=False,
    ):
        self.__class__._load_dispatch_table()
        previous = packed if packed is not None else packed_t
        if previous is None:
            raise ValueError("at least one compressed tensor is required")
        super().__init__(previous)
        self._logical_shape = tuple(int(value) for value in shape)
        self.packed = packed
        self.meta = meta
        self.packed_t = packed_t
        self.meta_t = meta_t
        self.compressed_swizzled_bitmask = compressed_swizzled_bitmask
        self.fuse_transpose_cusparselt = bool(fuse_transpose_cusparselt)
        self.alg_id_cusparselt = int(alg_id_cusparselt)
        self._source = source
        self._source_transposed = bool(source_transposed)
        if requires_grad:
            self.requires_grad = True

    @classmethod
    def __tensorplay_dispatch__(cls, func, types, args, kwargs=None):
        op_name = getattr(func, "__name__", func)
        if not isinstance(op_name, str):
            op_name = getattr(func, "name", str(func))
        dispatch = getattr(cls, "SPARSE_DISPATCH", None)
        if dispatch is None or op_name not in dispatch:
            raise NotImplementedError(
                f"{cls.__name__} does not implement operation {op_name}"
            )
        return dispatch[op_name](func, types, args, kwargs or {})

    @classmethod
    def _load_dispatch_table(cls, custom_dispatch_table=None):
        if getattr(cls, "SPARSE_DISPATCH", None) is None:
            cls.SPARSE_DISPATCH = {
                "values": semi_sparse_values,
                "indices": semi_sparse_indices,
                "is_same_size": semi_sparse_is_same_size,
                "detach_": semi_sparse_detach_,
                "detach": semi_sparse_detach,
                "t": semi_sparse_t,
                "transpose": semi_sparse_transpose,
                "view": semi_sparse_view,
                "reshape": semi_sparse_view,
                "mm": semi_sparse_mm,
                "matmul": semi_sparse_mm,
                "addmm": semi_sparse_addmm,
                "linear": semi_sparse_linear,
                "_to_copy": semi_sparse_to_copy,
                "_scaled_mm": semi_sparse_scaled_mm,
                "clone": semi_sparse_clone,
                "to": semi_sparse_to,
                "to_dense": semi_sparse_to_dense,
            }
            if custom_dispatch_table is not None:
                cls.SPARSE_DISPATCH.update(custom_dispatch_table)

    def __tensor_flatten__(self):
        names = [
            name
            for name in (
                "packed",
                "meta",
                "packed_t",
                "meta_t",
                "compressed_swizzled_bitmask",
            )
            if getattr(self, name) is not None
        ]
        return names, (
            self.shape,
            self.fuse_transpose_cusparselt,
            self.alg_id_cusparselt,
            self.requires_grad,
        )

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, tensor_meta, outer_size=None, outer_stride=None
    ):
        del outer_stride
        shape, fuse_transpose, alg_id, requires_grad = tensor_meta
        if outer_size is not None:
            shape = outer_size
        return cls(
            shape=shape,
            packed=inner_tensors.get("packed"),
            meta=inner_tensors.get("meta"),
            packed_t=inner_tensors.get("packed_t"),
            meta_t=inner_tensors.get("meta_t"),
            compressed_swizzled_bitmask=inner_tensors.get(
                "compressed_swizzled_bitmask"
            ),
            fuse_transpose_cusparselt=fuse_transpose,
            alg_id_cusparselt=alg_id,
            requires_grad=requires_grad,
        )

    @property
    def shape(self):
        return self._logical_shape

    @property
    def dtype(self):
        return self.packed.dtype if self.packed is not None else self.packed_t.dtype

    @property
    def device(self):
        return self.packed.device if self.packed is not None else self.packed_t.device

    @property
    def ndim(self):
        return len(self._logical_shape)

    @property
    def is_cuda(self):
        return self.device.is_cuda()

    @property
    def is_sparse(self):
        return False

    @property
    def layout(self):
        return 2

    @property
    def strides(self):
        if len(self.shape) != 2:
            return tuple(reversed(self.shape))
        return (self.shape[1], 1)

    def dim(self):
        return len(self.shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def stride(self, dim=None):
        if dim is None:
            return self.strides
        return self.strides[dim]

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def is_contiguous(self, *args, **kwargs):
        return True

    def is_sparse_csr(self):
        return False

    def values(self):
        if self.packed is None:
            raise RuntimeError("the logical transpose has no packed representation")
        if self.meta is not None:
            return self.packed.detach()
        kept = self.shape[0] * self.shape[1] // 2
        return self.packed.view(-1)[:kept].view(self.shape[0], -1)

    def indices(self):
        if self.packed is None:
            raise RuntimeError("the logical transpose has no metadata representation")
        if self.meta is not None:
            return self.meta
        kept = self.shape[0] * self.shape[1] // 2
        metadata = self.packed.view(-1)[kept:]
        metadata_dtype = (
            tensorplay.int32 if self.dtype == tensorplay.int8 else tensorplay.int16
        )
        return metadata.view(metadata_dtype).view(self.shape[0], -1)

    _values = values
    _indices = indices

    def to_dense(self):
        packed, meta, transposed = _raw_representation(self)
        dense = _native("_sparse_semi_structured_to_dense")(packed, meta)
        return dense.transpose(0, 1).contiguous() if transposed else dense

    def _mm(self, dense, *, bias=None, should_transpose_dense=False, **kwargs):
        if isinstance(dense, SparseSemiStructuredTensor):
            raise ValueError("compressed matrix multiplication needs a dense second operand")
        if self.ndim != 2 or dense.dim() != 2:
            raise NotImplementedError(
                "semi-structured matrix multiplication only supports 2-D operands"
            )
        if dense.dtype != self.dtype:
            raise TypeError("compressed and dense operands must have the same dtype")
        alpha = kwargs.pop("alpha", 1)
        beta = kwargs.pop("beta", 1)
        if kwargs:
            raise TypeError("unexpected compressed matmul arguments")
        if should_transpose_dense:
            packed, meta, transposed = _raw_representation(self)
            if transposed:
                result = _native_right(dense, packed, meta)
            else:
                result = _native_left(
                    packed, meta, dense.transpose(0, 1).contiguous()
                ).transpose(0, 1).contiguous()
            if bias is not None:
                result = _scale(result, alpha) + _scale(bias, beta)
            return result
        packed, meta, transposed = _raw_representation(self)
        if transposed:
            result = _native_right(
                dense.transpose(0, 1).contiguous(), packed, meta
            ).transpose(0, 1).contiguous()
            if bias is not None:
                result = _scale(result, alpha) + _scale(bias, beta)
            return result
        if bias is not None:
            return _native("_sparse_semi_structured_addmm")(
                bias,
                packed,
                meta,
                dense,
                alpha=tensorplay.Scalar(_scalar_value(alpha)),
                beta=tensorplay.Scalar(_scalar_value(beta)),
            )
        return _native_left(packed, meta, dense)

    def __matmul__(self, other):
        return tensorplay.mm(self, other)

    def __rmatmul__(self, other, out_dtype=None):
        if isinstance(other, SparseSemiStructuredTensor):
            raise ValueError("compressed matrix multiplication needs a dense left operand")
        if other.dim() != 2 or self.ndim != 2:
            raise NotImplementedError(
                "semi-structured matrix multiplication only supports 2-D operands"
            )
        if other.dtype != self.dtype:
            raise TypeError("compressed and dense operands must have the same dtype")
        if out_dtype is None:
            return tensorplay.mm(other, self)
        packed, meta, transposed = _raw_representation(self)
        if transposed:
            return _native("_sparse_semi_structured_mm")(
                packed,
                meta,
                other.transpose(0, 1).contiguous(),
                out_dtype=out_dtype,
            ).transpose(0, 1).contiguous()
        return _native("_sparse_semi_structured_mm_right")(
            other, packed, meta, out_dtype=out_dtype
        )

    def mm(self, other):
        return tensorplay.mm(self, other)

    def matmul(self, other):
        return tensorplay.matmul(self, other)

    def t(self):
        if len(self.shape) != 2:
            raise RuntimeError("t() expects a 2-D compressed tensor")
        return self.__class__(
            (self.shape[1], self.shape[0]),
            packed=self.packed_t,
            meta=self.meta_t,
            packed_t=self.packed,
            meta_t=self.meta,
            compressed_swizzled_bitmask=(
                self.compressed_swizzled_bitmask.transpose(0, 1)
                if self.compressed_swizzled_bitmask is not None
                else None
            ),
            fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
            alg_id_cusparselt=self.alg_id_cusparselt,
            requires_grad=self.requires_grad,
            source=self._source,
            source_transposed=not self._source_transposed,
        )

    def transpose(self, dim0=0, dim1=1):
        dim0 %= self.ndim
        dim1 %= self.ndim
        if dim0 == dim1:
            return self
        if self.ndim != 2 or {dim0, dim1} != {0, 1}:
            raise NotImplementedError("only a 2-D transpose is supported")
        return self.t()

    @property
    def T(self):
        return self.t()

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(value) for value in shape)
        if shape != self.shape:
            raise NotImplementedError("view is only available when the logical shape is unchanged")
        return self

    reshape = view

    def contiguous(self, *args, **kwargs):
        return self

    def clone(self, *args, **kwargs):
        return self.__class__(
            self.shape,
            packed=None if self.packed is None else self.packed.clone(),
            meta=None if self.meta is None else self.meta.clone(),
            packed_t=None if self.packed_t is None else self.packed_t.clone(),
            meta_t=None if self.meta_t is None else self.meta_t.clone(),
            compressed_swizzled_bitmask=(
                None
                if self.compressed_swizzled_bitmask is None
                else self.compressed_swizzled_bitmask.clone()
            ),
            fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
            alg_id_cusparselt=self.alg_id_cusparselt,
            requires_grad=self.requires_grad,
            source=None,
            source_transposed=False,
        )

    def detach(self):
        return self.__class__(
            self.shape,
            packed=None if self.packed is None else self.packed.detach(),
            meta=None if self.meta is None else self.meta.detach(),
            packed_t=None if self.packed_t is None else self.packed_t.detach(),
            meta_t=None if self.meta_t is None else self.meta_t.detach(),
            compressed_swizzled_bitmask=(
                None
                if self.compressed_swizzled_bitmask is None
                else self.compressed_swizzled_bitmask.detach()
            ),
            fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
            alg_id_cusparselt=self.alg_id_cusparselt,
            requires_grad=False,
            source=None,
            source_transposed=False,
        )

    def detach_(self):
        self.requires_grad = False
        return self

    def to(self, *args, **kwargs):
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        non_blocking = kwargs.pop("non_blocking", False)
        copy = kwargs.pop("copy", False)
        if kwargs:
            raise TypeError("unexpected arguments for compressed tensor conversion")
        if len(args) > 2:
            raise TypeError("too many arguments for compressed tensor conversion")
        if args:
            target = args[0]
            if isinstance(target, _C.DType):
                dtype = target
            elif isinstance(target, _C.Device):
                device = target
            elif isinstance(target, str):
                device = _C.Device(target)
            elif isinstance(target, Tensor):
                device = target.device
                dtype = target.dtype
            else:
                raise TypeError("invalid conversion target")
        if len(args) == 2:
            if not isinstance(args[1], _C.DType):
                raise TypeError("the second conversion argument must be a dtype")
            dtype = args[1]
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        if target_device == self.device and target_dtype == self.dtype:
            return self.clone() if copy else self
        dense = self.to_dense().to(
            target_device,
            target_dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        return self.__class__.from_dense(dense, alg_id=self.alg_id_cusparselt)

    def _linear(self, input, bias=None):
        if input.dim() == 0:
            raise RuntimeError("linear expects an input with at least one dimension")
        input_shape = tuple(input.shape)
        input_2d = input.view(-1, input_shape[-1])
        result = self._mm(input_2d, bias=bias, should_transpose_dense=True)
        return result.view(input_shape[:-1] + (self.shape[0],))

    @classmethod
    def _validate_device_dim_dtype_shape(cls, original_tensor):
        if not isinstance(original_tensor, Tensor):
            raise TypeError("compression expects a tensor")
        if original_tensor.dim() != 2:
            raise RuntimeError(
                f"compressed tensors require a 2-D input, got {original_tensor.dim()} dimensions"
            )
        if not original_tensor.is_contiguous():
            raise RuntimeError("compressed tensors require contiguous input")
        if original_tensor.dtype not in cls._DTYPE_SHAPE_CONSTRAINTS:
            raise RuntimeError(
                f"dtype {original_tensor.dtype} is not supported by {cls.__name__}"
            )
        group_size = 2 if original_tensor.dtype == tensorplay.float32 else 4
        if original_tensor.size(1) % group_size:
            raise RuntimeError("the input column dimension must be group-aligned")
        constraints = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype]
        if original_tensor.is_cuda:
            rows, cols = original_tensor.shape
            if (
                rows < constraints.sparse_min_rows
                or rows % constraints.sparse_min_rows
                or cols < constraints.sparse_min_cols
                or cols % constraints.sparse_min_cols
            ):
                raise RuntimeError(
                    f"compressed tensor shape {original_tensor.shape} is not supported"
                )

    @classmethod
    def from_dense(cls, original_tensor, alg_id=_DEFAULT_ALG_ID):
        cls._validate_device_dim_dtype_shape(original_tensor)
        packed, meta = _native("_to_sparse_semi_structured")(original_tensor)
        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=None,
            meta_t=None,
            fuse_transpose_cusparselt=cls._FUSE_TRANSPOSE,
            alg_id_cusparselt=alg_id,
            requires_grad=original_tensor.requires_grad,
            source=(original_tensor if original_tensor.requires_grad else None),
            source_transposed=False,
        )

    @classmethod
    def prune_dense_static_sort(cls, original_tensor, algorithm=""):
        return cls.from_dense(original_tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"


class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    BACKEND = "cutlass"
    _DTYPE_SHAPE_CONSTRAINTS = {
        tensorplay.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 128, 16, 16),
        tensorplay.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        tensorplay.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        tensorplay.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4),
    }


class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    BACKEND = "cusparselt"
    _DTYPE_SHAPE_CONSTRAINTS = {
        tensorplay.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        tensorplay.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        tensorplay.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
    }


def to_sparse_semi_structured(
    original_tensor, transposed=False, alg_id=SparseSemiStructuredTensor._DEFAULT_ALG_ID
):
    if transposed:
        warnings.warn(
            "the transposed argument is deprecated; pass contiguous input instead",
            FutureWarning,
            stacklevel=2,
        )
    sparse_class = (
        SparseSemiStructuredTensorCUTLASS
        if SparseSemiStructuredTensor._FORCE_CUTLASS
        else SparseSemiStructuredTensorCUSPARSELT
    )
    if original_tensor.dtype == tensorplay.float32 and sparse_class is SparseSemiStructuredTensorCUSPARSELT:
        sparse_class = SparseSemiStructuredTensorCUTLASS
    return sparse_class.from_dense(original_tensor, alg_id=alg_id)


__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
]
