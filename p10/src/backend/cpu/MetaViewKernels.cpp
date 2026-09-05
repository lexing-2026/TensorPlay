// Dispatcher kernels for view/alias metadata ops, detach, scalar reads,
// storage offsets, memory pinning and sparse layout component accessors.
//
// Every result here is built directly from TensorImpl primitives (shared
// storage handles, sizes/strides metadata, sparse component handles).  These
// kernels must never re-enter the dispatcher for their own op name: a
// generated Tensor method resolves through the dispatcher, so a kernel that
// called the same-named member would recurse without bound.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "SizesAndStrides.h"
#include "Storage.h"
#include "Allocator.h"
#include "SparseKernels.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cpu {

// -----------------------------------------------------------------------------
// as_strided / as_strided_
// -----------------------------------------------------------------------------

Tensor as_strided_cpu(const Tensor& self,
                      const std::vector<int64_t>& size,
                      const std::vector<int64_t>& stride,
                      std::optional<int64_t> storage_offset) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (size.size() != stride.size()) {
        TP_THROW(ValueError,
                 "as_strided(): sizes and strides must have the same length");
    }
    for (int64_t value : size) {
        if (value < 0) {
            TP_THROW(ValueError, "as_strided(): sizes must be non-negative");
        }
    }
    const int64_t offset =
        storage_offset.value_or(static_cast<int64_t>(impl->storage_offset()));
    if (offset < 0) {
        TP_THROW(ValueError,
                 "as_strided(): storage_offset must be non-negative");
    }
    Tensor out(impl->storage(), size, stride, impl->dtype(),
               static_cast<size_t>(offset));
    out.unsafeGetTensorImpl()->share_version_counter(*impl);
    // as_strided underlies every view op (transpose/squeeze/unsqueeze/
    // permute/reshape...): a quantized source's quantizer rides along since
    // the view aliases the same codes and mapping.
    if (impl->has_quantizer()) {
        out.unsafeGetTensorImpl()->set_quantizer(impl->quantizer());
    }
    return out;
}

Tensor& as_strided__cpu(Tensor& self,
                        const std::vector<int64_t>& size,
                        const std::vector<int64_t>& stride,
                        std::optional<int64_t> storage_offset) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (size.size() != stride.size()) {
        TP_THROW(ValueError,
                 "as_strided(): sizes and strides must have the same length");
    }
    for (int64_t value : size) {
        if (value < 0) {
            TP_THROW(ValueError, "as_strided(): sizes must be non-negative");
        }
    }
    const int64_t offset =
        storage_offset.value_or(static_cast<int64_t>(impl->storage_offset()));
    if (offset < 0) {
        TP_THROW(ValueError,
                 "as_strided(): storage_offset must be non-negative");
    }
    impl->set_sizes_and_strides(size, stride);
    impl->set_storage_offset(static_cast<size_t>(offset));
    return self;
}

// -----------------------------------------------------------------------------
// detach
// -----------------------------------------------------------------------------

// Copies the metadata while sharing the storage and the version counter; the
// metadata copy starts without autograd metadata, so the result is outside
// any recorded graph yet still aliases the original memory.
Tensor detach_cpu(const Tensor& self) {
    if (!self.defined()) {
        return Tensor();
    }
    return Tensor(std::make_shared<TensorImpl>(*self.unsafeGetTensorImpl()));
}

// -----------------------------------------------------------------------------
// view / view.dtype
// -----------------------------------------------------------------------------

Tensor view_cpu(const Tensor& self, const std::vector<int64_t>& shape) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    const std::vector<int64_t> inferred =
        SizesAndStrides::infer_size(shape, impl->numel());
    auto stride = SizesAndStrides::compute_view_strides(
        impl->sizes().vec(), impl->strides().vec(), inferred);
    if (!stride.has_value()) {
        TP_THROW(RuntimeError,
                 "view size is not compatible with input tensor's size and stride");
    }
    Tensor out(impl->storage(), inferred, *stride, impl->dtype(),
               impl->storage_offset());
    out.unsafeGetTensorImpl()->share_version_counter(*impl);
    return out;
}

// Reinterprets the element stream as `dtype` while aliasing the same storage.
// Same-size dtypes keep shape/strides; otherwise only the last dimension may
// change and the storage offset is rescaled between element units.
Tensor view_dtype_cpu(const Tensor& self, DType dtype) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    const DType self_dtype = impl->dtype();
    if (dtype == self_dtype) {
        Tensor out(impl->storage(), impl->sizes().vec(), impl->strides().vec(),
                   dtype, impl->storage_offset());
        out.unsafeGetTensorImpl()->share_version_counter(*impl);
        return out;
    }

    const std::vector<int64_t> self_sizes = impl->sizes().vec();
    const std::vector<int64_t> self_strides = impl->strides().vec();
    const size_t src_esize = elementSize(self_dtype);
    const size_t dst_esize = elementSize(dtype);

    if (self_strides.empty()) {
        TP_THROW(RuntimeError,
                 "view(): cannot reinterpret a 0-dim tensor to a dtype of a different element size");
    }
    if (self_strides.back() != 1 && src_esize != dst_esize) {
        TP_THROW(RuntimeError,
                 "view(): view(dtype) requires the last dimension to be contiguous when "
                 "element sizes differ");
    }

    std::vector<int64_t> new_sizes = self_sizes;
    std::vector<int64_t> new_strides = self_strides;
    size_t new_offset = impl->storage_offset();

    if (dst_esize < src_esize) {
        const int64_t ratio = static_cast<int64_t>(src_esize / dst_esize);
        new_sizes.back() *= ratio;
        for (size_t i = 0; i + 1 < new_strides.size(); ++i) new_strides[i] *= ratio;
        new_offset = impl->storage_offset() * static_cast<size_t>(ratio);
    } else if (dst_esize > src_esize) {
        const int64_t ratio = static_cast<int64_t>(dst_esize / src_esize);
        if (new_sizes.back() % ratio != 0) {
            TP_THROW(RuntimeError,
                     "view(): the last dimension must be divisible by the element size ratio");
        }
        for (size_t i = 0; i + 1 < new_strides.size(); ++i) {
            if (new_strides[i] % ratio != 0) {
                TP_THROW(RuntimeError,
                         "view(): strides must be divisible by the element size ratio");
            }
            new_strides[i] /= ratio;
        }
        new_sizes.back() /= ratio;
        if ((impl->storage_offset() * static_cast<int64_t>(src_esize)) %
                static_cast<int64_t>(dst_esize) != 0) {
            TP_THROW(RuntimeError,
                     "view(): storage offset is not aligned to the target element size");
        }
        new_offset = impl->storage_offset() / static_cast<size_t>(ratio);
    }

    Tensor out(impl->storage(), new_sizes, new_strides, dtype, new_offset);
    out.unsafeGetTensorImpl()->share_version_counter(*impl);
    return out;
}

// -----------------------------------------------------------------------------
// item / storage_offset
// -----------------------------------------------------------------------------

Scalar item_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (impl->is_sparse()) {
        TP_THROW(RuntimeError, "item() is not supported for sparse tensors");
    }
    if (impl->numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }
    if (!impl->device().is_cpu()) {
        TP_THROW(RuntimeError, "item(): expected a CPU tensor but got ",
                 impl->device().toString());
    }

    switch (impl->dtype()) {
        case DType::Float32: return Scalar(static_cast<double>(*impl->data<float>()));
        case DType::Float64: return Scalar(*impl->data<double>());
        case DType::Float16: return Scalar(static_cast<float>(*impl->data<Half>()));
        case DType::BFloat16: return Scalar(static_cast<float>(*impl->data<BFloat16>()));
        case DType::Float8_e4m3fn: return Scalar(static_cast<float>(*impl->data<Float8_e4m3fn>()));
        case DType::Float8_e5m2: return Scalar(static_cast<float>(*impl->data<Float8_e5m2>()));
        case DType::Float8_e4m3fnuz: return Scalar(static_cast<float>(*impl->data<Float8_e4m3fnuz>()));
        case DType::Float8_e5m2fnuz: return Scalar(static_cast<float>(*impl->data<Float8_e5m2fnuz>()));
        case DType::Float8_e8m0fnu: return Scalar(static_cast<float>(*impl->data<Float8_e8m0fnu>()));
        case DType::Int8: return Scalar(static_cast<int64_t>(*impl->data<int8_t>()));
        case DType::Int16: return Scalar(static_cast<int64_t>(*impl->data<int16_t>()));
        case DType::Int32: return Scalar(static_cast<int64_t>(*impl->data<int32_t>()));
        case DType::Int64: return Scalar(*impl->data<int64_t>());
        case DType::UInt8: return Scalar(static_cast<uint64_t>(*impl->data<uint8_t>()));
        case DType::UInt16: return Scalar(static_cast<uint64_t>(*impl->data<uint16_t>()));
        case DType::UInt32: return Scalar(static_cast<uint64_t>(*impl->data<uint32_t>()));
        case DType::UInt64: return Scalar(*impl->data<uint64_t>());
        case DType::Bool: return Scalar(static_cast<bool>(*impl->data<bool>()));
        case DType::ComplexHalf: {
            const auto value = *impl->data<std::complex<Half>>();
            return Scalar(std::complex<float>(static_cast<float>(value.real()),
                                              static_cast<float>(value.imag())));
        }
        case DType::ComplexFloat: return Scalar(*impl->data<std::complex<float>>());
        case DType::ComplexDouble: return Scalar(*impl->data<std::complex<double>>());
        case DType::BComplex32: {
            const auto value = *impl->data<std::complex<BFloat16>>();
            return Scalar(std::complex<float>(static_cast<float>(value.real()),
                                              static_cast<float>(value.imag())));
        }
        default:
            TP_THROW(NotImplementedError, "item() not implemented for this dtype");
    }
}

int64_t storage_offset_cpu(const Tensor& self) {
    if (!self.defined()) {
        return 0;
    }
    return static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
}

// -----------------------------------------------------------------------------
// is_pinned / pin_memory
// -----------------------------------------------------------------------------

bool is_pinned_cpu(const Tensor& self, std::optional<Device> device) {
    TP_CHECK(!device.has_value() || device->is_cpu(),
             "is_pinned(): expected the device to be CPU but got ",
             device.has_value() ? device->toString() : std::string("(none)"));
    if (!self.defined()) {
        return false;
    }
#ifdef USE_CUDA
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    return impl->device().is_cpu() && impl->has_storage() &&
           impl->storage().allocator() == getPinnedMemoryAllocator();
#else
    return false;
#endif
}

Tensor pin_memory_cpu(const Tensor& self, std::optional<Device> device) {
    TP_CHECK(!device.has_value() || device->is_cpu(),
             "pin_memory(): expected the device to be CPU but got ",
             device.has_value() ? device->toString() : std::string("(none)"));
    if (!self.defined()) {
        return Tensor();
    }
    if (!self.device().is_cpu()) {
        TP_THROW(RuntimeError, "cannot pin a tensor on " + self.device().toString() +
                               "; only dense CPU tensors can be pinned");
    }
#ifdef USE_CUDA
    if (is_pinned_cpu(self, std::nullopt)) {
        return self;
    }
    const auto sizes = self.unsafeGetTensorImpl()->sizes().vec();
    const size_t nbytes =
        static_cast<size_t>(self.unsafeGetTensorImpl()->numel()) *
        elementSize(self.unsafeGetTensorImpl()->dtype());
    Storage storage(nbytes, getPinnedMemoryAllocator(), Device(DeviceType::CPU));
    Tensor result(storage, sizes, self.dtype());
    result.copy_(self);
    return result;
#else
    TP_THROW(RuntimeError, "pin_memory requires a CUDA-enabled TensorPlay build");
#endif
}

// -----------------------------------------------------------------------------
// Sparse layout accessors
// -----------------------------------------------------------------------------

Tensor coalesce_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "coalesce() is only defined for sparse COO tensors");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (!impl->is_sparse() ||
        impl->sparse_layout() == TensorImpl::kSparseCSRLayout) {
        TP_THROW(RuntimeError,
                 "coalesce() is only defined for sparse COO tensors");
    }
    if (impl->is_coalesced()) {
        return self;
    }
    return coalesce_sparse_cpu(self);
}

Tensor _coalesce_cpu(const Tensor& self) {
    return coalesce_cpu(self);
}

bool is_coalesced_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "is_coalesced expected sparse coordinate tensor layout");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (!impl->is_sparse() ||
        impl->sparse_layout() == TensorImpl::kSparseCSRLayout) {
        TP_THROW(RuntimeError,
                 "is_coalesced expected sparse coordinate tensor layout");
    }
    return impl->is_coalesced();
}

int64_t sparse_dim_cpu(const Tensor& self) {
    if (!self.defined() || !self.unsafeGetTensorImpl()->is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_dim expected sparse tensor layout but got ",
                 self.defined() ? self.toString() : std::string("undefined"),
                 " tensor");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (impl->sparse_layout() == TensorImpl::kSparseCSRLayout) {
        // Row-compressed layouts span exactly two sparse dimensions
        // (compressed + plain).
        return 2;
    }
    auto indices = impl->sparse_indices_impl();
    if (!indices || indices->dim() == 0) {
        return 0;
    }
    return indices->size(0);
}

int64_t dense_dim_cpu(const Tensor& self) {
    std::shared_ptr<TensorImpl> impl =
        self.defined() ? self.unsafeGetTensorImpl() : nullptr;
    if (!impl || !impl->is_sparse()) {
        TP_THROW(RuntimeError,
                 "dense_dim expected sparse tensor layout but got ",
                 self.defined() ? self.toString() : std::string("undefined"),
                 " tensor");
    }
    auto values = impl->sparse_values_impl();
    if (!values || values->dim() == 0) {
        return 0;
    }
    return values->dim() - 1;
}

Tensor _indices_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "_indices() is only defined for sparse COO tensors");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    auto indices = impl->sparse_indices_impl();
    if (!indices) {
        TP_THROW(RuntimeError,
                 "_indices() is only defined for sparse COO tensors");
    }
    return Tensor(std::move(indices));
}

Tensor _values_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "_values() is only defined for sparse tensors");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    auto values = impl->sparse_values_impl();
    if (!values) {
        TP_THROW(RuntimeError,
                 "_values() is only defined for sparse tensors");
    }
    return Tensor(std::move(values));
}

Tensor crow_indices_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "crow_indices expected sparse row compressed tensor layout");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    auto crow = impl->sparse_crow_impl();
    if (!crow) {
        TP_THROW(RuntimeError,
                 "crow_indices expected sparse row compressed tensor layout");
    }
    return Tensor(std::move(crow));
}

Tensor col_indices_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "col_indices expected sparse row compressed tensor layout");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    auto col = impl->sparse_col_impl();
    if (!col) {
        TP_THROW(RuntimeError,
                 "col_indices expected sparse row compressed tensor layout");
    }
    return Tensor(std::move(col));
}

bool is_set_to_cpu(const Tensor& self, const Tensor& other) {
    const auto self_impl = self.unsafeGetTensorImpl();
    const auto other_impl = other.unsafeGetTensorImpl();
    if (!self_impl || !other_impl || !self_impl->has_storage() ||
        !other_impl->has_storage()) {
        return false;
    }
    if (!self_impl->storage().is_same(other_impl->storage()) ||
        self_impl->storage_offset() != other_impl->storage_offset() ||
        self.dim() != other.dim()) {
        return false;
    }
    for (int64_t dim = 0; dim < self.dim(); ++dim) {
        if (self.size(dim) != other.size(dim) ||
            self.stride(dim) != other.stride(dim)) {
            return false;
        }
    }
    return true;
}

TENSORPLAY_LIBRARY_IMPL(CPU, MetaViewOps) {
    m.impl("as_strided", as_strided_cpu);
    m.impl("as_strided_", as_strided__cpu);
    m.impl("detach", detach_cpu);
    m.impl("view", view_cpu);
    m.impl("view.dtype", view_dtype_cpu);
    m.impl("item", item_cpu);
    m.impl("storage_offset", storage_offset_cpu);
    m.impl("is_set_to", is_set_to_cpu);
    m.impl("is_pinned", is_pinned_cpu);
    m.impl("pin_memory", pin_memory_cpu);
    m.impl("coalesce", coalesce_cpu);
    m.impl("_coalesce", _coalesce_cpu);
    m.impl("is_coalesced", is_coalesced_cpu);
    m.impl("sparse_dim", sparse_dim_cpu);
    m.impl("dense_dim", dense_dim_cpu);
    m.impl("_indices", _indices_cpu);
    m.impl("_values", _values_cpu);
    m.impl("crow_indices", crow_indices_cpu);
    m.impl("col_indices", col_indices_cpu);
}

} // namespace cpu
} // namespace tensorplay
