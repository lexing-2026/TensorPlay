// Backend-neutral kernels for view/alias metadata, detach, storage offsets,
// memory pinning and the sparse layout component accessors.
//
// Every result here is derived from TensorImpl primitives alone -- shared
// storage handles, sizes/strides metadata, sparse component handles -- so no
// element is ever read or written and the same code answers for a tensor on
// any device.  Registering them once on the backend-neutral key keeps a
// single definition serving CPU, GPU and Vulkan tensors alike; a backend that
// needs different behaviour (a texture-backed as_strided, say) overrides the
// entry with its own registration.
//
// These kernels must never re-enter the dispatcher for their own op name: a
// generated Tensor method resolves through the dispatcher, so a kernel that
// called the same-named member would recurse without bound.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "SizesAndStrides.h"
#include "Storage.h"
#include "Allocator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace composite {

// -----------------------------------------------------------------------------
// as_strided / as_strided_
// -----------------------------------------------------------------------------

Tensor as_strided_kernel(const Tensor& self,
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

Tensor& as_strided__kernel(Tensor& self,
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
Tensor detach_kernel(const Tensor& self) {
    if (!self.defined()) {
        return Tensor();
    }
    return Tensor(std::make_shared<TensorImpl>(*self.unsafeGetTensorImpl()));
}

// -----------------------------------------------------------------------------
// view / view.dtype
// -----------------------------------------------------------------------------

Tensor view_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
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
Tensor view_dtype_kernel(const Tensor& self, DType dtype) {
    return self.view_dtype(dtype);
}

// -----------------------------------------------------------------------------
// storage_offset / is_set_to
// -----------------------------------------------------------------------------

int64_t storage_offset_kernel(const Tensor& self) {
    if (!self.defined()) {
        return 0;
    }
    return static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
}

bool is_set_to_kernel(const Tensor& self, const Tensor& other) {
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

// -----------------------------------------------------------------------------
// is_pinned / pin_memory
// -----------------------------------------------------------------------------

// Pinning is a host-allocator property: only a host tensor can carry it, so a
// tensor living on an accelerator answers false without inspecting storage.
bool is_pinned_kernel(const Tensor& self, std::optional<Device> device) {
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

Tensor pin_memory_kernel(const Tensor& self, std::optional<Device> device) {
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
    if (is_pinned_kernel(self, std::nullopt)) {
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

bool is_coalesced_kernel(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "is_coalesced expected sparse coordinate tensor layout");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (!impl->is_sparse() || impl->is_sparse_compressed()) {
        TP_THROW(RuntimeError,
                 "is_coalesced expected sparse coordinate tensor layout");
    }
    return impl->is_coalesced();
}

int64_t sparse_dim_kernel(const Tensor& self) {
    if (!self.defined() || !self.unsafeGetTensorImpl()->is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_dim expected sparse tensor layout but got ",
                 self.defined() ? self.toString() : std::string("undefined"),
                 " tensor");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (impl->is_sparse_compressed()) {
        // Compressed layouts span exactly two sparse dimensions
        // (compressed + plain).
        return 2;
    }
    auto indices = impl->sparse_indices_impl();
    if (!indices || indices->dim() == 0) {
        return 0;
    }
    return indices->size(0);
}

int64_t dense_dim_kernel(const Tensor& self) {
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

Tensor indices_kernel(const Tensor& self) {
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

Tensor values_kernel(const Tensor& self) {
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

Tensor crow_indices_kernel(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "crow_indices expected sparse row compressed tensor layout");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (!impl->is_sparse_compressed()) {
        TP_THROW(RuntimeError,
                 "crow_indices expected sparse row compressed tensor layout");
    }
    auto crow = impl->sparse_crow_impl();
    if (!crow) {
        TP_THROW(RuntimeError,
                 "crow_indices expected sparse row compressed tensor layout");
    }
    return Tensor(std::move(crow));
}

Tensor col_indices_kernel(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "col_indices expected sparse row compressed tensor layout");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (!impl->is_sparse_compressed()) {
        TP_THROW(RuntimeError,
                 "col_indices expected sparse row compressed tensor layout");
    }
    auto col = impl->sparse_col_impl();
    if (!col) {
        TP_THROW(RuntimeError,
                 "col_indices expected sparse row compressed tensor layout");
    }
    return Tensor(std::move(col));
}

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, MetaViewOps) {
    m.impl("as_strided", composite::as_strided_kernel);
    m.impl("as_strided_", composite::as_strided__kernel);
    m.impl("detach", composite::detach_kernel);
    m.impl("view", composite::view_kernel);
    m.impl("view.dtype", composite::view_dtype_kernel);
    m.impl("storage_offset", composite::storage_offset_kernel);
    m.impl("is_set_to", composite::is_set_to_kernel);
    m.impl("is_pinned", composite::is_pinned_kernel);
    m.impl("pin_memory", composite::pin_memory_kernel);
    m.impl("is_coalesced", composite::is_coalesced_kernel);
    m.impl("sparse_dim", composite::sparse_dim_kernel);
    m.impl("dense_dim", composite::dense_dim_kernel);
    m.impl("_indices", composite::indices_kernel);
    m.impl("_values", composite::values_kernel);
    m.impl("crow_indices", composite::crow_indices_kernel);
    m.impl("col_indices", composite::col_indices_kernel);
}

}  // namespace tensorplay
