#pragma once

#include "Allocator.h"
#include "Exception.h"
#include "Tensor.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace tensorplay::native {

inline uint64_t checked_add(uint64_t lhs, uint64_t rhs) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        TP_THROW(RuntimeError, "storage size calculation overflowed");
    }
    return lhs + rhs;
}

inline uint64_t checked_mul(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        TP_THROW(RuntimeError, "storage size calculation overflowed");
    }
    return lhs * rhs;
}

inline uint64_t checked_nonnegative(int64_t value, const char* what) {
    if (value < 0) {
        TP_THROW(ValueError, what, " must be non-negative");
    }
    return static_cast<uint64_t>(value);
}

inline size_t checked_storage_bytes_contiguous(
        const std::vector<int64_t>& size,
        size_t itemsize,
        int64_t storage_offset) {
    const uint64_t offset = checked_nonnegative(storage_offset, "storage offset");
    uint64_t elements = 1;
    for (int64_t value : size) {
        const uint64_t dimension = checked_nonnegative(value, "size");
        if (dimension == 0) {
            return 0;
        }
        elements = checked_mul(elements, dimension);
    }
    const uint64_t bytes = checked_mul(
        checked_add(offset, elements), static_cast<uint64_t>(itemsize));
    if (bytes > std::numeric_limits<size_t>::max()) {
        TP_THROW(RuntimeError, "storage size calculation overflowed");
    }
    return static_cast<size_t>(bytes);
}

inline size_t checked_storage_bytes_strided(
        const std::vector<int64_t>& size,
        const std::vector<int64_t>& stride,
        size_t itemsize,
        int64_t storage_offset) {
    if (size.size() != stride.size()) {
        TP_THROW(ValueError, "size and stride must have the same length");
    }
    const uint64_t offset = checked_nonnegative(storage_offset, "storage offset");
    uint64_t span = checked_add(offset, 1);
    for (size_t i = 0; i < size.size(); ++i) {
        const uint64_t dimension = checked_nonnegative(size[i], "size");
        if (dimension == 0) {
            return 0;
        }
        const uint64_t step = checked_nonnegative(stride[i], "stride");
        span = checked_add(span, checked_mul(step, dimension - 1));
    }
    const uint64_t bytes = checked_mul(span, static_cast<uint64_t>(itemsize));
    if (bytes > std::numeric_limits<size_t>::max()) {
        TP_THROW(RuntimeError, "storage size calculation overflowed");
    }
    return static_cast<size_t>(bytes);
}

inline Tensor& set_storage_and_metadata(
        Tensor& result,
        Storage storage,
        int64_t storage_offset,
        const std::vector<int64_t>& size,
        const std::vector<int64_t>& stride) {
    if (!result.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    if (!storage.defined()) {
        TP_THROW(RuntimeError, "cannot set a tensor from undefined storage");
    }
    auto impl = result.unsafeGetTensorImpl();
    if (impl->is_sparse()) {
        TP_THROW(RuntimeError, "set_ is unavailable for sparse tensors");
    }
    if (result.device() != storage.device()) {
        TP_THROW(DeviceMismatchError,
                 "cannot set tensor storage from a different device");
    }

    const size_t required = stride.empty()
        ? checked_storage_bytes_contiguous(size, impl->itemsize(), storage_offset)
        : checked_storage_bytes_strided(
              size, stride, impl->itemsize(), storage_offset);
    const bool same_size =
        static_cast<std::vector<int64_t>>(impl->sizes()) == size;
    const bool same_stride = stride.empty() ||
        static_cast<std::vector<int64_t>>(impl->strides()) == stride;
    if (same_size && same_stride && required > storage.nbytes()) {
        TP_THROW(RuntimeError,
                 "requested tensor geometry is out of bounds for storage");
    }
    if (required > storage.nbytes()) {
        Storage resized = storage;
        resized.set_nbytes(required);
        storage = std::move(resized);
    }

    if (!impl->storage().is_same(storage)) {
        impl->set_storage(std::move(storage));
    }
    impl->set_storage_offset(
        static_cast<size_t>(checked_nonnegative(storage_offset, "storage offset")));
    if (stride.empty()) {
        impl->set_sizes_contiguous(size);
    } else {
        impl->set_sizes_and_strides(size, stride);
    }
    return result;
}

inline int64_t storage_numel(const Storage& storage, size_t itemsize) {
    if (itemsize == 0) {
        TP_THROW(RuntimeError, "cannot derive storage size from an undefined dtype");
    }
    const uint64_t elements = storage.nbytes() / itemsize;
    if (elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        TP_THROW(RuntimeError, "storage size calculation overflowed");
    }
    return static_cast<int64_t>(elements);
}

inline Tensor& set_storage_native(Tensor& result, Storage source) {
    if (!result.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    const int64_t size = storage_numel(source, result.itemsize());
    return set_storage_and_metadata(
        result,
        std::move(source),
        0,
        {size},
        {});
}

inline Tensor& set_storage_offset_native(
        Tensor& result,
        Storage source,
        int64_t storage_offset,
        const std::vector<int64_t>& size,
        const std::vector<int64_t>& stride) {
    return set_storage_and_metadata(
        result, std::move(source), storage_offset, size, stride);
}

inline Tensor& set_tensor_native(Tensor& result, const Tensor& source) {
    if (!result.defined() || !source.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    if (result.unsafeGetTensorImpl() == source.unsafeGetTensorImpl()) {
        return result;
    }
    if (source.is_sparse()) {
        TP_THROW(RuntimeError, "set_ is unavailable from sparse tensors");
    }
    const uint64_t source_offset = source.unsafeGetTensorImpl()->storage_offset();
    if (source_offset >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        TP_THROW(RuntimeError, "storage offset calculation overflowed");
    }
    return set_storage_and_metadata(
        result,
        source.unsafeGetTensorImpl()->storage(),
        static_cast<int64_t>(source_offset),
        static_cast<std::vector<int64_t>>(source.shape()),
        source.strides());
}

inline Tensor& set_tensor_storage_offset_native(
        Tensor& result,
        const Tensor& source,
        int64_t storage_offset,
        const std::vector<int64_t>& size,
        const std::vector<int64_t>& stride) {
    if (!result.defined() || !source.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    if (source.is_sparse()) {
        TP_THROW(RuntimeError, "set_ is unavailable from sparse tensors");
    }
    if (!source.is_contiguous()) {
        TP_THROW(RuntimeError,
                 "passed in tensor to be used as storage must be contiguous");
    }
    const uint64_t source_offset = source.unsafeGetTensorImpl()->storage_offset();
    const uint64_t requested_offset =
        checked_nonnegative(storage_offset, "storage offset");
    if (source_offset > std::numeric_limits<uint64_t>::max() - requested_offset ||
        source_offset + requested_offset >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        TP_THROW(RuntimeError, "storage offset calculation overflowed");
    }
    return set_storage_and_metadata(
        result,
        source.unsafeGetTensorImpl()->storage(),
        static_cast<int64_t>(source_offset + requested_offset),
        size,
        stride);
}

inline Tensor& reset_tensor_storage_native(
        Tensor& result, DeviceType device_type) {
    if (!result.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    Storage storage(0, getAllocator(device_type), result.device());
    return set_storage_and_metadata(result, std::move(storage), 0, {0}, {});
}

} // namespace tensorplay::native
