#pragma once

#ifdef USE_VULKAN

#include "../api/Context.h"
#include "../api/Tensor.h"
#include "../api/Types.h"
#include "DataPtr.h"
#include "Storage.h"
#include "Tensor.h"

#include <memory>

namespace tensorplay {
namespace vulkan {
namespace ops {

//
// The Vulkan tensor payload lives in the tensor's Storage: its DataPtr wraps
// a shared_ptr<api::vTensorStorage> as the deleter context.  These helpers
// convert between the public Tensor surface and the api::vTensor view.
//
// NOTE: unlike a full opaque-TensorImpl setup, strides/contiguity bookkeeping
// stays on the ordinary TensorImpl; the backend only owns GPU resources.
//

//
// Restricts creation-time dtypes to the element types the backend implements:
// 1-, 2-, and 4-byte payloads backed by a matching VkFormat.  Requests outside
// the set are rejected at allocation time.
//
inline DType convert_dtype(const DType dtype) {
  switch (toUnderlyingStorageType(dtype)) {
    case DType::UInt8:
    case DType::Int8:
    case DType::Int32:
    case DType::Bool:
    case DType::Float16:
    case DType::Float32:
      return dtype;
    default:
      TP_THROW(NotImplementedError, "Not a supported Vulkan dtype!");
  }
}

namespace detail {

using StoragePtr = std::shared_ptr<api::vTensorStorage>;

inline void delete_vtensor_storage(void* ctx) {
  delete static_cast<StoragePtr*>(ctx);
}

inline const StoragePtr* storage_ptr_handle(const Tensor& tensor) {
  const Storage& storage = tensor.impl()->storage();
  return static_cast<const StoragePtr*>(
      storage.unsafeGetStorageImpl()->data_ptr.get_context());
}

inline StoragePtr* storage_ptr_handle(Tensor& tensor) {
  const Storage& storage = tensor.impl()->storage();
  return static_cast<StoragePtr*>(
      storage.unsafeGetStorageImpl()->data_ptr.get_context());
}

} // namespace detail

//
// Creates a Tensor whose storage shares the vTensor's payload: the DataPtr
// context holds the same vTensorStorage the shader writes go through.
//
inline Tensor convert(const api::vTensor& tensor) {
  auto* holder = new detail::StoragePtr(tensor.view());

  TP_CHECK(
      holder->get() != nullptr, "Vulkan tensor storage is missing!");

  const size_t nbytes = tensor.nbytes();

  // The DataPtr value carries the owning VulkanBuffer object (kept alive by
  // the context shared_ptr below); texture-backed storages have no buffer,
  // in which case the value stays null.
  DataPtr data_ptr(
      holder->get()->buffer().has_memory()
          ? static_cast<void*>(&holder->get()->buffer())
          : nullptr,
      holder,
      &detail::delete_vtensor_storage,
      Device(DeviceType::Vulkan));

  Storage storage(std::move(data_ptr), nbytes, nullptr);

  return Tensor(
      std::move(storage),
      std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()),
      tensor.dtype());
}

//
// Extracts the vTensor view for a Vulkan tensor.  The payload only follows
// the dense layout, so a TensorImpl carrying non-dense strides or a nonzero
// storage offset would silently mis-address reads; guard that case at the
// boundary.
//
inline api::vTensor convert(const Tensor& tensor) {
  TP_CHECK(tensor.device().is_vulkan(), "Vulkan tensor expected!");
  TP_CHECK(
      tensor.unsafeGetTensorImpl()->is_contiguous() &&
          tensor.unsafeGetTensorImpl()->storage_offset() == 0,
      "Vulkan tensor payload must be dense with zero storage offset; "
      "materialize strided views before device-level access");
  const detail::StoragePtr* holder = detail::storage_ptr_handle(tensor);
  TP_CHECK(holder && *holder, "Vulkan tensor storage is missing!");
  return api::vTensor(
      api::context(),
      static_cast<std::vector<int64_t>>(tensor.shape()),
      tensor.dtype(),
      **holder,
      (*holder)->gpu_memory_layout());
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
