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
// Creates a Tensor whose storage owns a fresh vTensorStorage allocation.
//
inline Tensor convert(const api::vTensor& tensor) {
  auto* holder = new detail::StoragePtr(
      std::make_shared<api::vTensorStorage>(
          api::context(),
          tensor.storage_type(),
          tensor.gpu_memory_layout(),
          tensor.gpu_sizes(),
          tensor.dtype()));

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
// Extracts the vTensor view for a Vulkan tensor.
//
inline api::vTensor convert(const Tensor& tensor) {
  TP_CHECK(tensor.device().is_vulkan(), "Vulkan tensor expected!");
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
