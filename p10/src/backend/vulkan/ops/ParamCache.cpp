#ifdef USE_VULKAN

#include "ParamCache.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

// Entry bound.  A decoder references one packed operand per weight tensor per
// layer, so the useful steady state is the whole model resident in packed
// form; the bound is there for call patterns that hand in a fresh storage on
// every invocation, which would otherwise grow the table without limit.
constexpr size_t kMaxCacheEntries = 4096u;

} // namespace

ParamTextureCache& ParamTextureCache::singleton() {
  static ParamTextureCache cache;
  return cache;
}

void ParamTextureCache::drop(EntryRef entry) {
  index_.erase(entry->first);
  const Identity identity{
      entry->first.storage_ptr, entry->first.layout, entry->first.tag};
  const auto current = current_.find(identity);
  if (current != current_.end() && current->second == entry) {
    current_.erase(current);
  }
  lru_.erase(entry);
}

api::vTensor ParamTextureCache::get_or_create(
    const Tensor& param,
    const std::vector<int64_t>& logical_sizes,
    api::GPUMemoryLayout layout,
    uint32_t tag,
    const std::function<api::vTensor()>& build) {
  const intrusive_ptr<TensorImpl>& impl = param.impl();
  TP_CHECK(
      impl->is_contiguous(), "Parameter texture cache expects dense tensors");

  const void* const storage_ptr = impl->storage().unsafeGetStorageImpl().get();
  const Key key{
      storage_ptr,
      impl->is_inference() ? 0u : impl->version(),
      static_cast<uint8_t>(layout),
      tag,
  };
  const Identity identity{storage_ptr, static_cast<uint8_t>(layout), tag};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = index_.find(key);
    if (it != index_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second);
      return convert(it->second->second.packed);
    }
  }

  api::vTensor v = build();

  Tensor device_tensor = convert(v);

  std::lock_guard<std::mutex> lock(mutex_);
  // A racing thread may have materialized this exact identity meanwhile; both
  // hold equivalent bytes, so keep whichever entry landed first.
  const auto raced = index_.find(key);
  if (raced != index_.end()) {
    lru_.splice(lru_.begin(), lru_, raced->second);
    return convert(raced->second->second.packed);
  }

  // The payload of an earlier version of the same parameter no longer
  // describes the storage contents, so it is dropped as this insert lands.
  const auto previous = current_.find(identity);
  if (previous != current_.end()) {
    drop(previous->second);
  }

  const EntryRef ref =
      lru_.emplace(lru_.begin(), key, Entry{Tensor(param), std::move(device_tensor)});
  index_.emplace(key, ref);
  current_.emplace(identity, ref);

  while (lru_.size() > kMaxCacheEntries) {
    drop(std::prev(lru_.end()));
  }

  return v;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
