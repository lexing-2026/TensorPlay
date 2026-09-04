#ifdef USE_VULKAN

#include "ParamCache.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

// Entry bound: each cached value holds one packed texture, and the working
// set of a steady-state loop touches a fixed handful of parameters.  When a
// call pattern streams distinct storages through the cache (fresh tensors
// every call), insertion evicts everything past the bound so the map cannot
// grow without limit.
constexpr size_t kMaxCacheEntries = 256u;

} // namespace

ParamTextureCache& ParamTextureCache::singleton() {
  static ParamTextureCache cache;
  return cache;
}

api::vTensor ParamTextureCache::get_or_create(
    const Tensor& param,
    const std::vector<int64_t>& logical_sizes,
    api::GPUMemoryLayout layout,
    uint32_t tag,
    const std::function<api::vTensor()>& build) {
  const std::shared_ptr<TensorImpl>& impl = param.impl();
  TP_CHECK(
      impl->is_contiguous(), "Parameter texture cache expects dense tensors");

  const Key key{
      impl->storage().unsafeGetStorageImpl().get(),
      impl->is_inference() ? 0u : impl->version(),
      static_cast<uint8_t>(layout),
      tag,
  };
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cache_.find(key);
    if (it != cache_.end()) {
      return convert(it->second.packed);
    }
  }

  api::vTensor v = build();

  Tensor device_tensor = convert(v);

  std::lock_guard<std::mutex> lock(mutex_);
  // Drop superseded versions of the same identity so in-place updates do not
  // accumulate stale payloads; a racing thread may have inserted the same
  // identity meanwhile, in which case keep whichever entry landed first
  // (both hold equivalent bytes).
  for (auto it = cache_.begin(); it != cache_.end();) {
    if (it->first.storage_ptr == key.storage_ptr &&
        it->first.layout == key.layout && it->first.tag == key.tag) {
      it = cache_.erase(it);
    } else {
      ++it;
    }
  }
  if (cache_.size() >= kMaxCacheEntries) {
    cache_.clear();
  }
  cache_.emplace(key, Entry{Tensor(param), std::move(device_tensor)});

  return v;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
