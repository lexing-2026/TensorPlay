#pragma once

#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"
#include "../api/Tensor.h"

#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

//
// Persistent texture cache for packed operator parameters (convolution
// weights, biases, product operands).
//
// A parameter payload is materialized once in its packed texture form and
// kept alive across calls, so steady-state inference does not re-stage or
// re-group the weights on every invocation.  Entries are keyed on the source
// tensor's identity (storage pointer + version counter + layout + tag):
// bumping the version through an in-place update invalidates the entry, and
// a released storage simply stops matching.  Superseded versions of the same
// storage are dropped on insert so training loops do not accumulate stale
// payloads.
//
// Over the entry bound the oldest-referenced entry is dropped, so a working
// set that fits stays resident instead of being flushed as a whole.
//
// `build` performs the actual materialization (host pack + upload, or a
// device-side regroup dispatch); it runs at most once per identity.
//
class ParamTextureCache final {
 public:
  static ParamTextureCache& singleton();

  api::vTensor get_or_create(
      const Tensor& param,
      const std::vector<int64_t>& logical_sizes,
      api::GPUMemoryLayout layout,
      uint32_t tag,
      const std::function<api::vTensor()>& build);

 private:
  // One entry holds the packed texture plus a strong reference to the source
  // tensor.  The reference pins the source storage while the entry lives, so
  // a released-and-recycled allocation cannot be re-keyed onto a stale
  // packed payload: the key is the source storage pointer, and as long as
  // this cache is the only thing extending its lifetime, a reused pointer
  // implies the previous owner is gone and its entry went with it.
  struct Entry final {
    Tensor source;
    Tensor packed;
  };

  struct Key {
    const void* storage_ptr;
    uint32_t version;
    uint8_t layout;
    uint32_t tag;

    bool operator==(const Key& other) const {
      return storage_ptr == other.storage_ptr && version == other.version &&
          layout == other.layout && tag == other.tag;
    }
  };

  struct Hasher {
    inline size_t operator()(const Key& key) const {
      size_t seed = std::hash<const void*>()(key.storage_ptr);
      seed = api::utils::hash_combine(
          seed, std::hash<uint32_t>()(key.version));
      seed = api::utils::hash_combine(seed, std::hash<uint8_t>()(key.layout));
      return api::utils::hash_combine(seed, std::hash<uint32_t>()(key.tag));
    }
  };

  // The key without its version counter: the parameter itself.  Holding the
  // live entry per parameter here makes dropping the superseded version of an
  // in-place update constant-time instead of a walk over every cached payload.
  struct Identity {
    const void* storage_ptr;
    uint8_t layout;
    uint32_t tag;

    bool operator==(const Identity& other) const {
      return storage_ptr == other.storage_ptr && layout == other.layout &&
          tag == other.tag;
    }
  };

  struct IdentityHasher {
    inline size_t operator()(const Identity& identity) const {
      size_t seed = std::hash<const void*>()(identity.storage_ptr);
      seed = api::utils::hash_combine(seed, std::hash<uint8_t>()(identity.layout));
      return api::utils::hash_combine(seed, std::hash<uint32_t>()(identity.tag));
    }
  };

  using Entries = std::list<std::pair<Key, Entry>>;
  using EntryRef = Entries::iterator;

  // Unlinks one entry from all three tables.
  void drop(EntryRef entry);

  std::mutex mutex_;
  // Recency order, most recently referenced at the front.
  Entries lru_;
  std::unordered_map<Key, EntryRef, Hasher> index_;
  std::unordered_map<Identity, EntryRef, IdentityHasher> current_;
};

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
