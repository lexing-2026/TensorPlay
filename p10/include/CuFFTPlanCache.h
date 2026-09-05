#pragma once

// Per-device LRU cache for cuFFT plans. A plan is described by a PlanSpec
// (transform type, transform sizes, embedded strides, batch); equivalent
// specs share one plan. Caches are bounded: when a lookup misses and the
// cache is full, the least recently used plan is destroyed. A max size of
// zero disables caching entirely, making every lookup create a transient
// plan the caller must destroy.

#include "Macros.h"

// The plan handle is a typedef of a scalar type in both supported FFT
// libraries, so it cannot be forward declared; include the real header.
#if defined(USE_ROCM)
#include <hipfft/hipfft.h>
#else
#include <cufft.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tensorplay {
namespace cuda {
namespace cufft {

// Upper bound on stored plans; the default capacity is a fraction of this.
inline constexpr int64_t kMaxPlanNum = INT64_MAX;
inline constexpr int64_t kDefaultCacheSize = 4096;

// Full description of one cuFFT plan. `rank` is 1 or 2. For rank 1 the
// embed arrays are unused and pass as null to the planner, matching the
// plain batched 1D planning call; for rank 2 they carry the embedded
// input/output layouts.
struct PlanSpec {
    int rank = 0;
    int type = 0;  // cufftType value (CUFFT_C2C, CUFFT_R2C, ...)
    std::array<int64_t, 2> n{};        // transform sizes
    std::array<int64_t, 2> inembed{};  // input embed
    std::array<int64_t, 2> onembed{};  // output embed
    int64_t istride = 0;
    int64_t idist = 0;
    int64_t ostride = 0;
    int64_t odist = 0;
    int64_t batch = 0;

    bool operator==(const PlanSpec& o) const {
        return rank == o.rank && type == o.type && n == o.n &&
               inembed == o.inembed && onembed == o.onembed &&
               istride == o.istride && idist == o.idist &&
               ostride == o.ostride && odist == o.odist && batch == o.batch;
    }
};

struct PlanSpecHash {
    size_t operator()(const PlanSpec& k) const {
        size_t h = std::hash<int>()(k.rank) ^ (std::hash<int>()(k.type) << 1);
        auto mix = [&h](int64_t v) {
            h ^= std::hash<int64_t>()(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        for (int i = 0; i < 2; ++i) {
            mix(k.n[i]);
            mix(k.inembed[i]);
            mix(k.onembed[i]);
        }
        mix(k.istride);
        mix(k.idist);
        mix(k.ostride);
        mix(k.odist);
        mix(k.batch);
        return h;
    }
};

class P10_API PlanLRUCache {
public:
    PlanLRUCache() : PlanLRUCache(kDefaultCacheSize) {}
    explicit PlanLRUCache(int64_t max_size) { set_max_size(max_size); }

    PlanLRUCache(PlanLRUCache&& other) noexcept
        : usage_(std::move(other.usage_)),
          map_(std::move(other.map_)),
          max_size_(other.max_size_) {}
    PlanLRUCache& operator=(PlanLRUCache&& other) noexcept {
        usage_ = std::move(other.usage_);
        map_ = std::move(other.map_);
        max_size_ = other.max_size_;
        return *this;
    }

    // Returns the plan for `spec`, creating and inserting it on a miss.
    // Requires max_size() > 0. The handle stays valid until eviction or
    // destruction; callers must not destroy it. With the internal mutex
    // held only during the call, distinct callers may race on the same
    // handle's stream assignment; set the execution stream right before
    // cufftExecute* to keep each launch ordered on its own stream.
    cufftHandle lookup(const PlanSpec& spec);

    void clear();
    void resize(int64_t new_size);
    size_t size() const { return map_.size(); }
    int64_t max_size() const noexcept { return max_size_; }

    std::mutex mutex;

private:
    void set_max_size(int64_t new_size);

    using Entry = std::pair<PlanSpec, cufftHandle>;
    std::list<Entry> usage_;  // front = most recently used
    std::unordered_map<PlanSpec, std::list<Entry>::iterator, PlanSpecHash> map_;
    int64_t max_size_ = kDefaultCacheSize;
};

// The cache for one device, created on demand. The registry grows with the
// highest device index queried; entries live until process exit.
P10_API PlanLRUCache& plan_cache(int64_t device_index);

// Cache introspection used by the Python bindings. Each accessor validates
// the index against the device count before touching the registry.
P10_API int64_t get_plan_cache_max_size(int64_t device_index);
P10_API void set_plan_cache_max_size(int64_t device_index, int64_t max_size);
P10_API int64_t get_plan_cache_size(int64_t device_index);
P10_API void clear_plan_cache(int64_t device_index);

// Returns a plan for `spec` on `device_index`. When the device cache is
// enabled the plan is owned by the cache; when caching is disabled
// (max size 0) a fresh transient plan is returned and the caller owns it.
P10_API cufftHandle acquire_plan(const PlanSpec& spec, int64_t device_index);

}  // namespace cufft
}  // namespace cuda
}  // namespace tensorplay
