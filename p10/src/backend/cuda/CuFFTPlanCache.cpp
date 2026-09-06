// Per-device cuFFT plan cache implementation. Plans are keyed by a full
// PlanSpec and kept in an LRU list so a bounded cache never grows without
// limit; eviction destroys the least recently used handle.

#include "CuFFTPlanCache.h"

#include "CUDARuntime.h"
#include "Exception.h"

#include <cufft.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace cufft {

namespace {

void check_cufft(cufftResult error, const char* what) {
    if (error != CUFFT_SUCCESS) {
        TP_THROW(RuntimeError, std::string(what), " failed: cuFFT error ",
                 static_cast<int>(error));
    }
}

// rocFFT historically double-freed handles returned by its planner, so
// eviction on the AMD backend drops the entry without destroying it.
void destroy_plan(cufftHandle handle) {
#if defined(USE_ROCM)
    (void)handle;
#else
    cufftDestroy(handle);
#endif
}

// Every rank goes through cufftPlanMany: null embeds mean unit stride, and
// routing 1D through the same call avoids the rank-specific cufftPlan1d
// failure mode observed on some driver / cuFFT 11 combinations.
cufftHandle make_plan(const PlanSpec& spec) {
    cufftHandle plan;
    check_cufft(cufftPlanMany(&plan, spec.rank,
                              reinterpret_cast<int*>(const_cast<int64_t*>(spec.n.data())),
                              spec.rank == 1
                                  ? nullptr
                                  : reinterpret_cast<int*>(const_cast<int64_t*>(spec.inembed.data())),
                              static_cast<int>(spec.istride),
                              static_cast<int>(spec.idist),
                              spec.rank == 1
                                  ? nullptr
                                  : reinterpret_cast<int*>(const_cast<int64_t*>(spec.onembed.data())),
                              static_cast<int>(spec.ostride),
                              static_cast<int>(spec.odist),
                              static_cast<cufftType>(spec.type),
                              static_cast<int>(spec.batch)),
                "cufftPlanMany");
    return plan;
}

}  // namespace

cufftHandle PlanLRUCache::lookup(const PlanSpec& spec) {
    if (max_size_ <= 0) {
        TP_THROW(RuntimeError, "cuFFT plan cache lookup requires a positive max size");
    }

    std::lock_guard<std::mutex> lock(mutex);

    auto it = map_.find(spec);
    if (it != map_.end()) {
        // Hit: move to the front of the usage list and rebind the plan to
        // the caller's current stream, since the previous user may have run
        // on a different one.
        usage_.splice(usage_.begin(), usage_, it->second);
        check_cufft(cufftSetStream(it->second->second, getCurrentCUDAStream().stream()),
                    "cufftSetStream");
        return it->second->second;
    }

    // Miss: evict the least recently used entry when full.
    if (static_cast<int64_t>(usage_.size()) >= max_size_) {
        auto last = usage_.end();
        --last;
        map_.erase(last->first);
        destroy_plan(last->second);
        usage_.pop_back();
    }

    cufftHandle plan = make_plan(spec);
    check_cufft(cufftSetStream(plan, getCurrentCUDAStream().stream()),
                "cufftSetStream");

    usage_.emplace_front(spec, plan);
    auto entry = usage_.begin();
    map_.emplace(entry->first, entry);
    return entry->second;
}

void PlanLRUCache::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& entry : usage_) {
        destroy_plan(entry.second);
    }
    map_.clear();
    usage_.clear();
}

void PlanLRUCache::set_max_size(int64_t new_size) {
    if (new_size < 0) {
        TP_THROW(RuntimeError,
                 "cuFFT plan cache size must be non-negative, but got ", new_size);
    }
    if (new_size > kMaxPlanNum) {
        TP_THROW(RuntimeError,
                 "cuFFT plan cache size can not be larger than ", kMaxPlanNum,
                 ", but got ", new_size);
    }
    max_size_ = new_size;
}

void PlanLRUCache::resize(int64_t new_size) {
    std::vector<cufftHandle> evicted;
    {
        std::lock_guard<std::mutex> lock(mutex);
        set_max_size(new_size);
        const int64_t current = static_cast<int64_t>(usage_.size());
        if (current > max_size_) {
            auto trim_from = usage_.end();
            for (int64_t i = 0; i < current - max_size_; ++i) {
                --trim_from;
                map_.erase(trim_from->first);
            }
            for (auto it = trim_from; it != usage_.end(); ++it) {
                evicted.push_back(it->second);
            }
            usage_.erase(trim_from, usage_.end());
        }
    }
    // Destroy outside the cache mutex: cufftDestroy can block on in-flight
    // work and must not serialize other cache operations.
    for (cufftHandle handle : evicted) {
        destroy_plan(handle);
    }
}

PlanLRUCache& plan_cache(int64_t device_index) {
    // One cache per device; created lazily on first touch. Entries live
    // until process exit, matching the lifetime of the underlying plans.
    static auto* caches = new std::vector<std::unique_ptr<PlanLRUCache>>();
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (device_index < 0) {
        TP_THROW(RuntimeError, "cuFFT plan cache: device index must be non-negative");
    }
    const size_t idx = static_cast<size_t>(device_index);
    if (idx >= caches->size()) {
        caches->resize(idx + 1);
    }
    if (!(*caches)[idx]) {
        (*caches)[idx] = std::make_unique<PlanLRUCache>();
    }
    return *(*caches)[idx];
}

namespace {

int device_count_or_zero() {
    const int count = ::tensorplay::cuda::deviceCount();
    return count > 0 ? count : 0;
}

void check_device_index(int64_t device_index, const char* who) {
    if (device_index < 0 || device_index >= device_count_or_zero()) {
        TP_THROW(RuntimeError, who, ": expected 0 <= device index < ",
                 device_count_or_zero(), ", but got device index ", device_index);
    }
}

}  // namespace

int64_t get_plan_cache_max_size(int64_t device_index) {
    check_device_index(device_index, "cufft_get_plan_cache_max_size");
    return plan_cache(device_index).max_size();
}

void set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
    check_device_index(device_index, "cufft_set_plan_cache_max_size");
    plan_cache(device_index).resize(max_size);
}

int64_t get_plan_cache_size(int64_t device_index) {
    check_device_index(device_index, "cufft_get_plan_cache_size");
    return static_cast<int64_t>(plan_cache(device_index).size());
}

void clear_plan_cache(int64_t device_index) {
    check_device_index(device_index, "cufft_clear_plan_cache");
    plan_cache(device_index).clear();
}

cufftHandle acquire_plan(const PlanSpec& spec, int64_t device_index) {
    PlanLRUCache& cache = plan_cache(device_index);
    if (cache.max_size() > 0) {
        return cache.lookup(spec);
    }
    // Caching disabled: build a transient plan the caller destroys.
    cufftHandle plan = make_plan(spec);
    check_cufft(cufftSetStream(plan, getCurrentCUDAStream().stream()),
                "cufftSetStream");
    return plan;
}

}  // namespace cufft
}  // namespace cuda
}  // namespace tensorplay
