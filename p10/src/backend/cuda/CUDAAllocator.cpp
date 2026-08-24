#include "Allocator.h"
#include "CUDAGraph.h"
#include "CUDARuntime.h"
#include "Device.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

constexpr size_t kSmallAllocation = 1024 * 1024;
constexpr size_t kSmallRound = 512;
// Keep large requests reasonably granular as well.  Without segment
// splitting, a coarse 2 MiB quantum leaves avoidable internal slack for the
// many differently-sized temporary tensors in decoder layers.
constexpr size_t kLargeRound = 512 * 1024;

size_t roundAllocation(size_t nbytes) {
    if (nbytes == 0) return 0;
    const size_t quantum = nbytes <= kSmallAllocation ? kSmallRound : kLargeRound;
    return ((nbytes + quantum - 1) / quantum) * quantum;
}

struct Segment {
    void* ptr = nullptr;
    size_t size = 0;
    int device = -1;
    // Number of live, cached, or event-pending block metadata objects that
    // still refer to this cudaMalloc allocation.  A segment is returned to
    // CUDA only after this reaches zero; a live split block therefore keeps
    // the rest of its segment reserved, exactly like Torch's inactive-split
    // accounting.
    size_t block_count = 0;
};

struct Block {
    void* ptr = nullptr;
    size_t size = 0;
    // ``size`` is the rounded backing allocation used by the pool.  Keep the
    // original request separately because memory_allocated() reports live
    // tensor bytes, whereas memory_reserved() reports backing bytes.
    size_t requested_size = 0;
    int device = -1;
    Segment* segment = nullptr;
    size_t offset = 0;
    // CUDA stream on which this block was allocated/last activated.  Work
    // submitted to one stream is totally ordered, so a block can be returned
    // to that stream's cache immediately after release.  Other streams are
    // protected by recorded events, matching the stream-ordered reuse rule of
    // c10::cuda::CUDACachingAllocator.
    cudaStream_t allocation_stream = nullptr;
    std::unordered_set<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    // Non-zero while the block belongs to a graph-private memory pool: its
    // address may be baked into a captured graph, so it is never recycled
    // outside that pool (see CUDAGraph.h).
    uint64_t pool_id = 0;
};

struct DeviceStats {
    size_t allocated = 0;
    size_t reserved = 0;
    size_t max_allocated = 0;
    size_t max_reserved = 0;
};

struct DeviceCache {
    // Small and large allocations are kept in separate best-fit pools so a
    // tiny request cannot consume a very large cached block.
    std::map<size_t, std::vector<std::shared_ptr<Block>>> small;
    std::map<size_t, std::vector<std::shared_ptr<Block>>> large;
    // Address ordering is used for adjacent split-block coalescing.  The
    // size-ordered maps above remain the best-fit lookup structure.
    std::map<uintptr_t, std::shared_ptr<Block>> free_by_addr;
    std::vector<std::shared_ptr<Block>> pending;
    DeviceStats stats;
};

struct PinnedEvent {
    cudaEvent_t event = nullptr;
    int device = -1;
};

struct PinnedStream {
    cudaStream_t stream = nullptr;
    int device = -1;
};

struct PinnedBlock {
    void* ptr = nullptr;
    size_t size = 0;
    std::vector<PinnedStream> streams;
    std::vector<PinnedEvent> events;
};

// Open capture scope: while set, allocations on {device, stream} are routed
// into the graph-private pool (CUDAGraph.h note).
struct ActiveCapture {
    uint64_t pool_id = 0;
    int device = -1;
    cudaStream_t stream = nullptr;
};

struct GraphPool {
    // Free blocks parked after release.  They stay exclusive to this pool so
    // a later capture-time allocation can reuse them, but no eager path can.
    std::map<size_t, std::vector<std::shared_ptr<Block>>> small;
    std::map<size_t, std::vector<std::shared_ptr<Block>>> large;
    // Segments cudaMalloc'd while this pool was the routing target.  Their
    // backing memory is returned to CUDA only by releasePool().
    std::vector<Segment*> segments;
};

class AllocatorState {
public:
    static AllocatorState& instance() {
        // Deliberately leaked: DataPtr destructors can run during interpreter
        // teardown after normal static destruction has started.
        static auto* state = new AllocatorState();
        return *state;
    }

    std::shared_ptr<Block> allocate(size_t nbytes, int device) {
        device = normalizeDevice(device);
        const size_t rounded = roundAllocation(nbytes);
        CUDAGuard guard(device);
        const cudaStream_t allocation_stream =
            getCurrentCUDAStream(device).stream();

        uint64_t pool_target = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ensureDevice(device);
            if (capture_.pool_id != 0 && capture_.device == device &&
                capture_.stream == allocation_stream) {
                // Graph-capture allocation: reuse is restricted to the
                // private pool so replay-visible addresses stay exclusive.
                pool_target = capture_.pool_id;
            } else {
                // Event queries are skipped while any capture is open: a
                // cudaEventQuery on a pending block cannot fail the capture,
                // but the queries themselves may target events recorded on
                // unrelated streams mid-capture; deferring them is free.
                if (capture_.pool_id == 0) reclaimCompletedLocked(device);
                if (auto block = takeCachedLocked(device, rounded, allocation_stream)) {
                    activateLocked(block, allocation_stream, nbytes);
                    return block;
                }
            }
            if (pool_target != 0) {
                if (auto block = takeGraphPoolLocked(pool_target, rounded,
                                                     allocation_stream)) {
                    activateLocked(block, allocation_stream, nbytes);
                    return block;
                }
            }
        }

        // Raw cudaMalloc is a "potentially unsafe" API: while a Global-mode
        // capture is open anywhere in the process it would abort the capture
        // with cudaErrorStreamCaptureUnsupported (error 900), yet segments
        // must grow mid-capture to serve graph-pool allocations.  Mirror
        // torch's CUDAStreamCaptureModeGuard by relaxing this thread's
        // capture mode around the driver call only.
        cudaStreamCaptureMode previous_mode = cudaStreamCaptureModeGlobal;
        const bool relax = capture_.pool_id != 0;
        if (relax) {
            cudaStreamCaptureMode relaxed = cudaStreamCaptureModeRelaxed;
            (void)cudaThreadExchangeStreamCaptureMode(&relaxed);
            previous_mode = relaxed;
        }
        void* ptr = nullptr;
        cudaError_t error = cudaMalloc(&ptr, rounded);
        if (error == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            emptyCacheDevice(device);
            error = cudaMalloc(&ptr, rounded);
        }
        if (relax) {
            cudaThreadExchangeStreamCaptureMode(&previous_mode);
        }
        if (error != cudaSuccess) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            (void)cudaMemGetInfo(&free_bytes, &total_bytes);
            std::ostringstream message;
            message << "CUDA out of memory on cuda:" << device
                    << ". Tried to allocate " << rounded << " bytes; "
                    << free_bytes << " bytes free of " << total_bytes << " total";
            checkCuda(error, message.str().c_str());
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            ensureDevice(device);
            auto segment = std::make_unique<Segment>();
            segment->ptr = ptr;
            segment->size = rounded;
            segment->device = device;
            segment->block_count = 1;
            Segment* segment_ptr = segment.get();
            segments_[ptr] = std::move(segment);

            auto block = std::make_shared<Block>();
            block->ptr = ptr;
            block->size = rounded;
            block->requested_size = nbytes;
            block->device = device;
            block->segment = segment_ptr;
            block->offset = 0;
            auto& stats = caches_[device].stats;
            if (pool_target != 0) {
                block->pool_id = pool_target;
                pools_[pool_target].segments.push_back(segment_ptr);
            }
            stats.reserved += rounded;
            stats.max_reserved = std::max(stats.max_reserved, stats.reserved);
            activateLocked(block, allocation_stream, nbytes);
            return block;
        }
    }

    // Raw-pointer entry point for DataPtr's function-pointer deleter: the
    // owning shared_ptr is recovered from the live-block map.
    void releaseRaw(Block* raw) noexcept {
        if (!raw) return;
        std::shared_ptr<Block> block;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = live_blocks_.find(raw->ptr);
            if (it == live_blocks_.end()) return;
            block = it->second;
        }
        release(block);
    }

    void release(const std::shared_ptr<Block>& block) noexcept {
        if (!block || !block->ptr) return;
        try {
            std::vector<cudaStream_t> streams;
            cudaStream_t allocation_stream = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ensureDevice(block->device);
                auto live = live_blocks_.find(block->ptr);
                if (live == live_blocks_.end()) return;
                live_blocks_.erase(live);
                auto& stats = caches_[block->device].stats;
                stats.allocated -= std::min(stats.allocated, block->requested_size);
                streams.assign(block->streams.begin(), block->streams.end());
                allocation_stream = block->allocation_stream;
                block->streams.clear();
            }

            // The allocation stream is ordered with all work that used the
            // block on that stream.  Do not create an event for it; only
            // streams which are not ordered with the owner need fencing.
            streams.erase(std::remove(streams.begin(), streams.end(), allocation_stream),
                          streams.end());

            if (block->pool_id != 0) {
                // Graph-pool blocks never re-enter the general cache and
                // never take event fences: event records on a capturing
                // stream abort the capture, and cross-stream reuse cannot
                // happen because only the pool's own same-stream free lists
                // can hand the block out again.  The memory stays reserved
                // until releasePool() frees its segment.
                std::lock_guard<std::mutex> lock(mutex_);
                block->requested_size = 0;
                auto pit = pools_.find(block->pool_id);
                if (pit == pools_.end()) return;
                auto& pmap = isSmall(block->size) ? pit->second.small
                                                  : pit->second.large;
                pmap[block->size].push_back(block);
                return;
            }

            if (streams.empty()) {
                std::lock_guard<std::mutex> lock(mutex_);
                insertCachedLocked(caches_[block->device], block);
                return;
            }

            CUDAGuard guard(block->device);
            bool event_failure = false;
            for (cudaStream_t stream : streams) {
                cudaEvent_t event = nullptr;
                cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                if (error == cudaSuccess) error = cudaEventRecord(event, stream);
                if (error != cudaSuccess) {
                    if (event) (void)cudaEventDestroy(event);
                    (void)cudaGetLastError();
                    event_failure = true;
                    break;
                }
                block->events.push_back(event);
            }

            if (event_failure) {
                for (cudaEvent_t event : block->events) (void)cudaEventDestroy(event);
                block->events.clear();
                // Correctness is more important than retaining asynchrony in
                // this exceptional path.
                if (cudaDeviceSynchronize() != cudaSuccess) return;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            auto& cache = caches_[block->device];
            if (block->events.empty()) {
                insertCachedLocked(cache, block);
            } else {
                cache.pending.push_back(block);
            }
        } catch (...) {
            // DataPtr deleters must never throw, especially during Python or
            // CUDA runtime shutdown. A driver-teardown failure may leak one
            // block, but cannot lead to unsafe early reuse.
        }
    }

    void record(void* ptr, const CUDAStream& stream) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = live_blocks_.find(ptr);
        if (it == live_blocks_.end()) return;
        const auto& block = it->second;
        if (block->device != stream.device_index()) {
            TP_THROW(DeviceMismatchError,
                     "CUDA storage on cuda:" + std::to_string(block->device) +
                     " was used with a stream on cuda:" +
                     std::to_string(stream.device_index()));
        }
        block->streams.insert(stream.stream());
    }

    size_t allocated(int device) {
        device = normalizeDevice(device);
        std::lock_guard<std::mutex> lock(mutex_);
        ensureDevice(device);
        return caches_[device].stats.allocated;
    }

    size_t reserved(int device) {
        device = normalizeDevice(device);
        std::lock_guard<std::mutex> lock(mutex_);
        ensureDevice(device);
        return caches_[device].stats.reserved;
    }

    size_t maxAllocated(int device) {
        device = normalizeDevice(device);
        std::lock_guard<std::mutex> lock(mutex_);
        ensureDevice(device);
        return caches_[device].stats.max_allocated;
    }

    size_t maxReserved(int device) {
        device = normalizeDevice(device);
        std::lock_guard<std::mutex> lock(mutex_);
        ensureDevice(device);
        return caches_[device].stats.max_reserved;
    }

    void resetPeaks(int device) {
        device = normalizeDevice(device);
        std::lock_guard<std::mutex> lock(mutex_);
        ensureDevice(device);
        auto& stats = caches_[device].stats;
        stats.max_allocated = stats.allocated;
        stats.max_reserved = stats.reserved;
    }

    void emptyCache() {
        const int count = deviceCount();
        for (int device = 0; device < count; ++device) emptyCacheDevice(device);
    }

    // -- graph-private pools (CUDAGraph.h) ------------------------------------

    uint64_t beginGraphCapture(int device, const CUDAStream& stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capture_.pool_id != 0) {
            TP_THROW(RuntimeError,
                     "nested CUDA graph capture pools are not supported");
        }
        const uint64_t pool_id = next_pool_id_++;
        pools_[pool_id] = GraphPool{};
        capture_ = ActiveCapture{pool_id, device, stream.stream()};
        return pool_id;
    }

    void endGraphCapture(uint64_t pool_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capture_.pool_id == pool_id) capture_ = ActiveCapture{};
    }

    bool isCapturing() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return capture_.pool_id != 0;
    }

    void releaseGraphPool(uint64_t pool_id) {
        std::vector<std::tuple<void*, size_t, int>> segments_to_free;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pools_.find(pool_id);
            if (it == pools_.end()) return;
            if (capture_.pool_id == pool_id) {
                TP_THROW(RuntimeError,
                         "cannot release a CUDA graph memory pool while its "
                         "capture is open");
            }
            GraphPool& pool = it->second;
            // A segment's block_count counts every metadata reference: live
            // tensors plus blocks parked on this pool's free lists.  Freeing
            // is safe only when nothing beyond those parked blocks remains.
            size_t parked = 0;
            for (const auto& [size, bucket] : pool.small) parked += bucket.size();
            for (const auto& [size, bucket] : pool.large) parked += bucket.size();
            size_t total_refs = 0;
            for (const Segment* segment : pool.segments) {
                total_refs += segment->block_count;
            }
            if (total_refs > parked) {
                TP_THROW(RuntimeError,
                         "refusing to release CUDA graph memory pool " +
                         std::to_string(pool_id) + ": " +
                         std::to_string(total_refs - parked) +
                         " tensor(s) allocated during capture are still "
                         "alive; destroy the graph executable and drop all "
                         "static input/output tensors first");
            }
            for (Segment* segment : pool.segments) {
                segments_to_free.emplace_back(segment->ptr, segment->size,
                                              segment->device);
                auto& stats = caches_[segment->device].stats;
                stats.reserved -= std::min(stats.reserved, segment->size);
                segments_.erase(segment->ptr);
            }
            pools_.erase(it);
        }
        for (const auto& segment : segments_to_free) {
            CUDAGuard segment_guard(std::get<2>(segment));
            checkCuda(cudaFree(std::get<0>(segment)), "cudaFree");
        }
    }

private:
    int normalizeDevice(int device) const {
        return device < 0 ? currentDevice() : device;
    }

    void ensureDevice(int device) {
        if (device < 0 || device >= deviceCount()) {
            TP_THROW(ValueError, "Invalid CUDA device index " + std::to_string(device));
        }
        if (caches_.size() <= static_cast<size_t>(device)) {
            caches_.resize(static_cast<size_t>(device) + 1);
        }
    }

    static bool isSmall(size_t size) { return size <= kSmallAllocation; }

    static void eraseCachedLocked(DeviceCache& cache,
                                  const std::shared_ptr<Block>& block) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(block->ptr);
        cache.free_by_addr.erase(address);
        auto& pool = isSmall(block->size) ? cache.small : cache.large;
        auto bucket = pool.find(block->size);
        if (bucket == pool.end()) return;
        auto it = std::find_if(bucket->second.begin(), bucket->second.end(),
                               [&](const auto& candidate) {
                                   return candidate.get() == block.get();
                               });
        if (it != bucket->second.end()) bucket->second.erase(it);
        if (bucket->second.empty()) pool.erase(bucket);
    }

    // Insert a completed block and merge adjacent free blocks from the same
    // allocation stream.  Blocks from different owner streams deliberately
    // remain separate: they cannot be handed to one another without an
    // explicit cross-stream event, matching Torch's per-stream free pools.
    static void insertCachedLocked(DeviceCache& cache,
                                   const std::shared_ptr<Block>& original) {
        auto block = original;
        block->requested_size = 0;
        block->streams.clear();
        const auto same_stream = [&](const std::shared_ptr<Block>& candidate) {
            return candidate->allocation_stream == block->allocation_stream &&
                   candidate->segment == block->segment;
        };

        auto it = cache.free_by_addr.lower_bound(
            reinterpret_cast<uintptr_t>(block->ptr));
        if (it != cache.free_by_addr.begin()) {
            auto prev = std::prev(it)->second;
            const auto prev_end = reinterpret_cast<uintptr_t>(prev->ptr) + prev->size;
            if (prev_end == reinterpret_cast<uintptr_t>(block->ptr) && same_stream(prev)) {
                eraseCachedLocked(cache, prev);
                prev->size += block->size;
                if (block->segment) block->segment->block_count--;
                block = prev;
            }
        }

        it = cache.free_by_addr.lower_bound(
            reinterpret_cast<uintptr_t>(block->ptr));
        if (it != cache.free_by_addr.end()) {
            auto next = it->second;
            const auto block_end = reinterpret_cast<uintptr_t>(block->ptr) + block->size;
            if (block_end == reinterpret_cast<uintptr_t>(next->ptr) && same_stream(next)) {
                eraseCachedLocked(cache, next);
                block->size += next->size;
                if (next->segment) next->segment->block_count--;
            }
        }

        auto& pool = isSmall(block->size) ? cache.small : cache.large;
        pool[block->size].push_back(block);
        cache.free_by_addr[reinterpret_cast<uintptr_t>(block->ptr)] = block;
    }

    std::shared_ptr<Block> takeCachedLocked(int device, size_t rounded,
                                             cudaStream_t allocation_stream) {
        auto& cache = caches_[device];
        auto& pool = isSmall(rounded) ? cache.small : cache.large;
        // Same-stream work is already ordered and can be reused immediately.
        // A block used by another stream is moved to ``pending`` on release
        // and only returns here after all recorded events complete.  Keeping
        // the owner-stream partition prevents an un-fenced owner block from
        // being handed to a different stream, as in Torch's stream-keyed
        // caching pools.
        for (auto it = pool.lower_bound(rounded); it != pool.end(); ++it) {
            auto& bucket = it->second;
            for (auto bucket_it = bucket.begin(); bucket_it != bucket.end(); ++bucket_it) {
                if ((*bucket_it)->allocation_stream != allocation_stream) continue;
                auto block = *bucket_it;
                bucket.erase(bucket_it);
                if (bucket.empty()) pool.erase(it);
                cache.free_by_addr.erase(reinterpret_cast<uintptr_t>(block->ptr));
                if (block->size > rounded) {
                    auto remainder = std::make_shared<Block>();
                    remainder->ptr = static_cast<char*>(block->ptr) + rounded;
                    remainder->size = block->size - rounded;
                    remainder->device = block->device;
                    remainder->segment = block->segment;
                    remainder->offset = block->offset + rounded;
                    remainder->allocation_stream = block->allocation_stream;
                    if (remainder->segment) remainder->segment->block_count++;
                    block->size = rounded;
                    insertCachedLocked(cache, remainder);
                }
                return block;
            }
        }
        return nullptr;
    }

    // Graph-pool counterpart of takeCachedLocked.  No coalescing and no
    // address index: pool blocks are parked per size and reused same-stream
    // only, which is enough because the pool is short-lived and private.
    std::shared_ptr<Block> takeGraphPoolLocked(uint64_t pool_id, size_t rounded,
                                               cudaStream_t allocation_stream) {
        auto pit = pools_.find(pool_id);
        if (pit == pools_.end()) return nullptr;
        GraphPool& pool = pit->second;
        auto& pmap = isSmall(rounded) ? pool.small : pool.large;
        for (auto it = pmap.lower_bound(rounded); it != pmap.end(); ++it) {
            auto& bucket = it->second;
            for (auto bucket_it = bucket.begin(); bucket_it != bucket.end(); ++bucket_it) {
                if ((*bucket_it)->allocation_stream != allocation_stream) continue;
                auto block = *bucket_it;
                bucket.erase(bucket_it);
                if (bucket.empty()) pmap.erase(it);
                if (block->size > rounded) {
                    auto remainder = std::make_shared<Block>();
                    remainder->ptr = static_cast<char*>(block->ptr) + rounded;
                    remainder->size = block->size - rounded;
                    remainder->device = block->device;
                    remainder->segment = block->segment;
                    remainder->offset = block->offset + rounded;
                    remainder->allocation_stream = block->allocation_stream;
                    remainder->pool_id = pool_id;
                    if (remainder->segment) remainder->segment->block_count++;
                    block->size = rounded;
                    auto& remainder_map = isSmall(remainder->size) ? pool.small
                                                                   : pool.large;
                    remainder_map[remainder->size].push_back(std::move(remainder));
                }
                return block;
            }
        }
        return nullptr;
    }

    void activateLocked(const std::shared_ptr<Block>& block,
                        cudaStream_t allocation_stream,
                        size_t requested_size) {
        auto& cache = caches_[block->device];
        block->events.clear();
        block->streams.clear();
        block->allocation_stream = allocation_stream;
        block->requested_size = requested_size;
        block->streams.insert(allocation_stream);
        live_blocks_[block->ptr] = block;
        cache.stats.allocated += requested_size;
        cache.stats.max_allocated = std::max(cache.stats.max_allocated, cache.stats.allocated);
    }

    void reclaimCompletedLocked(int device) {
        auto& cache = caches_[device];
        size_t output = 0;
        for (size_t i = 0; i < cache.pending.size(); ++i) {
            auto& block = cache.pending[i];
            bool ready = true;
            for (cudaEvent_t event : block->events) {
                cudaError_t error = cudaEventQuery(event);
                if (error == cudaErrorNotReady) {
                    (void)cudaGetLastError();
                    ready = false;
                    break;
                }
                if (error != cudaSuccess) {
                    (void)cudaGetLastError();
                    ready = false;
                    break;
                }
            }
            if (ready) {
                for (cudaEvent_t event : block->events) checkCuda(cudaEventDestroy(event), "cudaEventDestroy");
                block->events.clear();
                insertCachedLocked(cache, block);
            } else {
                cache.pending[output++] = block;
            }
        }
        cache.pending.resize(output);
    }

    void emptyCacheDevice(int device) {
        device = normalizeDevice(device);
        {
            // cudaDeviceSynchronize below would abort an open stream capture;
            // skip the release entirely while a graph is being captured.
            std::lock_guard<std::mutex> lock(mutex_);
            if (capture_.pool_id != 0) return;
        }
        CUDAGuard guard(device);
        std::vector<std::shared_ptr<Block>> blocks;
        std::vector<std::tuple<void*, size_t, int>> segments_to_free;
        {
            // Holding the allocator mutex across the synchronization prevents
            // a concurrent deleter from adding a not-yet-complete block to the
            // collection after the synchronization point.
            std::lock_guard<std::mutex> lock(mutex_);
            ensureDevice(device);
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            auto& cache = caches_[device];
            for (auto& [size, bucket] : cache.small) {
                blocks.insert(blocks.end(), bucket.begin(), bucket.end());
            }
            for (auto& [size, bucket] : cache.large) {
                blocks.insert(blocks.end(), bucket.begin(), bucket.end());
            }
            blocks.insert(blocks.end(), cache.pending.begin(), cache.pending.end());
            cache.small.clear();
            cache.large.clear();
            cache.free_by_addr.clear();
            cache.pending.clear();

            // A cached split block is only a slice of its segment.  Return
            // the complete cudaMalloc allocation once every slice is free;
            // keep it reserved when a live slice still exists.
            std::unordered_set<Segment*> touched;
            for (const auto& block : blocks) {
                if (!block->segment) continue;
                touched.insert(block->segment);
                if (block->segment->block_count > 0) block->segment->block_count--;
            }
            for (Segment* segment : touched) {
                if (segment->block_count != 0) continue;
                auto it = segments_.find(segment->ptr);
                if (it == segments_.end()) continue;
                segments_to_free.emplace_back(segment->ptr, segment->size, segment->device);
                cache.stats.reserved -= std::min(cache.stats.reserved, segment->size);
                segments_.erase(it);
            }
        }

        for (const auto& block : blocks) {
            for (cudaEvent_t event : block->events) (void)cudaEventDestroy(event);
            block->events.clear();
        }
        for (const auto& segment : segments_to_free) {
            CUDAGuard segment_guard(std::get<2>(segment));
            checkCuda(cudaFree(std::get<0>(segment)), "cudaFree");
        }
    }

    // mutable — isCapturing() is const and must be callable while any thread
    // holds the allocator lock.
    mutable std::mutex mutex_;
    std::vector<DeviceCache> caches_;
    std::unordered_map<void*, std::unique_ptr<Segment>> segments_;
    std::unordered_map<void*, std::shared_ptr<Block>> live_blocks_;
    ActiveCapture capture_;
    std::unordered_map<uint64_t, GraphPool> pools_;
    uint64_t next_pool_id_ = 1;
};

class PinnedAllocatorState {
public:
    static PinnedAllocatorState& instance() {
        // See AllocatorState::instance(): pinned buffers can also outlive the
        // Python module during interpreter shutdown.
        static auto* state = new PinnedAllocatorState();
        return *state;
    }

    std::shared_ptr<PinnedBlock> allocate(size_t nbytes) {
        const size_t rounded = roundPinned(nbytes);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            reclaimCompletedLocked();
            auto it = cache_.lower_bound(rounded);
            if (it != cache_.end()) {
                auto block = it->second.back();
                it->second.pop_back();
                if (it->second.empty()) cache_.erase(it);
                activateLocked(block);
                return block;
            }
        }

        void* ptr = nullptr;
        checkCuda(cudaHostAlloc(&ptr, rounded, cudaHostAllocPortable), "cudaHostAlloc");
        auto block = std::make_shared<PinnedBlock>();
        block->ptr = ptr;
        block->size = rounded;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            activateLocked(block);
        }
        return block;
    }

    void release(const std::shared_ptr<PinnedBlock>& block) noexcept {
        if (!block || !block->ptr) return;
        try {
            std::vector<PinnedStream> streams;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = live_.find(block->ptr);
                if (it == live_.end()) return;
                live_.erase(it);
                streams.swap(block->streams);
            }

            bool event_failure = false;
            for (const auto& recorded : streams) {
                CUDAGuard guard(recorded.device);
                cudaEvent_t event = nullptr;
                cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                if (error == cudaSuccess) error = cudaEventRecord(event, recorded.stream);
                if (error != cudaSuccess) {
                    if (event) (void)cudaEventDestroy(event);
                    (void)cudaGetLastError();
                    event_failure = true;
                    break;
                }
                block->events.push_back({event, recorded.device});
            }

            if (event_failure) {
                for (const auto& event : block->events) {
                    CUDAGuard guard(event.device);
                    (void)cudaEventDestroy(event.event);
                }
                block->events.clear();
                std::unordered_set<int> devices;
                for (const auto& recorded : streams) devices.insert(recorded.device);
                for (int device : devices) {
                    CUDAGuard guard(device);
                    if (cudaDeviceSynchronize() != cudaSuccess) return;
                }
            }

            std::lock_guard<std::mutex> lock(mutex_);
            if (block->events.empty()) {
                cache_[block->size].push_back(block);
            } else {
                pending_.push_back(block);
            }
        } catch (...) {
            // A shutdown-time CUDA failure may leak one host block, but it
            // must never cause early reuse or throw from a DataPtr deleter.
        }
    }

    void record(void* ptr, const CUDAStream& stream) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = live_.find(ptr);
        if (it == live_.end()) return;
        auto& streams = it->second->streams;
        auto duplicate = std::find_if(streams.begin(), streams.end(), [&](const auto& item) {
            return item.device == stream.device_index() && item.stream == stream.stream();
        });
        if (duplicate == streams.end()) {
            streams.push_back({stream.stream(), stream.device_index()});
        }
    }

private:
    static size_t roundPinned(size_t nbytes) {
        constexpr size_t quantum = 512;
        return ((nbytes + quantum - 1) / quantum) * quantum;
    }

    void activateLocked(const std::shared_ptr<PinnedBlock>& block) {
        block->streams.clear();
        block->events.clear();
        live_[block->ptr] = block;
    }

    void reclaimCompletedLocked() {
        size_t output = 0;
        for (size_t i = 0; i < pending_.size(); ++i) {
            auto& block = pending_[i];
            bool ready = true;
            for (const auto& event : block->events) {
                CUDAGuard guard(event.device);
                cudaError_t error = cudaEventQuery(event.event);
                if (error == cudaErrorNotReady) {
                    (void)cudaGetLastError();
                    ready = false;
                    break;
                }
                if (error != cudaSuccess) {
                    (void)cudaGetLastError();
                    ready = false;
                    break;
                }
            }
            if (ready) {
                for (const auto& event : block->events) {
                    CUDAGuard guard(event.device);
                    checkCuda(cudaEventDestroy(event.event), "cudaEventDestroy");
                }
                block->events.clear();
                cache_[block->size].push_back(block);
            } else {
                pending_[output++] = block;
            }
        }
        pending_.resize(output);
    }

    std::mutex mutex_;
    std::map<size_t, std::vector<std::shared_ptr<PinnedBlock>>> cache_;
    std::vector<std::shared_ptr<PinnedBlock>> pending_;
    std::unordered_map<void*, std::shared_ptr<PinnedBlock>> live_;

public:
    // Raw-pointer entry point for DataPtr's function-pointer deleter.
    void releaseRaw(PinnedBlock* raw) noexcept {
        if (!raw) return;
        std::shared_ptr<PinnedBlock> block;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = live_.find(raw->ptr);
            if (it == live_.end()) return;
            block = it->second;
        }
        release(block);
    }
};

} // namespace

size_t memory_allocated(int device) {
    return AllocatorState::instance().allocated(device);
}

size_t memory_reserved(int device) {
    return AllocatorState::instance().reserved(device);
}

size_t max_memory_allocated(int device) {
    return AllocatorState::instance().maxAllocated(device);
}

size_t max_memory_reserved(int device) {
    return AllocatorState::instance().maxReserved(device);
}

void reset_max_memory_allocated(int device) {
    AllocatorState::instance().resetPeaks(device);
}

void reset_peak_memory_stats(int device) {
    AllocatorState::instance().resetPeaks(device);
}

void empty_cache() {
    AllocatorState::instance().emptyCache();
}

void recordStream(void* base_ptr, const CUDAStream& stream) {
    AllocatorState::instance().record(base_ptr, stream);
}

void recordPinnedStream(void* base_ptr, const CUDAStream& stream) {
    PinnedAllocatorState::instance().record(base_ptr, stream);
}

bool isCapturing() {
    return AllocatorState::instance().isCapturing();
}

uint64_t beginAllocateToPool(int device, const CUDAStream& stream) {
    device = device < 0 ? currentDevice() : device;
    return AllocatorState::instance().beginGraphCapture(device, stream);
}

void endAllocateToPool(uint64_t pool_id) {
    AllocatorState::instance().endGraphCapture(pool_id);
}

void releasePool(uint64_t pool_id) {
    AllocatorState::instance().releaseGraphPool(pool_id);
}

} // namespace tensorplay::cuda

namespace {

// Capture-less deleter for device blocks: recovers the owning shared_ptr from
// the allocator's live-block map (the state singleton is deliberately leaked).
void cudaReleaseBlockCtx(void* ctx) noexcept {
    if (!ctx) return;
    cuda::AllocatorState::instance().releaseRaw(static_cast<cuda::Block*>(ctx));
}



class CUDAAllocator : public Allocator {
public:
    DataPtr allocate(size_t nbytes) const override {
        return allocate(nbytes, Device(DeviceType::CUDA, cuda::currentDevice()));
    }

    DataPtr allocate(size_t nbytes, const Device& requested_device) const override {
        if (!requested_device.is_cuda()) {
            TP_THROW(ValueError, "CUDAAllocator received a non-CUDA device");
        }
        const int device = requested_device.index() < 0
            ? cuda::currentDevice()
            : static_cast<int>(requested_device.index());
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, Device(DeviceType::CUDA, device));
        }
        auto block = cuda::AllocatorState::instance().allocate(nbytes, device);
        return DataPtr(
            block->ptr,
            block.get(),
            &cudaReleaseBlockCtx,
            Device(DeviceType::CUDA, device));
    }
};

// Capture-less deleter: recovers the owning shared_ptr from the allocator's
// live-block map (the state singleton is deliberately leaked).
void releasePinnedBlockCtx(void* ctx) noexcept {
    if (!ctx) return;
    cuda::PinnedAllocatorState::instance().releaseRaw(static_cast<cuda::PinnedBlock*>(ctx));
}

class PinnedMemoryAllocator : public Allocator {
public:
    DataPtr allocate(size_t nbytes) const override {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, Device(DeviceType::CPU));
        }
        auto block = cuda::PinnedAllocatorState::instance().allocate(nbytes);
        return DataPtr(
            block->ptr,
            block.get(),
            &releasePinnedBlockCtx,
            Device(DeviceType::CPU));
    }
};

} // namespace

Allocator* getCUDAAllocator() {
    static auto* allocator = new CUDAAllocator();
    return allocator;
}

Allocator* getPinnedMemoryAllocator() {
    static auto* allocator = new PinnedMemoryAllocator();
    return allocator;
}

} // namespace tensorplay
