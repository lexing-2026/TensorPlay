#include "Allocator.h"
#include "CUDAGraph.h"
#include "CUDARuntime.h"
#include "Device.h"
#include "Exception.h"
#include "Profiler.h"

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
// into the graph-private pool (CUDAGraph.h note).  Several scopes may be open
// concurrently (different devices / threads capture in parallel); routing is
// a tiny linear scan.
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
    // Address-ordered index of every parked block, enabling split-remainder
    // coalescing so mixed-size captures sharing one pool cannot shred it into
    // unusable slivers over time.
    std::map<uintptr_t, std::shared_ptr<Block>> free_by_addr;
    // Segments cudaMalloc'd while this pool was the routing target.  Their
    // backing memory is returned to CUDA only by releasePool().
    std::vector<Segment*> segments;
    // Set when the last referencing graph reset while captured tensors were
    // still alive.  Segments are reclaimed lazily by the free path (or
    // empty_cache) once those tensors die, mirroring torch's refcounted
    // private pools.
    bool pending_release = false;
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
            for (const auto& capture : active_captures_) {
                if (capture.device == device &&
                    capture.stream == allocation_stream) {
                    // Graph-capture allocation: reuse is restricted to the
                    // private pool so replay-visible addresses stay exclusive.
                    pool_target = capture.pool_id;
                    break;
                }
            }
            if (pool_target == 0) {
                // Event queries are skipped while any capture is open: a
                // cudaEventQuery on a pending block cannot fail the capture,
                // but the queries themselves may target events recorded on
                // unrelated streams mid-capture; deferring them is free.
                if (active_captures_.empty()) reclaimCompletedLocked(device);
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
        const bool relax = !active_captures_.empty();
        if (relax) {
            cudaStreamCaptureMode relaxed = cudaStreamCaptureModeRelaxed;
            (void)cudaThreadExchangeStreamCaptureMode(&relaxed);
            previous_mode = relaxed;
        }
        void* ptr = nullptr;
        cudaError_t error = cudaMalloc(&ptr, rounded);
        bool flushed = false;
        if (error == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            // Emergency defrag ladder (still inside the relaxed-capture-mode
            // window): first fence-drain this device's event-pending blocks
            // so their bytes become reusable immediately, then drop the whole
            // cache.  Only reachable on the OOM path; latency irrelevant.
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ensureDevice(device);
                drainPendingSyncLocked(device);
            }
            emptyCacheDevice(device);
            error = cudaMalloc(&ptr, rounded);
            flushed = true;
        }
        if (relax) {
            cudaThreadExchangeStreamCaptureMode(&previous_mode);
        }
        if (error != cudaSuccess) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            (void)cudaMemGetInfo(&free_bytes, &total_bytes);
            size_t live_bytes = 0;
            size_t reserved_bytes = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (device < static_cast<int>(caches_.size())) {
                    live_bytes = caches_[device].stats.allocated;
                    reserved_bytes = caches_[device].stats.reserved;
                }
            }
            std::ostringstream message;
            message << "CUDA out of memory on cuda:" << device
                    << ". Tried to allocate " << rounded << " bytes; "
                    << free_bytes << " bytes free of " << total_bytes << " total"
                    << (flushed ? " (after fence-drain + cache flush)" : "")
                    << "; allocator reserved=" << reserved_bytes
                    << ", allocated=" << live_bytes;
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
                // Allocator-level memory capture: exactly-once per block
                // (only this erase site can free a live block).  Runs under
                // mutex_ but the recorder takes its own lock afterwards and
                // never reaches back into the allocator.
                prof::mem_record_free(block->ptr,
                                      static_cast<int64_t>(block->requested_size),
                                      /*cuda=*/true,
                                      static_cast<int32_t>(block->device),
                                      reinterpret_cast<int64_t>(allocation_stream));
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
                auto pit = pools_.find(block->pool_id);
                if (pit == pools_.end()) return;
                insertGraphCachedLocked(pit->second, block);
                if (pit->second.pending_release) {
                    // Last static tensor of a reset graph just died.
                    tryReleasePendingPoolLocked(block->pool_id);
                }
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

    // Reserves a fresh pool id without creating the pool; capture-time routing
    // creates it lazily (mirrors c10's graph_pool_handle).
    uint64_t newGraphPoolId() {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_pool_id_++;
    }

    uint64_t beginGraphCapture(int device, const CUDAStream& stream,
                               uint64_t requested_pool_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& capture : active_captures_) {
            if (capture.pool_id == requested_pool_id && requested_pool_id != 0) {
                TP_THROW(RuntimeError,
                         "CUDA graph memory pool is already the routing "
                         "target of a concurrent capture");
            }
        }
        uint64_t pool_id = requested_pool_id;
        if (pool_id == 0) {
            pool_id = next_pool_id_++;
            pools_[pool_id] = GraphPool{};
        } else if (pools_.find(pool_id) == pools_.end()) {
            // Id from graph_pool_handle() whose pool was never used yet, or
            // already released: create it lazily so sharing still works.
            pools_[pool_id] = GraphPool{};
        }
        active_captures_.push_back(ActiveCapture{pool_id, device, stream.stream()});
        return pool_id;
    }

    void endGraphCapture(uint64_t pool_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_captures_.erase(
            std::remove_if(active_captures_.begin(), active_captures_.end(),
                           [&](const ActiveCapture& capture) {
                               return capture.pool_id == pool_id;
                           }),
            active_captures_.end());
    }

    // Registers an extra allocation-routing target (device, stream) for an
    // already-open capture: conditional-node bodies capture on their own
    // child stream while sharing the parent's graph pool.
    void routeStreamToGraphPool(int device, const CUDAStream& stream,
                                uint64_t pool_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_captures_.push_back(ActiveCapture{pool_id, device, stream.stream()});
    }

    void unrouteStreamFromGraphPool(const CUDAStream& stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(active_captures_.rbegin(), active_captures_.rend(),
                               [&](const ActiveCapture& capture) {
                                   return capture.stream == stream.stream();
                               });
        if (it != active_captures_.rend()) {
            active_captures_.erase(std::next(it).base());
        }
    }

    bool isCapturing() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !active_captures_.empty();
    }

    // Live-tensor reference count for a graph pool: every metadata ref that
    // is not a block already parked on the pool's free lists.
    static size_t poolLiveRefsLocked(const GraphPool& pool) {
        size_t parked = 0;
        for (const auto& [size, bucket] : pool.small) parked += bucket.size();
        for (const auto& [size, bucket] : pool.large) parked += bucket.size();
        size_t total_refs = 0;
        for (const Segment* segment : pool.segments) {
            total_refs += segment->block_count;
        }
        return total_refs > parked ? total_refs - parked : 0;
    }

    // Complete a pending release once no captured tensor remains alive.
    // Caller must hold ``mutex_``; returns true when the pool was freed.
    bool tryReleasePendingPoolLocked(uint64_t pool_id) {
        auto it = pools_.find(pool_id);
        if (it == pools_.end()) return false;
        GraphPool& pool = it->second;
        if (!pool.pending_release || poolLiveRefsLocked(pool) != 0) {
            return false;
        }
        std::vector<std::tuple<void*, size_t, int>> segments_to_free;
        for (Segment* segment : pool.segments) {
            segments_to_free.emplace_back(segment->ptr, segment->size,
                                          segment->device);
            auto& stats = caches_[segment->device].stats;
            stats.reserved -= std::min(stats.reserved, segment->size);
            segments_.erase(segment->ptr);
        }
        pools_.erase(it);
        for (const auto& segment : segments_to_free) {
            CUDAGuard segment_guard(std::get<2>(segment));
            checkCuda(cudaFree(std::get<0>(segment)), "cudaFree");
        }
        return true;
    }

    void releaseGraphPool(uint64_t pool_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pools_.find(pool_id);
        if (it == pools_.end()) return;
        for (const auto& capture : active_captures_) {
            if (capture.pool_id == pool_id) {
                TP_THROW(RuntimeError,
                         "cannot release a CUDA graph memory pool while "
                         "its capture is open");
            }
        }
        if (poolLiveRefsLocked(it->second) != 0) {
            // Static input/output tensors of a reset graph are still alive;
            // defer segment reclamation to their deallocation instead of
            // failing the reset (torch's private pools are refcounted).
            it->second.pending_release = true;
            return;
        }
        tryReleasePendingPoolLocked(pool_id);
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

    // -- graph-pool free-list maintenance (address-ordered coalescing) ------

    static void eraseGraphCachedLocked(GraphPool& pool,
                                       const std::shared_ptr<Block>& block) {
        pool.free_by_addr.erase(reinterpret_cast<uintptr_t>(block->ptr));
        auto& pmap = isSmall(block->size) ? pool.small : pool.large;
        auto bucket = pmap.find(block->size);
        if (bucket == pmap.end()) return;
        auto it = std::find_if(bucket->second.begin(), bucket->second.end(),
                               [&](const auto& candidate) {
                                   return candidate.get() == block.get();
                               });
        if (it != bucket->second.end()) bucket->second.erase(it);
        if (bucket->second.empty()) pmap.erase(bucket);
    }

    // Park a freed pool block, merging physically adjacent parked slices from
    // the same segment and stream.  Without this, alternating capture sizes
    // on one shared pool progressively fragment into blocks too small for any
    // later node.
    static void insertGraphCachedLocked(GraphPool& pool,
                                        const std::shared_ptr<Block>& original) {
        auto block = original;
        block->requested_size = 0;
        block->streams.clear();
        const auto same_slice = [&](const std::shared_ptr<Block>& candidate) {
            return candidate->allocation_stream == block->allocation_stream &&
                   candidate->segment == block->segment;
        };

        auto it = pool.free_by_addr.lower_bound(
            reinterpret_cast<uintptr_t>(block->ptr));
        if (it != pool.free_by_addr.begin()) {
            auto prev = std::prev(it)->second;
            const auto prev_end =
                reinterpret_cast<uintptr_t>(prev->ptr) + prev->size;
            if (prev_end == reinterpret_cast<uintptr_t>(block->ptr) &&
                same_slice(prev)) {
                eraseGraphCachedLocked(pool, prev);
                prev->size += block->size;
                if (block->segment) block->segment->block_count--;
                block = prev;
            }
        }

        it = pool.free_by_addr.lower_bound(
            reinterpret_cast<uintptr_t>(block->ptr));
        if (it != pool.free_by_addr.end()) {
            auto next = it->second;
            const auto block_end =
                reinterpret_cast<uintptr_t>(block->ptr) + block->size;
            if (block_end == reinterpret_cast<uintptr_t>(next->ptr) &&
                same_slice(next)) {
                eraseGraphCachedLocked(pool, next);
                block->size += next->size;
                if (next->segment) next->segment->block_count--;
            }
        }

        auto& pmap = isSmall(block->size) ? pool.small : pool.large;
        pmap[block->size].push_back(block);
        pool.free_by_addr[reinterpret_cast<uintptr_t>(block->ptr)] = block;
    }


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
                    insertGraphCachedLocked(pool, remainder);
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

    // Emergency OOM path only.  Blocks until every cross-stream fence of this
    // device's pending blocks completes, then returns those blocks to their
    // size caches so the next allocation attempt can reuse the bytes.
    // Caller must hold mutex_ (blocking syncs under the lock are acceptable
    // here: the process is already failing an allocation).
    void drainPendingSyncLocked(int device) {
        auto& cache = caches_[device];
        if (cache.pending.empty()) return;
        std::vector<std::shared_ptr<Block>> drained;
        drained.reserve(cache.pending.size());
        for (auto& block : cache.pending) {
            for (cudaEvent_t event : block->events) {
                (void)cudaEventSynchronize(event);
                (void)cudaGetLastError();
            }
            for (cudaEvent_t event : block->events) (void)cudaEventDestroy(event);
            block->events.clear();
            drained.push_back(std::move(block));
        }
        cache.pending.clear();
        for (auto& block : drained) insertCachedLocked(cache, block);
    }

    // Diagnostic snapshot for memory_stats(): free-block/segment accounting
    // that quantifies fragmentation (reserved >> allocated with a small
    // largest-free-block means the cache is shredded).
    std::unordered_map<std::string, uint64_t> collectStatsLocked(int device) {
        ensureDevice(device);
        auto& cache = caches_[device];
        std::unordered_map<std::string, uint64_t> out;
        out["allocated"] = cache.stats.allocated;
        out["reserved"] = cache.stats.reserved;
        out["max_allocated"] = cache.stats.max_allocated;
        out["max_reserved"] = cache.stats.max_reserved;
        out["pending_blocks"] = cache.pending.size();

        uint64_t segments = 0;
        for (const auto& [ptr, segment] : segments_) {
            if (segment->device == device) ++segments;
        }
        out["segments"] = segments;

        uint64_t free_blocks = 0;
        uint64_t free_bytes = 0;
        uint64_t largest_free = 0;
        uint64_t pending_bytes = 0;
        for (const auto* pmap : {&cache.small, &cache.large}) {
            for (const auto& [size, bucket] : *pmap) {
                free_blocks += bucket.size();
                free_bytes += static_cast<uint64_t>(size) * bucket.size();
                if (size > largest_free) largest_free = size;
            }
        }
        for (const auto& block : cache.pending) {
            ++free_blocks;
            pending_bytes += block->size;
        }
        out["pending_bytes"] = pending_bytes;
        out["free_blocks"] = free_blocks;
        out["free_bytes"] = free_bytes + pending_bytes;
        out["largest_free_block"] = largest_free;
        // Reserved bytes not backed by live tensors; high values with a low
        // largest_free_block indicate address-space fragmentation.
        const uint64_t reserved_total = cache.stats.reserved;
        const uint64_t inactive = reserved_total > cache.stats.allocated
                                      ? reserved_total - cache.stats.allocated
                                      : 0;
        out["inactive_split_bytes"] = inactive > free_bytes ? free_bytes : inactive;
        return out;
    }

    friend std::unordered_map<std::string, uint64_t> tensorplay::cuda::
        memory_stats(int device);

    std::unordered_map<std::string, uint64_t> memoryStatsSnapshot(int device) {
        device = normalizeDevice(device);
        std::lock_guard<std::mutex> lock(mutex_);
        auto out = collectStatsLocked(device);
        out["graph_pools"] = pools_.size();
        out["capturing"] = active_captures_.empty() ? 0 : 1;
        out["active_captures"] = active_captures_.size();
        return out;
    }

    void emptyCacheDevice(int device) {
        device = normalizeDevice(device);
        {
            // cudaDeviceSynchronize below would abort an open stream capture;
            // skip the release entirely while any graph is being captured.
            std::lock_guard<std::mutex> lock(mutex_);
            if (!active_captures_.empty()) return;
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
    // Open capture routing scopes (concurrent multi-device captures each
    // contribute one entry; conditional-node child streams add more).
    std::vector<ActiveCapture> active_captures_;
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

uint64_t graph_pool_handle() {
    return AllocatorState::instance().newGraphPoolId();
}

uint64_t beginAllocateToPool(int device, const CUDAStream& stream,
                             uint64_t requested_pool_id) {
    device = device < 0 ? currentDevice() : device;
    return AllocatorState::instance().beginGraphCapture(device, stream,
                                                        requested_pool_id);
}

void endAllocateToPool(uint64_t pool_id) {
    AllocatorState::instance().endGraphCapture(pool_id);
}

void routeStreamToPool(int device, const CUDAStream& stream, uint64_t pool_id) {
    AllocatorState::instance().routeStreamToGraphPool(device, stream, pool_id);
}

void unrouteStreamFromPool(const CUDAStream& stream) {
    AllocatorState::instance().unrouteStreamFromGraphPool(stream);
}

void releasePool(uint64_t pool_id) {
    AllocatorState::instance().releaseGraphPool(pool_id);
}

std::unordered_map<std::string, uint64_t> memory_stats(int device) {
    return AllocatorState::instance().memoryStatsSnapshot(device);
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
        // Allocator-level memory capture (profile_memory sessions).  The
        // allocation stream groups the event with the op that produced it.
        prof::mem_record_alloc(
            block->ptr, static_cast<int64_t>(nbytes), /*cuda=*/true,
            static_cast<int32_t>(device),
            reinterpret_cast<int64_t>(block->allocation_stream));
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
