#pragma once

#include "Macros.h"

#include <cstdint>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#endif

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDA

// --- caching-allocator capture routing -------------------------------------
//
// Addresses baked into a captured graph must stay exclusive for the whole
// life of the instantiated executable, otherwise a replay writes into memory
// the allocator handed to unrelated tensors.  While a capture scope is open,
// allocations issued on the capturing stream are routed into graph-private
// pool ``pool_id``; blocks in that pool are never recycled outside the pool
// until :cpp:func:`releasePool`, mirroring c10's beginAllocateToPool /
// endAllocateToPool / releasePool triple.

P10_API uint64_t beginAllocateToPool(int device, const CUDAStream& stream);
P10_API void endAllocateToPool(uint64_t pool_id);
// Frees every segment owned by the pool.  Throws when tensors allocated from
// the pool are still alive; destroy the graph executable and drop all static
// input/output references first.
P10_API void releasePool(uint64_t pool_id);

// True while a capture scope is open.  Allocator paths that would issue
// synchronizing CUDA calls (event queries, device synchronize) must no-op to
// keep the capture alive.
P10_API bool isCapturing();

// --- graph capture / execution ---------------------------------------------

namespace graph {

// The dedicated side stream capture runs on, created once per device and
// reused across captures (cuBLAS workspaces and other lazy per-stream state
// must see warmup and capture on the same stream).  Warmup should run with
// this stream current; beginCapture/endCapture switch to/from it themselves.
P10_API CUDAStream captureStream(int device_index = -1);
// Switches the calling thread's current stream to the capture side stream
// (the legacy default stream cannot capture) and starts stream capture on it.
P10_API void beginCapture();
// Ends capture and registers the captured template.  Returns an opaque
// handle usable with instantiate/launch/destroy; restores the previous
// current stream.
P10_API uint64_t endCapture();
// Compiles the captured template into an executable graph.  The handle stays
// valid and now refers to the executable; the template is released.
P10_API uint64_t instantiate(uint64_t handle);
// Enqueues the executable on the calling thread's current stream.  Does not
// synchronize; outputs are refreshed because kernels rewrite the exact
// virtual addresses baked at capture time.
P10_API void launch(uint64_t handle);
// Destroys the executable/template behind ``handle`` and releases its
// allocator pool.  All tensors allocated during the capture must be dead by
// this point.
P10_API void destroy(uint64_t handle);

} // namespace graph

#endif // USE_CUDA

} // namespace cuda
} // namespace tensorplay
