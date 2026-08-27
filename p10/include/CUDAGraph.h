#pragma once

#include "Macros.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#endif

namespace tensorplay {

class Tensor;

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
//
// ``requested_pool_id != 0`` routes into an existing pool instead of creating
// a fresh one, so several graphs can share one pool (torch's ``pool=`` /
// ``graph_pool_handle()``).  The pool is created on first use when unknown.

P10_API uint64_t beginAllocateToPool(int device, const CUDAStream& stream,
                                     uint64_t requested_pool_id = 0);
P10_API void endAllocateToPool(uint64_t pool_id);
// Extra routing targets for conditional-node child streams: the body of an
// if/while node captures on its own stream while sharing the parent's pool.
P10_API void routeStreamToPool(int device, const CUDAStream& stream,
                               uint64_t pool_id);
P10_API void unrouteStreamFromPool(const CUDAStream& stream);
P10_API uint64_t graph_pool_handle();
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

// Capture-safety mode passed to cudaStreamBeginCapture, mirroring torch's
// capture_error_mode ("global" | "thread_local" | "relaxed").
enum class CaptureMode {
    Global,
    ThreadLocal,
    Relaxed,
};

// Parses torch's capture_error_mode strings ("global", "thread_local",
// "relaxed"); throws ValueError on anything else.
P10_API CaptureMode captureModeFromName(const std::string& name);

// One captured CUDA graph: capture once, instantiate at capture_end, replay
// against static buffers.  Mirrors at::cuda::CUDAGraph:
//
//   CUDAGraph g;
//   g.capture_begin(pool_id);          // 0 = fresh private pool
//   /* work on the current stream */
//   g.capture_end();                   // instantiates eagerly
//   g.replay();                        // cached exec, no map lookups/locks
//
// The executable pointer is held directly in this object, so replay pays no
// registry mutex or lookup - just the RNG prologue (when the capture used
// random ops) and cudaGraphLaunch itself.
class P10_API CUDAGraph : public std::enable_shared_from_this<CUDAGraph> {
public:
    CUDAGraph();
    ~CUDAGraph();

    CUDAGraph(const CUDAGraph&) = delete;
    CUDAGraph& operator=(const CUDAGraph&) = delete;

    // Starts capture.  ``pool_id`` selects the allocator pool allocations are
    // routed to: 0 creates a fresh private pool, any id previously returned by
    // beginAllocateToPool / graph_pool_handle shares that pool with other
    // graphs.  ``stream`` overrides the dedicated per-device side stream.
    // ``mode`` maps onto torch's capture_error_mode.
    void capture_begin(uint64_t pool_id = 0,
                       CaptureMode mode = CaptureMode::Global,
                       const CUDAStream& stream = CUDAStream::undefined());
    // Ends the capture window and compiles the template into an executable
    // immediately (the expensive driver call is paid here, not on first
    // replay).  The template graph is retained for debug_dump().
    void capture_end();
    // No-op after capture_end; kept so late callers stay correct.
    void instantiate();
    bool has_graph_exec() const { return exec_ != nullptr; }
    // Enqueues the executable on the calling thread's current stream.  Does
    // not synchronize; outputs are refreshed because kernels rewrite the
    // exact virtual addresses baked at capture time.
    void replay();
    // Overload for hot loops pinned to one stream: skips the current-stream
    // query entirely.  The stream must belong to this graph's device.
    void replay(const CUDAStream& stream);
    // Destroys the executable and drops this graph's reference to its pool;
    // pool segments are freed when the last sharing graph resets and no
    // tensors allocated during capture remain alive.
    void reset();
    uint64_t pool_id() const { return pool_id_; }
    int device() const { return device_; }

    // Dump the graph as a DOT file (cudaGraphDebugDotPrint).  Call
    // enable_debug_mode() before capture_begin to retain the template for a
    // richer dump.
    void enable_debug_mode() { debug_ = true; }
    void debug_dump(const std::string& path);

    // --- conditional nodes (torch's if/while graph capture, CUDA >= 12.4) ---
    //
    // Inside an open capture, splits the remaining work into a driver-level
    // conditional node: ``pred`` (device Bool scalar) is sampled by a
    // captured kernel at replay time and decides whether the body captured
    // between begin_capture_to_{if,while}_node / end_capture_to_conditional_node
    // runs this iteration.
    void begin_capture_to_if_node(const Tensor& scalar_pred);
    void begin_capture_to_while_node(const Tensor& scalar_pred);
    // Refreshes the predicate consumed by the innermost open conditional
    // node (nested conditionals each own a handle).
    void set_conditional_handle_for_current_node(const Tensor& scalar_pred);
    void end_capture_to_conditional_node();

    cudaGraphExec_t raw_exec() const { return exec_; }

private:
    void instantiateLocked();
    bool hasPendingResources() const noexcept {
        return exec_ != nullptr || graph_ != nullptr || owns_pool_ref_ ||
               rng_state_id_ != 0 || !cond_child_streams_.empty();
    }
    void begin_capture_to_conditional_node(const Tensor& scalar_pred,
                                           int conditional_type);

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    uint64_t pool_id_ = 0;
    int device_ = -1;
    // RNG state registered for this graph's capture window (0 = none).
    uint64_t rng_state_id_ = 0;
    // Counter total consumed by RNG ops during capture; advanced into the
    // generator on every replay by rng_replay_prologue.
    uint64_t rng_wholegraph_increment_ = 0;
    // Set at capture_begin; reset() must release the pool even when capture
    // failed before capture_end (otherwise a failed capture would leak it).
    bool owns_pool_ref_ = false;
    bool debug_ = false;
    // Stream capture runs on (resolved side stream or caller-supplied), kept
    // so capture_end can close the window without re-entering the registry.
    CUDAStream capture_stream_ = CUDAStream::undefined();
    // Caller's stream, restored when the capture window closes.
    CUDAStream previous_stream_ = CUDAStream::undefined();
    // cudaStreamCaptureMode value passed at begin (needed again by
    // conditional-node child captures); stored as int to keep this header
    // free of CUDA version guards.
    int capture_mode_ = 0;
    // Open conditional nodes, outermost first.  Handles are
    // cudaGraphConditionalHandle (an unsigned long long) kept type-erased so
    // builds with CUDA < 12.4 still compile; entry points throw on them.
    std::vector<uint64_t> cond_handles_;
    std::vector<cudaStream_t> cond_child_streams_;
    // Ended conditional-body streams, kept alive until reset(): captured
    // tensors carry cross-stream fences against them and recording an event
    // on a destroyed stream faults inside libcuda.
    std::vector<cudaStream_t> cond_retired_streams_;
};

// The dedicated side stream capture runs on, created once per device and
// reused across captures (cuBLAS workspaces and other lazy per-stream state
// must see warmup and capture on the same stream).
P10_API CUDAStream captureStream(int device_index = -1);

// True while any graph capture scope is open in this process.
P10_API bool isCapturing();

// Captured-kernel helper (defined in CUDAGraphKernels.cu): samples the Bool
// device scalar into the conditional handle at replay time.  No-op (throws)
// on CUDA < 12.4.
P10_API void launchSetConditionalHandle(uint64_t handle, const void* pred_bool,
                                        cudaStream_t stream);
// True when the CUDA runtime supports conditional graph nodes.
P10_API bool conditionalNodesSupported();

} // namespace graph

#endif // USE_CUDA

} // namespace cuda
} // namespace tensorplay
