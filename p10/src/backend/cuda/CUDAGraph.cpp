#include "CUDAGraph.h"

#ifdef USE_CUDA
#include "CUDAContext.h"
#include "CUDAGenerator.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Tensor.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace tensorplay {
namespace cuda {

// Capture/handle bookkeeping lives here; the graph-private memory pools and
// their routing live inside the caching allocator (CUDAAllocator.cpp), which
// owns the segment/block types they refer to.  Graph-safe RNG state lives in
// CUDAGenerator.cpp: registration happens at capture_begin, kernels read the
// [seed, offset] device buffer through PhiloxCudaState, and every replay
// refreshes it (rng_replay_prologue) so each replay draws fresh randoms.

namespace graph {

namespace {

struct GraphState {
    static GraphState& instance() {
        // Deliberately leaked: graph objects may be destroyed during
        // interpreter teardown after static destruction has started.
        static auto* state = new GraphState();
        return *state;
    }

    std::mutex mutex;
    // Live capture sessions keyed by thread: captures on different devices
    // run concurrently from different threads (each lands on its own side
    // stream); a thread may only host one capture at a time.
    std::unordered_map<std::thread::id, std::shared_ptr<CUDAGraph>> capturing;

    // Pool refcounts: several graphs may share one pool via
    // capture_begin(pool).  Segments are freed only when the last referencing
    // graph resets, so an early reset can never release addresses another
    // executable still bakes.
    std::unordered_map<uint64_t, int> pool_refs;

    // One dedicated side stream per device, reused across captures so lazy
    // per-stream state (cuBLAS workspaces) sees warmup and capture equally.
    std::unordered_map<int, CUDAStream> side_streams;

    // Drops one reference for ``pool`` and frees its segments when it was the
    // last one.  Caller must hold ``mutex``.
    void unrefPoolLocked(uint64_t pool_id, bool release_segments) {
        auto rit = pool_refs.find(pool_id);
        if (rit == pool_refs.end()) return;
        if (--rit->second <= 0) {
            pool_refs.erase(rit);
            if (release_segments) releasePool(pool_id);
        }
    }
};

} // namespace

CaptureMode captureModeFromName(const std::string& name) {
    if (name == "global") return CaptureMode::Global;
    if (name == "thread_local") return CaptureMode::ThreadLocal;
    if (name == "relaxed") return CaptureMode::Relaxed;
    TP_THROW(ValueError,
             "capture_error_mode must be one of 'global', 'thread_local' or "
             "'relaxed'; got '" + name + "'");
}

bool isCapturing() {
    GraphState& state = GraphState::instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    return !state.capturing.empty();
}

CUDAStream captureStream(int device_index) {
    const int device = device_index < 0 ? currentDevice() : device_index;
    GraphState& state = GraphState::instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.side_streams.find(device);
    if (it == state.side_streams.end()) {
        it = state.side_streams.emplace(device, getStreamFromPool(0, device)).first;
    }
    return it->second;
}

CUDAGraph::CUDAGraph() = default;

CUDAGraph::~CUDAGraph() {
    try {
        reset();
    } catch (...) {
        // Destructors must not throw (e.g. static tensors from the capture
        // still alive during interpreter shutdown).  Drop the pool reference
        // silently; the segments leak rather than risk use-after-free.
        if (owns_pool_ref_) {
            GraphState& state = GraphState::instance();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.unrefPoolLocked(pool_id_, /*release_segments=*/false);
            owns_pool_ref_ = false;
        }
    }
}

void CUDAGraph::capture_begin(uint64_t pool_id, CaptureMode mode,
                              const CUDAStream& stream) {
    GraphState& state = GraphState::instance();
    const int device = currentDevice();

    // The legacy default stream cannot participate in capture; run the graph
    // on the dedicated side stream unless the caller supplied one, as
    // torch.cuda.graph does.
    CUDAStream side = CUDAStream::undefined();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.capturing.count(std::this_thread::get_id())) {
            TP_THROW(RuntimeError,
                     "nested CUDA graph capture is not supported on one "
                     "thread (captures on different devices/threads may "
                     "run concurrently)");
        }
        if (exec_ != nullptr || graph_ != nullptr || pool_id_ != 0 ||
            owns_pool_ref_) {
            TP_THROW(RuntimeError,
                     "this CUDAGraph already holds a capture or executable; "
                     "create a new one");
        }
        if (stream.stream() != nullptr) {
            side = stream;
        } else {
            auto it = state.side_streams.find(device);
            if (it == state.side_streams.end()) {
                it = state.side_streams
                         .emplace(device, getStreamFromPool(0, device))
                         .first;
            }
            side = it->second;
        }
    }
    previous_stream_ = getCurrentCUDAStream(device);

    // Pre-create every lazy library handle (cuBLAS/cuBLASLt/cuSOLVER/cuDNN)
    // for this device: handle creation allocates internally, which aborts a
    // live capture with an opaque library error (ATen warms hipblasLt for
    // the same reason).  Must run before pool routing so warmup allocations
    // land on the normal allocator.
    CUDAContext::warmupHandles();

    // Pool routing must be armed before cudaStreamBeginCapture: once capture
    // starts, an allocator free() issuing an event record would abort it
    // (same ordering as ATen's CUDAGraph::capture_begin).  RNG state goes
    // first: it allocates on the default stream and synchronizes, both of
    // which are unsafe inside a live capture.
    const uint64_t rng_state_id = rng_register_graph(device);
    uint64_t actual_pool = 0;
    try {
        actual_pool = beginAllocateToPool(device, side, pool_id);
    } catch (...) {
        rng_unregister_graph(rng_state_id);
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto& refs = state.pool_refs[actual_pool];
        if (refs < 0) refs = 0;
        ++refs;
    }

    cudaStreamCaptureMode cuda_mode = cudaStreamCaptureModeGlobal;
    switch (mode) {
        case CaptureMode::Global:
            cuda_mode = cudaStreamCaptureModeGlobal;
            break;
        case CaptureMode::ThreadLocal:
            cuda_mode = cudaStreamCaptureModeThreadLocal;
            break;
        case CaptureMode::Relaxed:
            cuda_mode = cudaStreamCaptureModeRelaxed;
            break;
    }
    capture_mode_ = static_cast<int>(cuda_mode);

    setCurrentCUDAStream(side);
    cudaError_t error = cudaStreamBeginCapture(side.stream(), cuda_mode);
    if (error != cudaSuccess) {
        setCurrentCUDAStream(previous_stream_);
        endAllocateToPool(actual_pool);
        rng_unregister_graph(rng_state_id);
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            // Nothing was captured, so the pool holds no segments yet; just
            // drop this graph's reference (a shared pool must survive).
            state.unrefPoolLocked(actual_pool, /*release_segments=*/false);
        }
        checkCuda(error, "cudaStreamBeginCapture");
    }

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.capturing.emplace(std::this_thread::get_id(),
                                shared_from_this());
    }
    capture_stream_ = side;
    device_ = device;
    pool_id_ = actual_pool;
    owns_pool_ref_ = true;
    rng_state_id_ = rng_state_id;
}

void CUDAGraph::capture_end() {
    {
        GraphState& state = GraphState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.capturing.find(std::this_thread::get_id());
        if (it == state.capturing.end() || it->second.get() != this) {
            TP_THROW(RuntimeError,
                     "capture_end called without a live capture on this "
                     "CUDAGraph");
        }
        state.capturing.erase(it);
    }
    if (!owns_pool_ref_ || rng_state_id_ == 0) {
        TP_THROW(RuntimeError, "capture_begin must run before capture_end");
    }
    if (!cond_handles_.empty()) {
        TP_THROW(RuntimeError,
                 "unclosed conditional node: call end_capture_to_conditional_"
                 "node() for every begin_capture_to_{if,while}_node()");
    }

    cudaGraph_t captured = nullptr;
    cudaError_t error =
        cudaStreamEndCapture(capture_stream_.stream(), &captured);
    setCurrentCUDAStream(previous_stream_);
    endAllocateToPool(pool_id_);
    if (error != cudaSuccess || captured == nullptr) {
        // Failed capture: tear down everything capture_begin armed so the
        // object returns to its pristine state.
        rng_unregister_graph(rng_state_id_);
        rng_state_id_ = 0;
        {
            GraphState& state = GraphState::instance();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.unrefPoolLocked(pool_id_, /*release_segments=*/false);
        }
        owns_pool_ref_ = false;
        pool_id_ = 0;
        device_ = -1;
        capture_stream_ = CUDAStream::undefined();
        (void)cudaGetLastError();
        checkCuda(error, "cudaStreamEndCapture");
    }
    graph_ = captured;
    rng_wholegraph_increment_ =
        rng_state_id_ != 0 ? rng_capture_epilogue(rng_state_id_) : 0;

    {
        static const bool dbg = std::getenv("TP_RNG_DEBUG") != nullptr;
        size_t num_nodes = 0;
        (void)cudaGraphGetNodes(graph_, nullptr, &num_nodes);
        if (dbg)
            fprintf(stderr, "[gdbg] capture_end nodes=%zu\n", num_nodes);
    }

    instantiateLocked();
}

void CUDAGraph::instantiateLocked() {
    if (graph_ == nullptr) {
        if (exec_ != nullptr) return; // already instantiated
        TP_THROW(RuntimeError, "capture_end must run before instantiate");
    }
    // The expensive driver call, paid eagerly here instead of on first
    // replay.  No special flags: AutoFreeOnLaunch (for cudaMallocAsync-pool
    // graphs) made repeated launches of plain-cudaMalloc graphs silently skip
    // node execution on this driver; torch instantiates with flags=0 as well.
    cudaGraphExec_t exec = nullptr;
    cudaError_t error = cudaGraphInstantiateWithFlags(&exec, graph_, 0);
    if (error != cudaSuccess) {
        (void)cudaGetLastError();
        checkCuda(error, "cudaGraphInstantiateWithFlags");
    }
    exec_ = exec;
    // The template is retained: cudaGraphDebugDotPrint only accepts raw
    // cudaGraph_t handles, so destroying it would make debug_dump fail with
    // "invalid argument" after every capture.  The template is topology
    // metadata (a few hundred bytes per node), not device memory.
}

void CUDAGraph::instantiate() { instantiateLocked(); }

void CUDAGraph::replay() {
    if (exec_ == nullptr) {
        TP_THROW(RuntimeError,
                 "cannot replay before completing a capture "
                 "(capture_begin/capture_end)");
    }
    // Refresh the graph's RNG buffers with the generator's current state and
    // advance past what this replay consumes, so captured RNG kernels draw
    // fresh randoms each replay.  Enqueued on the same stream as the launch,
    // so ordering guarantees the graph reads the new values.
    if (rng_state_id_ != 0) {
        rng_replay_prologue(rng_state_id_, rng_wholegraph_increment_);
    }
    // Device switch only when actually needed (cheap TLS compare now that
    // the current device is cached per-thread).
    if (currentDevice() != device_) {
        CUDAGuard guard(device_);
        checkCuda(cudaGraphLaunch(exec_, getCurrentCUDAStream(device_).stream()),
                  "cudaGraphLaunch");
        return;
    }
    checkCuda(cudaGraphLaunch(exec_, getCurrentCUDAStream(device_).stream()),
              "cudaGraphLaunch");
}

void CUDAGraph::replay(const CUDAStream& stream) {
    if (exec_ == nullptr) {
        TP_THROW(RuntimeError,
                 "cannot replay before completing a capture "
                 "(capture_begin/capture_end)");
    }
    if (stream.stream() == nullptr) {
        TP_THROW(ValueError,
                 "replay(stream=...) requires a real CUDA stream");
    }
    if (stream.device_index() >= 0 && stream.device_index() != device_) {
        TP_THROW(ValueError,
                 "graph was captured on cuda:" + std::to_string(device_) +
                     " but replay was requested on cuda:" +
                     std::to_string(stream.device_index()));
    }
    if (rng_state_id_ != 0) {
        rng_replay_prologue(rng_state_id_, rng_wholegraph_increment_);
    }
    if (currentDevice() != device_) {
        CUDAGuard guard(device_);
        checkCuda(cudaGraphLaunch(exec_, stream.stream()), "cudaGraphLaunch");
        return;
    }
    checkCuda(cudaGraphLaunch(exec_, stream.stream()), "cudaGraphLaunch");
}

void CUDAGraph::reset() {
    if (!hasPendingResources()) return;
    if (exec_ != nullptr) {
        checkCuda(cudaGraphExecDestroy(exec_), "cudaGraphExecDestroy");
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        checkCuda(cudaGraphDestroy(graph_), "cudaGraphDestroy");
        graph_ = nullptr;
    }
    // Defensive: a capture aborted mid-conditional may have left child
    // streams open; their graphs die with the parent template.  Ended-body
    // streams are retired (not destroyed at end-capture) because captured
    // tensors still hold fences against them; by reset() time every such
    // tensor is required to be gone, so destroying here is safe.
    for (cudaStream_t child : cond_child_streams_) {
        (void)cudaStreamDestroy(child);
    }
    cond_child_streams_.clear();
    for (cudaStream_t child : cond_retired_streams_) {
        (void)cudaStreamDestroy(child);
    }
    cond_retired_streams_.clear();
    cond_handles_.clear();
    if (rng_state_id_ != 0) {
        rng_unregister_graph(rng_state_id_);
        rng_state_id_ = 0;
    }
    // The executables are gone, so baked addresses are no longer referenced
    // by CUDA work; drop this graph's pool reference and free the segments
    // when we were the last user (throws if static tensors remain alive).
    if (owns_pool_ref_) {
        owns_pool_ref_ = false;
        GraphState& state = GraphState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.unrefPoolLocked(pool_id_, /*release_segments=*/true);
    }
    pool_id_ = 0;
    device_ = -1;
    capture_stream_ = CUDAStream::undefined();
    rng_wholegraph_increment_ = 0;
}

void CUDAGraph::debug_dump(const std::string& path) {
    if (graph_ == nullptr && exec_ == nullptr) {
        TP_THROW(RuntimeError,
                 "no captured CUDA graph to dump; capture first");
    }
#if CUDART_VERSION >= 11030
    // The template carries full node attributes; without enable_debug_mode
    // the dump is topology-only (flags gate attribute verbosity).
    cudaGraph_t template_graph = graph_;
    if (template_graph == nullptr) {
        TP_THROW(RuntimeError,
                 "no captured CUDA graph to dump; capture first");
    }
    cudaError_t error = cudaGraphDebugDotPrint(
        template_graph, path.c_str(), debug_
            ? cudaGraphDebugDotFlagsVerbose
            : cudaGraphDebugDotFlags(0));
    checkCuda(error, "cudaGraphDebugDotPrint");
#else
    (void)path;
    TP_THROW(RuntimeError,
             "cudaGraphDebugDotPrint requires CUDA >= 11.3");
#endif
}

// --- conditional nodes (if/while bodies, CUDA >= 12.4) ----------------------

namespace {

void requireLiveCaptureOnThisThread(CUDAGraph& self) {
    GraphState& state = GraphState::instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.capturing.find(std::this_thread::get_id());
    if (it == state.capturing.end() || it->second.get() != &self) {
        TP_THROW(RuntimeError,
                 "conditional-node capture requires a live capture_begin() "
                 "on this CUDAGraph");
    }
}

void validatePredicate(const Tensor& pred) {
    if (!pred.device().is_cuda()) {
        TP_THROW(ValueError,
                 "conditional predicate must be a CUDA Bool scalar tensor");
    }
    if (pred.dtype() != DType::Bool || pred.numel() != 1) {
        TP_THROW(ValueError,
                 "conditional predicate must be a single-element Bool "
                 "tensor; got numel=" + std::to_string(pred.numel()));
    }
}

} // namespace

void CUDAGraph::begin_capture_to_if_node(const Tensor& scalar_pred) {
    // cudaGraphCondTypeIf
    begin_capture_to_conditional_node(scalar_pred, /*type=*/0);
}

void CUDAGraph::begin_capture_to_while_node(const Tensor& scalar_pred) {
    // cudaGraphCondTypeWhile
    begin_capture_to_conditional_node(scalar_pred, /*type=*/1);
}

void CUDAGraph::set_conditional_handle_for_current_node(const Tensor& scalar_pred) {
    if (cond_handles_.empty()) {
        TP_THROW(RuntimeError, "no active CUDA graph conditional node");
    }
#if CUDART_VERSION >= 12040
    validatePredicate(scalar_pred);
    launchSetConditionalHandle(cond_handles_.back(), scalar_pred.data_ptr(),
                               getCurrentCUDAStream(device_).stream());
#else
    (void)scalar_pred;
    TP_THROW(RuntimeError,
             "CUDA graphs conditional nodes require CUDA >= 12.4");
#endif
}

void CUDAGraph::end_capture_to_conditional_node() {
    if (cond_handles_.empty() || cond_child_streams_.empty()) {
        TP_THROW(RuntimeError,
                 "end_capture_to_conditional_node without an open "
                 "conditional body");
    }
    const cudaStream_t child = cond_child_streams_.back();
    cond_child_streams_.pop_back();
    cond_handles_.pop_back();
    cond_retired_streams_.push_back(child);

    cudaGraph_t ended = nullptr;
    checkCuda(cudaStreamEndCapture(child, &ended), "cudaStreamEndCapture");
    // ``ended`` is the child graph already owned by the parent's conditional
    // node; the driver keeps it alive with the parent.
    (void)ended;
    unrouteStreamFromPool(
        CUDAStream::fromExternal(child, device_));
    setCurrentCUDAStream(capture_stream_);
    // The stream itself stays alive until reset(): tensors written inside
    // the body carry cross-stream fences against it, and recording such an
    // event on a destroyed stream faults inside libcuda.  reset() drains
    // ``cond_child_streams_`` after the executable (and every tensor that
    // could reference the stream) is gone.
}

void CUDAGraph::begin_capture_to_conditional_node(const Tensor& scalar_pred,
                                                  int conditional_type) {
#if CUDART_VERSION >= 12040
    requireLiveCaptureOnThisThread(*this);
    validatePredicate(scalar_pred);

    cudaStream_t parent = capture_stream_.stream();
    cudaStreamCaptureStatus status{};
    cudaGraph_t parent_graph{};
#if CUDART_VERSION >= 13000
    cudaError_t error = cudaStreamGetCaptureInfo(
        parent, &status, nullptr, &parent_graph);
#else
    cudaError_t error =
        cudaStreamGetCaptureInfo_v2(parent, &status, nullptr, &parent_graph);
#endif
    checkCuda(error, "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive || parent_graph == nullptr) {
        TP_THROW(RuntimeError,
                 "capture_begin() must be called before "
                 "begin_capture_to_{if,while}_node()");
    }

    cudaGraphConditionalHandle handle{};
    checkCuda(cudaGraphConditionalHandleCreate(&handle, parent_graph, 0, 0),
              "cudaGraphConditionalHandleCreate");
    // Captured kernel: samples the predicate at replay time.
    launchSetConditionalHandle(handle, scalar_pred.data_ptr(),
                               getCurrentCUDAStream(device_).stream());

    // Current capture dependencies become the parents of the new
    // conditional node, and the stream's dependency set is replaced by the
    // node itself so subsequent captured work lands inside its body.
    const cudaGraphNode_t* dependencies = nullptr;
    const cudaGraphEdgeData* dependency_edges = nullptr;
    size_t num_dependencies = 0;
#if CUDART_VERSION >= 13000
    checkCuda(cudaStreamGetCaptureInfo(parent, &status, nullptr,
                                       &parent_graph, &dependencies,
                                       &dependency_edges, &num_dependencies),
              "cudaStreamGetCaptureInfo");
#else
    checkCuda(cudaStreamGetCaptureInfo_v3(parent, &status, nullptr,
                                          &parent_graph, &dependencies,
                                          &dependency_edges,
                                          &num_dependencies),
              "cudaStreamGetCaptureInfo_v3");
#endif

    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type =
        static_cast<cudaGraphConditionalNodeType>(conditional_type);
    params.conditional.size = 1;

    cudaGraphNode_t cond_node{};
#if CUDART_VERSION >= 13000
    checkCuda(cudaGraphAddNode(&cond_node, parent_graph, dependencies,
                               dependency_edges, num_dependencies, &params),
              "cudaGraphAddNode");
#else
    checkCuda(cudaGraphAddNode_v2(&cond_node, parent_graph, dependencies,
                                  dependency_edges, num_dependencies, &params),
              "cudaGraphAddNode_v2");
#endif
    cudaGraph_t child_graph = params.conditional.phGraph_out[0];

#if CUDART_VERSION >= 13000
    checkCuda(cudaStreamUpdateCaptureDependencies(
                  parent, &cond_node, nullptr, 1,
                  cudaStreamSetCaptureDependencies),
              "cudaStreamUpdateCaptureDependencies");
#else
    checkCuda(cudaStreamUpdateCaptureDependencies_v2(
                  parent, &cond_node, nullptr, 1,
                  cudaStreamSetCaptureDependencies),
              "cudaStreamUpdateCaptureDependencies_v2");
#endif

    cudaStream_t child_stream = nullptr;
    checkCuda(
        cudaStreamCreateWithFlags(&child_stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
    routeStreamToPool(device_, CUDAStream::fromExternal(child_stream, device_),
                      pool_id_);
    setCurrentCUDAStream(CUDAStream::fromExternal(child_stream, device_));
    error = cudaStreamBeginCaptureToGraph(child_stream, child_graph, nullptr,
                                          nullptr, 0,
                                          static_cast<cudaStreamCaptureMode>(
                                              capture_mode_));
    if (error != cudaSuccess) {
        (void)cudaGetLastError();
        setCurrentCUDAStream(capture_stream_);
        unrouteStreamFromPool(
            CUDAStream::fromExternal(child_stream, device_));
        (void)cudaStreamDestroy(child_stream);
        checkCuda(error, "cudaStreamBeginCaptureToGraph");
    }
    cond_handles_.push_back(handle);
    cond_child_streams_.push_back(child_stream);
#else
    (void)scalar_pred;
    (void)conditional_type;
    TP_THROW(RuntimeError,
             "CUDA graphs conditional nodes require CUDA >= 12.4");
#endif
}

} // namespace graph

} // namespace cuda
} // namespace tensorplay

#endif // USE_CUDA
