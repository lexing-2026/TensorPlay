#include "CUDAGraph.h"

#ifdef USE_CUDA
#include "CUDAGenerator.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace tensorplay {
namespace cuda {

// Capture/handle bookkeeping lives here; the graph-private memory pools and
// their routing live inside the caching allocator (CUDAAllocator.cpp), which
// owns the segment/block types they refer to.  Graph-safe RNG state lives in
// CUDAGenerator.cpp: registration happens here at capture begin, kernels read
// the [seed, offset] device buffer through PhiloxCudaState, and every replay
// refreshes it (rng_replay_prologue) so each replay draws fresh randoms.

struct GraphHandle {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    uint64_t pool_id = 0;
    int device = -1;
    // RNG state registered for this graph's capture window (0 = none).
    uint64_t rng_state_id = 0;
    // Counter total consumed by RNG ops during capture; advanced into the
    // generator on every replay by rng_replay_prologue.
    uint64_t rng_wholegraph_increment = 0;
};

struct GraphState {
    static GraphState& instance() {
        static auto* state = new GraphState();
        return *state;
    }

    std::mutex mutex;
    std::unordered_map<uint64_t, GraphHandle> handles;
    uint64_t next_handle = 1;

    // Capture bookkeeping; only meaningful while ``capturing`` is true.
    bool capturing = false;
    CUDAStream capture_stream = CUDAStream::undefined();
    CUDAStream previous_stream = CUDAStream::undefined();
    uint64_t capture_pool_id = 0;
    uint64_t capture_rng_id = 0;

    // One dedicated side stream per device, reused across captures so lazy
    // per-stream state (cuBLAS workspaces) sees warmup and capture equally.
    std::unordered_map<int, CUDAStream> side_streams;
};

// Public graph API declared under tensorplay::cuda::graph (CUDAGraph.h);
// definitions live here so the mangled names match the bindings.
namespace graph {

CUDAStream captureStream(int device_index) {
    GraphState& state = GraphState::instance();
    const int device = device_index < 0 ? currentDevice() : device_index;
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.side_streams.find(device);
    if (it == state.side_streams.end()) {
        it = state.side_streams.emplace(device, getStreamFromPool(0, device)).first;
    }
    return it->second;
}

void beginCapture() {
    GraphState& state = GraphState::instance();
    const int device = currentDevice();

    // The legacy default stream cannot participate in capture; run the graph
    // on the dedicated side stream instead, as torch.cuda.graph does.
    CUDAStream side = captureStream(device);
    CUDAStream previous = getCurrentCUDAStream(device);

    // Pool routing must be armed before cudaStreamBeginCapture: once capture
    // starts, an allocator free() issuing an event record would abort it
    // (same ordering as ATen's CUDAGraph::capture_begin).  RNG state goes
    // first: it allocates on the default stream and synchronizes, both of
    // which are unsafe inside a live capture.
    const uint64_t rng_state_id = rng_register_graph(device);
    const uint64_t pool_id = beginAllocateToPool(device, side);

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.capturing) {
        endAllocateToPool(pool_id);
        rng_unregister_graph(rng_state_id);
        TP_THROW(RuntimeError, "nested CUDA graph capture is not supported");
    }
    setCurrentCUDAStream(side);
    cudaError_t error = cudaStreamBeginCapture(
        side.stream(), cudaStreamCaptureModeGlobal);
    if (error != cudaSuccess) {
        setCurrentCUDAStream(previous);
        endAllocateToPool(pool_id);
        rng_unregister_graph(rng_state_id);
        checkCuda(error, "cudaStreamBeginCapture");
    }
    state.capturing = true;
    state.capture_stream = side;
    state.previous_stream = previous;
    state.capture_pool_id = pool_id;
    state.capture_rng_id = rng_state_id;
}

uint64_t endCapture() {
    GraphState& state = GraphState::instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.capturing) {
        TP_THROW(RuntimeError,
                 "cuda_graph_end_capture called without a live capture");
    }

    cudaGraph_t graph = nullptr;
    cudaError_t error = cudaStreamEndCapture(state.capture_stream.stream(), &graph);
    setCurrentCUDAStream(state.previous_stream);
    endAllocateToPool(state.capture_pool_id);
    state.capturing = false;
    if (error != cudaSuccess || graph == nullptr) {
        rng_unregister_graph(state.capture_rng_id);
        state.capture_rng_id = 0;
        (void)cudaGetLastError();
        checkCuda(error, "cudaStreamEndCapture");
    }

    {
        static const bool dbg = std::getenv("TP_RNG_DEBUG") != nullptr;
        size_t num_nodes = 0;
        (void)cudaGraphGetNodes(graph, nullptr, &num_nodes);
        if (dbg)
            fprintf(stderr, "[gdbg] endCapture nodes=%zu\n", num_nodes);
    }

    GraphHandle handle;
    handle.graph = graph;
    handle.pool_id = state.capture_pool_id;
    handle.device = state.capture_stream.device_index();
    // Close the RNG capture window and remember the counter total so every
    // replay can advance the generator past what the graph consumes.
    handle.rng_state_id = state.capture_rng_id;
    handle.rng_wholegraph_increment = rng_capture_epilogue(state.capture_rng_id);
    state.capture_rng_id = 0;
    const uint64_t id = state.next_handle++;
    state.handles[id] = handle;
    return id;
}

uint64_t instantiate(uint64_t handle_id) {
    cudaGraph_t graph = nullptr;
    {
        GraphState& state = GraphState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.handles.find(handle_id);
        if (it == state.handles.end()) {
            TP_THROW(ValueError, "unknown CUDA graph handle");
        }
        if (it->second.graph == nullptr) {
            if (it->second.exec != nullptr) {
                // Already instantiated; the Python wrapper calls this before
                // every replay, so treat it as a no-op.
                return handle_id;
            }
            TP_THROW(RuntimeError,
                     "CUDA graph was already destroyed");
        }
        graph = it->second.graph;
    }

    // Instantiate outside the lock: this is the expensive driver call.
    cudaGraphExec_t exec = nullptr;
    // No special flags: AutoFreeOnLaunch (for cudaMallocAsync-pool graphs)
    // made repeated launches of plain-cudaMalloc graphs silently skip node
    // execution on this driver; torch instantiates with flags=0 as well.
    cudaError_t error = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    if (error != cudaSuccess) {
        (void)cudaGetLastError();
        checkCuda(error, "cudaGraphInstantiateWithFlags");
    }

    GraphState& state = GraphState::instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.handles.find(handle_id);
    if (it == state.handles.end()) {
        cudaGraphExecDestroy(exec);
        TP_THROW(ValueError, "CUDA graph destroyed during instantiation");
    }
    it->second.exec = exec;
    // The executable is self-contained; release the template (keep_graph=false).
    (void)cudaGraphDestroy(it->second.graph);
    it->second.graph = nullptr;
    return handle_id;
}

void launch(uint64_t handle_id) {
    cudaGraphExec_t exec = nullptr;
    int device = -1;
    uint64_t rng_state_id = 0;
    uint64_t rng_wholegraph_increment = 0;
    {
        GraphState& state = GraphState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.handles.find(handle_id);
        if (it == state.handles.end() || it->second.exec == nullptr) {
            TP_THROW(RuntimeError,
                     "cuda_graph_launch requires an instantiated graph; "
                     "call cuda_graph_instantiate first");
        }
        exec = it->second.exec;
        device = it->second.device;
        rng_state_id = it->second.rng_state_id;
        rng_wholegraph_increment = it->second.rng_wholegraph_increment;
    }
    // Refresh the graph's RNG buffers with the generator's current state and
    // advance past what this replay consumes, so captured RNG kernels draw
    // fresh randoms each replay.  Enqueued on the same stream as the launch,
    // so ordering guarantees the graph reads the new values.
    if (rng_state_id != 0) {
        rng_replay_prologue(rng_state_id, rng_wholegraph_increment);
    }
    CUDAGuard guard(device);
    {
        static const bool dbg = std::getenv("TP_RNG_DEBUG") != nullptr;
        cudaStream_t s = getCurrentCUDAStream(device).stream();
        cudaError_t le = cudaGraphLaunch(exec, s);
        if (dbg)
            fprintf(stderr, "[gdbg] launch h=%llu exec=%p stream=%p err=%d\n",
                    (unsigned long long)handle_id, (void*)exec, (void*)s, (int)le);
        checkCuda(le, "cudaGraphLaunch");
    }
}

void destroy(uint64_t handle_id) {
    uint64_t pool_id = 0;
    uint64_t rng_state_id = 0;
    {
        GraphState& state = GraphState::instance();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.handles.find(handle_id);
        if (it == state.handles.end()) return;
        if (it->second.exec != nullptr) {
            checkCuda(cudaGraphExecDestroy(it->second.exec), "cudaGraphExecDestroy");
        } else if (it->second.graph != nullptr) {
            checkCuda(cudaGraphDestroy(it->second.graph), "cudaGraphDestroy");
        }
        pool_id = it->second.pool_id;
        rng_state_id = it->second.rng_state_id;
        state.handles.erase(it);
    }
    // The executable is gone, so baked addresses are no longer referenced by
    // CUDA work; free the pool and the graph's RNG buffer (throws if static
    // tensors are still alive).
    releasePool(pool_id);
    if (rng_state_id != 0) {
        rng_unregister_graph(rng_state_id);
    }
}

} // namespace graph

} // namespace cuda
} // namespace tensorplay

#endif // USE_CUDA
