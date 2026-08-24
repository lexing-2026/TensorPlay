#include "CUDAGenerator.h"
#include "CUDAContext.h"
#include "CUDAGraph.h"
#include "CUDARuntime.h"
#include "Device.h"
#include "DType.h"
#include "Generator.h"
#include "Tensor.h"
#include "Exception.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

// Per-device Philox state (seed, offset). Pure host-side bookkeeping: kernels
// read (seed, offset) at launch and manage counters on the device.
struct PhiloxState {
    std::mutex mutex;
    uint64_t seed = default_rng_seed_val;
    uint64_t offset = 0;
};

std::mutex g_states_mutex;
std::unordered_map<int, PhiloxState*> g_states;

// Seed stashed by a pre-initialization manual_seed call; applied at the
// first real CUDA runtime call (see noteCudaRuntimeCall).
std::atomic<bool> g_has_pending_seed{false};
std::atomic<uint64_t> g_pending_seed{0};

PhiloxState& state_for(int device) {
    std::lock_guard<std::mutex> lock(g_states_mutex);
    auto it = g_states.find(device);
    if (it != g_states.end()) return *it->second;
    auto* created = new PhiloxState();
    g_states.emplace(device, created);
    return *created;
}

void apply_seed(int device, uint64_t seed) {
    auto& state = state_for(device);
    std::lock_guard<std::mutex> lock(state.mutex);
    state.seed = seed;
    state.offset = 0;
}

} // namespace

void stash_pending_seed_all(uint64_t seed) {
    g_pending_seed.store(seed, std::memory_order_relaxed);
    g_has_pending_seed.store(true, std::memory_order_release);
}

void apply_pending_seed() {
    if (g_has_pending_seed.exchange(false, std::memory_order_acq_rel)) {
        manual_seed_all(g_pending_seed.load(std::memory_order_relaxed));
    }
}

void manual_seed(uint64_t seed) {
    // Lazy: never initialize CUDA from a seeding call. If CUDA is not
    // initialized yet, stash the seed and apply it at first real CUDA use.
    if (!isCudaInitialized()) {
        stash_pending_seed_all(seed);
        return;
    }
    if (isInBadFork()) return;
    apply_seed(currentDevice(), seed);
}

void manual_seed_all(uint64_t seed) {
    if (!isCudaInitialized()) {
        stash_pending_seed_all(seed);
        return;
    }
    if (isInBadFork()) return;
    const int count = deviceCount();
    for (int device = 0; device < count; ++device) {
        apply_seed(device, seed);
    }
}

uint64_t current_seed() {
    auto& state = state_for(currentDevice());
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.seed;
}

uint64_t current_offset() {
    auto& state = state_for(currentDevice());
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.offset;
}

void set_offset(uint64_t offset) {
    auto& state = state_for(currentDevice());
    std::lock_guard<std::mutex> lock(state.mutex);
    state.offset = offset;
}

std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) {
    auto& state = state_for(currentDevice());
    std::lock_guard<std::mutex> lock(state.mutex);
    uint64_t offset = state.offset;
    state.offset += increment;
    return {state.seed, offset};
}

// --- graph-safe RNG (see CUDAGenerator.h) ------------------------------------

namespace {

const bool rng_dbg = std::getenv("TP_RNG_DEBUG") != nullptr;

struct RngGraphState {
    int device = -1;
    // Packed [seed, offset] device buffer; allocated on the default stream so
    // graph-pool routing (keyed on the capture side stream) never claims it.
    // Its addresses are baked into captured RNG kernels, so it must stay alive
    // until rng_unregister_graph.
    Tensor bufs;
    // Counter consumption accumulated by RNG ops while this graph's capture
    // window was open; reported whole by rng_capture_epilogue.
    uint64_t intragraph_offset = 0;
};

std::mutex g_rng_graphs_mutex;
std::unordered_map<uint64_t, RngGraphState> g_rng_graphs;
uint64_t g_next_rng_graph_id = 1;
// The single live capture slot: set by rng_register_graph at beginCapture,
// cleared by rng_capture_epilogue.  philox_cuda_state consults isCapturing()
// for the mode switch and this id to locate the buffers.
uint64_t g_capturing_rng_id = 0;

void fill_rng_device_buffer(const Tensor& bufs,
                            const CUDAStream& stream,
                            uint64_t seed,
                            uint64_t offset) {
    const int64_t host[2] = {static_cast<int64_t>(seed),
                             static_cast<int64_t>(offset)};
    checkCuda(cudaMemcpyAsync(bufs.data_ptr<int64_t>(), host, sizeof(host),
                              cudaMemcpyHostToDevice, stream.stream()),
              "rng state H2D copy");
}

} // namespace

PhiloxCudaState philox_cuda_state(uint64_t increment) {
    if (!isCapturing()) {
        auto& state = state_for(currentDevice());
        std::lock_guard<std::mutex> lock(state.mutex);
        uint64_t offset = state.offset;
        state.offset += increment;
        if (rng_dbg) fprintf(stderr, "[rngdbg] value-mode inc=%llu -> (seed=%llu off=%llu)\n",
                             (unsigned long long)increment, (unsigned long long)state.seed, (unsigned long long)offset);
        PhiloxCudaState out;
        out.seed = state.seed;
        out.offset = offset;
        return out;
    }
    if (rng_dbg) fprintf(stderr, "[rngdbg] CAPTURE-MODE enter\n");
    std::lock_guard<std::mutex> graphs_lock(g_rng_graphs_mutex);
    auto it = g_rng_graphs.find(g_capturing_rng_id);
    if (it == g_rng_graphs.end()) {
        TP_THROW(RuntimeError,
                 "RNG op during CUDA graph capture without registered "
                 "generator state");
    }
    RngGraphState& st = it->second;
    if (st.intragraph_offset >
        std::numeric_limits<uint64_t>::max() - increment) {
        TP_THROW(RuntimeError,
                 "increment causes overflow in the intragraph philox offset");
    }
    const uint64_t base = st.intragraph_offset;
    st.intragraph_offset += increment;
    if (rng_dbg) fprintf(stderr, "[rngdbg] ptr-mode id=%llu inc=%llu base=%llu\n",
                         (unsigned long long)g_capturing_rng_id, (unsigned long long)increment, (unsigned long long)base);
    const auto* base_ptr =
        reinterpret_cast<const uint64_t*>(st.bufs.data_ptr<int64_t>());
    PhiloxCudaState out;
    out.captured = true;
    out.seed_dev = base_ptr;
    out.offset_dev = base_ptr + 1;
    out.offset_intragraph = base;
    return out;
}

uint64_t rng_register_graph(int device) {
    if (device < 0) device = currentDevice();
    // Allocate outside the capture window on the default stream so the buffer
    // lands in the regular pool and its contents are settled before kernels
    // that will read them are captured.
    const CUDAStream previous = getCurrentCUDAStream(device);
    const CUDAStream deflt = getDefaultCUDAStream(device);
    setCurrentCUDAStream(deflt);
    Tensor bufs;
    try {
        bufs = Tensor::empty({2}, DType::Int64,
                             Device(DeviceType::CUDA, device));
        auto& state = state_for(device);
        uint64_t seed;
        uint64_t offset;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            seed = state.seed;
            offset = state.offset;
        }
        fill_rng_device_buffer(bufs, deflt, seed, offset);
        if (rng_dbg) fprintf(stderr, "[rngdbg] register id-seq dev=%d seed=%llu off=%llu\n",
                             device, (unsigned long long)seed, (unsigned long long)offset);
        checkCuda(cudaStreamSynchronize(deflt.stream()), "rng state sync");
    } catch (...) {
        setCurrentCUDAStream(previous);
        throw;
    }
    setCurrentCUDAStream(previous);

    std::lock_guard<std::mutex> lock(g_rng_graphs_mutex);
    RngGraphState st;
    st.device = device;
    st.bufs = std::move(bufs);
    const uint64_t id = g_next_rng_graph_id++;
    g_rng_graphs.emplace(id, std::move(st));
    g_capturing_rng_id = id;
    return id;
}

uint64_t rng_capture_epilogue(uint64_t id) {
    std::lock_guard<std::mutex> lock(g_rng_graphs_mutex);
    if (rng_dbg) fprintf(stderr, "[rngdbg] epilogue id=%llu inc=%llu\n",
                         (unsigned long long)id, (unsigned long long)(g_rng_graphs.count(id) ? g_rng_graphs[id].intragraph_offset : 0));
    auto it = g_rng_graphs.find(id);
    if (it == g_rng_graphs.end()) {
        TP_THROW(ValueError, "unknown RNG graph state");
    }
    if (g_capturing_rng_id == id) g_capturing_rng_id = 0;
    return std::exchange(it->second.intragraph_offset, 0);
}

void rng_replay_prologue(uint64_t id, uint64_t wholegraph_increment) {
    std::lock_guard<std::mutex> graphs_lock(g_rng_graphs_mutex);
    auto it = g_rng_graphs.find(id);
    if (it == g_rng_graphs.end()) {
        TP_THROW(ValueError, "unknown RNG graph state");
    }
    RngGraphState& st = it->second;

    auto& state = state_for(st.device);
    uint64_t seed;
    uint64_t offset;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        seed = state.seed;
        offset = state.offset;
    }

    if (rng_dbg) fprintf(stderr, "[rngdbg] replay-prologue id=%llu wholegraph_inc=%llu seed=%llu off=%llu\n",
                         (unsigned long long)id, (unsigned long long)wholegraph_increment,
                         (unsigned long long)seed, (unsigned long long)offset);
    const CUDAStream launch_stream = getCurrentCUDAStream(st.device);
    // fill_rng_device_buffer dereferences the buffer via data_ptr(); TensorImpl
    // records every access against the then-current stream, so this pins the
    // buffer to `launch_stream` until any in-flight replay completes even if
    // the graph is destroyed right after launch.
    fill_rng_device_buffer(st.bufs, launch_stream, seed, offset);

    std::lock_guard<std::mutex> lock(state.mutex);
    state.offset += wholegraph_increment;
}

void rng_unregister_graph(uint64_t id) {
    std::lock_guard<std::mutex> lock(g_rng_graphs_mutex);
    if (g_capturing_rng_id == id) g_capturing_rng_id = 0;
    g_rng_graphs.erase(id);
}

Tensor get_rng_state() {
    constexpr size_t seed_size = sizeof(uint64_t);
    constexpr size_t offset_size = sizeof(int64_t);
    constexpr size_t total_size = seed_size + offset_size;

    Tensor state_tensor(std::vector<int64_t>{static_cast<int64_t>(total_size)}, DType::UInt8,
                        Device(DeviceType::CPU));
    auto rng_state = state_tensor.data_ptr<uint8_t>();
    const uint64_t seed = current_seed();
    const int64_t offset = static_cast<int64_t>(current_offset());
    std::memcpy(rng_state, &seed, seed_size);
    std::memcpy(rng_state + seed_size, &offset, offset_size);
    return state_tensor;
}

void set_rng_state(const Tensor& new_state) {
    constexpr size_t seed_size = sizeof(uint64_t);
    constexpr size_t offset_size = sizeof(int64_t);
    constexpr size_t total_size = seed_size + offset_size;

    if (new_state.device().type() != DeviceType::CPU || new_state.dtype() != DType::UInt8) {
        TP_THROW(RuntimeError, "RNG state must be a CPU UInt8 tensor");
    }
    const size_t new_state_size = static_cast<size_t>(new_state.numel());
    uint64_t input_seed = 0;
    const auto* state_bytes = new_state.data_ptr<uint8_t>();
    if (new_state_size == total_size - offset_size) {
        // Legacy state with no philox offset.
        std::memcpy(&input_seed, state_bytes, seed_size);
        manual_seed(input_seed);
        return;
    }
    if (new_state_size != total_size) {
        TP_THROW(RuntimeError, "RNG state is wrong size");
    }
    std::memcpy(&input_seed, state_bytes, seed_size);
    int64_t philox_offset = 0;
    std::memcpy(&philox_offset, state_bytes + seed_size, offset_size);
    manual_seed(input_seed);
    set_offset(static_cast<uint64_t>(philox_offset));
}

} // namespace cuda
} // namespace tensorplay
