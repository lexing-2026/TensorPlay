#include "CUDAGenerator.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Device.h"
#include "DType.h"
#include "Generator.h"
#include "Tensor.h"
#include "Exception.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
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
