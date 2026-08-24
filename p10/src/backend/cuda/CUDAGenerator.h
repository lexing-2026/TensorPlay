#pragma once

#include "Macros.h"
#include <cstdint>
#include <utility>

namespace tensorplay {

class Tensor;

namespace cuda {

// Philox4_32_10 counter-based RNG state, mirroring torch's CUDAGeneratorImpl:
// the state is (seed, offset) and each kernel launch atomically reserves
// `increment` counter values so results are independent of launch geometry
// and reproducible across runs and architectures.

P10_API void manual_seed(uint64_t seed);
P10_API void manual_seed_all(uint64_t seed);

P10_API uint64_t current_seed();
P10_API uint64_t current_offset();
P10_API void set_offset(uint64_t offset);

// Atomically reserves `increment` philox counter values for the current
// device and returns the (seed, offset) the launching kernel should consume.
P10_API std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

// Graph-safe philox state handed to RNG kernels, mirroring torch's
// at::PhiloxCudaState (Note [CUDA Graph-safe RNG states]).  Outside a CUDA
// graph capture kernels consume the plain (seed, offset) values; while a
// capture is underway they instead read seed/offset from device buffers owned
// by the capturing graph plus a per-kernel intragraph offset, so replay
// prologues can refresh the buffers and every replay consumes a fresh slice
// of the random stream instead of repeating capture-time values.
struct PhiloxCudaState {
    // Value mode (`captured == false`).
    uint64_t seed = 0;
    uint64_t offset = 0;
    // Pointer mode (`captured == true`): both point into one packed device
    // buffer [seed, offset]; kernels dereference them at execution time.
    bool captured = false;
    const uint64_t* seed_dev = nullptr;
    const uint64_t* offset_dev = nullptr;
    // Base counter this kernel consumes within the graph; added to the
    // device-side offset so sibling RNG ops in one graph never collide.
    uint64_t offset_intragraph = 0;
};

// Capture-aware counterpart of philox_engine_inputs: returns value mode when
// no capture is live and pointer mode into the registered graph's buffers
// otherwise, reserving `increment` counters in either regime.
P10_API PhiloxCudaState philox_cuda_state(uint64_t increment);

// --- graph hooks (called by CUDAGraph.cpp; see Note above) -------------------
//
// Registers per-graph RNG state for `device`: allocates the packed [seed,
// offset] Int64 device buffer on the default stream (so it escapes graph-pool
// routing) pre-filled with the generator's current values.  Returns an id the
// graph stores alongside its handle.  Must run before cudaStreamBeginCapture.
P10_API uint64_t rng_register_graph(int device);
// Ends the capture window: returns the total counter increment consumed by
// RNG ops inside the graph ("wholegraph_increment") and clears the active
// slot.  Call once after a successful cudaStreamEndCapture.
P10_API uint64_t rng_capture_epilogue(uint64_t id);
// Replay preamble: refills the graph's device buffer with the generator's
// current (seed, offset) on the calling stream, records that stream against
// the buffer, then advances the generator offset by `wholegraph_increment`.
// Call before every cudaGraphLaunch.
P10_API void rng_replay_prologue(uint64_t id, uint64_t wholegraph_increment);
// Drops the graph's RNG state (frees the device buffer).  Call at graph
// destroy.
P10_API void rng_unregister_graph(uint64_t id);

// Serializes (seed, offset) to a 16-byte CPU UInt8 tensor, matching
// torch.cuda.get_rng_state.
P10_API Tensor get_rng_state();
P10_API void set_rng_state(const Tensor& new_state);

// Seeding before CUDA initialization is stashed and replayed at the first
// real CUDA runtime call (mirrors torch.cuda._lazy_call).
P10_API void stash_pending_seed_all(uint64_t seed);
P10_API void apply_pending_seed();

} // namespace cuda
} // namespace tensorplay
