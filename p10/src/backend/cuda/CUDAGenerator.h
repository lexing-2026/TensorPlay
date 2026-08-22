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
