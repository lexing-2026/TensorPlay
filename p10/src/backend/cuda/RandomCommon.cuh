#pragma once

// Shared Philox helpers for the random-sampling translation units.  Every
// distribution kernel unpacks the generator state the same way, and the
// unpacking must agree bit-for-bit between files so a given (seed, offset)
// pair reproduces the same stream everywhere.

#include <cstdint>

#include "CUDAGenerator.h"

namespace tensorplay {
namespace cuda {

namespace {

// Unpacks PhiloxCudaState into the effective (seed, offset) this launch
// consumes; in pointer mode the device buffer is dereferenced at kernel
// execution time, both during capture and on every later replay.
__device__ inline void philox_unpack(const PhiloxCudaState& state,
                                     uint64_t* seed, uint64_t* offset) {
    if (state.captured) {
        *seed = *state.seed_dev;
        *offset = *state.offset_dev + state.offset_intragraph;
    } else {
        *seed = state.seed;
        *offset = state.offset;
    }
}

} // namespace

// curand device API consumes at most 4 counter values per call.
constexpr uint64_t kMaxGeneratorOffsetsPerCall = 4;

} // namespace cuda
} // namespace tensorplay

