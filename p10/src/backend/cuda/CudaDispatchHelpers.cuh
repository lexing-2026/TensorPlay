#pragma once

// CUDA registrations for operator spellings whose semantics are already
// served by kernels in this backend.  Two groups live here:
//
// 1. Alias spellings: internal names for ops tp implements under its
//    public name (the argument lists match, so the kernel is reused as-is).
// 2. Helper spellings: decomposition-level operators composed here from the
//    dispatched primitives, so the CUDA key holds a direct registration and
//    the dispatcher's composite fallthrough stays a fallback rather than the
//    only path.
//
// Spellings whose semantics tp does not provide (sparse-only helpers) get an
// explicit registration that reports the missing backend instead of falling
// through to a misleading dense result.
//
// Every kernel's signature must match the generated dispatcher stub for its
// spelling byte for byte: registrations are stored as type-erased function
// pointers, so a mismatch is silent undefined behaviour at call time.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "Generator.h"
#include "CUDARuntime.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include "OutWrite.h"

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;


// Kernels living in other translation units' anonymous namespaces are
// reached through their registered public operators.
template <typename Return, typename... Args>
Return dispatch_cuda(const char* op, Args... args) {
    return DispatchStub<Return, Args...>::call(
        std::string(op), DispatchKey::CUDA, std::forward<Args>(args)...);
}

// Defined in the repeat-interleave unit, used by the triangular builders.
Tensor repeat_interleave_tensor_cuda(const Tensor& repeats,
                                     std::optional<int64_t> output_size);


// ---------------------------------------------------------------------------
// repeat_interleave.Tensor returns the flat source-index list
// [0 x r0, 1 x r1, ...] from cumulative repeat boundaries.
// ---------------------------------------------------------------------------

Tensor repeat_interleave_tensor_cuda(const Tensor& repeats,
                                     std::optional<int64_t> output_size);

} // namespace cuda
} // namespace tensorplay
