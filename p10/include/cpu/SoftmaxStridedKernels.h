#pragma once

// Dispatch stubs for the softmax over a dimension that is not the last one.
//
// The walk reduces across the strided dimension while the values of the inner
// dimension are handled several at a time, so it is written against the vector
// layer.  The layer is compiled per CPU_CAPABILITY like the reductions, and
// DispatchStub picks the best registered tier at runtime.
// See p10/CMakeLists.txt Note [CPU_CAPABILITY namespace] and DispatchStub.h.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// The walk of a strided softmax: `size` rows of `stride` elements are reduced
// into `after` answers, `before` times over.
using softmax_strided_fn = void (*)(const float*, float*, int64_t, int64_t,
                                    int64_t, int64_t, bool);
DECLARE_DISPATCH(softmax_strided_fn, softmax_strided_f32_stub)

using softmax_strided_f64_fn = void (*)(const double*, double*, int64_t, int64_t,
                                        int64_t, int64_t, bool);
DECLARE_DISPATCH(softmax_strided_f64_fn, softmax_strided_f64_stub)

} // namespace cpu
} // namespace tensorplay
