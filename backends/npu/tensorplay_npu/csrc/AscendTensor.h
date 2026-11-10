#pragma once

// Tensor helpers shared by the kernels of this backend: device allocation,
// strided copies and the host staging every layout change goes through.

#include <cstdint>
#include <vector>

#include "AscendAllocator.h"
#include "AscendRuntime.h"
#include "Tensor.h"

namespace tensorplay {
namespace npu {

// Allocate an uninitialised contiguous tensor on device ``index``.
Tensor empty_on(const std::vector<int64_t>& sizes, DType dtype, int index);

// ``t`` itself when contiguous, otherwise a contiguous copy on its device.
Tensor contiguous_on_device(const Tensor& t);

// A view whose elements run backwards through memory cannot be handed to
// the operator library as a descriptor; operands like that take the host.
inline bool has_negative_stride(const Tensor& t) {
    const IntArrayRef strides = t.stride();
    for (int64_t stride : strides) {
        if (stride < 0) return true;
    }
    return false;
}

// Copy ``src`` into ``dst`` on one device (same dtype, same shape; either
// may be strided).
void strided_copy(const Tensor& dst, const Tensor& src);

// Run the host copy kernel: the call that reaches this helper has already
// passed autograd.
void host_copy_into(Tensor& dst, const Tensor& src);

} // namespace npu
} // namespace tensorplay
