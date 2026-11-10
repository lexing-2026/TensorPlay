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

// Copy ``src`` into ``dst`` on one device (same dtype, same shape; either
// may be strided).
void strided_copy(const Tensor& dst, const Tensor& src);

// Run the host copy kernel: the call that reaches this helper has already
// passed autograd.
void host_copy_into(Tensor& dst, const Tensor& src);

} // namespace npu
} // namespace tensorplay
