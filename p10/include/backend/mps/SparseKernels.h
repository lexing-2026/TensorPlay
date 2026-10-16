#pragma once

#include "Tensor.h"

namespace tensorplay {
namespace mps {

// Coalesces an MPS sparse COO tensor in place semantics (returns a new
// coalesced COO tensor with duplicate coordinates folded).
Tensor coalesce_sparse_mps(const Tensor& self);

} // namespace mps
} // namespace tensorplay