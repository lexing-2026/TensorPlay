#pragma once

#include "DispatchStub.h"
#include "Generator.h"
#include "Tensor.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cpu {

// Internal Stax lowering hook.  The expression program is kept at the Stax
// boundary, while p10 owns CPU ISA dispatch and the actual vector loop.
using stax_pointwise_fn = Tensor (*)(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants);

using stax_pointwise_multi_fn = std::vector<Tensor> (*)(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs);

DECLARE_DISPATCH(stax_pointwise_fn, stax_pointwise_stub)
DECLARE_DISPATCH(stax_pointwise_multi_fn, stax_pointwise_multi_stub)

P10_API Tensor stax_fused_pointwise_cpu(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants);

P10_API std::vector<Tensor> stax_fused_pointwise_cpu_multi(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs);

} // namespace cpu

namespace cuda {

// Device-side counterpart of the CPU program runner: evaluates the same
// instruction-triple program elementwise in one kernel launch.  Inputs must
// share one contiguous shape; outputs inherit the input dtype.
P10_API Tensor stax_fused_pointwise_cuda(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    int64_t out_dtype = -1);

P10_API std::vector<Tensor> stax_fused_pointwise_cuda_multi(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs,
    const std::vector<int64_t>& out_dtypes = {});

} // namespace cuda

} // namespace tensorplay
