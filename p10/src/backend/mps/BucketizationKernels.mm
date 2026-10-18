#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace tensorplay {
namespace mps {

// The bundled Metal shader library (Bucketization.metal).
static auto& lib = MetalShaderLibrary::getBundledLibrary();

namespace {

Tensor searchsorted_impl(const Tensor& seq_in, const Tensor& self_in,
                         bool out_int32, bool right, const Tensor& sorter) {
  // The shader requires contiguous inputs of the same dtype.
  Tensor boundaries = seq_in.contiguous();
  Tensor queries = self_in.contiguous();
  if (boundaries.dtype() != queries.dtype()) {
    boundaries = boundaries.to(queries.dtype()).contiguous();
  }
  if (sorter.defined() && !sorter.is_contiguous()) {
    TP_THROW(RuntimeError, "searchsorted: sorter must be contiguous");
  }
  const int64_t numel_in = queries.numel();
  const bool is_scalar_input = queries.dim() == 0 && numel_in == 1;
  const int64_t idim_in = is_scalar_input ? 1 : queries.size(-1);
  const int64_t idim_bd = boundaries.size(-1);
  const int64_t right_i64 = right ? 1 : 0;
  const int64_t is_1d_boundaries = boundaries.dim() == 1 ? 1 : 0;

  const DType out_dtype = out_int32 ? DType::Int32 : DType::Int64;
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(queries.shape()),
                                out_dtype, queries.device());
  if (numel_in == 0) {
    return result;
  }

  const std::string name =
      "searchsorted_" + scalarToMetalTypeString(queries.dtype()) + "_" +
      scalarToMetalTypeString(out_dtype);

  // One power-of-two boundary run without a sorter: the search takes a fixed
  // number of bit-fixing steps with 32-bit positions, four queries per thread.
  constexpr int64_t kPow2QueryLimit = int64_t{1} << 30;
  const bool pow2 = is_1d_boundaries != 0 && !sorter.defined() &&
      idim_bd > 0 && (idim_bd & (idim_bd - 1)) == 0 &&
      idim_bd <= std::numeric_limits<int32_t>::max() &&
      numel_in <= kPow2QueryLimit;
  if (pow2) {
    auto pso = lib.getPipelineStateForFunc(
        name + (right ? "_pow2_right" : "_pow2_left"));
    MPSStream* stream = getCurrentMPSStream();
    const uint32_t n_bd = static_cast<uint32_t>(idim_bd);
    const uint32_t n_in = static_cast<uint32_t>(numel_in);
    const NSUInteger threads = static_cast<NSUInteger>((numel_in + 3) / 4);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, queries, boundaries, result, n_bd, n_in);
        mtl_dispatch1DJob(encoder, pso, threads);
      }
    });
    return result;
  }

  const std::string kernel = name + (sorter.defined() ? "_sorter" : "");
  auto pso = lib.getPipelineStateForFunc(kernel);
  const int64_t max_tptg = [pso maxTotalThreadsPerThreadgroup];
  const int64_t tg_size = std::min(max_tptg, numel_in);
  const int64_t num_tg = (numel_in + tg_size - 1) / tg_size;

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, queries, boundaries, result, idim_in, idim_bd,
                  numel_in, right_i64, is_1d_boundaries);
      if (sorter.defined()) {
        set_shader_arg(encoder, sorter, 8);
      }
      [encoder dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
  });
  return result;
}

} // namespace

Tensor searchsorted_kernel(const Tensor& sorted_sequence, const Tensor& self,
                           bool out_int32, bool right,
                           const std::optional<std::string>& side_opt,
                           const std::optional<Tensor>& sorter_opt) {
  bool use_right = right;
  if (side_opt.has_value()) {
    use_right = *side_opt == "right";
  }
  Tensor sorter = sorter_opt.has_value() ? *sorter_opt : Tensor();
  return searchsorted_impl(sorted_sequence, self, out_int32, use_right,
                           sorter);
}

Tensor searchsorted_scalar_kernel(const Tensor& sorted_sequence,
                                  const Scalar& self, bool out_int32,
                                  bool right,
                                  const std::optional<std::string>& side_opt,
                                  const std::optional<Tensor>& sorter_opt) {
  Tensor query({}, sorted_sequence.dtype(), sorted_sequence.device());
  query.fill_(self);
  bool use_right = right;
  if (side_opt.has_value()) {
    use_right = *side_opt == "right";
  }
  Tensor sorter = sorter_opt.has_value() ? *sorter_opt : Tensor();
  return searchsorted_impl(sorted_sequence, query, out_int32, use_right,
                           sorter);
}

Tensor bucketize_kernel(const Tensor& self, const Tensor& boundaries,
                        bool out_int32, bool right) {
  if (boundaries.dim() != 1) {
    TP_THROW(RuntimeError,
             "bucketize(): boundaries tensor must be 1 dimension");
  }
  return searchsorted_impl(boundaries, self, out_int32, right, Tensor());
}

Tensor bucketize_scalar_kernel(const Scalar& self, const Tensor& boundaries,
                               bool out_int32, bool right) {
  Tensor query({}, boundaries.dtype(), boundaries.device());
  query.fill_(self);
  return searchsorted_impl(boundaries, query, out_int32, right, Tensor());
}

TENSORPLAY_LIBRARY_IMPL(MPS, BucketizationKernels) {
  m.impl("searchsorted.Tensor", searchsorted_kernel);
  m.impl("searchsorted.Scalar", searchsorted_scalar_kernel);
  m.impl("bucketize.Tensor", bucketize_kernel);
  m.impl("bucketize.Scalar", bucketize_scalar_kernel);
}

} // namespace mps
} // namespace tensorplay