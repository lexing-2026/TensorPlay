#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Generator.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <array>
#include <atomic>
#include <tuple>

namespace tensorplay {
namespace mps {

// The bundled Metal shader library (Dropout.metal).
static auto& lib = MetalShaderLibrary::getBundledLibrary();

namespace {

// Per-process offset for the shader-side Philox RNG.  The seed comes from the
// framework's default generator so manual_seed stays reproducible.
std::atomic<int64_t> g_rng_offset{0};

} // namespace

std::tuple<Tensor, Tensor> native_dropout_kernel(const Tensor& input, double p,
                                                  std::optional<bool> train) {
  if (input.numel() == 0 || !train.value_or(false) || p == 0) {
    return std::make_tuple(input.clone(), Tensor::ones_like(input, DType::Bool));
  }
  if (p < 0 || p > 1) {
    TP_THROW(ValueError, "native_dropout: p must be in [0, 1]");
  }

  const float p_comp = static_cast<float>(1.0 - p);
  const float scale = p_comp == 0.0f ? 0.0f : 1.0f / p_comp;

  Tensor input_c = input.contiguous();
  Tensor output = Tensor::empty(static_cast<std::vector<int64_t>>(input_c.shape()),
                                input_c.dtype(), input_c.device());
  Tensor mask = Tensor::empty(static_cast<std::vector<int64_t>>(input_c.shape()),
                              DType::Bool, input_c.device());
  const int64_t numel = input_c.numel();
  const int64_t seed = static_cast<int64_t>(default_generator().current_seed());
  const int64_t base_offset = g_rng_offset.fetch_add(numel);
  const uint32_t numel_u32 = static_cast<uint32_t>(numel);

  auto pso = lib.getPipelineStateForFunc("dropout_fwd_" +
                                         scalarToMetalTypeString(input_c.dtype()));
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      stream->endKernelCoalescing();
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, output, mask, input_c,
                  std::array<float, 2>{p_comp, scale},
                  std::array<long, 2>{seed, base_offset});
      mtl_setBytes(encoder, numel_u32, 5);
      mtl_dispatch1DJob(encoder, pso, (numel + 3) / 4);
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
  return std::make_tuple(output, mask);
}

Tensor native_dropout_backward_kernel(const Tensor& grad_output,
                                      const Tensor& mask, double scale) {
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(mask.shape()),
                                grad_output.dtype(), grad_output.device());
  if (grad_output.numel() == 0) {
    return result;
  }
  auto pso = lib.getPipelineStateForFunc("dropout_bwd_" +
                                         scalarToMetalTypeString(grad_output.dtype()));
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      stream->endKernelCoalescing();
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, result, grad_output, mask, static_cast<float>(scale));
      mtl_dispatch1DJob(encoder, pso, grad_output.numel());
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
  return result;
}

TENSORPLAY_LIBRARY_IMPL(MPS, DropoutKernels) {
  m.impl("native_dropout", native_dropout_kernel);
  m.impl("native_dropout_backward", native_dropout_backward_kernel);
}

} // namespace mps
} // namespace tensorplay