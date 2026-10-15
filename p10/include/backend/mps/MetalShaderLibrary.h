#pragma once

#include <Metal/Metal.h>

#include <string>
#include <unordered_map>

#include "Macros.h"

namespace tensorplay {
namespace mps {

typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;

// Loads a precompiled Metal shader library (embedded as a byte array by
// tools/write_metallib_headers.py) and caches compute pipeline states for
// the shader functions used by the MPS backend kernels.
class P10_API MetalShaderLibrary {
 public:
  explicit MetalShaderLibrary(const unsigned char* data, unsigned long size);
  ~MetalShaderLibrary();

  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  MetalShaderLibrary& operator=(const MetalShaderLibrary&) = delete;

  // Returns (and caches) the compute pipeline state for `fname`.
  MTLComputePipelineState_t getPipelineStateForFunc(const std::string& fname);

  // Returns a new (retained) function for `fname`; caller must release it.
  MTLFunction_t getFunction(const std::string& fname);

  // The process-lifetime bundled shader library (Quantized.metal).
  static MetalShaderLibrary& getBundledLibrary();

 private:
  MTLLibrary_t library_ = nil;
  std::unordered_map<std::string, MTLComputePipelineState_t> pipeline_cache_;
};

} // namespace mps
} // namespace tensorplay